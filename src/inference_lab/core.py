"""Dense-Qwen3 inference with no, dynamic, or static KV cache.

Batches are equal-length and unpadded. Both baselines project only the last
hidden position to vocabulary logits. Positions are explicit; Transformers
constructs the causal mask, including unused static-cache slots.
"""

from dataclasses import dataclass
import time

import torch
from transformers import DynamicCache, StaticCache

CACHE_MODES = ("no_cache", "dynamic", "static")


@dataclass
class DecodeResult:
    output_ids: torch.Tensor
    prefill_first_token_ms: float
    decode_total_ms: float
    decode_steps: int
    decode_step_ms: float | None
    total_ms: float
    output_tokens: int
    output_tokens_per_second: float
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    kv_cache_mib: float
    cache_tokens: int
    cache_mode: str
    baseline_allocated_mib: float | None
    extra_peak_allocated_mib: float | None
    cache_capacity_tokens: int


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def cache_tensors(cache):
    return [] if cache is None else [(key, value) for key, value in cache]


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for pair in cache_tensors(cache) for t in pair)


def expected_kv_bytes(config, batch_size, sequence_length, dtype):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return (2 * config.num_hidden_layers * batch_size * sequence_length
            * config.num_key_value_heads * head_dim * torch.empty((), dtype=dtype).element_size())


def make_cache(model, mode, capacity):
    if mode == "no_cache":
        return None
    if mode == "dynamic":
        return DynamicCache(config=model.config)
    if mode == "static":
        return StaticCache(config=model.config, max_cache_len=capacity)
    raise ValueError(f"Unknown cache mode: {mode}")


def forward_last(model, input_ids, *, use_cache, past_key_values=None, cache_position=None):
    return model(input_ids=input_ids, use_cache=use_cache, past_key_values=past_key_values,
                 cache_position=cache_position, logits_to_keep=1)


def resolve_mode(use_cache, cache_mode):
    mode = cache_mode or ("no_cache" if use_cache is False else "dynamic")
    if mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}")
    if use_cache is not None and use_cache != (mode != "no_cache"):
        raise ValueError("use_cache and cache_mode disagree")
    return mode


@torch.inference_mode()
def decode(model, input_ids, *, new_tokens, use_cache=None, cache_mode=None):
    """Generate N tokens: prefill produces 1, decode produces N-1.

    Wall-clock phase timings include sampling and Python work. Model loading,
    tokenization, transfer and warmup are excluded. A fresh cache is used on
    every call; its lazy allocation is included in prefill. EOS is ignored.
    """
    if input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("input_ids must have nonempty [batch, sequence] dimensions")
    if new_tokens < 1:
        raise ValueError("new_tokens must be at least 1")
    mode = resolve_mode(use_cache, cache_mode)
    cached = mode != "no_cache"
    model.eval()
    device = input_ids.device
    batch, prompt_length = input_ids.shape
    capacity = prompt_length + new_tokens - 1
    if capacity > model.config.max_position_embeddings:
        raise ValueError("Requested sequence exceeds the model context limit")
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    tokens = torch.empty((batch, prompt_length + new_tokens), dtype=input_ids.dtype, device=device)
    tokens[:, :prompt_length] = input_ids
    positions = torch.arange(capacity, device=device)
    cache = make_cache(model, mode, capacity)
    synchronize(device)

    start = time.perf_counter()
    with torch.profiler.record_function("prefill"):
        output = forward_last(model, input_ids, use_cache=cached,
                              past_key_values=cache, cache_position=positions[:prompt_length])
        tokens[:, prompt_length] = output.logits[:, -1].argmax(dim=-1)
        cache = output.past_key_values if cached else None
        del output
    synchronize(device)
    prefill_end = time.perf_counter()
    with torch.profiler.record_function("decode"):
        for step in range(1, new_tokens):
            position = prompt_length + step
            current = tokens[:, position - 1:position] if cached else tokens[:, :position]
            current_positions = positions[position - 1:position] if cached else positions[:position]
            output = forward_last(model, current, use_cache=cached,
                                  past_key_values=cache, cache_position=current_positions)
            tokens[:, position] = output.logits[:, -1].argmax(dim=-1)
            cache = output.past_key_values if cached else None
            del output
    synchronize(device)
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    total_ms = (end - start) * 1000
    decode_ms = (end - prefill_end) * 1000
    steps = new_tokens - 1
    return DecodeResult(
        output_ids=tokens, prefill_first_token_ms=(prefill_end - start) * 1000,
        decode_total_ms=decode_ms, decode_steps=steps,
        decode_step_ms=decode_ms / steps if steps else None, total_ms=total_ms,
        output_tokens=batch * new_tokens,
        output_tokens_per_second=batch * new_tokens * 1000 / total_ms,
        peak_allocated_mib=peak / 2**20 if peak is not None else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        kv_cache_mib=cache_bytes(cache) / 2**20, cache_tokens=capacity if cached else 0,
        cache_mode=mode, baseline_allocated_mib=baseline / 2**20 if baseline is not None else None,
        extra_peak_allocated_mib=(peak - baseline) / 2**20 if peak is not None else None,
        cache_capacity_tokens=capacity if cached else 0,
    )


@torch.inference_mode()
def diagnose_cache(model, fixed_ids, *, prompt_length, cache_mode="dynamic", atol=1e-4, rtol=1e-4):
    """Teacher-forced comparison: failures remain failures in the report."""
    if fixed_ids.ndim != 2 or not 1 <= prompt_length <= fixed_ids.shape[1]:
        raise ValueError("Require 1 <= prompt_length <= fixed sequence length")
    if fixed_ids.shape[1] > model.config.max_position_embeddings:
        raise ValueError("Fixed sequence exceeds the model context limit")
    if cache_mode not in ("dynamic", "static"):
        raise ValueError("Correctness comparison requires dynamic or static cache")
    model.eval()
    cache = make_cache(model, cache_mode, fixed_ids.shape[1])
    positions = torch.arange(fixed_ids.shape[1], device=fixed_ids.device)
    steps = []
    for length in range(prompt_length, fixed_ids.shape[1] + 1):
        full = forward_last(model, fixed_ids[:, :length], use_cache=False,
                            cache_position=positions[:length]).logits.float()  # full-prefix logits
        first = length == prompt_length
        current = fixed_ids[:, :length] if first else fixed_ids[:, length - 1:length]  # current token
        position = positions[:length] if first else positions[length - 1:length]
        result = forward_last(model, current, use_cache=True, past_key_values=cache, cache_position=position)
        cache = result.past_key_values
        candidate = result.logits.float()
        error = (full - candidate).abs()
        close = torch.isclose(candidate, full, atol=atol, rtol=rtol)
        steps.append({
            "sequence_length": length, "passed": bool(close.all().item()),
            "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
            "rmse": error.square().mean().sqrt().item(),
            "mismatch_fraction": (~close).float().mean().item(),
            "top1_agreement": (candidate.argmax(-1) == full.argmax(-1)).float().mean().item(),
            "cached_input_tokens": current.shape[1],
        })
    return {
        "passed": all(s["passed"] for s in steps), "comparisons": len(steps),
        "max_abs_error": max(s["max_abs_error"] for s in steps),
        "atol": atol, "rtol": rtol, "cache_tokens": int(cache.get_seq_length()),
        "cache_mode": cache_mode, "steps": steps,
        "interpretation": "Elementwise logits test; top1 agreement alone is not a correctness proof.",
    }


def check_cache(model, fixed_ids, *, prompt_length, atol=1e-4, rtol=1e-4, cache_mode="dynamic"):
    report = diagnose_cache(model, fixed_ids, prompt_length=prompt_length,
                            cache_mode=cache_mode, atol=atol, rtol=rtol)
    if not report["passed"]:
        first = next(s for s in report["steps"] if not s["passed"])
        raise AssertionError(f"Cache logits differ at length {first['sequence_length']}: {first}. "
                             "Use check --diagnostic to retain the report; do not silently widen tolerance.")
    return report
