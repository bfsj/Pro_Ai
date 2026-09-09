"""Equal-length, unpadded Qwen3 decoding. No serving or scheduling abstraction."""

from dataclasses import dataclass
import time

import torch


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


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def cache_tensors(cache):
    if cache is None:
        return []
    return [(key, value) for key, value in cache]


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for pair in cache_tensors(cache) for t in pair)


def expected_kv_bytes(config, batch_size, sequence_length, dtype):
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return (2 * config.num_hidden_layers * batch_size * sequence_length
            * config.num_key_value_heads * head_dim * torch.empty((), dtype=dtype).element_size())


def forward_last(model, input_ids, *, use_cache, past_key_values=None):
    # All inputs have equal length and no padding. Qwen3 derives positions from
    # past_key_values; its attention backend constructs the causal mask.
    return model(
        input_ids=input_ids,
        use_cache=use_cache,
        past_key_values=past_key_values,
        logits_to_keep=1,
    )


@torch.inference_mode()
def decode(model, input_ids, *, new_tokens, use_cache):
    """Generate exactly N tokens, including the first token from prefill.

    Greedy decoding intentionally ignores EOS for equal-work benchmarking.
    Wall-clock timings synchronize at the prefill and decode boundaries, not
    on each token. Tokenization, model loading and input transfer are excluded.
    """
    if input_ids.ndim != 2 or min(input_ids.shape) < 1:
        raise ValueError("input_ids must have nonempty [batch, sequence] dimensions")
    if new_tokens < 1:
        raise ValueError("new_tokens must be at least 1")
    model.eval()
    device = input_ids.device
    batch, prompt_length = input_ids.shape
    if prompt_length + new_tokens - 1 > model.config.max_position_embeddings:
        raise ValueError("Requested sequence exceeds the model context limit")
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Preallocate the generated sequence so concatenation is not another variable.
    tokens = torch.empty((batch, prompt_length + new_tokens), dtype=input_ids.dtype, device=device)
    tokens[:, :prompt_length] = input_ids
    synchronize(device)

    start = time.perf_counter()
    with torch.profiler.record_function("prefill"):
        output = forward_last(model, input_ids, use_cache=use_cache)
        tokens[:, prompt_length] = output.logits[:, -1].argmax(dim=-1)
        cache = output.past_key_values if use_cache else None
    synchronize(device)
    prefill_end = time.perf_counter()

    with torch.profiler.record_function("decode"):
        for step in range(1, new_tokens):
            position = prompt_length + step
            current = tokens[:, position - 1:position] if use_cache else tokens[:, :position]
            output = forward_last(model, current, use_cache=use_cache, past_key_values=cache)
            tokens[:, position] = output.logits[:, -1].argmax(dim=-1)
            cache = output.past_key_values if use_cache else None
    synchronize(device)
    end = time.perf_counter()

    total_ms = (end - start) * 1000
    decode_ms = (end - prefill_end) * 1000
    steps = new_tokens - 1
    return DecodeResult(
        output_ids=tokens,
        prefill_first_token_ms=(prefill_end - start) * 1000,
        decode_total_ms=decode_ms,
        decode_steps=steps,
        decode_step_ms=decode_ms / steps if steps else None,
        total_ms=total_ms,
        output_tokens=batch * new_tokens,
        output_tokens_per_second=batch * new_tokens * 1000 / total_ms,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        kv_cache_mib=cache_bytes(cache) / 2**20,
        cache_tokens=int(cache.get_seq_length()) if cache is not None else 0,
    )


@torch.inference_mode()
def check_cache(model, fixed_ids, *, prompt_length, atol=1e-4, rtol=1e-4):
    """Compare full-prefix vs cached logits on identical teacher-forced tokens."""
    if fixed_ids.ndim != 2 or not 1 <= prompt_length <= fixed_ids.shape[1]:
        raise ValueError("Require 1 <= prompt_length <= fixed sequence length")
    model.eval()
    cache = None
    max_abs_error = 0.0
    comparisons = 0
    for length in range(prompt_length, fixed_ids.shape[1] + 1):
        full = forward_last(model, fixed_ids[:, :length], use_cache=False).logits.float()
        current = fixed_ids[:, :length] if cache is None else fixed_ids[:, length - 1:length]
        cached = forward_last(model, current, use_cache=True, past_key_values=cache)
        cache = cached.past_key_values
        candidate = cached.logits.float()
        max_abs_error = max(max_abs_error, (full - candidate).abs().max().item())
        torch.testing.assert_close(candidate, full, atol=atol, rtol=rtol)
        comparisons += 1
    return {"passed": True, "comparisons": comparisons, "max_abs_error": max_abs_error,
            "atol": atol, "rtol": rtol, "cache_tokens": int(cache.get_seq_length())}
