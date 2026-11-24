"""Variable-length prefill for the dense, full-attention Qwen3 decoder."""
from dataclasses import dataclass
from itertools import accumulate
from time import perf_counter

import torch
from torch.nn.functional import pad, scaled_dot_product_attention
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

from .core import make_cache, synchronize


def _padded(model, sequences):
    lengths = [s.numel() for s in sequences]
    width = max(lengths)
    ids = torch.stack([pad(s, (0, width - s.numel())) for s in sequences])
    ends = ids.new_tensor(lengths) - 1
    valid = torch.arange(width, device=ids.device) <= ends[:, None]
    hidden = model.model(ids, attention_mask=valid, use_cache=False).last_hidden_state
    return model.lm_head(hidden[torch.arange(len(sequences), device=ids.device), ends])


def _packed(model, sequences, cache=None, *, flash=False):
    lengths = [s.numel() for s in sequences]
    ids = torch.cat(sequences)
    positions = torch.cat([torch.arange(n, device=ids.device) for n in lengths])
    width = max(lengths)
    offsets = ids.new_tensor([0, *accumulate(lengths)], dtype=torch.int32)
    if flash:
        try:
            from torch.nn.attention.varlen import varlen_attn
        except ImportError as exc:
            raise RuntimeError("Packed CUDA requires torch.nn.attention.varlen (tested on torch 2.11)") from exc
    if cache is not None:
        # Decode uses a left-padded HF cache. Map each packed token exactly once.
        slots = torch.cat([torch.arange((i + 1) * width - n, (i + 1) * width,
                                        device=ids.device) for i, n in enumerate(lengths)])
    base = model.model
    hidden = base.embed_tokens(ids)[None]
    cos, sin = base.rotary_emb(hidden, positions[None])
    for layer_idx, layer in enumerate(base.layers):
        attn = layer.self_attn
        norm = layer.input_layernorm(hidden)
        shape = (1, ids.numel(), -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(norm).view(shape)).transpose(1, 2)
        k = attn.k_norm(attn.k_proj(norm).view(shape)).transpose(1, 2)
        v = attn.v_proj(norm).view(shape).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if cache is not None:
            cache.update(_cache_rows(k, slots, len(lengths), width),
                         _cache_rows(v, slots, len(lengths), width), layer_idx)
        if flash:
            qt, kt, vt = (t[0].transpose(0, 1) for t in (q, k, v))
            out = varlen_attn(qt, kt, vt, offsets, offsets, width, width,
                              scale=attn.scaling, window_size=(-1, 0)).reshape(1, ids.numel(), -1)
        else:
            parts = [
                scaled_dot_product_attention(qi, ki, vi, is_causal=True,
                                             scale=attn.scaling, enable_gqa=True)
                for qi, ki, vi in zip(*(t.split(lengths, dim=2) for t in (q, k, v)))
            ]
            out = torch.cat(parts, dim=2).transpose(1, 2).reshape(1, ids.numel(), -1)
        hidden = hidden + attn.o_proj(out)
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    ends = offsets[1:].long() - 1
    return model.lm_head(base.norm(hidden[0, ends]))


@torch.inference_mode()
def _validate(model, sequences, new_tokens):
    if model.training or model.config.model_type != "qwen3" or any(
        kind != "full_attention" for kind in model.config.layer_types
    ):
        raise ValueError("Requires an eval-mode full-attention Qwen3 decoder")
    if new_tokens < 1 or not sequences or any(
        s.ndim != 1 or s.numel() < 1 or s.numel() + new_tokens - 1 > model.config.max_position_embeddings
        for s in sequences
    ):
        raise ValueError("Expected nonempty prompts and positive generation length within the context limit")


def _buckets(sequences):
    buckets = {}
    for i, s in enumerate(sequences):
        buckets.setdefault((s.numel() - 1).bit_length(), []).append(i)
    return buckets.values()


@torch.inference_mode()
def prefill_logits(model, sequences, method):
    """Return last-valid-token logits in original request order, without a KV cache."""
    _validate(model, sequences, 1)
    if method == "padded":
        return _padded(model, sequences)
    if method == "packed":
        return _packed(model, sequences)
    if method != "bucketed":
        raise ValueError(f"Unknown prefill method: {method}")
    outputs = [None] * len(sequences)
    for indices in _buckets(sequences):
        logits = _padded(model, [sequences[i] for i in indices])
        for index, row in zip(indices, logits):
            outputs[index] = row
    return torch.stack(outputs)


def _cache_rows(tensor, slots, batch, width):
    values = tensor[0].transpose(0, 1)
    rows = values.new_zeros(batch * width, *values.shape[1:])
    rows.index_copy_(0, slots, values)
    return rows.unflatten(0, (batch, width)).transpose(1, 2)


@dataclass
class BatchGeneration:
    token_ids: torch.Tensor
    first_logits: torch.Tensor
    first_token_ms: float
    total_ms: float
    prefill_backend: str
    method: str


@torch.inference_mode()
def generate_batch(model, sequences, method, new_tokens, *, pack_at_padding=None):
    """Fixed-length greedy generation. TTFT measures readiness of the whole batch."""
    _validate(model, sequences, new_tokens)
    device = sequences[0].device
    synchronize(device)
    start = perf_counter()
    if method == "auto":
        if pack_at_padding is None or not 0 < pack_at_padding <= 1:
            raise ValueError("Auto requires an explicit pack_at_padding threshold in (0, 1]")
        lengths = [s.numel() for s in sequences]
        capacity = len(lengths) * max(lengths)
        wasted = capacity - sum(lengths)
        flash_dtype = model.dtype in (torch.float16, torch.bfloat16)
        pack = device.type == "cuda" and flash_dtype and wasted >= capacity * pack_at_padding
        method = "packed" if pack else "padded"
    elif pack_at_padding is not None:
        raise ValueError("pack_at_padding is only used with the auto method")
    if method not in ("padded", "bucketed", "packed"):
        raise ValueError(f"Unknown generation method: {method}")
    flash = method == "packed" and device.type == "cuda"
    if flash and model.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Packed CUDA requires float16 or bfloat16")
    groups = _buckets(sequences) if method == "bucketed" else [range(len(sequences))]
    states = []
    first_rows = [None] * len(sequences)
    generated = torch.empty((len(sequences), new_tokens), dtype=torch.long, device=device)
    for indices in groups:
        prompts = [sequences[i] for i in indices]
        lengths = [s.numel() for s in prompts]
        width = max(lengths)
        logical = prompts[0].new_tensor(lengths, dtype=torch.long)
        mask = None
        if min(lengths) != width:
            mask = torch.arange(width, device=device) >= (width - logical[:, None])
        cache = make_cache(model, "dynamic", width + new_tokens - 1)
        if method == "packed":
            logits = _packed(model, prompts, cache, flash=flash)
        else:
            ids = torch.stack([pad(s, (width - s.numel(), 0)) for s in prompts])
            pos = None if mask is None else (mask.long().cumsum(1) - 1).clamp_min_(0)
            logits = model(ids, position_ids=pos, attention_mask=mask, past_key_values=cache,
                           use_cache=True, logits_to_keep=1).logits[:, -1]
        current = logits.argmax(-1)
        rows = logical.new_tensor(list(indices))
        generated[:, 0].index_copy_(0, rows, current)
        for i, value in zip(indices, logits.unbind()):
            first_rows[i] = value
        states.append((rows, cache, mask, logical, current))
    synchronize(device)
    first = perf_counter()
    for rows, cache, mask, logical, current in states:
        for step in range(1, new_tokens):
            if mask is not None:
                mask = pad(mask, (0, 1), value=True)
            logits = model(current[:, None], attention_mask=mask, position_ids=logical[:, None],
                           past_key_values=cache, use_cache=True, logits_to_keep=1).logits[:, -1]
            current = logits.argmax(-1)
            generated[:, step].index_copy_(0, rows, current)
            logical = logical + 1
    synchronize(device)
    end = perf_counter()
    backend = ("torch-varlen-flash" if flash else "segmented-sdpa") if method == "packed" else model.config._attn_implementation
    return BatchGeneration(generated, torch.stack(first_rows), (first - start) * 1000,
                           (end - start) * 1000, backend, method)
