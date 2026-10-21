"""Isolated prefill diagnostic, not a performance optimization or correctness fix.

Run in a separate process: temporarily intercept SDPA to record layout and to
test whether making Q/K/V contiguous removes a prefill discrepancy. This does
not investigate incremental steps or establish low-precision model quality.
"""

import argparse
from contextlib import contextmanager

import torch
import torch.nn.functional as functional

from inference_lab.benchmark import environment, synthetic_ids
from inference_lab.cli import add_model_arguments, load_model
from inference_lab.core import forward_last, make_cache
from inference_lab.experiments import save_json


@contextmanager
def intercept_sdpa(capture, force_contiguous, uniform_explicit_mask=False):
    original = functional.scaled_dot_product_attention

    def wrapped(query, key, value, *args, **kwargs):
        if not capture:
            mask = kwargs.get("attn_mask", args[0] if args else None)
            capture.update(
                layout={name: {"shape": list(tensor.shape), "stride": list(tensor.stride()),
                               "contiguous": tensor.is_contiguous()}
                        for name, tensor in (("q", query), ("k", key), ("v", value))},
                mask_shape=list(mask.shape) if mask is not None else None,
                mask_dtype=str(mask.dtype) if mask is not None else None,
                is_causal=kwargs.get("is_causal", False),
                enable_gqa=kwargs.get("enable_gqa", False),
                first_layer_values=[tensor.detach().cpu().clone() for tensor in (query, key, value)],
            )
        if uniform_explicit_mask:
            if args or query.shape[-2] != key.shape[-2]:
                raise ValueError("Uniform-mask intervention is for equal-length prefill only")
            groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
            length = query.shape[-2]
            mask = torch.full((length, length), torch.finfo(query.dtype).min,
                              dtype=query.dtype, device=query.device).triu(1)
            kwargs.update(attn_mask=mask[None, None].expand(query.shape[0], 1, length, length).contiguous(),
                          is_causal=False, enable_gqa=False)
        if force_contiguous or uniform_explicit_mask:
            query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        return original(query, key, value, *args, **kwargs)

    functional.scaled_dot_product_attention = wrapped
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


@torch.inference_mode()
def probe(model, ids, *, force_contiguous=False, uniform_explicit_mask=False):
    results = {}
    reference = None
    first_values = None
    positions = torch.arange(ids.shape[1], device=ids.device)
    # Repeating the full forward helps distinguish ordinary run-to-run noise.
    for label, mode in (("no_cache", "no_cache"), ("no_cache_repeat", "no_cache"),
                        ("dynamic", "dynamic"), ("static", "static")):
        capture = {}
        cache = make_cache(model, mode, ids.shape[1])
        with intercept_sdpa(capture, force_contiguous, uniform_explicit_mask):
            logits = forward_last(model, ids, use_cache=mode != "no_cache",
                                  past_key_values=cache, cache_position=positions).logits.float()
        if not capture:
            raise RuntimeError("SDPA was not intercepted; use --attention sdpa")
        values = capture.pop("first_layer_values")
        if reference is None:
            reference, first_values = logits, values
        error = (logits - reference).abs()
        results[label] = dict(
            capture, max_abs_error=error.max().item(), mean_abs_error=error.mean().item(),
            top1_agreement=(logits.argmax(-1) == reference.argmax(-1)).float().mean().item(),
            first_layer_qkv_equal=[torch.equal(left, right) for left, right in zip(values, first_values)],
        )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.set_defaults(dtype="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--extra-ids", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.attention != "sdpa" or args.norm_backend != "torch":
        parser.error("This diagnostic requires --attention sdpa --norm-backend torch")
    model = load_model(args)
    ids = synthetic_ids(model.config.vocab_size, args.batch_size,
                        args.input_length + args.extra_ids, args.device, args.seed)[:, :args.input_length]
    result = {"environment": environment(), "configuration": vars(args),
              "original_layout": probe(model, ids),
              "force_contiguous_qkv": probe(model, ids, force_contiguous=True),
              "uniform_explicit_mask": probe(model, ids, uniform_explicit_mask=True),
              "scope": "Prefill only; layout intervention changes SDPA execution, not a production fix. "
                       "Does not establish incremental BF16 correctness or isolate a specific hardware kernel."}
    save_json(args.output, result)
    for intervention in ("original_layout", "force_contiguous_qkv", "uniform_explicit_mask"):
        print(intervention, {mode: row["max_abs_error"] for mode, row in result[intervention].items()})


if __name__ == "__main__":
    main()
