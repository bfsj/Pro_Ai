"""Paired RTX GPU ablations and separate CUDA-event diagnostics for batch prefill."""
import argparse
from contextlib import ExitStack, nullcontext
from functools import wraps
import json
from pathlib import Path
import random
from unittest.mock import patch

import torch
import torch.nn.functional as functional
import transformers
from transformers import AutoModelForCausalLM, DynamicCache
from torch.nn.attention import varlen

from benchmark_batch_generation import gpu_snapshot, make_prompts
from inference_lab import prefill


SHAPES = ((128,) * 8, (256,) * 4, (96, 96, 112, 112, 144, 144, 160, 160),
          (8, 16, 24, 32, 40, 48, 88, 768))


def cache_view(tensor, slots, batch, width):
    # Only valid when every request fills its entire row in original order.
    values = tensor[0].transpose(0, 1)
    return values.unflatten(0, (batch, width)).transpose(1, 2)


def dense_attention(q, k, v, cu_q, cu_k, max_q, max_k, *, scale, window_size):
    # Keep packed projections, RoPE, KV handoff and sampling. Swap only attention.
    batch = cu_q.numel() - 1
    assert q.size(0) == batch * max_q and k.size(0) == batch * max_k and window_size == (-1, 0)
    q, k, v = (t.unflatten(0, (batch, max_q)).transpose(1, 2) for t in (q, k, v))
    out = functional.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True, enable_gqa=True)
    return out.transpose(1, 2).flatten(0, 1)


def online_layout(sequences, need_slots):
    lengths = [s.numel() for s in sequences]
    width = max(lengths)
    ids = torch.cat(sequences)
    flat = torch.arange(ids.numel(), device=ids.device)
    if min(lengths) == width:
        offsets = torch.arange(0, ids.numel() + 1, width, device=ids.device, dtype=torch.int32)
        return lengths, ids, flat % width, width, offsets, flat if need_slots else None
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    offsets = ids.new_tensor(boundaries, dtype=torch.int32)
    row = torch.searchsorted(offsets[1:], flat.int(), right=True)
    starts = offsets[row]
    positions = flat - starts
    slots = (row + 1) * width - offsets[row + 1] + flat if need_slots else None
    return lengths, ids, positions, width, offsets, slots


def online_with_view():
    stack = ExitStack()
    stack.enter_context(patch.object(prefill, "_pack_inputs", online_layout))
    stack.enter_context(patch.object(prefill, "_cache_rows", cache_view))
    return stack


def variants(prompts, online):
    # Precomputation applies only to this exact batch; it models an upstream
    # provider of packed inputs, not free preprocessing of arbitrary requests.
    layout = prefill._pack_inputs(prompts, True)
    options = {
        "padded": nullcontext,
        "packed": nullcontext,
    }
    uniform = min(layout[0]) == max(layout[0])
    if online:
        rebuilt = online_layout(prompts, True)
        for i in (1, 2, 4, 5):
            torch.testing.assert_close(rebuilt[i], layout[i], atol=0, rtol=0)
        options["online_layout"] = lambda: patch.object(prefill, "_pack_inputs", online_layout)
        if uniform:
            options["online_with_view"] = online_with_view
    else:
        options["prepared_input"] = lambda: patch.object(prefill, "_pack_inputs", lambda *unused: layout)
        if uniform:
            options.update(cache_view=lambda: patch.object(prefill, "_cache_rows", cache_view),
                           dense_attention=lambda: patch.object(varlen, "varlen_attn", dense_attention))
    return options


def orders(names, repeats, seed):
    rng = random.Random(seed)
    cycle = []
    for _ in range(repeats):
        if not cycle:
            order = list(names)
            rng.shuffle(order)
            rotations = [order[i:] + order[:i] for i in range(len(order))]
            cycle = rotations + [row[::-1] for row in rotations]
            rng.shuffle(cycle)
        yield cycle.pop()


def run(model, prompts, name, options, steps):
    with options[name]():
        return prefill.generate_batch(model, prompts, "padded" if name == "padded" else "packed", steps)


def check_changes(model, lengths, seed, seeds, online):
    results = []
    for i in range(seeds):
        prompts = make_prompts(model, lengths, seed + i)
        options = variants(prompts, online)
        base = run(model, prompts, "packed", options, 4)
        checks = {}
        for name in (key for key in options if key != "packed"):
            sample = run(model, prompts, name, options, 4)
            diff = (sample.first_logits.float() - base.first_logits.float()).abs().max().item()
            equal = sample.token_ids.eq(base.token_ids).all(dim=1)
            if name not in ("padded", "dense_attention") and (diff or not bool(equal.all())):
                raise AssertionError(f"{name} changed outputs: {lengths}, seed {seed+i}, max error {diff}")
            checks[name] = dict(max_first_logit_difference=diff, matched_sequences=int(equal.sum()))
        results.append(dict(seed=seed+i, checks=checks))
    return results


def diagnostic(model, prompts, method):
    # Time disjoint stream intervals. Events include launch gaps and can perturb
    # the run. The A/B runs never install these wrappers or allocate events.
    layers = len(model.model.layers)
    targets = [(DynamicCache, "update", "cache_update", layers)]
    if method == "packed":
        targets += [(prefill, "_pack_inputs", "pack_inputs", 1),
                    (prefill, "_cache_rows", "cache_rows", layers * 2),
                    (varlen, "varlen_attn", "attention", layers)]
    else:
        targets.append((functional, "scaled_dot_product_attention", "attention", layers))
    event_pairs = {}
    observed = {}

    def wrap(fn, label, count):
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                  for _ in range(count)]
        for start, end in events:
            start.record()
            end.record()
        event_pairs[label] = events
        observed[label] = 0
        calls = iter(events)

        @wraps(fn)
        def measured(*args, **kwargs):
            start, end = next(calls)
            observed[label] += 1
            start.record()
            out = fn(*args, **kwargs)
            end.record()
            return out
        return measured

    with ExitStack() as stack:
        for owner, attr, label, count in targets:
            stack.enter_context(patch.object(owner, attr, wrap(getattr(owner, attr), label, count)))
        torch.cuda.synchronize()
        prefill.generate_batch(model, prompts, method, 1)
    torch.cuda.synchronize()
    if any(observed[label] != len(pairs) for label, pairs in event_pairs.items()):
        raise RuntimeError(f"Profiler hooks missed expected calls: {observed}")
    return {name: sum(a.elapsed_time(b) for a, b in pairs) for name, pairs in event_pairs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--check-seeds", type=int, default=6)
    parser.add_argument("--online", action="store_true", help="Time rebuilding packed inputs on every call")
    args = parser.parse_args()
    if args.output.exists() or min(args.rounds, args.check_seeds) < 1:
        parser.error("Use positive counts and a fresh output file")
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda().eval()
    result = dict(device=torch.cuda.get_device_name(), torch=str(torch.__version__), dtype="float16",
                  transformers=transformers.__version__, model=Path(args.model).name, seed=args.seed,
                  layers=len(model.model.layers), rounds=args.rounds, online=args.online, shapes=[])
    for lengths in SHAPES:
        prompts = make_prompts(model, lengths, args.seed)
        options = variants(prompts, args.online)
        before = gpu_snapshot()
        checks = check_changes(model, lengths, args.seed, args.check_seeds, args.online)
        for name in options:
            run(model, prompts, name, options, 1)
        timings = []
        for order in orders(options, args.rounds, args.seed):
            timings.append({name: run(model, prompts, name, options, 1).first_token_ms for name in order})
        diagnostics = {}
        if not args.online:
            diagnostics = {m: [] for m in ("padded", "packed")}
            for i in range(5):
                for method in (("padded", "packed") if i % 2 == 0 else ("packed", "padded")):
                    diagnostics[method].append(diagnostic(model, prompts, method))
        result["shapes"].append(dict(lengths=lengths, ttft_ms=timings, event_ms=diagnostics,
                                     correctness=checks, before=before, after=gpu_snapshot()))
        print(f"Finished B={len(lengths)} Lmax={max(lengths)}", flush=True)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
