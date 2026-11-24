"""Compare fixed-length generation with three prefill layouts and a solo reference."""
import argparse
from itertools import permutations
import json
import math
from pathlib import Path
from statistics import median
import subprocess

import torch
import transformers
from transformers import AutoModelForCausalLM

from inference_lab.prefill import generate_batch


METHODS = ("padded", "bucketed", "packed")
# Hold total prompt tokens at 1024 for all six workloads.
SHAPES = ((256,) * 4, (192, 224, 288, 320), (16, 32, 80, 896),
          (128,) * 8, (96, 96, 112, 112, 144, 144, 160, 160),
          (8, 16, 24, 32, 40, 48, 88, 768))


def gpu_snapshot():
    proc = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
                           "--format=csv,noheader"], capture_output=True, text=True, check=False)
    return proc.stdout.splitlines()


def make_prompts(model, lengths, seed):
    rng = torch.Generator().manual_seed(seed)
    return [torch.randint(model.config.vocab_size, (n,), generator=rng).to(model.device) for n in lengths]


@torch.inference_mode()
def solo_reference(model, prompts, steps):
    # Full-prefix recomputation shares neither packed attention nor our cache layout.
    first, outputs = [], []
    for prompt in prompts:
        ids = prompt[None]
        for step in range(steps):
            logits = model(ids, use_cache=False, logits_to_keep=1).logits[0, -1]
            if step == 0:
                first.append(logits)
            ids = torch.cat([ids, logits.argmax().view(1, 1)], dim=1)
        outputs.append(ids[0, -steps:])
    return torch.stack(first), torch.stack(outputs)


def evaluate(model, lengths, args):
    checks = []
    for seed in range(args.seed, args.seed + args.seeds):
        prompts = make_prompts(model, lengths, seed)
        first, tokens = solo_reference(model, prompts, args.new_tokens)
        for method in METHODS:
            run = generate_batch(model, prompts, method, args.new_tokens)
            matches = run.token_ids == tokens
            checks.append(dict(seed=seed, method=method, backend=run.prefill_backend,
                               sequences_equal=matches.all(1).tolist(),
                               tokens_equal=int(matches.sum()), first_token_equal=int(matches[:, 0].sum()),
                               max_first_logit_error=float((run.first_logits.float() - first.float()).abs().max())))
    prompts = make_prompts(model, lengths, args.seed)
    timings = {m: [] for m in METHODS}
    orders = list(permutations(METHODS))
    for method in METHODS:
        generate_batch(model, prompts, method, args.new_tokens)
    for repeat in range(args.repeats):
        for method in orders[repeat % len(orders)]:
            run = generate_batch(model, prompts, method, args.new_tokens)
            timings[method].append([run.first_token_ms, run.total_ms])
    summary = {}
    for method, samples in timings.items():
        ttft, end = zip(*samples)
        summary[method] = dict(batch_ttft_p50_ms=median(ttft),
                               batch_ttft_p95_ms=sorted(ttft)[math.ceil(.95 * len(ttft)) - 1],
                               end_to_end_p50_ms=median(end),
                               useful_prefill_tokens_per_s=sum(lengths) / (median(ttft) / 1000))
    return dict(lengths=lengths, checks=checks, timings_ms=timings, summary=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--seeds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.repeats, args.seeds, args.new_tokens) < 1:
        parser.error("Choose a fresh output path and positive counts")
    if args.device == "cuda" and args.dtype == "float32":
        parser.error("The packed GPU backend requires float16 or bfloat16")
    if args.device == "cpu" and args.dtype != "float32":
        parser.error("Use float32 for CPU reference runs")
    torch.set_num_threads(4)
    cuda = args.device == "cuda"
    report = dict(torch=str(torch.__version__), transformers=transformers.__version__, dtype=args.dtype,
                  device=torch.cuda.get_device_name() if cuda else "cpu", model=Path(args.model).name,
                  new_tokens=args.new_tokens, seeds=args.seeds, repeats=args.repeats,
                  timing_columns=["batch_first_token_ms", "total_ms"], shapes=[])
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa"
    ).to(args.device).eval()
    for lengths in SHAPES:
        before = gpu_snapshot() if cuda else []
        shape = evaluate(model, lengths, args)
        shape.update(gpus_before=before, gpus_after=gpu_snapshot() if cuda else [])
        report["shapes"].append(shape)
        print(lengths, shape["summary"], flush=True)
    with args.output.open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
