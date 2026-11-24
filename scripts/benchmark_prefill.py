"""Check per-request logits and measure local batch prefill latency."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM

from inference_lab.core import synchronize
from inference_lab.prefill import prefill_logits


@torch.inference_mode()
def benchmark(model, lengths, repeats, seed):
    device = model.device
    rng = torch.Generator().manual_seed(seed)
    seqs = [torch.randint(model.config.vocab_size, (n,), generator=rng).to(device) for n in lengths]
    reference = torch.cat([model(s[None], use_cache=False, logits_to_keep=1).logits[:, 0]
                           for s in seqs])
    methods = ["padded", "bucketed", "packed"]
    errors, times = {}, {m: [] for m in methods}
    for method in methods:
        actual = prefill_logits(model, seqs, method)
        errors[method] = (actual - reference).abs().max().item()
        torch.testing.assert_close(actual, reference, atol=1e-3, rtol=1e-4)
    for i in range(repeats):
        for method in methods[i % 3:] + methods[:i % 3]:
            synchronize(device)
            start = time.perf_counter()
            prefill_logits(model, seqs, method)
            synchronize(device)
            times[method].append(1000 * (time.perf_counter() - start))
    result = {}
    for method, ms in times.items():
        result[method] = dict(batch_ms=ms, median_ms=statistics.median(ms),
                              p95_ms=sorted(ms)[math.ceil(.95 * repeats) - 1],
                              useful_tokens_per_s=sum(lengths) * 1000 / statistics.mean(ms),
                              max_logit_error=errors[method])
    return dict(lengths=lengths, methods=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.repeats, args.threads) < 1 or args.output.exists():
        parser.error("Require positive repeats/threads and a new output file")
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                 attn_implementation="sdpa").to(args.device).eval()
    report = dict(torch=torch.__version__, transformers=transformers.__version__, model=args.model,
                  device=args.device, dtype="float32", threads=args.threads, seed=42, batches=[])
    for lengths in ([64, 64, 64, 64], [60, 67, 62, 67], [8, 64, 24, 160]):
        report["batches"].append(benchmark(model, lengths, args.repeats, 42))
    with args.output.open("x", encoding="utf8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
