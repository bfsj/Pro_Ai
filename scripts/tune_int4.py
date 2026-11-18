"""Select INT4 settings on development text, then score the locked selection on new text."""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from statistics import median
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from inference_lab.core import decode
from inference_lab.int4 import quantize_decoder

ROOT = Path(__file__).resolve().parent


def read_text(tokenizer, filename, device):
    data = (ROOT / filename).read_bytes()
    rows = [
        tokenizer(line, return_tensors="pt", add_special_tokens=False).input_ids.to(
            device
        )
        for line in data.decode().splitlines()
        if line.strip()
    ]
    return rows, hashlib.sha256(data).hexdigest()


def score(model, rows):
    nll = []
    counts = []
    for ids in rows:
        logits = model(ids, use_cache=False).logits[:, :-1].float()
        nll.append(
            torch.nn.functional.cross_entropy(
                logits[0], ids[0, 1:], reduction="sum"
            ).item()
        )
        counts.append(ids.numel() - 1)
    return dict(
        ppl=math.exp(sum(nll) / sum(counts)),
        tokens=sum(counts),
        nll_by_doc=nll,
        tokens_by_doc=counts,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", required=True, choices=["cpu", "cuda"])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output path")
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dev, dev_hash = read_text(tokenizer, "quantization_text.txt", args.device)
    holdout, holdout_hash = read_text(
        tokenizer, "quantization_holdout.txt", args.device
    )

    def load(group, clip):
        m = (
            AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to(args.device)
            .eval()
        )
        if group:
            quantize_decoder(m, group_size=group, mse_clip=clip)
        return m

    sweep = []
    for group, clip in [
        (0, False),
        (128, False),
        (128, True),
        (64, False),
        (64, True),
        (32, False),
        (32, True),
    ]:
        m = load(group, clip)
        size = (
            sum(t.numel() * t.element_size() for t in [*m.parameters(), *m.buffers()])
            / 2**20
        )
        row = dict(group=group, clip=clip, mib=size, dev=score(m, dev))
        sweep.append(row)
        print(group, clip, row["dev"]["ppl"], flush=True)
        del m
        gc.collect()
    winner = min(sweep[1:], key=lambda r: r["dev"]["ppl"])
    ids = torch.randint(
        10, 20000, (1, 128), generator=torch.Generator().manual_seed(10128)
    ).to(args.device)
    for row in sweep:
        if row is not winner and row not in sweep[:2]:
            continue
        m = load(row["group"], row["clip"])
        row["holdout"] = score(m, holdout)
        decode(m, ids, new_tokens=4, cache_mode="dynamic")
        samples = [
            decode(m, ids, new_tokens=32, cache_mode="dynamic") for _ in range(3)
        ]
        row["tokens_s"] = [s.output_tokens_per_second for s in samples]
        row["median_tokens_s"] = median(row["tokens_s"])
        row["peak_gpu_mib"] = samples[-1].peak_allocated_mib
        del m
        gc.collect()
    result = dict(
        torch=torch.__version__,
        device=args.device,
        threads=torch.get_num_threads(),
        dev_sha256=dev_hash,
        holdout_sha256=holdout_hash,
        winner=[winner["group"], winner["clip"]],
        sweep=sweep,
    )
    with args.output.open("x", encoding="utf8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
