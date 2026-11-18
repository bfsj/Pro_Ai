"""Compare BF16 and groupwise INT4 on Qwen3 decoder projections (PyTorch 2.11)."""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from inference_lab.core import decode
from inference_lab.int4 import PackedLinear, quantize_decoder
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["bf16", "int4"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--group-size", type=int, choices=[32, 64, 128], default=128)
    parser.add_argument("--mse-clip", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "bf16" and (args.group_size != 128 or args.mse_clip):
        parser.error("Quantization settings require --mode int4")
    torch.set_num_threads(4)
    torch.manual_seed(18)
    if args.mode == "int4":
        # Exactly representable integer weights expose nibble order or offset errors.
        probe = nn.Linear(128, 32, bias=False, device=args.device, dtype=torch.bfloat16)
        probe.weight.copy_(
            torch.arange(128, device=args.device).remainder(16).sub(8).repeat(32, 1)
        )
        x = torch.eye(128, device=args.device, dtype=torch.bfloat16)
        packed = PackedLinear(probe, group_size=args.group_size, mse_clip=args.mse_clip)
        torch.testing.assert_close(packed(x), probe(x), atol=0, rtol=0)
        del packed
        del probe, x
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(args.device)
        .eval()
    )
    if model.config.model_type != "qwen3":
        raise ValueError("Expected dense Qwen3")
    converted = 0
    if args.mode == "int4":
        converted = quantize_decoder(
            model, group_size=args.group_size, mse_clip=args.mse_clip
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    corpus = Path(__file__).with_name("quantization_text.txt").read_bytes()
    nll_sum = 0.0
    scored = 0
    for paragraph in corpus.decode("utf8").splitlines():
        if not paragraph.strip():
            continue
        ids = tokenizer(
            paragraph, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(args.device)
        logits = model(ids, use_cache=False).logits[0, :-1].float()
        nll_sum += nn.functional.cross_entropy(
            logits, ids[0, 1:], reduction="sum"
        ).item()
        scored += ids.size(1) - 1
        del logits, ids
    report = dict(
        mode=args.mode,
        group_size=args.group_size if converted else None,
        mse_clip=args.mse_clip,
        device=args.device,
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        hardware=torch.cuda.get_device_name()
        if args.device == "cuda"
        else platform.machine(),
        converted_layers=converted,
        tokens_scored=scored,
        text_sha256=hashlib.sha256(corpus).hexdigest(),
        perplexity=math.exp(nll_sum / scored),
        stored_model_mib=sum(
            t.numel() * t.element_size()
            for t in [*model.parameters(), *model.buffers()]
        )
        / 2**20,
        samples=[],
    )
    for batch in (1, 4):
        for length in (128, 256, 512, 1024):
            ids = torch.randint(
                10,
                20000,
                (batch, length),
                generator=torch.Generator().manual_seed(batch * 10000 + length),
            ).to(args.device)
            decode(model, ids, new_tokens=4, cache_mode="dynamic")
            for _ in range(3):
                result = decode(model, ids, new_tokens=32, cache_mode="dynamic")
                report["samples"].append(
                    dict(
                        batch=batch,
                        length=length,
                        ttft_ms=result.prefill_first_token_ms,
                        step_ms=result.decode_step_ms,
                        tokens_s=result.output_tokens_per_second,
                        peak_gpu_mib=result.peak_allocated_mib,
                    )
                )
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2)


if __name__ == "__main__":
    main()
