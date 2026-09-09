import argparse
from dataclasses import fields
import json
from pathlib import Path

import torch

from .benchmark import environment, run_benchmark, synthetic_ids
from .core import cache_bytes, cache_tensors, check_cache, decode, expected_kv_bytes, forward_last


def add_model_arguments(parser):
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)


def load_model(args):
    from transformers import AutoModelForCausalLM
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Use a CUDA-enabled PyTorch build on the GPU machine, or --device cpu --dtype float32.")
        torch.cuda.set_device(device.index if device.index is not None else 0)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support BF16; try --dtype float16.")
    if device.type == "cpu" and args.dtype != "float32":
        raise ValueError("Use --dtype float32 for the CPU smoke check.")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attention,
    ).to(device).eval()
    if model.config.model_type != "qwen3":
        raise ValueError("This starter lab supports dense Qwen3 models only.")
    return model


def parser():
    root = argparse.ArgumentParser(description="Single-GPU LLM inference learning lab")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="Print runtime and GPU information")
    inspect = commands.add_parser("inspect", help="Inspect logits, per-layer cache shapes and memory")
    generate = commands.add_parser("generate", help="Generate with an explicit greedy loop")
    check = commands.add_parser("check", help="Compare cached and uncached logits on fixed tokens")
    bench = commands.add_parser("benchmark", help="Run controlled cache/no-cache experiments")
    profile = commands.add_parser("profile", help="Export one separate PyTorch profiling trace")
    plot = commands.add_parser("plot", help="Plot a measured benchmark JSON file")
    for command in (inspect, generate, check, bench, profile):
        add_model_arguments(command)
    for command in (inspect, check, profile):
        command.add_argument("--input-length", type=int, default=128)
        command.add_argument("--batch-size", type=int, default=1)
    generate.add_argument("--prompt", default="用三句话解释什么是 GPU 显存。")
    generate.add_argument("--new-tokens", type=int, default=64)
    generate.add_argument("--no-cache", action="store_true")
    check.add_argument("--steps", type=int, default=8)
    check.add_argument("--atol", type=float, default=None)
    check.add_argument("--rtol", type=float, default=None)
    bench.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    bench.add_argument("--input-lengths", nargs="+", type=int, default=[128, 512, 2048])
    bench.add_argument("--new-tokens", type=int, default=128)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--output", required=True)
    profile.add_argument("--new-tokens", type=int, default=32)
    profile.add_argument("--no-cache", action="store_true")
    profile.add_argument("--output", required=True)
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    return root


def main():
    args = parser().parse_args()
    if args.command == "env":
        print(json.dumps(environment(), indent=2, ensure_ascii=False))
        return
    if args.command == "plot":
        from .report import plot_report
        print(plot_report(args.input, args.output))
        return
    for name in ("input_length", "batch_size", "new_tokens", "steps", "repeats"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise SystemExit(f"{name} must be positive")
    if hasattr(args, "output") and Path(args.output).exists():
        raise SystemExit("Output already exists. Choose a new path to preserve previous results.")
    model = load_model(args)
    if args.command == "generate":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(args.device)
        result = decode(model, ids, new_tokens=args.new_tokens, use_cache=not args.no_cache)
        print(tokenizer.decode(result.output_ids[0, ids.shape[1]:].tolist(), skip_special_tokens=True))
        print(json.dumps({field.name: getattr(result, field.name) for field in fields(result)
                          if field.name != "output_ids"}, indent=2))
    elif args.command == "benchmark":
        report = run_benchmark(
            model, device=args.device, batches=args.batch_sizes, lengths=args.input_lengths,
            new_tokens=args.new_tokens, warmup=args.warmup, repeats=args.repeats, seed=args.seed,
            output_path=args.output,
            metadata={"model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
                      "device": args.device, "dtype": args.dtype, "attention": args.attention},
        )
        print(json.dumps(report["summary"], indent=2))
    else:
        length = args.input_length + (args.steps if args.command == "check" else 0)
        ids = synthetic_ids(model.config.vocab_size, args.batch_size, length, args.device, args.seed)
        if args.command == "check":
            low_precision = args.dtype != "float32"
            result = check_cache(
                model, ids, prompt_length=args.input_length,
                atol=args.atol if args.atol is not None else (0.05 if low_precision else 1e-4),
                rtol=args.rtol if args.rtol is not None else (0.05 if low_precision else 1e-4),
            )
            print(json.dumps(result, indent=2))
        elif args.command == "inspect":
            with torch.inference_mode():
                output = forward_last(model, ids, use_cache=True)
            layers = [{"layer": i, "key": list(k.shape), "value": list(v.shape)}
                      for i, (k, v) in enumerate(cache_tensors(output.past_key_values))]
            print(json.dumps({
                "input_shape": list(ids.shape), "last_position_logits_shape": list(output.logits.shape),
                "dtype": str(next(model.parameters()).dtype),
                "weights_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
                "actual_kv_mib": cache_bytes(output.past_key_values) / 2**20,
                "expected_kv_mib": expected_kv_bytes(model.config, args.batch_size, args.input_length,
                                                     next(model.parameters()).dtype) / 2**20,
                "cache_layers": layers,
            }, indent=2))
        elif args.command == "profile":
            decode(model, ids, new_tokens=args.new_tokens, use_cache=not args.no_cache)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.device(args.device).type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
                decode(model, ids, new_tokens=args.new_tokens, use_cache=not args.no_cache)
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(str(output_path))
            print(f"Saved diagnostic trace: {output_path}. Profiling timings are not benchmark results.")


if __name__ == "__main__":
    main()
