"""Command-line entry points for reproducible inference experiments."""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path

import torch

from .benchmark import environment, run_benchmark, synthetic_ids
from .core import CACHE_MODES, cache_bytes, cache_tensors, decode, diagnose_cache, expected_kv_bytes, forward_last, make_cache
from .kernels import configure_norm


def default_model():
    explicit = os.environ.get("PRO_AI_MODEL")
    if explicit:
        return explicit
    local = Path(__file__).resolve().parents[3] / "models" / "Qwen3-0.6B"
    return str(local) if (local / "config.json").is_file() else "Qwen/Qwen3-0.6B"


def add_model_arguments(command):
    command.add_argument("--model", default=default_model())
    command.add_argument("--device", default="cuda")
    command.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    command.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    command.add_argument("--norm-backend", choices=("torch", "triton"), default="torch")
    command.add_argument("--cpu-threads", type=int, default=2)
    command.add_argument("--seed", type=int, default=42)


def load_model(args):
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable. Use --device cpu --dtype float32 for a smoke check.")
        torch.cuda.set_device(device.index if device.index is not None else 0)
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported on selected GPU")
    if device.type == "cpu" and args.dtype != "float32":
        raise ValueError("CPU smoke check requires --dtype float32")
    if device.type != "cuda" and args.norm_backend == "triton":
        raise ValueError("Triton backend requires CUDA")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation=args.attention,
    ).to(device).eval()
    if model.config.model_type != "qwen3":
        raise ValueError("This lab supports dense Qwen3 models only")
    configure_norm(model, args.norm_backend)
    return model


def parser():
    root = argparse.ArgumentParser(description="Pro_Ai: single-GPU inference, KV cache and Triton experiments")
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("env", help="Print runtime, GPU and source provenance")
    inspect = commands.add_parser("inspect", help="Inspect weights, logits and per-layer cache")
    generate = commands.add_parser("generate", help="Run an explicit greedy loop")
    check = commands.add_parser("check", help="Teacher-forced cache correctness and precision diagnostics")
    bench = commands.add_parser("benchmark", help="Controlled, interleaved cache/norm benchmark")
    profile = commands.add_parser("profile", help="Export a separate diagnostic trace and operator summary")
    plot = commands.add_parser("plot", help="Render performance curves")
    report = commands.add_parser("report", help="Create Markdown, CSV, raw JSON and figures")
    suite = commands.add_parser("suite", help="FP32 correctness gate, benchmark, traces and report")
    norm = commands.add_parser("kernel-benchmark", help="Triton vs torch eager vs torch.compile RMSNorm")
    for command in (inspect, generate, check, bench, profile, suite):
        add_model_arguments(command)
    for command in (inspect, check, profile):
        command.add_argument("--input-length", type=int, default=128)
        command.add_argument("--batch-size", type=int, default=1)
    for command in (inspect, generate, profile):
        command.add_argument("--cache-mode", choices=CACHE_MODES, default="dynamic")
    for command in (generate, profile):
        command.add_argument("--new-tokens", type=int, default=32)
        command.add_argument("--no-cache", action="store_true", help="Alias for --cache-mode no_cache")
    generate.add_argument("--prompt", default="用三句话解释什么是 KV Cache。")
    check.add_argument("--cache-mode", choices=("dynamic", "static"), default="dynamic")
    check.add_argument("--steps", type=int, default=8)
    check.add_argument("--atol", type=float)
    check.add_argument("--rtol", type=float)
    check.add_argument("--diagnostic", action="store_true", help="Report failures without a nonzero exit")
    check.add_argument("--output")
    bench.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    bench.add_argument("--input-lengths", nargs="+", type=int, default=[128, 512, 2048])
    bench.add_argument("--cache-modes", nargs="+", choices=CACHE_MODES, default=list(CACHE_MODES))
    bench.add_argument("--norm-backends", nargs="+", choices=("torch", "triton"), default=["torch"])
    bench.add_argument("--new-tokens", type=int, default=128)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--output", required=True)
    profile.add_argument("--output", required=True)
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    report.add_argument("--input", required=True)
    report.add_argument("--output-dir", required=True)
    report.add_argument("--kind", choices=("inference", "kernel"), default="inference")
    suite.add_argument("--config", required=True)
    suite.add_argument("--output-dir", required=True)
    norm.add_argument("--device", default="cuda")
    norm.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    norm.add_argument("--rows", nargs="+", type=int, default=[1, 8, 128, 512])
    norm.add_argument("--widths", nargs="+", type=int, default=[1024, 2048, 4096])
    norm.add_argument("--repeats", type=int, default=3)
    norm.add_argument("--seed", type=int, default=42)
    norm.add_argument("--output", required=True)
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
    if args.command == "report":
        from .artifacts import create_report, create_kernel_report
        function = create_kernel_report if args.kind == "kernel" else create_report
        print(function(args.input, args.output_dir))
        return
    if args.command == "kernel-benchmark":
        from .kernels import benchmark_norm
        torch.cuda.set_device(torch.device(args.device).index or 0)
        benchmark_norm(device=args.device, dtype=args.dtype, rows=args.rows, widths=args.widths,
                       repeats=args.repeats, output_path=args.output, seed=args.seed)
        return
    for name in ("input_length", "batch_size", "new_tokens", "steps", "repeats", "cpu_threads"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise SystemExit(f"{name} must be positive")
    if getattr(args, "output", None) and Path(args.output).exists():
        raise SystemExit("Output already exists; choose a new path.")
    if args.command == "suite":
        from .experiments import run_suite
        run_suite(args)
        return
    model = load_model(args)
    mode = "no_cache" if getattr(args, "no_cache", False) else getattr(args, "cache_mode", "dynamic")
    if args.command == "generate":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(args.device)
        result = decode(model, ids, new_tokens=args.new_tokens, cache_mode=mode)
        print(tokenizer.decode(result.output_ids[0, ids.shape[1]:].tolist(), skip_special_tokens=True))
        print(json.dumps({f.name: getattr(result, f.name) for f in fields(result) if f.name != "output_ids"}, indent=2))
    elif args.command == "benchmark":
        run_benchmark(
            model, device=args.device, batches=args.batch_sizes, lengths=args.input_lengths,
            new_tokens=args.new_tokens, warmup=args.warmup, repeats=args.repeats, seed=args.seed,
            cache_modes=args.cache_modes, norm_backends=args.norm_backends, output_path=args.output,
            metadata={"model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
                      "device": args.device, "dtype": args.dtype, "attention": args.attention},
        )
        print(f"Saved benchmark: {args.output}")
    else:
        length = args.input_length + (args.steps if args.command == "check" else 0)
        ids = synthetic_ids(model.config.vocab_size, args.batch_size, length, args.device, args.seed)
        if args.command == "check":
            from .experiments import save_json
            low_precision = args.dtype != "float32"
            result = diagnose_cache(
                model, ids, prompt_length=args.input_length, cache_mode=mode,
                atol=args.atol if args.atol is not None else (0.05 if low_precision else 1e-4),
                rtol=args.rtol if args.rtol is not None else (0.05 if low_precision else 1e-4),
            )
            if args.output:
                save_json(args.output, dict(environment=environment(), dtype=args.dtype, **result))
            print(json.dumps(result, indent=2))
            if not result["passed"] and not args.diagnostic:
                raise SystemExit(1)
        elif args.command == "inspect":
            cache = make_cache(model, mode, args.input_length)
            with torch.inference_mode():
                output = forward_last(model, ids, use_cache=mode != "no_cache", past_key_values=cache,
                                      cache_position=torch.arange(args.input_length, device=ids.device))
            layers = [{"layer": i, "key": list(k.shape), "value": list(v.shape)}
                      for i, (k, v) in enumerate(cache_tensors(output.past_key_values))]
            print(json.dumps({
                "input_shape": list(ids.shape), "last_position_logits_shape": list(output.logits.shape),
                "dtype": str(next(model.parameters()).dtype), "cache_mode": mode,
                "weights_mib": sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20,
                "actual_kv_mib": cache_bytes(output.past_key_values) / 2**20,
                "expected_kv_mib": expected_kv_bytes(model.config, args.batch_size, args.input_length,
                                                     next(model.parameters()).dtype) / 2**20,
                "cache_layers": layers,
            }, indent=2))
        elif args.command == "profile":
            from .experiments import profile_run
            profile_run(model, ids, new_tokens=args.new_tokens, mode=mode, output_path=args.output)
            print(f"Saved diagnostic trace: {args.output}; profiling timings are not benchmark results.")


if __name__ == "__main__":
    main()
