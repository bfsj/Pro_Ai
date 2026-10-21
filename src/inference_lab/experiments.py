"""A complete experiment: FP32 gate, measured variants, trace and report."""

import argparse
import gc
import json
from pathlib import Path

import torch

from .artifacts import create_report
from .benchmark import environment, run_benchmark, synthetic_ids
from .core import diagnose_cache, decode, forward_last
from .kernels import configure_norm


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, allow_nan=False)


def profile_run(model, ids, *, new_tokens, mode, output_path):
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decode(model, ids, new_tokens=new_tokens, cache_mode=mode)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if ids.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        decode(model, ids, new_tokens=new_tokens, cache_mode=mode)
    prof.export_chrome_trace(str(path))
    sort_by = "self_device_time_total" if ids.is_cuda else "self_cpu_time_total"
    path.with_suffix(".txt").write_text(prof.key_averages().table(sort_by=sort_by, row_limit=30), encoding="utf-8")


@torch.inference_mode()
def norm_integration_check(model, ids):
    configure_norm(model, "torch")
    reference = forward_last(model, ids, use_cache=False).logits.float()
    try:
        count = configure_norm(model, "triton")
        candidate = forward_last(model, ids, use_cache=False).logits.float()
    finally:
        configure_norm(model, "torch")
    tolerance = 1e-4 if next(model.parameters()).dtype == torch.float32 else 0.05
    error = (candidate - reference).abs()
    return {"passed": torch.allclose(candidate, reference, atol=tolerance, rtol=tolerance),
            "max_abs_error": error.max().item(), "mean_abs_error": error.mean().item(),
            "top1_agreement": (candidate.argmax(-1) == reference.argmax(-1)).float().mean().item(),
            "atol": tolerance, "rtol": tolerance, "replaced_modules": count}


def run_suite(args):
    from .cli import load_model
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "requested-config.json", config)
    check_args = argparse.Namespace(**vars(args))
    check_args.dtype = "float32"
    check_args.norm_backend = "torch"
    model = load_model(check_args)
    length = config.get("check_input_length", 32)
    steps = config.get("check_steps", 4)
    batch = config.get("check_batch_size", 1)
    ids = synthetic_ids(model.config.vocab_size, batch, length + steps, args.device, args.seed)
    checks = {mode: diagnose_cache(model, ids, prompt_length=length, cache_mode=mode)
              for mode in ("dynamic", "static")}
    validation = {"environment": environment(), "dtype": "float32", "checks": checks}
    if "triton" in config.get("norm_backends", ["torch"]):
        validation["triton_norm_integration"] = norm_integration_check(model, ids[:, :length])
        try:
            configure_norm(model, "triton")
            validation["triton_cache_checks"] = {
                mode: diagnose_cache(model, ids, prompt_length=length, cache_mode=mode)
                for mode in ("dynamic", "static")
            }
        finally:
            configure_norm(model, "torch")
    save_json(destination / "correctness-fp32.json", validation)
    if not all(report["passed"] for report in checks.values()):
        raise RuntimeError("FP32 cache correctness gate failed; see correctness-fp32.json")
    if not validation.get("triton_norm_integration", {"passed": True})["passed"]:
        raise RuntimeError("FP32 Triton integration gate failed; see correctness-fp32.json")
    if not all(report["passed"] for report in validation.get("triton_cache_checks", {}).values()):
        raise RuntimeError("FP32 Triton cache correctness gate failed; see correctness-fp32.json")
    print("FP32 correctness gate passed.", flush=True)
    if args.dtype != "float32":
        del model, ids
        gc.collect()
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
        model = load_model(args)
        ids = synthetic_ids(model.config.vocab_size, batch, length + steps, args.device, args.seed)
        checks = {mode: diagnose_cache(model, ids, prompt_length=length, cache_mode=mode, atol=0.05, rtol=0.05)
                  for mode in ("dynamic", "static")}
        save_json(destination / f"correctness-{args.dtype}.json", {"dtype": args.dtype, "checks": checks})
        print("Low-precision diagnostics recorded; failed checks remain marked failed.", flush=True)
    del ids
    run_benchmark(
        model, device=args.device, batches=config["batches"], lengths=config["input_lengths"],
        new_tokens=config["new_tokens"], warmup=config["warmup"], repeats=config["repeats"], seed=args.seed,
        cache_modes=config.get("cache_modes", ["no_cache", "dynamic", "static"]),
        norm_backends=config.get("norm_backends", ["torch"]), output_path=destination / "benchmark.json",
        metadata={"model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
                  "device": args.device, "dtype": args.dtype, "attention": args.attention},
    )
    if config.get("profile", False):
        ids = synthetic_ids(model.config.vocab_size, 1, config.get("profile_input_length", 128), args.device, args.seed)
        for mode in ("dynamic", "static"):
            profile_run(model, ids, new_tokens=8, mode=mode, output_path=destination / "traces" / f"{mode}.json")
    create_report(destination / "benchmark.json", destination / "report")
    print(f"Completed experiment: {destination / 'report' / 'report.md'}", flush=True)
    return destination
