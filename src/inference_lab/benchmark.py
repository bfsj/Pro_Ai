"""Controlled, repeated inference experiments with raw data and provenance."""

from dataclasses import fields
from datetime import datetime, timezone
import csv
import gc
import importlib.metadata
import io
import itertools
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess

import torch

from .core import CACHE_MODES, decode, synchronize


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,driver_version",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode:
            return []
        keys = ("physical_index", "uuid", "name", "memory_used_mib", "memory_total_mib", "utilization_percent", "driver_version")
        return [dict(zip(keys, (value.strip() for value in row))) for row in csv.reader(io.StringIO(result.stdout))]
    except (OSError, subprocess.TimeoutExpired):
        return []


def environment():
    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_threads": torch.get_num_threads(),
    }
    try:
        info["triton"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        info["triton"] = None
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        info.update(gpu_index=index, gpu=props.name, gpu_memory_mib=props.total_memory / 2**20,
                    gpu_uuid=str(getattr(props, "uuid", "unavailable")),
                    gpu_snapshot=gpu_snapshot())
    root = Path(__file__).resolve().parents[2]
    try:
        prefix = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
        revision = subprocess.run(prefix + ["rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        status = subprocess.run(prefix + ["status", "--porcelain"], capture_output=True, text=True, timeout=10)
        if revision.returncode == 0:
            info["source_commit"] = revision.stdout.strip()
            info["source_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        pass
    return info


def synthetic_ids(vocab_size, batch, length, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, length), generator=generator).to(device)


def result_metrics(result):
    return {field.name: getattr(result, field.name) for field in fields(result) if field.name != "output_ids"}


def summarize(records):
    groups = {}
    for record in records:
        if record["status"] != "ok":
            continue
        mode = record.get("cache_mode", "dynamic" if record["use_cache"] else "no_cache")
        group = (record["batch_size"], record["input_length"], mode, record.get("norm_backend", "torch"))
        groups.setdefault(group, []).append(record)
    summaries = []
    metrics = ("prefill_first_token_ms", "decode_step_ms", "total_ms", "output_tokens_per_second",
               "peak_allocated_mib", "peak_reserved_mib", "extra_peak_allocated_mib", "kv_cache_mib")
    for group, rows in sorted(groups.items()):
        batch, length, mode, norm = group
        summary = dict(batch_size=batch, input_length=length, cache_mode=mode, norm_backend=norm,
                       use_cache=mode != "no_cache", successful_repeats=len(rows))
        for metric in metrics:
            values = [row[metric] for row in rows if row.get(metric) is not None]
            summary[metric + "_median"] = statistics.median(values) if values else None
        speeds = [row["output_tokens_per_second"] for row in rows]
        summary["output_tokens_per_second_min"] = min(speeds)
        summary["output_tokens_per_second_max"] = max(speeds)
        summary["throughput_cv"] = statistics.stdev(speeds) / statistics.mean(speeds) if len(speeds) > 1 else 0.0
        summaries.append(summary)
    return summaries


def _trial(model, ids, new_tokens, mode):
    try:
        result = decode(model, ids, new_tokens=new_tokens, cache_mode=mode)
        return dict(status="ok", **result_metrics(result))
    except torch.cuda.OutOfMemoryError:
        # The failing decode frame is released before outer-loop cleanup.
        return dict(status="oom")


def run_benchmark(model, *, device, batches, lengths, new_tokens, warmup, repeats, seed,
                  output_path, metadata, cache_modes=None, norm_backends=None):
    from .kernels import configure_norm
    if not batches or not lengths or min(batches + lengths) < 1:
        raise ValueError("Batch sizes and input lengths must be positive")
    if repeats < 1 or warmup < 0 or new_tokens < 1:
        raise ValueError("Require repeats >= 1, warmup >= 0 and new_tokens >= 1")
    modes = list(cache_modes or ("no_cache", "dynamic"))
    norms = list(norm_backends or ("torch",))
    if any(mode not in CACHE_MODES for mode in modes):
        raise ValueError(f"cache_modes must be selected from {CACHE_MODES}")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(), "status": "running",
        "configuration": dict(metadata, batches=batches, input_lengths=lengths, new_tokens=new_tokens,
                              cache_modes=modes, norm_backends=norms, warmup=warmup, repeats=repeats, seed=seed,
                              workload="equal-length seeded random token IDs; fixed output length; EOS ignored",
                              timing="synchronized wall clock, prefill boundary synchronized, no per-token synchronization",
                              cache_lifecycle="fresh cache on every trial; initial allocation included in prefill"),
        "records": [], "summary": [],
    }
    with path.open("x", encoding="utf-8") as file:
        def checkpoint():
            report["summary"] = summarize(report["records"])
            file.seek(0)
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.truncate()
            file.flush()
        checkpoint()
        try:
            cases = list(itertools.product(batches, lengths))
            random.Random(seed).shuffle(cases)
            for batch, length in cases:
                ids = synthetic_ids(model.config.vocab_size, batch, length, device, seed + batch * 100000 + length)
                variants = list(itertools.product(norms, modes))
                failed = set()
                for norm, mode in variants:
                    configure_norm(model, norm)
                    for _ in range(warmup):
                        record = _trial(model, ids, new_tokens, mode)
                        if record["status"] != "ok":
                            failed.add((norm, mode))
                            break
                # Interleave modes and randomize within each repeat to reduce
                # systematic order/temperature bias. Inputs remain identical.
                for repeat in range(repeats):
                    order = variants.copy()
                    random.Random(seed + repeat + batch * 10000 + length).shuffle(order)
                    for norm, mode in order:
                        configure_norm(model, norm)
                        case = dict(batch_size=batch, input_length=length, cache_mode=mode,
                                    norm_backend=norm, use_cache=mode != "no_cache", repeat=repeat)
                        record = dict(status="oom") if (norm, mode) in failed else _trial(model, ids, new_tokens, mode)
                        report["records"].append(dict(case, **record))
                        if record["status"] == "oom":
                            gc.collect()
                            if torch.device(device).type == "cuda":
                                torch.cuda.empty_cache()
                    checkpoint()
                del ids
                print(f"batch={batch}, input={length}: {len(variants)} variants x {repeats} repeats", flush=True)
            report["status"] = "completed"
        except BaseException:
            report["status"] = "interrupted_or_failed"
            raise
        finally:
            configure_norm(model, "torch")
            if torch.device(device).type == "cuda":
                report["gpu_snapshot_end"] = gpu_snapshot()
            checkpoint()
    return report
