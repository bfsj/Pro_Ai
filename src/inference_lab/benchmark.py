"""Reproducible synthetic workloads and raw per-repeat results."""

from dataclasses import fields
from datetime import datetime, timezone
import gc
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import random
import statistics
import subprocess

import torch

from .core import decode, synchronize


def environment():
    info = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
        "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(gpu_index=index, gpu=properties.name, gpu_memory_mib=properties.total_memory / 2**20)
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            info["nvidia_smi"] = driver.stdout.strip() if driver.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            info["nvidia_smi"] = "unavailable"
    return info


def synthetic_ids(vocab_size, batch, length, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab_size, (batch, length), generator=generator).to(device)


def result_metrics(result):
    return {field.name: getattr(result, field.name) for field in fields(result) if field.name != "output_ids"}


def summarize(records):
    keys = ("batch_size", "input_length", "use_cache")
    groups = {}
    for record in records:
        if record["status"] == "ok":
            groups.setdefault(tuple(record[key] for key in keys), []).append(record)
    summaries = []
    for group, rows in groups.items():
        summary = dict(zip(keys, group))
        summary["successful_repeats"] = len(rows)
        for metric in ("prefill_first_token_ms", "decode_step_ms", "total_ms", "output_tokens_per_second",
                       "peak_allocated_mib", "peak_reserved_mib", "kv_cache_mib"):
            values = [row[metric] for row in rows if row[metric] is not None]
            summary[metric + "_median"] = statistics.median(values) if values else None
        speeds = [row["output_tokens_per_second"] for row in rows]
        summary["output_tokens_per_second_min"] = min(speeds)
        summary["output_tokens_per_second_max"] = max(speeds)
        summaries.append(summary)
    return summaries


def run_benchmark(model, *, device, batches, lengths, new_tokens, warmup, repeats, seed, output_path, metadata):
    if not batches or not lengths or min(batches + lengths) < 1:
        raise ValueError("Batch sizes and input lengths must be positive")
    if repeats < 1 or warmup < 0 or new_tokens < 1:
        raise ValueError("Require repeats >= 1, warmup >= 0 and new_tokens >= 1")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing an earlier experiment.
    with path.open("x", encoding="utf-8") as file:
        cases = list(itertools.product(batches, lengths, (False, True)))
        random.Random(seed).shuffle(cases)
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment(),
            "configuration": dict(metadata, batches=batches, input_lengths=lengths, new_tokens=new_tokens,
                                  warmup=warmup, repeats=repeats, seed=seed,
                                  workload="equal-length random token IDs; fixed-length greedy decoding; EOS ignored"),
            "records": [], "summary": [],
        }
        for batch, length, use_cache in cases:
            case = {"batch_size": batch, "input_length": length, "use_cache": use_cache}
            try:
                gc.collect()
                if torch.device(device).type == "cuda":
                    synchronize(device)
                    torch.cuda.empty_cache()
                ids = synthetic_ids(model.config.vocab_size, batch, length, device, seed + batch * 100000 + length)
                for _ in range(warmup):
                    decode(model, ids, new_tokens=new_tokens, use_cache=use_cache)
                for repeat in range(repeats):
                    result = decode(model, ids, new_tokens=new_tokens, use_cache=use_cache)
                    report["records"].append(dict(case, repeat=repeat, status="ok", **result_metrics(result)))
                    # Do not keep the previous repeat's output alive during the next run.
                    del result
                del ids
                print(f"batch={batch}, input={length}, cache={use_cache}: {repeats} runs complete", flush=True)
            except torch.cuda.OutOfMemoryError:
                report["records"].append(dict(case, status="oom"))
                print(f"batch={batch}, input={length}, cache={use_cache}: CUDA OOM", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            report["summary"] = summarize(report["records"])
            file.seek(0)
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.truncate()
            file.flush()
    return report
