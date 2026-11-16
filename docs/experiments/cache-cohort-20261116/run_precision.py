"""Run the matched-precision cache benchmark on a dedicated NVIDIA GPU."""

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def smi_snapshot():
    def query(flag, fields):
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-{flag}={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return [
            dict(zip(fields, (v.strip() for v in row)))
            for row in csv.reader(io.StringIO(result.stdout))
            if row
        ]

    return {
        "utc": utc_now(),
        "gpus": query(
            "gpu",
            [
                "index",
                "uuid",
                "name",
                "memory.used",
                "memory.free",
                "utilization.gpu",
                "temperature.gpu",
                "power.draw",
                "clocks.sm",
                "clocks.mem",
                "driver_version",
            ],
        ),
        "processes": query(
            "compute-apps", ["gpu_uuid", "pid", "process_name", "used_gpu_memory"]
        ),
    }


def resource_reason(
    snapshot,
    uuid,
    *,
    own_pid=None,
    require_idle=True,
    max_idle_util=5,
    min_free_mib=6000,
):
    gpu = next((g for g in snapshot["gpus"] if g["uuid"] == uuid), None)
    if gpu is None:
        return "Selected GPU UUID not found"
    foreign = [
        p
        for p in snapshot["processes"]
        if p["gpu_uuid"] == uuid and str(p["pid"]) != str(own_pid)
    ]
    if foreign:
        return f"Other compute processes present: {[p['pid'] for p in foreign]}"
    try:
        memory = float(gpu["memory.free"])
        utilization = float(gpu["utilization.gpu"])
        if (
            not math.isfinite(memory)
            or not math.isfinite(utilization)
            or memory < 0
            or not 0 <= utilization <= 100
        ):
            return "GPU accounting unavailable; cannot establish idle state"
        if memory < min_free_mib:
            return "Insufficient free GPU memory"
        if require_idle and utilization > max_idle_util:
            return "GPU utilization above idle threshold"
    except (ValueError, TypeError):
        return "GPU accounting unavailable; cannot establish idle state"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == args.gpu

    def guard(idle=False):
        reason = resource_reason(
            smi_snapshot(), args.gpu, own_pid=os.getpid(), require_idle=idle
        )
        if reason:
            raise RuntimeError(reason)

    for _ in range(3):
        guard(True)
        time.sleep(2)
    import torch
    import transformers
    from inference_lab.cli import load_model
    from inference_lab.core import decode

    models = {}
    for dtype in ("float32", "float16", "bfloat16"):
        guard()
        models[dtype] = load_model(
            argparse.Namespace(
                model=args.model,
                device="cuda",
                dtype=dtype,
                attention="sdpa",
                norm_backend="torch",
                cpu_threads=2,
                seed=2026,
            )
        )
    cases = [
        (batch, length, i)
        for batch in (1, 4)
        for length in (128, 256, 512, 1024)
        for i in range(13 if length >= 512 else 12)
    ]
    assert len(cases) == 100
    random.Random(617).shuffle(cases)
    orders = list(itertools.permutations(models))
    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    with torch.inference_mode(), args.output.open("x") as stream:
        for batch in (1, 4):
            for length in (128, 256, 512, 1024):
                ids = torch.ones((batch, length), device="cuda", dtype=torch.long)
                for model in models.values():
                    guard()
                    for mode in ("dynamic", "no_cache"):
                        decode(model, ids, new_tokens=4, cache_mode=mode)
        for idx, (batch, length, i) in enumerate(cases):
            seed = batch * 100000 + length * 26 + i
            ids = torch.randint(
                10,
                20000,
                (batch, length),
                generator=torch.Generator().manual_seed(seed),
            ).cuda()
            for dtype in orders[idx % 6]:
                modes = ["dynamic", "no_cache"] if i % 2 else ["no_cache", "dynamic"]
                row = {
                    "seed": seed,
                    "batch": batch,
                    "length": length,
                    "new_tokens": 32,
                    "dtype": dtype,
                    "order": modes,
                    "env": env,
                }
                tokens = {}
                for mode in modes:
                    guard()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    out = decode(
                        models[dtype], ids, new_tokens=32, cache_mode=mode
                    ).output_ids
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t
                    guard()
                    tokens[mode] = out[:, -32:].cpu().contiguous()
                    row[mode + "_seconds"] = elapsed
                    row[mode + "_tokens_s"] = batch * 32 / elapsed
                    row[mode + "_output_sha256"] = hashlib.sha256(
                        tokens[mode].numpy().tobytes()
                    ).hexdigest()
                row["exact"] = torch.equal(tokens["no_cache"], tokens["dynamic"])
                row["sequences_equal"] = int(
                    (tokens["no_cache"] == tokens["dynamic"]).all(dim=1).sum()
                )
                row["ratio"] = row["no_cache_seconds"] / row["dynamic_seconds"]
                stream.write(json.dumps(row) + "\n")
                stream.flush()
    print("Completed", len(cases) * len(models))


if __name__ == "__main__":
    main()
