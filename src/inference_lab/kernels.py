"""Optional inference-only Triton RMSNorm and controlled kernel experiments.

Triton is imported lazily so CPU/Windows installations can use the rest of the
project. Qwen3 rounds the normalized value to input dtype *before* applying
the weight; the kernel preserves that rounding point.
"""

import json
from pathlib import Path
import statistics
import types

import torch


def torch_rms_norm(x, weight, eps):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return weight * value.to(x.dtype)


def triton_rms_norm(x, weight, eps):
    if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Triton RMSNorm requires contiguous CUDA inputs and weight")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise ValueError("This Triton kernel is inference-only; use torch.inference_mode()")
    from .triton_ops import launch_rms_norm
    return launch_rms_norm(x, weight, eps)


def configure_norm(model, backend):
    if backend not in ("torch", "triton"):
        raise ValueError("norm backend must be torch or triton")
    count = 0
    for module in model.modules():
        if module.__class__.__name__ != "Qwen3RMSNorm":
            continue
        if not hasattr(module, "_pro_ai_torch_forward"):
            module._pro_ai_torch_forward = module.forward
        if backend == "torch":
            module.forward = module._pro_ai_torch_forward
        else:
            def forward(self, hidden_states):
                return triton_rms_norm(hidden_states, self.weight, self.variance_epsilon)
            module.forward = types.MethodType(forward, module)
        count += 1
    return count


@torch.inference_mode()
def benchmark_norm(*, device, dtype, rows, widths, repeats, output_path, seed=42):
    if torch.device(device).type != "cuda":
        raise ValueError("Kernel benchmark requires CUDA")
    if min(rows + widths) < 1 or repeats < 1:
        raise ValueError("Positive shapes and repeats are required")
    from .benchmark import environment
    from triton.testing import do_bench
    torch.manual_seed(seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"environment": environment(), "dtype": dtype, "seed": seed,
              "method": "Triton do_bench, compilation/warmup excluded, repeated median, operator-only",
              "records": [], "status": "running"}
    compiled = torch.compile(torch_rms_norm)
    with path.open("x", encoding="utf-8") as file:
        for width in widths:
            for row in rows:
                x = torch.randn((row, width), device=device, dtype=getattr(torch, dtype))
                weight = torch.randn((width,), device=device, dtype=x.dtype)
                reference = torch_rms_norm(x, weight, 1e-6)
                tolerance = 1e-5 if dtype == "float32" else (0.02 if dtype == "bfloat16" else 0.002)
                shape_passed = True
                for name, function in (("torch_eager", torch_rms_norm), ("torch_compile", compiled),
                                       ("triton", triton_rms_norm)):
                    candidate = function(x, weight, 1e-6)
                    error = (reference.float() - candidate.float()).abs()
                    passed = torch.allclose(candidate, reference, atol=tolerance, rtol=tolerance)
                    shape_passed = shape_passed and passed
                    correctness = {"passed": passed, "max_abs_error": error.max().item(),
                                   "mean_abs_error": error.mean().item(), "atol": tolerance, "rtol": tolerance}
                    torch.cuda.synchronize(device)
                    measurements = [do_bench(lambda: function(x, weight, 1e-6), warmup=100, rep=200)
                                    for _ in range(repeats)]
                    # Logical traffic: input read, weight read once per row, output write.
                    logical_bytes = 3 * x.numel() * x.element_size()
                    ms = statistics.median(measurements)
                    report["records"].append({"rows": row, "width": width, "backend": name,
                                              "milliseconds": measurements, "median_ms": ms,
                                              "logical_gbps": logical_bytes / (ms * 1e6),
                                              "correctness": correctness})
                file.seek(0)
                json.dump(report, file, indent=2, allow_nan=False)
                file.truncate()
                file.flush()
                print(f"RMSNorm rows={row}, width={width}: all backends correct={shape_passed}", flush=True)
        report["status"] = "completed"
        file.seek(0)
        json.dump(report, file, indent=2, allow_nan=False)
        file.truncate()
    return report
