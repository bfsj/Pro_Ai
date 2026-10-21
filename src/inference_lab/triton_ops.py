"""One Triton program per row; forward inference only."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(X, W, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(X + row * N + offsets, mask=mask, other=0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / N
    normalized = values * tl.rsqrt(variance + EPS)
    # Match Hugging Face Qwen3's cast before multiplying the learned weight.
    rounded = normalized.to(Y.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + offsets, mask=mask, other=0).to(tl.float32)
    tl.store(Y + row * N + offsets, rounded * weight, mask=mask)


def launch_rms_norm(x, weight, eps):
    width = x.shape[-1]
    if weight.shape != (width,) or weight.dtype != x.dtype or weight.device != x.device:
        raise ValueError("RMSNorm weight must match the last dimension, dtype and device")
    if width > 16384:
        raise ValueError("This educational RMSNorm kernel supports hidden sizes <= 16384")
    output = torch.empty_like(x)
    block = triton.next_power_of_2(width)
    _rms_norm[(x.numel() // width,)](x, weight, output, width, eps, block,
                                   num_warps=4 if block <= 2048 else 8)
    return output
