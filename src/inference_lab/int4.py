"""Groupwise INT4 inference for BF16 dense Qwen3 decoders on PyTorch 2.11."""

import torch
from torch import nn


class PackedLinear(nn.Module):
    """Groupwise affine weights, BF16 activations and packed ATen matmul."""

    def __init__(self, linear, *, group_size=128, mse_clip=False):
        super().__init__()
        weight = linear.weight.detach()
        n, k = weight.shape
        if group_size not in (32, 64, 128):
            raise ValueError("group_size must be 32, 64 or 128")
        self.group_size = group_size
        if k % 128 or n % 16 or linear.bias is not None:
            raise ValueError(
                "Expected bias-free Qwen3 projection aligned to INT4 tiles"
            )
        groups = weight.float().reshape(n, -1, group_size)
        low, high = groups.amin(-1), groups.amax(-1)
        scale = ((high - low) / 15).clamp_min(1e-10)
        codes = (
            ((groups - low.unsqueeze(-1)) / scale.unsqueeze(-1)).round().clamp(0, 15)
        )
        offset = low + 8 * scale
        if mse_clip:

            def error(q, step, zero):
                restored = (q - 8) * step.bfloat16().float().unsqueeze(-1)
                return (
                    (restored + zero.bfloat16().float().unsqueeze(-1) - groups)
                    .square()
                    .mean(-1)
                )

            best = error(codes, scale, offset)
            full_scale = scale
            midpoint = (low + high) / 2
            for fraction in (0.95, 0.90, 0.85):
                step = full_scale * fraction
                zero = midpoint + step / 2
                q = (
                    ((groups - zero.unsqueeze(-1)) / step.unsqueeze(-1) + 8)
                    .round()
                    .clamp(0, 15)
                )
                loss = error(q, step, zero)
                take = loss < best
                codes = torch.where(take.unsqueeze(-1), q, codes)
                scale = torch.where(take, step, scale)
                offset = torch.where(take, zero, offset)
                best = torch.minimum(best, loss)
        codes = codes.reshape(n, k)
        if weight.is_cuda:
            codes = codes.to(torch.uint8)
            packed = torch.ops.aten._convert_weight_to_int4pack(
                ((codes[:, ::2] << 4) | codes[:, 1::2]).contiguous(), 8
            )
        else:
            packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(codes.int(), 8)
        self.register_buffer("packed", packed)
        params = torch.stack([scale, offset], -1).transpose(0, 1)
        self.register_buffer("params", params.to(torch.bfloat16).contiguous())

    def forward(self, x):
        op = (
            torch.ops.aten._weight_int4pack_mm
            if x.is_cuda
            else torch.ops.aten._weight_int4pack_mm_for_cpu
        )
        out = op(
            x.reshape(-1, x.size(-1)).contiguous(),
            self.packed,
            self.group_size,
            self.params,
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])


@torch.no_grad()
def quantize_decoder(model, *, group_size=128, mse_clip=False):
    """Replace BF16 Qwen3 projections in place. Return the number converted."""
    if model.config.model_type != "qwen3" or model.dtype != torch.bfloat16:
        raise ValueError("Expected a dense Qwen3 model loaded in BF16")
    converted = 0
    for layer in model.model.layers:
        for block in (layer.self_attn, layer.mlp):
            for name, linear in list(block.named_children()):
                if isinstance(linear, nn.Linear):
                    setattr(
                        block,
                        name,
                        PackedLinear(linear, group_size=group_size, mse_clip=mse_clip),
                    )
                    converted += 1
    assert converted == 7 * model.config.num_hidden_layers
    return converted
