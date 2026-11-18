"""Rebuild the report from two independent processes per device."""

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent


def load(mode, device):
    data = json.loads((HERE / f"{mode}-{device}.json").read_text())
    assert data["mode"] == mode and data["device"] == device
    assert data["converted_layers"] == (196 if mode == "int4" else 0)
    assert len(data["samples"]) == 24
    return data


def main():
    lines = [
        "# Qwen3-0.6B · BF16 与 INT4 权重量化",
        "",
        "量化对象为 28 个 Transformer 块内的 Q、K、V、输出投影和 MLP 的 gate、up、down 投影，共 196 个线性层。每 128 个输入权重共享一对 BF16 缩放与偏移，4 bit 编码交给 PyTorch 的 packed matmul。共享词嵌入与输出头、激活和动态 KV Cache 保持 BF16。",
        "",
        "每个模式在独立进程中加载同一检查点。每种负载预热一次，固定输入重复三次，各列取中位数。每序列贪心生成 32 token，首 token 时间含 prefill，后续延迟为余下 31 步的平均。共享 CPU 主机固定四线程，两种模式分时执行。每个负载的三次计时重复使用同一输入。模型存储统计参数和缓冲区的字节数。CUDA 峰值单列生成时 PyTorch 已分配的显存，计时排除加载与打包。",
        "",
    ]
    has_data = False
    for device, label in [
        ("cpu", "Intel Xeon Platinum 8352V"),
        ("cuda", "NVIDIA GeForce RTX 4090"),
    ]:
        if not (HERE / f"bf16-{device}.json").exists():
            continue
        has_data = True
        base, quant = [load(mode, device) for mode in ("bf16", "int4")]
        assert (
            base["tokens_scored"] == quant["tokens_scored"]
            and base["text_sha256"] == quant["text_sha256"]
        )
        lines.extend(
            [
                f"## {label}",
                "",
                f"模型张量存储由 {base['stored_model_mib']:.2f} 降至 {quant['stored_model_mib']:.2f} MiB，减少 {100 * (1 - quant['stored_model_mib'] / base['stored_model_mib']):.1f}%。12 段手写中英文短文共计 {base['tokens_scored']} 个受评 token，困惑度从 {base['perplexity']:.3f} 变为 {quant['perplexity']:.3f}。每段独立分词，以真实前文预测后续 token，按 token 汇总交叉熵再取指数。该语料用于小规模质量回归。",
                "",
                "| Batch | 输入长度 | 模式 | 首 token ms | 后续 ms/token | 整批 tokens/s | 生成峰值显存 MiB |",
                "|---:|---:|---|---:|---:|---:|---:|",
            ]
        )
        for batch in (1, 4):
            for length in (128, 256, 512, 1024):
                for d in (base, quant):
                    rows = [
                        r
                        for r in d["samples"]
                        if (r["batch"], r["length"]) == (batch, length)
                    ]
                    assert len(rows) == 3
                    peak = (
                        f"{median(r['peak_gpu_mib'] for r in rows):.1f}"
                        if device == "cuda"
                        else "—"
                    )
                    lines.append(
                        f"| {batch} | {length} | {d['mode']} | {median(r['ttft_ms'] for r in rows):.2f} | {median(r['step_ms'] for r in rows):.2f} | {median(r['tokens_s'] for r in rows):.2f} | {peak} |"
                    )
        lines.extend(
            [
                "",
                f"[BF16 原始记录](bf16-{device}.json) · [INT4 原始记录](int4-{device}.json)",
                "",
            ]
        )
    if not (HERE / "bf16-cuda.json").exists():
        lines.extend(
            [
                "GPU 检查期间三张 RTX 4090 均有其他任务，未取得完整的 BF16 / INT4 配对计时。表中 CPU 数字仅描述本次硬件和线程设置。",
                "",
            ]
        )
    assert has_data
    lines.extend(
        [
            "## 复现",
            "",
            "在仓库根目录安装项目及其依赖，使用 PyTorch 2.11。将 MODEL 指向同一份 Qwen3-0.6B 检查点。切换到空闲 GPU 时将 device 改为 cuda。每个输出使用新文件路径。",
            "",
            "```sh",
            'python scripts/quantization_bench.py --model "$MODEL" --mode bf16 --device cpu --output bf16.json',
            'python scripts/quantization_bench.py --model "$MODEL" --mode int4 --device cpu --output int4.json',
            "```",
            "",
            "加载 BF16 的 Qwen3 模型后调用 `from inference_lab.int4 import quantize_decoder`、`quantize_decoder(model)` 即可通过已有 generate 或 decode 使用打包权重。权重量化在进程内进行。",
            "",
            "[固定评测文本](../../../scripts/quantization_text.txt) · [权重指纹](../20261021-4090/model-sha256.txt) · [PyTorch AO 的 INT4 tile-packed 实现参考](https://github.com/pytorch/ao/blob/main/torchao/quantization/quantize_/workflows/int4/int4_tile_packed_to_4d_tensor.py)",
        ]
    )
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
