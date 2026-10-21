"""Portable run artifacts: CSV, measured figures and a factual Markdown report."""

import csv
import json
from pathlib import Path
import shutil

from .report import plot_report


def number(value):
    return "N/A" if value is None else f"{value:.3f}"


def create_report(input_path, output_dir):
    source = Path(input_path)
    data = json.loads(source.read_text(encoding="utf-8"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, destination / "measurements.json")
    rows = data["summary"]
    if not rows:
        raise ValueError("No successful benchmark measurements")
    with (destination / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_report(input_path, destination / "performance.png")
    env = data["environment"]
    config = data["configuration"]
    lines = [
        "# 单卡 LLM 推理实验报告", "",
        f"生成时间（UTC）：{data.get('created_at', '未记录')}。状态：{data.get('status', 'legacy')}。", "",
        "## 环境与方法", "",
        f"- GPU：{env.get('gpu', 'CPU')}；PyTorch {env['torch']}；Transformers {env['transformers']}。",
        f"- 源码提交：{env.get('source_commit', '未记录')}；工作区有未提交修改：{env.get('source_dirty', '未知')}。",
        f"- 模型：{config['model']}；精度：{config.get('dtype', '未记录')}；Attention：{config.get('attention', '未记录')}。",
        f"- 输出长度：{config['new_tokens']}；预热：{config['warmup']}；正式重复：{config['repeats']}。",
        "- 等长随机 token 输入、固定输出长度、忽略 EOS；各策略使用相同输入，重复内交错顺序。",
        "- 计时为同步后的本地 wall-clock，包含 Python、采样与缓存初始分配；排除权重加载、分词、输入搬运和预热。",
        "- Prefill 包含首 token；后续 Decode 共 N-1 步。平均每步耗时不是在线 ITL/P95 或服务 TTFT。", "",
        "## 实测结果", "",
        "![实测曲线](performance.png)", "",
        "| Batch | 输入 tokens | Cache | Norm | 吞吐 tokens/s | Prefill+首 token ms | Decode/步 ms | 峰值 allocated MiB | 额外峰值 MiB | CV |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append("| " + " | ".join([
            str(row["batch_size"]), str(row["input_length"]), row.get("cache_mode", "dynamic"),
            row.get("norm_backend", "torch"), number(row["output_tokens_per_second_median"]),
            number(row["prefill_first_token_ms_median"]), number(row["decode_step_ms_median"]),
            number(row["peak_allocated_mib_median"]), number(row.get("extra_peak_allocated_mib_median")),
            number(row.get("throughput_cv")),
        ]) + " |")
    lines += ["", "## 配对比较", "",
              "以下倍数只适用于同一行指定的工作负载，分子/分母均为重复测量的吞吐中位数。", "",
              "| Batch | 输入 | 比较（分子 / 分母） | 吞吐倍数 |", "| --- | --- | --- | --- |"]
    index = {(r["batch_size"], r["input_length"], r.get("cache_mode", "dynamic"),
              r.get("norm_backend", "torch")): r for r in rows}
    for batch, length in sorted({(r["batch_size"], r["input_length"]) for r in rows}):
        comparisons = [(("dynamic", "torch"), ("no_cache", "torch")),
                       (("static", "torch"), ("dynamic", "torch")),
                       (("dynamic", "triton"), ("dynamic", "torch")),
                       (("static", "triton"), ("static", "torch"))]
        for numerator, denominator in comparisons:
            top = index.get((batch, length, *numerator))
            bottom = index.get((batch, length, *denominator))
            if top and bottom:
                ratio = top["output_tokens_per_second_median"] / bottom["output_tokens_per_second_median"]
                lines.append(f"| {batch} | {length} | {'+'.join(numerator)} / {'+'.join(denominator)} | {ratio:.3f}x |")
    noisy = [r for r in rows if r.get("throughput_cv", 0) > 0.1]
    failed = [r for r in data["records"] if r["status"] != "ok"]
    lines += ["", "## 验证边界与局限", "",
              "- 本报告是性能结果，不能替代逐位置 logits 校验。请同时保留 suite 的 correctness-fp32.json；低精度失败不能写成通过。",
              "- 静态缓存采用 Transformers StaticCache，动态缓存采用 DynamicCache；原创工作是策略集成、显式位置管理、控制实验与可选 Triton 算子。",
              "- 动态缓存的拼接、静态缓存额外 mask 和容量开销都会影响结果，静态缓存不保证更快。",
              "- 峰值 allocated 包含模型；额外峰值扣除运行前活跃分配。reserved 和整个进程显存是不同口径。",
              "- 数据来自单机单卡和指定模型。没有验证多卡、生产服务、真实到达率或生产质量。",
              f"- 失败/OOM 记录：{len(failed)}；吞吐 CV > 10% 的配置：{len(noisy)}。原始数据均保留。",
              "- 初始与结束 GPU 快照在 measurements.json 中；共享机器的干扰仍可能影响速度。", "",
              "## 复现配置", "", "```json", json.dumps(config, ensure_ascii=False, indent=2), "```", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination


def create_kernel_report(input_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(input_path, destination / "measurements.json")
    widths = sorted({r["width"] for r in data["records"]})
    fig, axes = plt.subplots(1, len(widths), figsize=(5 * len(widths), 4), squeeze=False)
    lines = ["# Triton RMSNorm 算子实验", "", f"精度：{data['dtype']}。GPU：{data['environment'].get('gpu')}。", "",
             "同时对比 eager 和 torch.compile；下面是算子微基准，不代表完整推理加速。", "",
             "![算子耗时](kernel.png)", "",
             "| Rows | Width | Backend | 中位耗时 μs | 正确性 | 最大绝对误差 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for width, ax in zip(widths, axes[0]):
        for backend in ("torch_eager", "torch_compile", "triton"):
            selected = sorted([r for r in data["records"] if r["width"] == width and r["backend"] == backend],
                              key=lambda r: r["rows"])
            ax.plot([r["rows"] for r in selected], [r["median_ms"] * 1000 for r in selected], marker="o", label=backend)
            for row in selected:
                lines.append(f"| {row['rows']} | {width} | {backend} | {row['median_ms'] * 1000:.3f} | "
                             f"{row['correctness']['passed']} | {row['correctness']['max_abs_error']:.6g} |")
        ax.set(xscale="log", yscale="log", xlabel="Token rows", ylabel="Latency (us)", title=f"Hidden size {width}")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination / "kernel.png", dpi=160)
    plt.close(fig)
    lines += ["", "逻辑带宽估计假设每行读取输入与权重、写回输出；并非硬件 DRAM 实测带宽。",
              "Triton 实现保留 Qwen3 的 FP32 reduction 和归一化结果先回转输入 dtype 的舍入顺序。",
              "原始重复测量、误差和环境在 measurements.json。", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return destination
