"""Rebuild the matched-precision report from the archived RTX 4090 measurements."""

import json
from pathlib import Path
from statistics import median

DIRECTORY = Path(__file__).resolve().parent
DTYPES = {"float32": "FP32", "float16": "FP16", "bfloat16": "BF16"}
SHAPES = [(b, length) for b in (1, 4) for length in (128, 256, 512, 1024)]


def read_rows(filename):
    text = (DIRECTORY / filename).read_text(encoding="utf8")
    return [json.loads(line) for line in text.splitlines()]


def case_key(row):
    return row["batch"], row["length"], row["seed"]


def select_shape(rows, batch, length):
    return [r for r in rows if (r["batch"], r["length"]) == (batch, length)]


def stats(rows):
    return {
        "n": len(rows),
        "base": median(r["no_cache_tokens_s"] for r in rows),
        "dynamic": median(r["dynamic_tokens_s"] for r in rows),
        "speedup": median(r["ratio"] for r in rows),
        "exact": sum(r["exact"] for r in rows),
    }


def main():
    rows = read_rows("precision_cache_300.jsonl")
    old = read_rows("gpu_cache_cohort.jsonl")
    groups = {dtype: {} for dtype in DTYPES}
    for row in rows:
        data = groups[row["dtype"]]
        key = case_key(row)
        assert key not in data and row["new_tokens"] == 32
        assert 0 <= row["sequences_equal"] <= row["batch"]
        assert row["exact"] == (row["sequences_equal"] == row["batch"])
        assert row["exact"] == (
            row["dynamic_output_sha256"] == row["no_cache_output_sha256"]
        )
        assert row["ratio"] == row["no_cache_seconds"] / row["dynamic_seconds"]
        data[key] = row
    fp32 = groups["float32"]
    assert len(fp32) == 100 and all(set(g) == set(fp32) for g in groups.values())
    assert len(rows) == 300 and all(r["env"] == rows[0]["env"] for r in rows)
    old_subset = [r for r in old if case_key(r) in fp32]
    assert len(old) == len(set(map(case_key, old))) == 208
    assert set(map(case_key, old_subset)) == set(fp32)
    report = {}
    lines = [
        "# RTX 4090 · FP32 / FP16 / BF16 各 100 组",
        "",
        "Qwen3-0.6B，SDPA，贪心生成 32 个 token。三种精度共用 100 个种子与输入批次，覆盖 8 种形状，每种 12 或 13 个输入。每组分别测完整前缀重算和动态 KV Cache，合计 600 次正式生成。缓存路径交替先后，精度轮换顺序。CUDA 同步后计时，预热与加载时间排除在外。",
        "",
        "| 精度 | 全部形状的配对加速比中位数 | 同精度两路径整批一致 | 同精度两路径逐序列一致 | 动态缓存对 FP32 动态缓存整批一致 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dtype, label in DTYPES.items():
        data = groups[dtype]
        summary = stats(list(data.values()))
        summary["matched_fp32_dynamic"] = sum(
            r["dynamic_output_sha256"] == fp32[key]["dynamic_output_sha256"]
            for key, r in data.items()
        )
        summary["sequence_match"] = sum(r["sequences_equal"] for r in data.values())
        summary["sequences"] = sum(r["batch"] for r in data.values())
        report[dtype] = {"summary": summary, "shapes": []}
        lines.append(
            f"| {label} | {summary['speedup']:.2f}× | {summary['exact']}/100 | "
            f"{summary['sequence_match']}/{summary['sequences']} | {summary['matched_fp32_dynamic']}/100 |"
        )
    lines.extend(
        [
            "",
            "下表两列吞吐分别取中位数，加速比先在同一输入上求比值再取中位数。整批一致要求该批所有序列的 32 个生成 token 完全一致。随机 token 输入用于测试数值敏感性和吞吐，语言任务准确率需要另外评测。",
            "",
            "| 精度 | Batch | 输入长度 | 组数 | 完整前缀 tokens/s | 动态缓存 tokens/s | 配对加速比 | 整批一致 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dtype, label in DTYPES.items():
        for batch, length in SHAPES:
            data = select_shape(groups[dtype].values(), batch, length)
            assert len(data) == (13 if length >= 512 else 12)
            result = stats(data)
            report[dtype]["shapes"].append({"batch": batch, "length": length, **result})
            lines.append(
                f"| {label} | {batch} | {length} | {result['n']} | {result['base']:.2f} | "
                f"{result['dynamic']:.2f} | {result['speedup']:.2f}× | {result['exact']}/{result['n']} |"
            )
    full = stats(select_shape(old, 4, 1024))
    prev = stats(select_shape(old_subset, 4, 1024))
    new = stats(select_shape(fp32.values(), 4, 1024))
    report["previous_fp32_comparison_b4_l1024"] = {
        "old_all": full,
        "old_matched": prev,
        "new_matched": new,
    }
    lines.extend(
        [
            "",
            (
                f"上一轮纯 FP32 共 208 组，其中 B=4、输入 1024 的 {full['n']} 组配对加速比中位数为 {full['speedup']:.2f}×。"
                f"只比较两轮共有的 {prev['n']} 个输入，上一轮为 {prev['speedup']:.2f}×，本轮为 {new['speedup']:.2f}×。两轮在不同时间执行。"
            ),
            "",
            "## 复现",
            "",
            "在安装了本仓库的环境中使用独占 GPU，设置 CUDA_VISIBLE_DEVICES 为该卡的完整 UUID。运行命令中的 MODEL_DIR 指向同一份 Qwen3-0.6B 检查点，输出路径须为新文件。归档驱动已内联原测量驱动使用的 GPU 状态查询函数。",
            "",
            "```sh",
            'python run_precision.py --gpu "$CUDA_VISIBLE_DEVICES" --model "$MODEL_DIR" --output rerun.jsonl',
            "python report_precision.py",
            "```",
            "",
            "第二条命令使用本目录已归档的两份 JSONL 重建报告。",
            "",
            "[300 组原始计时、输出摘要和环境](precision_cache_300.jsonl) · [上一轮 208 组原始计时](gpu_cache_cohort.jsonl) · [权重 SHA-256](../20261021-4090/model-sha256.txt)",
        ]
    )
    (DIRECTORY / "precision-comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf8"
    )
    (DIRECTORY / "precision-summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf8"
    )
    print(
        json.dumps({d: [report[d]["summary"], report[d]["shapes"][-1]] for d in DTYPES})
    )


if __name__ == "__main__":
    main()
