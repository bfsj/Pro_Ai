"""Verify both FP32 cohorts and print their per-shape Markdown comparison."""

import json
from math import isclose
from pathlib import Path
from statistics import median

DIRECTORY = Path(__file__).resolve().parent


def load(filename):
    return [
        json.loads(line) for line in (DIRECTORY / filename).read_text().splitlines()
    ]


def verify(rows, start, extended):
    expected = {
        (batch, length, batch * 100000 + length * 26 + index)
        for batch in (1, 4)
        for length in (128, 256, 512, 1024)
        for index in range(start, (38 if length >= 512 else 37) if extended else 26)
    }
    assert len(rows) == len(expected)
    assert {(r["batch"], r["length"], r["seed"]) for r in rows} == expected
    for row in rows:
        assert row["new_tokens"] == 32
        assert set(row["order"]) == {"no_cache", "dynamic"} and len(row["order"]) == 2
        assert isinstance(row["exact"], bool)
        for mode in row["order"]:
            seconds = row[f"{mode}_seconds"]
            assert seconds > 0 and isclose(
                row[f"{mode}_tokens_s"], row["batch"] * 32 / seconds
            )
        assert isclose(row["ratio"], row["no_cache_seconds"] / row["dynamic_seconds"])


def main():
    original = load("gpu_cache_cohort.jsonl")
    extension = load("gpu_cache_extra92.jsonl")
    verify(original, 0, False)
    verify(extension, 26, True)
    combined = original + extension
    assert len({row["seed"] for row in combined}) == len(combined) == 300
    print(
        "| Batch | 输入 token | 原组数 → 扩测组数 | 原配对加速比 | 扩测配对加速比 | 扩测完整重算 tokens/s | 扩测动态缓存 tokens/s | 输出一致 |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for batch in (1, 4):
        for length in (128, 256, 512, 1024):
            before = [
                r for r in original if (r["batch"], r["length"]) == (batch, length)
            ]
            after = [
                r for r in combined if (r["batch"], r["length"]) == (batch, length)
            ]
            print(
                f"| {batch} | {length} | {len(before)} → {len(after)} | "
                f"{median(r['ratio'] for r in before):.2f}× | {median(r['ratio'] for r in after):.2f}× | "
                f"{median(r['no_cache_tokens_s'] for r in after):.2f} | "
                f"{median(r['dynamic_tokens_s'] for r in after):.2f} | "
                f"{sum(r['exact'] for r in after)}/{len(after)} |"
            )


if __name__ == "__main__":
    main()
