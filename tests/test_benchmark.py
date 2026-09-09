import json

import pytest
import torch

from inference_lab.benchmark import run_benchmark, synthetic_ids
from inference_lab.report import plot_report


def test_benchmark_report_and_plot(tiny_model, tmp_path):
    path = tmp_path / "measurements.json"
    report = run_benchmark(
        tiny_model, device="cpu", batches=[1, 2], lengths=[4, 8], new_tokens=3,
        warmup=1, repeats=2, seed=7, output_path=path, metadata={"model": "random-test-qwen3"},
    )
    assert json.loads(path.read_text()) == report
    assert len(report["records"]) == 16
    assert len(report["summary"]) == 8
    for row in report["summary"]:
        assert row["successful_repeats"] == 2
        assert row["output_tokens_per_second_min"] <= row["output_tokens_per_second_median"] <= row["output_tokens_per_second_max"]
    image_path = plot_report(path, tmp_path / "plot.png")
    assert image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(FileExistsError):
        plot_report(path, image_path)


def test_preserve_existing_results(tiny_model, tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("previous experiment", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_benchmark(tiny_model, device="cpu", batches=[1], lengths=[4], new_tokens=2,
                      warmup=0, repeats=1, seed=0, output_path=path, metadata={})
    assert path.read_text() == "previous experiment"


def test_reproducible_workload():
    first = synthetic_ids(97, 2, 8, "cpu", 42)
    assert torch.equal(first, synthetic_ids(97, 2, 8, "cpu", 42))
    assert not torch.equal(first, synthetic_ids(97, 2, 8, "cpu", 43))
