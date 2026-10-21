import json
import sys

from inference_lab.artifacts import create_report
from inference_lab.benchmark import run_benchmark
from inference_lab.cli import main


def test_report_contains_only_measured_rows(tiny_model, tmp_path):
    path = tmp_path / "raw.json"
    run_benchmark(tiny_model, device="cpu", batches=[1], lengths=[4], new_tokens=3,
                  warmup=0, repeats=2, seed=0, output_path=path, metadata={"model": "local-test"},
                  cache_modes=["no_cache", "dynamic", "static"])
    folder = create_report(path, tmp_path / "report")
    assert json.loads((folder / "measurements.json").read_text()) == json.loads(path.read_text())
    text = (folder / "report.md").read_text(encoding="utf-8")
    assert "static" in text
    assert "未提交" in text
    assert (folder / "summary.csv").is_file()
    assert (folder / "performance.png").is_file()


def test_cpu_suite(tiny_model, tmp_path, monkeypatch):
    checkpoint = tmp_path / "model"
    tiny_model.save_pretrained(checkpoint)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"batches": [1], "input_lengths": [4], "new_tokens": 2,
                                 "warmup": 0, "repeats": 1, "norm_backends": ["torch"],
                                 "check_input_length": 4, "check_steps": 2}), encoding="utf-8")
    output = tmp_path / "suite"
    monkeypatch.setattr(sys, "argv", ["inference-lab", "suite", "--model", str(checkpoint),
                                     "--device", "cpu", "--dtype", "float32", "--config", str(config),
                                     "--output-dir", str(output)])
    main()
    assert json.loads((output / "correctness-fp32.json").read_text())["checks"]["static"]["passed"]
    assert (output / "report" / "report.md").is_file()
