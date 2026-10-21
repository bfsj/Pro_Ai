import importlib.util
from pathlib import Path

import torch.nn.functional as functional
import pytest

from inference_lab.benchmark import synthetic_ids


def load_probe():
    source = Path(__file__).resolve().parents[1] / "scripts" / "attention_layout_probe.py"
    spec = importlib.util.spec_from_file_location("attention_layout_probe", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_records_and_restores_sdpa(tiny_model):
    tiny_model.set_attn_implementation("sdpa")
    original = functional.scaled_dot_product_attention
    module = load_probe()
    result = module.probe(tiny_model, synthetic_ids(97, 2, 4, "cpu", 7))
    assert functional.scaled_dot_product_attention is original
    assert result["no_cache_repeat"]["max_abs_error"] == 0
    assert result["dynamic"]["max_abs_error"] < 1e-4
    # A different mask/GQA branch can expand K/V heads before SDPA. torch.equal
    # then reports False for the different shapes; Q must remain unchanged.
    assert result["dynamic"]["first_layer_qkv_equal"][0]
    assert result["no_cache_repeat"]["first_layer_qkv_equal"] == [True, True, True]


def test_probe_restores_on_failure():
    original = functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError):
        with load_probe().intercept_sdpa({}, True):
            raise RuntimeError("failure")
    assert functional.scaled_dot_product_attention is original
