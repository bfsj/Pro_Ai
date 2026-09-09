import pytest
import torch

from inference_lab.benchmark import synthetic_ids
from inference_lab.core import cache_bytes, check_cache, decode, expected_kv_bytes, forward_last


@pytest.mark.parametrize("attention", ["eager", "sdpa"])
@pytest.mark.parametrize("batch,prompt_length", [(1, 1), (1, 7), (3, 5)])
def test_cached_logits_match_full_prefix(tiny_model, attention, batch, prompt_length):
    tiny_model.set_attn_implementation(attention)
    ids = synthetic_ids(97, batch, prompt_length + 4, "cpu", 5)
    report = check_cache(tiny_model, ids, prompt_length=prompt_length)
    assert report["comparisons"] == 5
    assert report["cache_tokens"] == prompt_length + 4


@pytest.mark.parametrize("new_tokens", [1, 5])
def test_generation_and_accounting(tiny_model, new_tokens):
    ids = synthetic_ids(97, 2, 6, "cpu", 8)
    full = decode(tiny_model, ids, new_tokens=new_tokens, use_cache=False)
    cached = decode(tiny_model, ids, new_tokens=new_tokens, use_cache=True)
    torch.testing.assert_close(full.output_ids, cached.output_ids)
    torch.testing.assert_close(cached.output_ids[:, :6], ids)
    assert cached.output_ids.shape == (2, 6 + new_tokens)
    assert cached.output_tokens == 2 * new_tokens
    assert cached.decode_steps == new_tokens - 1
    assert (cached.decode_step_ms is None) == (new_tokens == 1)
    assert cached.cache_tokens == 6 + new_tokens - 1
    assert cached.peak_allocated_mib is None
    assert full.kv_cache_mib == 0
    assert cached.kv_cache_mib * 2**20 == expected_kv_bytes(
        tiny_model.config, 2, cached.cache_tokens, torch.float32,
    )


def test_gqa_cache_memory(tiny_model):
    ids = synthetic_ids(97, 3, 9, "cpu", 0)
    with torch.inference_mode():
        output = forward_last(tiny_model, ids, use_cache=True)
    assert cache_bytes(output.past_key_values) == expected_kv_bytes(tiny_model.config, 3, 9, torch.float32)


def test_invalid_generation_lengths(tiny_model):
    ids = synthetic_ids(97, 1, 4, "cpu", 0)
    with pytest.raises(ValueError, match="at least 1"):
        decode(tiny_model, ids, new_tokens=0, use_cache=True)
    with pytest.raises(ValueError, match="context limit"):
        decode(tiny_model, ids, new_tokens=300, use_cache=True)


def test_correctness_check_detects_corruption(tiny_model, monkeypatch):
    import inference_lab.core as core
    original = core.forward_last

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("past_key_values") is not None:
            result.logits = result.logits + 1
        return result

    monkeypatch.setattr(core, "forward_last", corrupt)
    ids = synthetic_ids(97, 1, 8, "cpu", 0)
    with pytest.raises(AssertionError):
        check_cache(tiny_model, ids, prompt_length=4)
