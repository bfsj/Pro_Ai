import pytest
import torch

from inference_lab.benchmark import synthetic_ids
from inference_lab.core import cache_tensors, check_cache, decode, diagnose_cache, forward_last, make_cache


@pytest.mark.parametrize("mode", ["dynamic", "static"])
@pytest.mark.parametrize("batch,length", [(1, 1), (1, 9), (3, 5)])
@pytest.mark.parametrize("attention", ["eager", "sdpa"])
def test_three_paths_agree(tiny_model, mode, batch, length, attention):
    tiny_model.set_attn_implementation(attention)
    ids = synthetic_ids(97, batch, length + 4, "cpu", 2)
    report = check_cache(tiny_model, ids, prompt_length=length, cache_mode=mode)
    assert report["passed"]
    assert report["comparisons"] == 5
    assert [s["cached_input_tokens"] for s in report["steps"]] == [length, 1, 1, 1, 1]


@pytest.mark.parametrize("new_tokens", [1, 5])
def test_static_generation_matches_no_cache(tiny_model, new_tokens):
    ids = synthetic_ids(97, 2, 7, "cpu", 5)
    reference = decode(tiny_model, ids, new_tokens=new_tokens, cache_mode="no_cache")
    candidate = decode(tiny_model, ids, new_tokens=new_tokens, cache_mode="static")
    assert torch.equal(reference.output_ids, candidate.output_ids)
    assert candidate.cache_tokens == 7 + new_tokens - 1
    assert candidate.cache_capacity_tokens == candidate.cache_tokens


def test_static_future_slots_are_masked(tiny_model):
    ids = synthetic_ids(97, 1, 4, "cpu", 12)
    cache = make_cache(tiny_model, "static", 16)
    with torch.inference_mode():
        forward_last(tiny_model, ids[:, :3], use_cache=True, past_key_values=cache,
                     cache_position=torch.arange(3))
        for key, value in cache_tensors(cache):
            key[:, :, 3:] = 1000
            value[:, :, 3:] = -1000
        candidate = forward_last(tiny_model, ids[:, 3:4], use_cache=True, past_key_values=cache,
                                 cache_position=torch.tensor([3])).logits
        reference = forward_last(tiny_model, ids, use_cache=False).logits
    torch.testing.assert_close(candidate, reference, atol=1e-4, rtol=1e-4)


def test_diagnostic_keeps_failure(tiny_model, monkeypatch):
    import inference_lab.core as core
    original = core.forward_last
    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["use_cache"]:
            result.logits += 1
        return result
    monkeypatch.setattr(core, "forward_last", corrupt)
    report = diagnose_cache(tiny_model, synthetic_ids(97, 1, 8, "cpu", 0), prompt_length=4)
    assert report["passed"] is False
    assert report["comparisons"] == 5
    assert report["steps"][0]["mismatch_fraction"] > 0.5


def test_conflicting_cache_flags(tiny_model):
    with pytest.raises(ValueError, match="disagree"):
        decode(tiny_model, torch.tensor([[1, 2]]), new_tokens=1, use_cache=False, cache_mode="static")
