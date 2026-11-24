import pytest
import torch

from inference_lab.prefill import generate_batch, prefill_logits


@pytest.mark.parametrize("method", ["padded", "bucketed", "packed"])
def test_mixed_lengths_isolated_and_restored(tiny_model, method):
    prompts = [torch.tensor([2, 5, 7]), torch.tensor([3]), torch.tensor([8, 2, 5])]
    expected = torch.stack([tiny_model(s[None]).logits[0, -1] for s in prompts])
    actual = prefill_logits(tiny_model, prompts, method)
    torch.testing.assert_close(actual, expected)
    prompts[0] = torch.tensor([9, 1, 3])
    changed = prefill_logits(tiny_model, prompts, method)
    torch.testing.assert_close(changed[1:], actual[1:])


def reference(model, prompt, steps):
    ids = prompt[None]
    tokens = []
    with torch.inference_mode():
        for step in range(steps):
            logits = model(ids, use_cache=False).logits[:, -1]
            if step == 0:
                first = logits[0]
            token = logits.argmax(-1, keepdim=True)
            tokens.append(token)
            ids = torch.cat([ids, token], dim=1)
    return first, torch.cat(tokens, dim=1)[0]


@pytest.mark.parametrize("method", ["padded", "bucketed", "packed"])
@pytest.mark.parametrize("lengths,steps", [([5, 1, 3, 5, 2], 4), ([3, 3], 1), ([1], 3)])
def test_generation_matches_unbatched_recomputation(tiny_model, method, lengths, steps):
    prompts = [torch.randint(0, 97, (n,)) for n in lengths]
    result = generate_batch(tiny_model, prompts, method, steps)
    for i, prompt in enumerate(prompts):
        logits, tokens = reference(tiny_model, prompt, steps)
        torch.testing.assert_close(result.first_logits[i], logits)
        torch.testing.assert_close(result.token_ids[i], tokens)


def test_context_boundary(tiny_model):
    prompt = torch.ones(255, dtype=torch.long)
    generate_batch(tiny_model, [prompt], "packed", 2)
    with pytest.raises(ValueError, match="context limit"):
        generate_batch(tiny_model, [prompt], "packed", 3)


def test_auto_requires_threshold_and_cpu_uses_padded(tiny_model):
    prompts = [torch.tensor([1, 2, 3]), torch.tensor([8])]
    with pytest.raises(ValueError, match="threshold"):
        generate_batch(tiny_model, prompts, "auto", 2)
    result = generate_batch(tiny_model, prompts, "auto", 2, pack_at_padding=0.2)
    assert result.method == "padded"
    torch.testing.assert_close(result.token_ids, generate_batch(tiny_model, prompts, "padded", 2).token_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA varlen attention")
def test_cuda_packed_causal_gqa_and_isolation(tiny_model):
    pytest.importorskip("torch.nn.attention.varlen")
    model = tiny_model.cuda().half()
    prompts = [torch.tensor(p, device="cuda") for p in ([4, 3, 6], [8], [9, 2])]
    result = generate_batch(model, prompts, "packed", 4)
    assert result.prefill_backend == "torch-varlen-flash"
    for i, prompt in enumerate(prompts):
        logits, tokens = reference(model, prompt, 4)
        torch.testing.assert_close(result.first_logits[i], logits, atol=0.001, rtol=0.01)
        torch.testing.assert_close(result.token_ids[i], tokens)
    prompts[0] = prompts[0].flip(0)
    changed = generate_batch(model, prompts, "packed", 4)
    torch.testing.assert_close(changed.first_logits[1:], result.first_logits[1:], atol=0, rtol=0)
    torch.testing.assert_close(changed.token_ids[1:], result.token_ids[1:])
    selected = generate_batch(model, prompts, "auto", 4, pack_at_padding=1/3)
    assert selected.method == "packed"
    torch.testing.assert_close(selected.token_ids, changed.token_ids)
    selected = generate_batch(model, prompts, "auto", 4, pack_at_padding=0.5)
    assert selected.method == "padded"
