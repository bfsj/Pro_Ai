import pytest
import torch

from inference_lab.kernels import configure_norm, torch_rms_norm, triton_rms_norm


def test_torch_norm_matches_qwen(tiny_model):
    norm = tiny_model.model.norm
    x = torch.randn(3, 5, 32)
    torch.testing.assert_close(torch_rms_norm(x, norm.weight, norm.variance_epsilon), norm(x))
    # Each Qwen3 layer has input/post-attention norms plus Q/K norms.
    assert configure_norm(tiny_model, "torch") == 4 * tiny_model.config.num_hidden_layers + 1


def test_triton_cpu_rejected():
    with pytest.raises(ValueError, match="CUDA"):
        triton_rms_norm(torch.ones(1, 32), torch.ones(32), 1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 32), (7, 127), (128, 1024)])
def test_triton_norm_correctness(dtype, shape):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    with torch.inference_mode():
        expected = torch_rms_norm(x, weight, 1e-6)
        actual = triton_rms_norm(x, weight, 1e-6)
    tolerance = 1e-5 if dtype == torch.float32 else (0.02 if dtype == torch.bfloat16 else 0.002)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_model_integration(tiny_model):
    from inference_lab.experiments import norm_integration_check
    pytest.importorskip("triton")
    tiny_model = tiny_model.cuda()
    report = norm_integration_check(tiny_model, torch.tensor([[3, 4, 5]], device="cuda"))
    assert report["passed"]
