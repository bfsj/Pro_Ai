import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_model():
    # A random, local model exercises the real architecture without downloads.
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = Qwen3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).eval()
