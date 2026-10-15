import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch

from config import GPTConfig
from model.multi_head_attention import MultiHeadAttention


def test_output_shape():
    mha = MultiHeadAttention(GPTConfig())

    x = torch.randn(2, 8, 56)
    y = mha(x)

    assert y.shape == x.shape


def test_full_block_size_shape():
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    x = torch.randn(2, config.block_size, config.n_embd)

    assert mha(x).shape == x.shape


def test_parameter_count():
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    C = config.n_embd
    # Fused QKV (weight + bias) plus output projection (weight + bias).
    expected = (C * 3 * C + 3 * C) + (C * C + C)

    assert sum(p.numel() for p in mha.parameters()) == expected


def test_head_dim():
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    assert mha.head_dim * mha.n_head == config.n_embd


def test_split_heads_shape_and_content():
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    B, T, C = 2, 5, config.n_embd
    x = torch.randn(B, T, C)

    heads = mha._split_heads(x)

    assert heads.shape == (B, config.n_head, T, mha.head_dim)
    # Head 1 holds channels [head_dim : 2 * head_dim].
    assert torch.equal(heads[:, 1], x[:, :, mha.head_dim:2 * mha.head_dim])


def test_causality_future_tokens_do_not_change_past_outputs():
    torch.manual_seed(42)

    mha = MultiHeadAttention(GPTConfig()).eval()

    x1 = torch.randn(1, 10, 56)
    x2 = x1.clone()
    x2[:, 6:] = torch.randn(1, 4, 56)  # change only future positions

    with torch.no_grad():
        y1 = mha(x1)
        y2 = mha(x2)

    assert torch.allclose(y1[:, :6], y2[:, :6], atol=1e-6)
    assert not torch.allclose(y1[:, 6:], y2[:, 6:], atol=1e-6)


def test_batch_independence():
    torch.manual_seed(42)

    mha = MultiHeadAttention(GPTConfig()).eval()

    x = torch.randn(3, 8, 56)

    with torch.no_grad():
        batched = mha(x)
        single = mha(x[1:2])

    assert torch.allclose(batched[1:2], single, atol=1e-6)


def test_deterministic_in_eval_mode():
    mha = MultiHeadAttention(GPTConfig()).eval()

    x = torch.randn(2, 8, 56)

    with torch.no_grad():
        assert torch.equal(mha(x), mha(x))


def test_gradients_reach_all_parameters():
    mha = MultiHeadAttention(GPTConfig())

    x = torch.randn(2, 8, 56)
    mha(x).sum().backward()

    for name, p in mha.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_sequence_longer_than_block_size_raises():
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    x = torch.randn(1, config.block_size + 1, config.n_embd)

    with pytest.raises(ValueError):
        mha(x)
