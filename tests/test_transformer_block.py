
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import GPTConfig
from model.transformer_block import TransformerBlock
from model.transformer_stack import TransformerStack


def test_block_shape():
    block = TransformerBlock(GPTConfig())

    x = torch.randn(2, 8, 56)
    y = block(x)

    assert y.shape == x.shape


def test_block_parameters():
    block = TransformerBlock(GPTConfig())

    params = sum(p.numel() for p in block.parameters())

    assert params == 38360


def test_stack_parameters():
    stack = TransformerStack(GPTConfig())

    params = sum(p.numel() for p in stack.parameters())

    assert params == 76720


def test_blocks_have_independent_weights():
    stack = TransformerStack(GPTConfig())

    weight1 = stack.blocks[0].attention.qkv.qkv_proj.weight
    weight2 = stack.blocks[1].attention.qkv.qkv_proj.weight

    # The blocks must not share parameter storage.
    assert weight1.data_ptr() != weight2.data_ptr()


def test_gradients():
    stack = TransformerStack(GPTConfig())

    x = torch.randn(2, 8, 56, requires_grad=True)

    y = stack(x)
    loss = y.square().mean()
    loss.backward()

    assert x.grad is not None

    for block in stack.blocks:
        assert block.attention.qkv.qkv_proj.weight.grad is not None
        assert block.mlp.fc1.weight.grad is not None


def test_causality():
    torch.manual_seed(42)

    stack = TransformerStack(GPTConfig())
    stack.eval()

    x = torch.randn(1, 8, 56)

    original = stack(x)

    changed = x.clone()
    changed[:, -1, :] += 100

    modified = stack(changed)

    # Changing the last token must not affect
    # the representations of earlier tokens.
    assert torch.allclose(
        original[:, :-1, :],
        modified[:, :-1, :],
        atol=1e-5
    )


if __name__ == "__main__":
    test_block_shape()
    test_block_parameters()
    test_stack_parameters()
    test_blocks_have_independent_weights()
    test_gradients()
    test_causality()

    print("All Transformer block tests passed!")
