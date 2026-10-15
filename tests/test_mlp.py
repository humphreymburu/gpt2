
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import GPTConfig
from model.mlp import MLP


def test_output_shape():
    model = MLP(GPTConfig())

    x = torch.randn(4, 64, 56)
    y = model(x)

    assert y.shape == x.shape


def test_parameter_count():
    model = MLP(GPTConfig())

    params = sum(p.numel() for p in model.parameters())

    assert params == 25368


def test_gradients():
    model = MLP(GPTConfig())

    x = torch.randn(2, 5, 56, requires_grad=True)

    output = model(x)
    loss = output.square().mean()

    loss.backward()

    assert x.grad is not None
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None


def test_token_independence():
    torch.manual_seed(42)

    model = MLP(GPTConfig())

    x = torch.randn(1, 4, 56)

    original = model(x)

    changed = x.clone()
    changed[:, 3, :] += 100

    modified = model(changed)

    # Changing token 3 must not change
    # the outputs of tokens 0, 1, and 2.
    assert torch.allclose(
        original[:, :3, :],
        modified[:, :3, :],
        atol=1e-6
    )


if __name__ == "__main__":
    test_output_shape()
    test_parameter_count()
    test_gradients()
    test_token_independence()

    print("All MLP tests passed!")
