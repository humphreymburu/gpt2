
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import GPTConfig
from model.gpt import GPT


def test_logits_shape():
    config = GPTConfig()
    model = GPT(config)

    x = torch.randint(0, 256, (4, 64))

    logits, loss = model(x)

    assert logits.shape == (4, 64, 256)
    assert loss is None


def test_parameter_count():
    model = GPT(GPTConfig())

    params = sum(p.numel() for p in model.parameters())

    assert params == 94752


def test_weight_tying():
    model = GPT(GPTConfig())

    embedding_weight = (
        model.embeddings.token_embedding.embedding.weight
    )

    output_weight = model.lm_head.weight

    # They must reference the same parameter
    assert embedding_weight is output_weight


def test_training_loss():
    model = GPT(GPTConfig())

    x = torch.randint(0, 256, (2, 8))
    y = torch.randint(0, 256, (2, 8))

    logits, loss = model(x, y)

    assert logits.shape == (2, 8, 256)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()

    assert model.lm_head.weight.grad is not None


def test_context_limit():
    model = GPT(GPTConfig())

    x = torch.randint(0, 256, (1, 65))

    try:
        model(x)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected ValueError for sequence length 65"
        )


def test_causality():
    torch.manual_seed(42)

    model = GPT(GPTConfig())
    model.eval()

    x = torch.randint(0, 256, (1, 8))

    logits1, _ = model(x)

    modified = x.clone()
    modified[:, -1] = (modified[:, -1] + 1) % 256

    logits2, _ = model(modified)

    # Earlier positions must not change.
    assert torch.allclose(
        logits1[:, :-1, :],
        logits2[:, :-1, :],
        atol=1e-5
    )


if __name__ == "__main__":
    test_logits_shape()
    test_parameter_count()
    test_weight_tying()
    test_training_loss()
    test_context_limit()
    test_causality()

    print("All GPT model tests passed!")
