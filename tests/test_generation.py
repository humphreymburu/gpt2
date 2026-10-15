import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import GPTConfig
from model.gpt import GPT
from generate import generate


def test_generation_length():
    torch.manual_seed(42)

    model = GPT(GPTConfig())

    prompt = torch.tensor(
        [[100, 101, 102]],
        dtype=torch.long
    )

    output = generate(
        model,
        prompt,
        max_new_tokens=10,
        temperature=1.0,
        top_k=20
    )

    assert output.shape == (1, 13)


def test_generation_preserves_prompt():
    model = GPT(GPTConfig())

    prompt = torch.tensor(
        [[100, 101, 102]],
        dtype=torch.long
    )

    output = generate(
        model,
        prompt,
        max_new_tokens=5
    )

    assert torch.equal(
        output[:, :3],
        prompt
    )


def test_generation_respects_context_window():
    model = GPT(GPTConfig())

    # Prompt longer than block_size.
    prompt = torch.randint(
        0, 256, (1, 80)
    )

    output = generate(
        model,
        prompt,
        max_new_tokens=5
    )

    assert output.shape == (1, 85)


if __name__ == "__main__":
    test_generation_length()
    test_generation_preserves_prompt()
    test_generation_respects_context_window()

    print("All generation tests passed!")