import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import GPTConfig
from model.input_embeddings import InputEmbeddings


def test_positions_distinguish_repeated_tokens():
    torch.manual_seed(42)

    model = InputEmbeddings(GPTConfig())

    # The same byte token appears three times.
    ids = torch.tensor([[100, 100, 100]])

    # Token embeddings alone.
    tok = model.token_embedding(ids)

    # Combined token + position embeddings.
    combined = model(ids)

    # Same token ID gives identical token vectors.
    assert torch.equal(tok[0, 0], tok[0, 1])

    # But position makes combined vectors different.
    assert not torch.equal(combined[0, 0], combined[0, 1])

    print("Token vectors equal:", torch.equal(tok[0, 0], tok[0, 1]))
    print("Combined vectors equal:", torch.equal(combined[0, 0], combined[0, 1]))


if __name__ == "__main__":
    test_positions_distinguish_repeated_tokens()
