import torch
import torch.nn as nn

from config import GPTConfig
from model.embeddings import TokenEmbedding
from model.positional_embeddings import PositionalEmbedding

class InputEmbeddings(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)

    def forward(self, token_ids):
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(token_ids)

        x = token_vectors + position_vectors

        return x


if __name__ == "__main__":
    config = GPTConfig()
    model = InputEmbeddings(config)

    token_ids = torch.tensor([
        [100, 101, 102],
        [97, 98, 99]
    ], dtype=torch.long)

    output = model(token_ids)

    print("Input shape:", token_ids.shape)
    print("Output shape:", output.shape)

    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    assert output.shape == (2, 3, 56)
