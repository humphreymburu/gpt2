
import torch
import torch.nn as nn

from config import GPTConfig
from model.input_embeddings import InputEmbeddings
from model.layer_norm import LayerNorm


class NormalizedInput(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.embeddings = InputEmbeddings(config)
        self.ln = LayerNorm(config.n_embd)

    def forward(self, token_ids):
        x = self.embeddings(token_ids)
        x = self.ln(x)
        return x


if __name__ == "__main__":
    config = GPTConfig()
    model = NormalizedInput(config)

    token_ids = torch.tensor([
        [100, 101, 102],
        [97, 98, 99]
    ], dtype=torch.long)

    output = model(token_ids)

    print("Input:", token_ids.shape)
    print("Output:", output.shape)

    assert output.shape == (2, 3, 56)
