
import torch
import torch.nn as nn

from config import GPTConfig

class TokenEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

    def forward(self, token_ids):
        return self.embedding(token_ids)


if __name__ == "__main__":
    config = GPTConfig()
    model = TokenEmbedding(config)

    x = torch.tensor([[100, 101, 102]])
    output = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)
