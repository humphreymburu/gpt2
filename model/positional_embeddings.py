
import torch
import torch.nn as nn

from config import GPTConfig


class PositionalEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.block_size,
            embedding_dim=config.n_embd
        )

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        if seq_len > self.embedding.num_embeddings:
            raise ValueError(
                "Sequence exceeds block_size"
            )

        positions = torch.arange(
            seq_len,
            device=token_ids.device
        )

        return self.embedding(positions)


if __name__ == "__main__":
    config = GPTConfig()
    model = PositionalEmbedding(config)

    x = torch.tensor([
        [100, 101, 102],
        [97, 98, 99]
    ], dtype=torch.long)

    output = model(x)

    print("Positions:", torch.arange(x.shape[1]))
    print("Position embeddings:", output.shape)
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))
