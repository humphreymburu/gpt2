
import torch
import torch.nn as nn

from config import GPTConfig
from model.layer_norm import LayerNorm
from model.multi_head_attention import MultiHeadAttention
from model.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        # First sublayer: attention
        self.ln1 = LayerNorm(config.n_embd)
        self.attention = MultiHeadAttention(config)

        # Second sublayer: feed-forward network
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # x: [B, T, 56]

        # Residual connection 1
        x = x + self.attention(self.ln1(x))

        # Residual connection 2
        x = x + self.mlp(self.ln2(x))

        # Output: [B, T, 56]
        return x


if __name__ == "__main__":
    config = GPTConfig()

    block = TransformerBlock(config)

    x = torch.randn(4, 64, config.n_embd)

    y = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    params = sum(
        p.numel() for p in block.parameters()
    )

    print("Parameters:", params)

    assert y.shape == x.shape
    assert params == 38360
