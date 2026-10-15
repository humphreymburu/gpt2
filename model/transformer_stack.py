import torch
import torch.nn as nn

from config import GPTConfig
from model.transformer_block import TransformerBlock


class TransformerStack(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

    def forward(self, x):
        # x: [B, T, C] -> [B, T, C]
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    config = GPTConfig()
    stack = TransformerStack(config)

    x = torch.randn(4, 64, config.n_embd)
    y = stack(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    params = sum(p.numel() for p in stack.parameters())
    print("Parameters:", params)

    assert y.shape == x.shape
    assert params == config.n_layer * 38360
