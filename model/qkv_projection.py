
import torch
import torch.nn as nn

from config import GPTConfig


class QKVProjection(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_embd = config.n_embd

        # Fused Query, Key, Value projection
        self.qkv_proj = nn.Linear(
            in_features=config.n_embd,
            out_features=3 * config.n_embd,
            bias=True
        )

    def forward(self, x):
        # x: [B, T, C]
        qkv = self.qkv_proj(x)

        # qkv: [B, T, 3*C]
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # Each: [B, T, C]
        return q, k, v


if __name__ == "__main__":
    config = GPTConfig()

    model = QKVProjection(config)

    # Simulated normalized token embeddings
    x = torch.randn(4, 64, 56)

    q, k, v = model(x)

    print("Input:", x.shape)
    print("Query:", q.shape)
    print("Key:", k.shape)
    print("Value:", v.shape)

    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    assert q.shape == (4, 64, 56)
    assert k.shape == (4, 64, 56)
    assert v.shape == (4, 64, 56)
