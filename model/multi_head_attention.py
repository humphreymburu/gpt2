import torch
import torch.nn as nn

from config import GPTConfig
from model.qkv_projection import QKVProjection
from model.attention import CausalSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = QKVProjection(config)
        self.attn = CausalSelfAttention(config.block_size)

        # Mixes the concatenated heads back together.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

    def _split_heads(self, x):
        # [B, T, C] -> [B, H, T, D]
        B, T, C = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        q, k, v = self.qkv(x)
        q, k, v = map(self._split_heads, (q, k, v))

        out = self.attn(q, k, v)
        # [B, H, T, D] -> [B, T, C]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


if __name__ == "__main__":
    config = GPTConfig()
    mha = MultiHeadAttention(config)

    x = torch.randn(4, 64, config.n_embd)
    y = mha(x)

    print("Input:", x.shape)
    print("Output:", y.shape)
    print("Parameters:", sum(p.numel() for p in mha.parameters()))

    assert y.shape == x.shape
