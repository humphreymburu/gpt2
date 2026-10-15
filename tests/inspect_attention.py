
import torch

from config import GPTConfig
from qkv_projection import QKVProjection
from attention import CausalSelfAttention

config = GPTConfig()

projection = QKVProjection(config)
attention = CausalSelfAttention(config.block_size)

x = torch.randn(4, 64, 56)

q, k, v = projection(x)

B, T, C = q.shape
H = config.n_head
D = C // H

def split_heads(tensor):
    return tensor.view(B, T, H, D).transpose(1, 2)

q = split_heads(q)
k = split_heads(k)
v = split_heads(v)

print("Q:", q.shape)
print("K:", k.shape)
print("V:", v.shape)

output = attention(q, k, v)

print("Attention output:", output.shape)

assert output.shape == (4, 4, 64, 14)
