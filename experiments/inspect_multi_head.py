import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from config import GPTConfig
from model.multi_head_attention import MultiHeadAttention

torch.manual_seed(42)
torch.set_printoptions(precision=3, linewidth=140, sci_mode=False)

config = GPTConfig()
mha = MultiHeadAttention(config).eval()

B, T, C = 1, 6, config.n_embd
H, D = config.n_head, mha.head_dim

x = torch.randn(B, T, C)
print(f"x:        [B={B}, T={T}, C={C}]")

with torch.no_grad():
    # Step 1: fused projection, then split into Q, K, V.
    q, k, v = mha.qkv(x)
    print(f"q, k, v:  {tuple(q.shape)} each")

    # Step 2: split channels into heads.
    q, k, v = map(mha._split_heads, (q, k, v))
    print(f"per head: {tuple(q.shape)}  = [B, H={H}, T, D={D}]")

    # Step 3: scores = Q K^T / sqrt(D)
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)
    print(f"scores:   {tuple(scores.shape)}  = [B, H, T, T]")

    # Step 4: causal mask, then softmax.
    mask = mha.attn.causal_mask[:T, :T]
    masked = scores.masked_fill(~mask, float("-inf"))
    weights = F.softmax(masked, dim=-1)

    print("\nCausal mask (True = allowed):")
    print(mask.int())

    for h in range(H):
        print(f"\nAttention weights, head {h}:")
        print(weights[0, h])

    # Step 5: weighted sum of values, merge heads, output projection.
    out = weights @ v
    merged = out.transpose(1, 2).contiguous().view(B, T, C)
    y = mha.out_proj(merged)
    print(f"\nper-head output: {tuple(out.shape)}")
    print(f"merged heads:    {tuple(merged.shape)}")
    print(f"final output:    {tuple(y.shape)}")

    # Checks.
    assert torch.allclose(weights.sum(-1), torch.ones(B, H, T), atol=1e-6)
    assert torch.all(weights.triu(1) == 0)
    assert torch.allclose(y, mha(x), atol=1e-6)

print("\nOK: rows sum to 1, no weight on future tokens, "
      "manual result matches module output.")
