import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import GPTConfig
from model.transformer_block import TransformerBlock

torch.manual_seed(42)

config = GPTConfig()
block = TransformerBlock(config).eval()

B, T, C = 2, 16, config.n_embd
x = torch.randn(B, T, C)


def rms(t):
    return t.pow(2).mean().sqrt().item()


print(f"x: [B={B}, T={T}, C={C}]  rms = {rms(x):.4f}\n")

with torch.no_grad():
    # Sublayer 1: LayerNorm -> attention -> residual add.
    n1 = block.ln1(x)
    attn_out = block.attention(n1)
    h = x + attn_out

    # Sublayer 2: LayerNorm -> MLP -> residual add.
    n2 = block.ln2(h)
    mlp_out = block.mlp(n2)
    y = h + mlp_out

    print(f"ln1(x)        rms = {rms(n1):.4f}   (normalized, near 1)")
    print(f"attention     rms = {rms(attn_out):.4f}   (update to add)")
    print(f"x + attention rms = {rms(h):.4f}")
    print(f"ln2(h)        rms = {rms(n2):.4f}")
    print(f"mlp           rms = {rms(mlp_out):.4f}   (update to add)")
    print(f"output        rms = {rms(y):.4f}")

    assert torch.allclose(y, block(x), atol=1e-6)

    # Causality through the whole block.
    x2 = x.clone()
    x2[:, 10:] = torch.randn(B, T - 10, C)
    y2 = block(x2)
    assert torch.allclose(y[:, :10], y2[:, :10], atol=1e-6)
    print("\nOK: changing tokens 10+ does not change outputs 0-9.")

print("OK: manual step-by-step result matches block(x).")

print("\nParameters per part:")
for name, module in [
    ("ln1", block.ln1), ("attention", block.attention),
    ("ln2", block.ln2), ("mlp", block.mlp),
]:
    print(f"  {name:10s} {sum(p.numel() for p in module.parameters()):6d}")
print(f"  {'total':10s} {sum(p.numel() for p in block.parameters()):6d}")
