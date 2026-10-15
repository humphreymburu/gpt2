import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from config import GPTConfig
from model.qkv_projection import QKVProjection


def test_fused_matches_separate_projections():
    torch.manual_seed(42)

    config = GPTConfig()
    C = config.n_embd

    fused = QKVProjection(config)

    q_layer = nn.Linear(C, C)
    k_layer = nn.Linear(C, C)
    v_layer = nn.Linear(C, C)

    # Copy each portion of the fused weights and biases
    with torch.no_grad():
        q_layer.weight.copy_(fused.qkv_proj.weight[:C])
        k_layer.weight.copy_(fused.qkv_proj.weight[C:2*C])
        v_layer.weight.copy_(fused.qkv_proj.weight[2*C:])

        q_layer.bias.copy_(fused.qkv_proj.bias[:C])
        k_layer.bias.copy_(fused.qkv_proj.bias[C:2*C])
        v_layer.bias.copy_(fused.qkv_proj.bias[2*C:])

    x = torch.randn(4, 64, C)

    q, k, v = fused(x)

    assert torch.allclose(q, q_layer(x), atol=1e-6)
    assert torch.allclose(k, k_layer(x), atol=1e-6)
    assert torch.allclose(v, v_layer(x), atol=1e-6)

    print("Fused and separate projections match!")


if __name__ == "__main__":
    test_fused_matches_separate_projections()
