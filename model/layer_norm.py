import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd))

    def forward(self, x):
        # normalize each token vector across the embedding dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        return x * self.weight + self.bias


if __name__ == "__main__":
    ln = LayerNorm(56)

    x = torch.randn(2, 3, 56)
    output = ln(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Mean (should be ~0):", output.mean(dim=-1)[0, 0].item())
    print("Std  (should be ~1):", output.std(dim=-1, unbiased=False)[0, 0].item())

    # matches PyTorch's built-in LayerNorm
    assert torch.allclose(output, nn.LayerNorm(56)(x), atol=1e-5)
