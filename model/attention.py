
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, block_size: int):
        super().__init__()

        self.block_size = block_size

        # True means the position is allowed.
        mask = torch.tril(
            torch.ones(block_size, block_size, dtype=torch.bool)
        )

        # A buffer moves with the model to CPU/GPU
        # but is not a trainable parameter.
        self.register_buffer("causal_mask", mask)

    def forward(self, q, k, v):
        # Each tensor: [B, H, T, D]
        B, H, T, D = q.shape

        if T > self.block_size:
            raise ValueError("Sequence exceeds block_size")

        # Step 1: Q @ K transpose
        scores = q @ k.transpose(-2, -1)

        # Shape: [B, H, T, T]

        # Step 2: Scale
        scores = scores / math.sqrt(D)

        # Step 3: Causal mask
        mask = self.causal_mask[:T, :T]
        scores = scores.masked_fill(~mask, float("-inf"))

        # Step 4: Softmax
        weights = F.softmax(scores, dim=-1)

        # Step 5: Weighted sum of Values
        output = weights @ v

        # Shape: [B, H, T, D]
        return output
