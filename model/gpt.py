import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from model.input_embeddings import InputEmbeddings
from model.transformer_stack import TransformerStack
from model.layer_norm import LayerNorm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        # Token + positional embeddings
        self.embeddings = InputEmbeddings(config)

        # Two independent Transformer blocks
        self.transformer = TransformerStack(config)

        # Final LayerNorm
        self.ln_f = LayerNorm(config.n_embd)

        # Vocabulary projection
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # Tie output weights to token embeddings
        self.lm_head.weight = (
            self.embeddings.token_embedding.embedding.weight
        )

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block_size {self.config.block_size}"
            )

        # Step 1: Embeddings
        x = self.embeddings(idx)
        # [B, T, 56]

        # Step 2: Transformer blocks
        x = self.transformer(x)
        # [B, T, 56]

        # Step 3: Final normalization
        x = self.ln_f(x)
        # [B, T, 56]

        # Step 4: Vocabulary logits
        logits = self.lm_head(x)
        # [B, T, 256]

        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError(
                    "targets must have the same shape as idx"
                )

            # Flatten batch and sequence dimensions
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1)
            )

        return logits, loss


if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)

    idx = torch.randint(
        0, config.vocab_size, (4, 64)
    )

    targets = torch.randint(
        0, config.vocab_size, (4, 64)
    )

    logits, loss = model(idx, targets)

    print("Input shape:", idx.shape)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())

    params = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", params)

    assert logits.shape == (4, 64, 256)
    assert params == 94752
