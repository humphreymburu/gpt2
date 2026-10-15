import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from config import GPTConfig
from tokenizer import ByteTokenizer

torch.manual_seed(42)

config = GPTConfig()
tokenizer = ByteTokenizer()

tokens = torch.load(
    ROOT / "data" / "tokens.pt",
    map_location="cpu",
    weights_only=True,
).long().flatten()

T = config.block_size
B = 2

print("Total tokens:", len(tokens))
print("Token range:", tokens.min().item(), "to", tokens.max().item())

# Same sampling rule as train.py.
starts = torch.randint(0, len(tokens) - T, (B,))
x = torch.stack([tokens[i:i + T] for i in starts])
y = torch.stack([tokens[i + 1:i + T + 1] for i in starts])

print("\nx shape:", tuple(x.shape))
print("y shape:", tuple(y.shape))

for b in range(B):
    print(f"\n--- sequence {b} (start = {starts[b].item()}) ---")
    print("x text:", repr(tokenizer.decode(x[b].tolist())))
    print("y text:", repr(tokenizer.decode(y[b].tolist())))

# Show the next-token pairs for the first sequence.
print("\nFirst 8 (input -> target) pairs of sequence 0:")
for t in range(8):
    xi, yi = x[0, t].item(), y[0, t].item()
    print(f"  {xi:3d} {chr(xi)!r:>8} -> {yi:3d} {chr(yi)!r}")

assert torch.equal(x[:, 1:], y[:, :-1])
print("\nOK: y is x shifted left by one token.")

# Loss of a model that guesses uniformly. Untrained GPT should start near this.
print(f"Uniform-guess loss: ln({config.vocab_size}) = "
      f"{torch.log(torch.tensor(float(config.vocab_size))).item():.4f}")
