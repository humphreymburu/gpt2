
from pathlib import Path
import torch
from tokenizer import ByteTokenizer

text = Path("data/input.txt").read_text(
    encoding="utf-8"
)

tokenizer = ByteTokenizer()
tokens = tokenizer.encode(text)

data = torch.tensor(tokens, dtype=torch.long)

print("Characters:", len(text))
print("Tokens:", len(data))
print("First 20 token IDs:", data[:20].tolist())

torch.save(data, "data/tokens.pt")
