
import torch

data = torch.load(
    "data/tokens.pt",
    weights_only=True
)

block_size = 64
batch_size = 4

def get_batch():
    if len(data) <= block_size:
        raise ValueError(
            "Dataset must contain more than 64 tokens"
        )

    starts = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in starts
    ])

    return x, y


x, y = get_batch()

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("Input:", x[0, :10].tolist())
print("Target:", y[0, :10].tolist())

assert x.shape == (4, 64)
assert y.shape == (4, 64)
assert torch.equal(x[:, 1:], y[:, :-1])
