
from pathlib import Path

import torch

from config import GPTConfig
from model.gpt import GPT


# -----------------------------------------
# Configuration
# -----------------------------------------

config = GPTConfig()

BATCH_SIZE = 4
MAX_STEPS = 2000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 200
EVAL_BATCHES = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)


# -----------------------------------------
# Load dataset
# -----------------------------------------

tokens = torch.load(
    "data/tokens.pt",
    map_location="cpu",
    weights_only=True,
).long().flatten()

assert len(tokens) > config.block_size + 1

# Split into train and validation data.
split = int(0.9 * len(tokens))

train_data = tokens[:split]
val_data = tokens[split:]

# Both splits must support a complete shifted window.
assert len(train_data) > config.block_size
assert len(val_data) > config.block_size, (
    "Dataset is too small for a 90/10 split "
    "with block_size=64"
)

print("Training tokens:", len(train_data))
print("Validation tokens:", len(val_data))


# -----------------------------------------
# Batch creation
# -----------------------------------------

def get_batch(data):
    T = config.block_size

    starts = torch.randint(
        0,
        len(data) - T,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[i:i + T]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1:i + T + 1]
        for i in starts
    ])

    return x.to(DEVICE), y.to(DEVICE)


# -----------------------------------------
# Model and optimizer
# -----------------------------------------

model = GPT(config).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

print("Device:", DEVICE)
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# -----------------------------------------
# Evaluation
# -----------------------------------------

@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for name, data in [
        ("train", train_data),
        ("val", val_data),
    ]:
        losses = []

        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data)

            _, loss = model(x, y)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results


# -----------------------------------------
# Checkpoint saving
# -----------------------------------------

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)


def save_checkpoint(step, filename):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.__dict__,
            "step": step,
        },
        checkpoint_dir / filename,
    )


# -----------------------------------------
# Training loop
# -----------------------------------------

model.train()

best_val_loss = float("inf")

for step in range(MAX_STEPS):

    # Evaluate before training at step 0,
    # then at regular intervals.
    if step % EVAL_INTERVAL == 0:
        losses = estimate_loss()

        print(
            f"step {step:4d} | "
            f"train loss {losses['train']:.4f} | "
            f"val loss {losses['val']:.4f}"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(step, "tiny_gpt_best.pt")

    # Get a training batch
    x, y = get_batch(train_data)

    # Clear previous gradients
    optimizer.zero_grad(set_to_none=True)

    # Forward pass
    logits, loss = model(x, y)

    # Backward pass
    loss.backward()

    # Clip gradients to improve stability
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Update model parameters
    optimizer.step()


# -----------------------------------------
# Final evaluation and checkpoint
# -----------------------------------------

losses = estimate_loss()

print(
    f"Final | "
    f"train loss {losses['train']:.4f} | "
    f"val loss {losses['val']:.4f}"
)

save_checkpoint(MAX_STEPS, "tiny_gpt.pt")

print("Training complete!")
print("Saved checkpoints in checkpoints/")
