import torch
import torch.nn as nn

from config import GPTConfig

class MLP(nn.Module):
    def __init__(self,config: GPTConfig):
        super().__init__()

        hidden_dim = 4 * config.n_embd

        self.fc1 = nn.Linear(
            config.n_embd,
            hidden_dim
        )

        #GPT-2 STYLE GELU
        self.activation = nn.GELU(approximate="tanh")

        #project back 224 -56
        self.fc2 = nn.Linear(
            hidden_dim,
            config.n_embd
        )

    def forward(self, x):
        # x: [B, T, 56]

        x = self.fc1(x)
        # [B, T, 224]

        x = self.activation(x)
        # [B, T, 224]

        x = self.fc2(x)
        # [B, T, 56]

        return x


if __name__ == "__main__":
    config = GPTConfig()
    model = MLP(config)

    x = torch.randn(4, 64, 56)
    output = model(x)

    print("Input:", x.shape)
    print("Output:", output.shape)

    params = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", params)

    assert output.shape == (4, 64, 56)
    assert params == 25368
