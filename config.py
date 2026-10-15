from dataclasses import dataclass

@dataclass(frozen=True)
class GPTConfig: 
    vocab_size: int = 256
    block_size: int = 64
    n_embd: int = 56
    n_head: int = 4
    n_layer: int = 2

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.vocab_size > 0
        assert self.block_size > 0
