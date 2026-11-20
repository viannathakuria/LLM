import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the small decoder-only Transformer."""

    vocab_size: int
    context_length: int = 64
    num_layers: int = 2
    num_heads: int = 2
    embedding_dim: int = 64
    batch_size: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "context_length",
            "num_layers",
            "num_heads",
            "embedding_dim",
            "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be a finite number in [0, 1)")
