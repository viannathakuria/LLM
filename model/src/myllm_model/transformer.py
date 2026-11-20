import math

import torch
from torch import Tensor, nn

from .config import TransformerConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.qkv_projection = nn.Linear(
            config.embedding_dim,
            3 * config.embedding_dim,
        )
        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, sequence_length, embedding_dim = inputs.shape
        qkv = self.qkv_projection(inputs).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        future_positions = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=inputs.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(future_positions, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended = attention_weights @ value
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size,
            sequence_length,
            embedding_dim,
        )
        return self.residual_dropout(self.output_projection(attended))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.embedding_dim, 4 * config.embedding_dim),
            nn.GELU(),
            nn.Linear(4 * config.embedding_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.attention(self.attention_norm(inputs))
        return inputs + self.feed_forward(self.feed_forward_norm(inputs))


class TinyGPT(nn.Module):
    """A randomly initialized decoder-only Transformer that returns logits."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(
            config.context_length,
            config.embedding_dim,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(
            *(TransformerBlock(config) for _ in range(config.num_layers))
        )
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_model_head = nn.Linear(config.embedding_dim, config.vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        if not isinstance(input_ids, Tensor):
            raise TypeError("input_ids must be a torch.Tensor")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs")

        batch_size, sequence_length = input_ids.shape
        if batch_size < 1 or sequence_length < 1:
            raise ValueError(
                "input_ids must have non-empty batch and sequence dimensions"
            )
        if sequence_length > self.config.context_length:
            raise ValueError(
                "input sequence exceeds the configured context length "
                f"({self.config.context_length})"
            )
        if torch.any(input_ids < 0).item() or torch.any(
            input_ids >= self.config.vocab_size
        ).item():
            raise ValueError("input_ids contain token IDs outside the vocabulary")

        positions = torch.arange(sequence_length, device=input_ids.device)
        hidden_states = self.token_embeddings(input_ids)
        hidden_states = hidden_states + self.position_embeddings(positions)
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.blocks(hidden_states)
        return self.language_model_head(self.final_norm(hidden_states))
