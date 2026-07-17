from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hiro_llm.config import ModelConfig


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads: int, max_distance: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        self.embedding = nn.Embedding(max_distance, num_heads)

    def forward(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        distances = positions[:, None] - positions[None, :]
        distances = distances.clamp(min=0, max=self.max_distance - 1)
        return self.embedding(distances).permute(2, 0, 1).unsqueeze(0)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embedding_dim // config.num_attention_heads
        self.qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=False)
        self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.position_bias = RelativePositionBias(
            config.num_attention_heads, config.num_relative_positions
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden_states.shape
        qkv = self.qkv(hidden_states).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1) * self.head_dim**-0.5
        scores = scores + self.position_bias(sequence_length, hidden_states.device)
        causal_mask = torch.ones(
            sequence_length, sequence_length, device=hidden_states.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = self.dropout(F.softmax(scores, dim=-1))
        context = probabilities @ value
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        )
        return self.output(context)
