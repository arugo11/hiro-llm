from torch import nn

from hiro_llm.config import ModelConfig
from hiro_llm.model.attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attention(self.attention_norm(hidden_states))
        return hidden_states + self.feed_forward(self.feed_forward_norm(hidden_states))
