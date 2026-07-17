from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hiro_llm.config import ModelConfig
from hiro_llm.model.transformer import TransformerBlock

IGNORE_INDEX = -100


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids)

    def logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden_states = embeddings
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output_projection(self.final_norm(hidden_states))

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.shape[-1] > self.config.max_sequence_length:
            raise ValueError("input sequence exceeds model.max_sequence_length")
        logits = self.logits_from_embeddings(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            conditioned = input_ids[:, -self.config.max_sequence_length :]
            logits, _ = self(conditioned)
            next_logits = logits[:, -1] / temperature
            if top_k is not None:
                if top_k <= 0:
                    raise ValueError("top_k must be positive")
                threshold = torch.topk(next_logits, min(top_k, next_logits.shape[-1])).values[
                    :, -1:
                ]
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))
            probabilities = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids
