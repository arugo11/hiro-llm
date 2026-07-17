from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from hiro_llm.config import VisionConfig
from hiro_llm.model.language_model import IGNORE_INDEX, LanguageModel


class VisionLanguageModel(nn.Module):
    def __init__(
        self,
        language_model: LanguageModel,
        config: VisionConfig,
        vision_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.config = config
        if vision_encoder is None:
            try:
                from transformers import CLIPVisionModel
            except ImportError as exc:
                raise ImportError("Install hiro-llm[vision] to use vision models") from exc
            vision_encoder = CLIPVisionModel.from_pretrained(config.encoder_name)
        self.vision_encoder = vision_encoder
        encoder_dim = int(self.vision_encoder.config.hidden_size)
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, config.projector_hidden_dim),
            nn.GELU(),
            nn.Linear(config.projector_hidden_dim, language_model.config.embedding_dim),
        )

    def freeze_for_stage(self, instruction_tuning: bool) -> None:
        self.vision_encoder.requires_grad_(False)
        self.projector.requires_grad_(True)
        self.language_model.requires_grad_(instruction_tuning)
        self.vision_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_encoder.eval()
        return self

    def multimodal_embeddings(
        self, pixel_values: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_features = vision_outputs.last_hidden_state[:, 1:, :]
        if image_features.shape[1] != self.config.num_image_tokens:
            raise ValueError(
                f"vision encoder produced {image_features.shape[1]} patches; "
                f"expected {self.config.num_image_tokens}"
            )
        if input_ids.shape[1] < self.config.num_image_tokens:
            raise ValueError("input sequence is shorter than num_image_tokens")
        text_embeddings = self.language_model.embed(input_ids)
        image_embeddings = self.projector(image_features).to(text_embeddings.dtype)
        return torch.cat(
            (image_embeddings, text_embeddings[:, self.config.num_image_tokens :, :]), dim=1
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        embeddings = self.multimodal_embeddings(pixel_values, input_ids)
        logits = self.language_model.logits_from_embeddings(embeddings)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=IGNORE_INDEX
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            text_budget = (
                self.language_model.config.max_sequence_length - self.config.num_image_tokens
            )
            image_placeholders = input_ids[:, : self.config.num_image_tokens]
            text_tokens = input_ids[:, self.config.num_image_tokens :]
            conditioned = torch.cat((image_placeholders, text_tokens[:, -text_budget:]), dim=1)
            logits, _ = self(pixel_values, conditioned)
            probabilities = F.softmax(logits[:, -1] / temperature, dim=-1)
            input_ids = torch.cat(
                (input_ids, torch.multinomial(probabilities, num_samples=1)), dim=1
            )
        return input_ids
