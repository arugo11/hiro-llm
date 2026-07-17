from types import SimpleNamespace

import torch
from torch import nn

from hiro_llm.config import ModelConfig, VisionConfig
from hiro_llm.model.attention import RelativePositionBias
from hiro_llm.model.language_model import IGNORE_INDEX, LanguageModel
from hiro_llm.model.vision_language_model import VisionLanguageModel


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32,
        max_sequence_length=8,
        embedding_dim=16,
        hidden_dim=32,
        num_attention_heads=4,
        num_layers=2,
        num_relative_positions=8,
    )


class FakeVisionEncoder(nn.Module):
    config = SimpleNamespace(hidden_size=8)

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 8)

    def forward(self, pixel_values: torch.Tensor):
        pooled = pixel_values.mean(dim=(-2, -1))
        patch = self.projection(pooled).unsqueeze(1).repeat(1, 4, 1)
        cls = torch.zeros(patch.shape[0], 1, patch.shape[-1], device=patch.device)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patch), dim=1))


def test_relative_position_bias_shape() -> None:
    bias = RelativePositionBias(num_heads=4, max_distance=8)(5, torch.device("cpu"))
    assert bias.shape == (1, 4, 5, 5)


def test_language_model_forward_loss_and_generation() -> None:
    torch.manual_seed(0)
    model = LanguageModel(tiny_model_config()).eval()
    inputs = torch.randint(0, 32, (2, 8))
    labels = inputs.clone()
    labels[:, :4] = IGNORE_INDEX
    logits, loss = model(inputs, labels)
    generated = model.generate(inputs[:, :3], max_new_tokens=2, temperature=0.8, top_k=5)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and torch.isfinite(loss)
    assert generated.shape == (2, 5)


def test_causal_attention_does_not_observe_future_tokens() -> None:
    torch.manual_seed(0)
    model = LanguageModel(tiny_model_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5]])
    second = torch.tensor([[1, 2, 3, 9, 10]])
    first_logits, _ = model(first)
    second_logits, _ = model(second)
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3])


def test_vision_model_forward_and_freezing() -> None:
    language_model = LanguageModel(tiny_model_config())
    vision_config = VisionConfig(encoder_name="fake", num_image_tokens=4, projector_hidden_dim=12)
    model = VisionLanguageModel(language_model, vision_config, FakeVisionEncoder())
    pixels = torch.randn(2, 3, 4, 4)
    inputs = torch.randint(0, 32, (2, 8))
    labels = inputs.clone()
    labels[:, :4] = IGNORE_INDEX
    logits, loss = model(pixels, inputs, labels)
    generated = model.generate(pixels[:1], inputs[:1, :6], max_new_tokens=2)
    assert logits.shape == (2, 8, 32)
    assert loss is not None and torch.isfinite(loss)
    assert generated.shape == (1, 8)
    model.freeze_for_stage(instruction_tuning=False)
    assert not any(parameter.requires_grad for parameter in model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    model.train()
    assert not model.vision_encoder.training
    model.freeze_for_stage(instruction_tuning=True)
    assert all(parameter.requires_grad for parameter in model.language_model.parameters())
