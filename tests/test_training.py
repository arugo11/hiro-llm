from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hiro_llm.config import (
    DataConfig,
    HubConfig,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
)
from hiro_llm.model.language_model import LanguageModel
from hiro_llm.training.checkpoint import load_training_state, read_checkpoint
from hiro_llm.training.trainer import Trainer


class RecordingPublisher:
    def __init__(self) -> None:
        self.prepared = False
        self.uploads: list[Path] = []

    def prepare(self) -> None:
        self.prepared = True

    def upload(self, path: str | Path) -> None:
        self.uploads.append(Path(path))


class TinyVisionTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, pixels, input_ids, labels):
        prediction = pixels.mean() * self.scale + input_ids.float().mean() * self.scale
        loss = (prediction - labels.float().mean()).square()
        return prediction, loss


def test_training_checkpoint_upload_and_round_trip(tmp_path: Path) -> None:
    model_config = ModelConfig(
        vocab_size=16,
        max_sequence_length=4,
        embedding_dim=8,
        hidden_dim=16,
        num_attention_heads=2,
        num_layers=1,
        num_relative_positions=4,
    )
    config = ProjectConfig(
        model=model_config,
        training=TrainingConfig(
            batch_size=2,
            total_steps=2,
            checkpoint_every=1,
            log_every=1,
            warmup_steps=0,
            amp=False,
        ),
        data=DataConfig(kind="text"),
        hub=HubConfig(repo_id="owner/model"),
    )
    inputs = torch.randint(0, 16, (4, 4))
    batches = DataLoader(TensorDataset(inputs, inputs.clone()), batch_size=2)
    model = LanguageModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    publisher = RecordingPublisher()
    trainer = Trainer(
        model,
        optimizer,
        batches,
        config,
        torch.device("cpu"),
        publisher,  # type: ignore[arg-type]
        tmp_path,
    )
    checkpoint = trainer.train()
    assert publisher.prepared
    assert len(publisher.uploads) == 2
    payload = read_checkpoint(checkpoint)
    assert payload["format_version"] == 1 and payload["step"] == 2

    restored = LanguageModel(model_config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    restored_payload = load_training_state(checkpoint, restored, restored_optimizer)
    assert restored_payload["step"] == 2
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


@pytest.mark.parametrize("task", ["vision_pretrain", "vision_instruction_tuning"])
def test_vision_training_stages_accept_three_tensor_batches(task: str, tmp_path: Path) -> None:
    config = ProjectConfig(
        task=task,  # type: ignore[arg-type]
        training=TrainingConfig(
            batch_size=1,
            total_steps=1,
            checkpoint_every=1,
            log_every=1,
            warmup_steps=0,
            amp=False,
        ),
        data=DataConfig(kind="vision"),
        hub=HubConfig(repo_id="owner/model"),
    )
    pixels = torch.randn(1, 3, 2, 2)
    tokens = torch.ones(1, 4, dtype=torch.long)
    labels = torch.zeros(1, 4, dtype=torch.long)
    batches = DataLoader(TensorDataset(pixels, tokens, labels), batch_size=1)
    model = TinyVisionTrainingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    publisher = RecordingPublisher()
    checkpoint = Trainer(
        model,
        optimizer,
        batches,
        config,
        torch.device("cpu"),
        publisher,  # type: ignore[arg-type]
        tmp_path / task,
    ).train()
    assert checkpoint.is_file()
    assert len(publisher.uploads) == 1
