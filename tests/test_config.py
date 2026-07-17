from pathlib import Path

import pytest

from hiro_llm.config import load_config
from hiro_llm.model import LanguageModel


def write_config(path: Path, extra: str = "") -> Path:
    path.write_text(
        """
task: pretrain
model:
  vocab_size: 32
  max_sequence_length: 8
  embedding_dim: 16
  hidden_dim: 32
  num_attention_heads: 4
  num_layers: 2
  num_relative_positions: 8
training:
  total_steps: 2
  warmup_steps: 0
data:
  kind: text
  source: sample.txt
vision: {}
hub: {}
runtime: {}
"""
        + extra,
        encoding="utf-8",
    )
    return path


def test_load_config_and_override(tmp_path: Path) -> None:
    config = load_config(
        write_config(tmp_path / "config.yaml"), ["training.batch_size=3", "runtime.device=cpu"]
    )
    assert config.training.batch_size == 3
    assert config.runtime.device == "cpu"


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8").replace("  vocab_size: 32", "  unknown: 1")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_invalid_model_dimensions_are_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path / "config.yaml")
    text = path.read_text(encoding="utf-8").replace("embedding_dim: 16", "embedding_dim: 15")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="divisible"):
        load_config(path)


def test_checked_in_text_and_vision_stages_are_weight_compatible() -> None:
    root = Path(__file__).parents[1]
    text_config = load_config(root / "configs/instruction_tuning.yaml")
    vision_config = load_config(root / "configs/vision_pretrain.yaml")
    text_model = LanguageModel(text_config.model)
    vision_model = LanguageModel(vision_config.model)
    vision_model.load_state_dict(text_model.state_dict())
