import json
from pathlib import Path

import numpy as np
import pytest

from hiro_llm.config import DataConfig, ModelConfig, VisionConfig
from hiro_llm.data.text import prepare_instruction_data, prepare_pretraining_data
from hiro_llm.data.vision import prepare_vision_data
from hiro_llm.model.language_model import IGNORE_INDEX


def model_config(sequence_length: int = 16) -> ModelConfig:
    return ModelConfig(
        vocab_size=50_257,
        max_sequence_length=sequence_length,
        embedding_dim=16,
        hidden_dim=32,
        num_attention_heads=4,
        num_layers=1,
        num_relative_positions=sequence_length,
    )


def test_prepare_pretraining_text(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("A short training sentence repeated. " * 10, encoding="utf-8")
    output = prepare_pretraining_data(
        DataConfig(source=str(source), output_dir=str(tmp_path / "processed"))
    )
    tokens = np.load(output)
    assert tokens.ndim == 1 and len(tokens) > 10


def test_pretraining_split_manifest_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("deterministic text " * 100, encoding="utf-8")
    config = DataConfig(
        source=str(source), output_dir=str(tmp_path / "processed"), validation_fraction=0.1
    )
    prepare_pretraining_data(config)
    first = np.load(tmp_path / "processed" / "train_tokens.npy").copy()
    prepare_pretraining_data(config)
    second = np.load(tmp_path / "processed" / "train_tokens.npy").copy()
    np.testing.assert_array_equal(first, second)
    manifest = json.loads((tmp_path / "processed" / "manifest.json").read_text())
    assert manifest["validation_token_count"] > 0


def test_messages_are_converted_to_prompt_and_response(tmp_path: Path) -> None:
    source = tmp_path / "messages.jsonl"
    source.write_text(
        json.dumps({
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]
        }) + "\n",
        encoding="utf-8",
    )
    inputs, labels = prepare_instruction_data(
        DataConfig(kind="instruction", source=str(source), output_dir=str(tmp_path / "out")),
        model_config(),
    )
    assert np.any(np.load(labels) != IGNORE_INDEX)
    assert np.load(inputs).shape == (1, 16)


def test_instruction_prompt_is_masked(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    source.write_text(json.dumps({"question": "Hello?", "answer": "Hi."}) + "\n", encoding="utf-8")
    config = DataConfig(
        kind="instruction",
        source=str(source),
        output_dir=str(tmp_path / "processed"),
        prompt_field="question",
        response_field="answer",
    )
    input_path, label_path = prepare_instruction_data(config, model_config())
    inputs = np.load(input_path)
    labels = np.load(label_path)
    assert inputs.shape == labels.shape == (1, 16)
    assert labels[0, 0] == IGNORE_INDEX
    assert np.any(labels[0] != IGNORE_INDEX)


def test_invalid_jsonl_is_reported_with_line(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text("not json\n", encoding="utf-8")
    config = DataConfig(kind="instruction", source=str(source), output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="line 1"):
        prepare_instruction_data(config, model_config())


def test_prepare_vision_data_checks_and_resolves_images(tmp_path: Path) -> None:
    image = tmp_path / "image.ppm"
    image.write_text("P3\n1 1\n255\n255 0 0\n", encoding="ascii")
    source = tmp_path / "vision.jsonl"
    source.write_text(
        json.dumps({"image": image.name, "prompt": "Color?", "response": "Red."}) + "\n",
        encoding="utf-8",
    )
    config = DataConfig(
        kind="vision",
        source=str(source),
        output_dir=str(tmp_path / "processed"),
        image_root=str(tmp_path),
    )
    paths, inputs, labels = prepare_vision_data(
        config,
        model_config(sequence_length=16),
        VisionConfig(encoder_name="fake", num_image_tokens=4, projector_hidden_dim=8),
    )
    assert json.loads(paths.read_text(encoding="utf-8")) == [str(image.resolve())]
    assert np.load(inputs).shape == np.load(labels).shape == (1, 16)


def test_prepare_vision_data_rejects_missing_image(tmp_path: Path) -> None:
    source = tmp_path / "vision.jsonl"
    source.write_text(
        json.dumps({"image": "missing.png", "prompt": "What?", "response": "Nothing."}) + "\n",
        encoding="utf-8",
    )
    config = DataConfig(
        kind="vision",
        source=str(source),
        output_dir=str(tmp_path / "processed"),
    )
    with pytest.raises(FileNotFoundError, match="missing.png"):
        prepare_vision_data(
            config,
            model_config(sequence_length=16),
            VisionConfig(encoder_name="fake", num_image_tokens=4, projector_hidden_dim=8),
        )
