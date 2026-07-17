from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from hiro_llm.config import DataConfig, ModelConfig
from hiro_llm.data.sources import acquire_source
from hiro_llm.model.language_model import IGNORE_INDEX


def _tokenizer(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def prepare_pretraining_data(config: DataConfig) -> Path:
    destination_dir = Path(config.output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    source = None
    text = None
    if config.source_type == "huggingface":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for Hugging Face dataset sources") from exc
        dataset = load_dataset(config.source, split=config.split, revision=config.revision)
        column = config.prompt_field if config.prompt_field in dataset.column_names else "text"
        text = "\n".join(str(item) for item in dataset[column])
    else:
        source = acquire_source(config)
    destination = destination_dir / "tokens.npy"
    if source is not None and source.suffix == ".npy":
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination
    if text is None and source is not None:
        text = source.read_text(encoding="utf-8")
    assert text is not None
    tokens = _tokenizer(config.tokenizer).encode(text, allowed_special="all")
    token_array = np.asarray(tokens, dtype=np.int32)
    split_at = int(len(token_array) * (1.0 - config.validation_fraction))
    if config.validation_fraction > 0:
        np.save(destination_dir / "train_tokens.npy", token_array[:split_at])
        np.save(destination_dir / "validation_tokens.npy", token_array[split_at:])
    np.save(destination, token_array[:split_at] if config.validation_fraction > 0 else token_array)
    manifest = {
        "source": config.source,
        "source_type": config.source_type,
        "revision": config.revision,
        "split": config.split,
        "tokenizer": config.tokenizer,
        "token_count": int(len(token_array)),
        "train_token_count": int(split_at),
        "validation_token_count": int(len(token_array) - split_at),
        "validation_fraction": config.validation_fraction,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (destination_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def _encode_instruction(
    sample: dict,
    config: DataConfig,
    model: ModelConfig,
    image_tokens: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        prompt = str(sample[config.prompt_field])
        response = str(sample[config.response_field])
    except KeyError as exc:
        raise ValueError(f"missing JSONL field: {exc.args[0]}") from exc
    encoder = _tokenizer(config.tokenizer)
    prompt_text = config.prompt_template.format(prompt=prompt)
    prompt_ids = encoder.encode(prompt_text, allowed_special="all")
    response_ids = encoder.encode(response + config.eos_text, allowed_special="all")
    max_tokens = model.max_sequence_length + 1 - image_tokens
    response_ids = response_ids[:max_tokens]
    prompt_budget = max(0, max_tokens - len(response_ids))
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    token_ids = [encoder.eot_token] * image_tokens + prompt_ids + response_ids
    target_ids = [IGNORE_INDEX] * (image_tokens + len(prompt_ids)) + response_ids
    token_ids = token_ids[: model.max_sequence_length + 1]
    target_ids = target_ids[: model.max_sequence_length + 1]
    inputs = token_ids[:-1]
    labels = target_ids[1:]
    padding = model.max_sequence_length - len(inputs)
    inputs.extend([encoder.eot_token] * padding)
    labels.extend([IGNORE_INDEX] * padding)
    return np.asarray(inputs, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def prepare_instruction_data(config: DataConfig, model: ModelConfig) -> tuple[Path, Path]:
    source = acquire_source(config) if config.source_type != "huggingface" else None
    records = []
    if config.source_type == "huggingface":
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("datasets is required for Hugging Face dataset sources") from exc
        dataset = load_dataset(config.source, split=config.split, revision=config.revision)
        records = list(dataset)
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    iterator = (
        enumerate(records, start=1)
        if records
        else enumerate(source.open(encoding="utf-8"), start=1)
    )
    for line_number, line in iterator:
        if isinstance(line, dict):
            sample = line
        else:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
        if "messages" in sample:
            messages = sample["messages"]
            if (
                not isinstance(messages, list)
                or not messages
                or messages[-1].get("role") != "assistant"
            ):
                continue
            prompt_parts = [
                str(message.get("content", ""))
                for message in messages[:-1]
                if message.get("role") in {"system", "user"}
            ]
            sample = {
                "prompt": "\n".join(prompt_parts),
                "response": messages[-1].get("content", ""),
            }
        input_ids, target_ids = _encode_instruction(sample, config, model)
        inputs.append(input_ids)
        labels.append(target_ids)
        if config.max_samples is not None and len(inputs) >= config.max_samples:
            break
    if not inputs:
        raise ValueError("instruction dataset contains no samples")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_ids.npy"
    label_path = output_dir / "labels.npy"
    np.save(input_path, np.stack(inputs))
    np.save(label_path, np.stack(labels))
    return input_path, label_path


class PretrainingDataset(Dataset):
    def __init__(self, token_path: str | Path, sequence_length: int) -> None:
        self.tokens = np.load(token_path, mmap_mode="r")
        self.sequence_length = sequence_length
        if self.tokens.ndim != 1:
            raise ValueError("pretraining token array must be one-dimensional")
        if len(self.tokens) <= sequence_length:
            raise ValueError("token array is too short for the configured sequence length")

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.sequence_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sequence_length
        chunk = np.asarray(self.tokens[start : start + self.sequence_length + 1], dtype=np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


class TokenPairDataset(Dataset):
    def __init__(self, input_path: str | Path, label_path: str | Path) -> None:
        self.inputs = np.load(input_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if self.inputs.shape != self.labels.shape or self.inputs.ndim != 2:
            raise ValueError("input and label arrays must be matching two-dimensional arrays")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = np.asarray(self.inputs[index], dtype=np.int64)
        labels = np.asarray(self.labels[index], dtype=np.int64)
        return torch.from_numpy(inputs.copy()), torch.from_numpy(labels.copy())
