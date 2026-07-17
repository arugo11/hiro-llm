from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

Task = Literal["pretrain", "instruction_tuning", "vision_pretrain", "vision_instruction_tuning"]


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 50_257
    max_sequence_length: int = 1024
    embedding_dim: int = 768
    hidden_dim: int = 3072
    num_attention_heads: int = 12
    num_layers: int = 12
    num_relative_positions: int = 1024
    dropout: float = 0.0

    def validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "max_sequence_length": self.max_sequence_length,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_attention_heads": self.num_attention_heads,
            "num_layers": self.num_layers,
            "num_relative_positions": self.num_relative_positions,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.embedding_dim % self.num_attention_heads:
            raise ValueError("model.embedding_dim must be divisible by num_attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 8
    total_steps: int = 1000
    checkpoint_every: int = 100
    log_every: int = 10
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0
    seed: int = 42
    amp: bool = True
    compile_model: bool = False
    init_from: str | None = None
    resume_from: str | None = None

    def validate(self) -> None:
        for name in ("batch_size", "total_steps", "checkpoint_every", "log_every"):
            if getattr(self, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("training.warmup_steps must be in [0, total_steps)")
        if not 0 < self.min_learning_rate <= self.max_learning_rate:
            raise ValueError("learning rates must satisfy 0 < min <= max")
        if self.weight_decay < 0 or self.gradient_clip_norm < 0:
            raise ValueError("weight_decay and gradient_clip_norm cannot be negative")


@dataclass(slots=True)
class DataConfig:
    kind: Literal["text", "instruction", "vision"] = "text"
    source_type: Literal["local", "http", "huggingface"] = "local"
    source: str = "data/raw/input.txt"
    source_filename: str | None = None
    output_dir: str = "data/processed"
    tokenizer: str = "gpt2"
    prompt_field: str = "prompt"
    response_field: str = "response"
    image_field: str = "image"
    image_root: str | None = None
    prompt_template: str = "<user>{prompt}<assistant>"
    eos_text: str = "<|endoftext|>"
    max_samples: int | None = None
    revision: str | None = None
    split: str = "train"
    validation_fraction: float = 0.0
    seed: int = 42

    def validate(self) -> None:
        if not self.source:
            raise ValueError("data.source is required")
        if self.source_type == "huggingface" and not self.source:
            raise ValueError("data.source is required for a Hugging Face source")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("data.max_samples must be positive")
        if "{prompt}" not in self.prompt_template:
            raise ValueError("data.prompt_template must contain {prompt}")
        if not 0 <= self.validation_fraction < 1:
            raise ValueError("data.validation_fraction must be in [0, 1)")


@dataclass(slots=True)
class VisionConfig:
    encoder_name: str = "openai/clip-vit-large-patch14"
    num_image_tokens: int = 256
    projector_hidden_dim: int = 2048
    language_model_checkpoint: str | None = None

    def validate(self) -> None:
        if not self.encoder_name:
            raise ValueError("vision.encoder_name is required")
        if self.num_image_tokens <= 0 or self.projector_hidden_dim <= 0:
            raise ValueError("vision dimensions must be positive")


@dataclass(slots=True)
class HubConfig:
    repo_id: str = ""
    private: bool = False
    token_env: str = "HF_TOKEN"
    checkpoint_filename: str | None = None

    def validate(self) -> None:
        if not self.token_env:
            raise ValueError("hub.token_env is required")


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("runtime.device must be auto, cpu, cuda, or mps")
        if self.num_workers < 0:
            raise ValueError("runtime.num_workers cannot be negative")


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = True
    project: str = "hiro-llm"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    offline_dir: str = "wandb_offline"
    retries: int = 3

    def validate(self) -> None:
        if not self.project:
            raise ValueError("wandb.project is required")
        if self.retries < 0:
            raise ValueError("wandb.retries cannot be negative")


@dataclass(slots=True)
class BenchmarkConfig:
    datasets_revision: str | None = None
    context_stride: int | None = None
    max_examples: int | None = None

    def validate(self) -> None:
        if self.context_stride is not None and self.context_stride <= 0:
            raise ValueError("benchmark.context_stride must be positive")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("benchmark.max_examples must be positive")


@dataclass(slots=True)
class ProjectConfig:
    task: Task = "pretrain"
    variant: str = "baseline"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    hub: HubConfig = field(default_factory=HubConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def validate(self) -> None:
        if not self.variant or any(char.isspace() for char in self.variant):
            raise ValueError("variant must be a non-empty identifier without whitespace")
        if self.task.startswith("vision") and self.data.kind != "vision":
            raise ValueError("vision tasks require data.kind=vision")
        if self.task.startswith("vision") and (
            self.model.max_sequence_length <= self.vision.num_image_tokens
        ):
            raise ValueError("model.max_sequence_length must exceed vision.num_image_tokens")
        if self.task == "pretrain" and self.data.kind != "text":
            raise ValueError("pretrain requires data.kind=text")
        if self.task == "instruction_tuning" and self.data.kind != "instruction":
            raise ValueError("instruction_tuning requires data.kind=instruction")
        if self.task == "instruction_tuning" and not (
            self.training.init_from or self.training.resume_from
        ):
            raise ValueError("instruction_tuning requires training.init_from or resume_from")
        if self.task == "vision_pretrain" and not self.vision.language_model_checkpoint:
            raise ValueError("vision_pretrain requires vision.language_model_checkpoint")
        if self.task == "vision_instruction_tuning" and not (
            self.training.init_from or self.training.resume_from
        ):
            raise ValueError("vision_instruction_tuning requires training.init_from or resume_from")
        for section in (
            self.model,
            self.training,
            self.data,
            self.vision,
            self.hub,
            self.runtime,
            self.wandb,
            self.benchmark,
        ):
            section.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _convert_value(expected_type: Any, value: Any, path: str) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)
    if dataclasses.is_dataclass(expected_type):
        if not isinstance(value, dict):
            raise TypeError(f"{path} must be a mapping")
        return _dataclass_from_dict(expected_type, value, path)
    if origin is Literal:
        if value not in args:
            raise ValueError(f"{path} must be one of {args}")
        return value
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        candidates = [item for item in args if item is not type(None)]
        if len(candidates) == 1:
            return _convert_value(candidates[0], value, path)
    if expected_type is bool and not isinstance(value, bool):
        raise TypeError(f"{path} must be a boolean")
    if expected_type in (int, float) and isinstance(value, bool):
        raise TypeError(f"{path} must be {expected_type.__name__}")
    if expected_type in (str, int, float) and not isinstance(value, expected_type):
        if expected_type is float and isinstance(value, int):
            return float(value)
        raise TypeError(f"{path} must be {expected_type.__name__}")
    return value


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any], path: str = "config") -> Any:
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ValueError(f"unknown keys under {path}: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    kwargs = {
        name: _convert_value(hints[name], value, f"{path}.{name}") for name, value in values.items()
    }
    return cls(**kwargs)


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"override must be key=value: {raw}")
    key, raw_value = raw.split("=", 1)
    return key.split("."), yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> ProjectConfig:
    config_path = Path(path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise TypeError("configuration root must be a mapping")
    for raw in overrides or []:
        keys, value = _parse_override(raw)
        cursor = values
        for key in keys[:-1]:
            current = cursor.setdefault(key, {})
            if not isinstance(current, dict):
                raise ValueError(f"cannot set nested override under {key}")
            cursor = current
        cursor[keys[-1]] = value
    config = _dataclass_from_dict(ProjectConfig, values)
    config.validate()
    return config


def config_from_dict(values: dict[str, Any], overrides: list[str] | None = None) -> ProjectConfig:
    values = dict(values)
    for raw in overrides or []:
        keys, value = _parse_override(raw)
        cursor = values
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    config = _dataclass_from_dict(ProjectConfig, values)
    config.validate()
    return config
