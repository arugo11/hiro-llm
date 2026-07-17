from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from hiro_llm.config import HubConfig, ProjectConfig

CHECKPOINT_FORMAT_VERSION = 1


def _unwrapped(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ProjectConfig,
    step: int,
    metrics: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "task": config.task,
        "step": step,
        "config": config.to_dict(),
        "model_state_dict": _unwrapped(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(payload, destination)
    return destination


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format")
    required = {"task", "step", "config", "model_state_dict", "optimizer_state_dict"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(missing)}")
    return payload


def load_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    expected_task: str | None = None,
) -> dict[str, Any]:
    payload = read_checkpoint(path, map_location)
    if expected_task is not None and payload["task"] != expected_task:
        raise ValueError(f"cannot resume {expected_task} from a {payload['task']} checkpoint")
    _unwrapped(model).load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def resolve_checkpoint(reference: str, local_dir: str | Path = "models/checkpoints") -> Path:
    if not reference.startswith("hf://"):
        path = Path(reference).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return path.resolve()
    parts = reference.removeprefix("hf://").split("/", 2)
    if len(parts) != 3:
        raise ValueError("Hub checkpoint must use hf://owner/repository/path")
    repo_id = "/".join(parts[:2])
    filename = parts[2]
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=repo_id, repo_type="model", filename=filename, local_dir=local_dir)
    ).resolve()


class HubPublisher:
    def __init__(self, config: HubConfig) -> None:
        self.config = config
        self.token = os.getenv(config.token_env)
        if not self.token and config.token_env == "HF_TOKEN":
            try:
                from huggingface_hub import get_token
                self.token = get_token()
            except ImportError:
                self.token = None

    def prepare(self) -> None:
        if not self.config.repo_id or "/" not in self.config.repo_id:
            raise ValueError("hub.repo_id must be set to owner/repository")
        if not self.token:
            raise RuntimeError(f"Hugging Face token is missing from {self.config.token_env}")
        from huggingface_hub import HfApi

        api = HfApi(token=self.token)
        api.whoami()
        api.create_repo(
            repo_id=self.config.repo_id,
            repo_type="model",
            private=self.config.private,
            exist_ok=True,
        )

    def upload(self, checkpoint_path: str | Path) -> None:
        from huggingface_hub import HfApi

        path = Path(checkpoint_path)
        HfApi(token=self.token).upload_file(
            repo_id=self.config.repo_id,
            repo_type="model",
            path_or_fileobj=path,
            path_in_repo=path.name,
        )
