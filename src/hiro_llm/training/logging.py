from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from hiro_llm.config import WandbConfig


class ExperimentLogger:
    """W&B logger with a local JSONL fallback for transient outages."""

    def __init__(
        self, config: WandbConfig, *, default_name: str, run_config: dict[str, Any], tags: list[str]
    ):
        self.config = config
        self.run = None
        self.local_path: Path | None = None
        self.run_id: str | None = None
        if not config.enabled:
            return
        try:
            import wandb

            if not getattr(wandb.api, "api_key", None) and not config.entity:
                raise RuntimeError("W&B entity could not be resolved; run wandb login first")
            api = wandb.Api()
            entity = config.entity or getattr(api, "default_entity", None)
            if not entity:
                raise RuntimeError("W&B entity could not be resolved; run wandb login first")
            self.run = wandb.init(
                project=config.project,
                entity=entity,
                name=config.run_name or default_name,
                tags=[*config.tags, *tags],
                config=run_config,
                reinit="finish_previous",
            )
            self.run_id = self.run.id
        except ImportError as exc:
            raise ImportError("wandb is required when wandb.enabled=true") from exc
        except RuntimeError as exc:
            if "entity" in str(exc).lower():
                raise
            self.local_path = Path(config.offline_dir) / f"run-{int(time.time())}.jsonl"
            self.local_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            self.local_path = Path(config.offline_dir) / f"run-{int(time.time())}.jsonl"
            self.local_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is None:
            if self.config.enabled:
                self._write_local({"step": step, **metrics})
            return
        for attempt in range(self.config.retries + 1):
            try:
                self.run.log(metrics, step=step)
                return
            except Exception:
                if attempt == self.config.retries:
                    self._write_local({"step": step, **metrics})

    def _write_local(self, record: dict[str, Any]) -> None:
        if self.local_path is None:
            directory = Path(self.config.offline_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.local_path = directory / f"run-{self.run_id or int(time.time())}.jsonl"
        with self.local_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
        if self.local_path:
            print(
                f"W&B offline metrics saved to {self.local_path}; run `wandb sync {self.local_path.parent}`"
            )
