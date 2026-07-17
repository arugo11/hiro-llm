from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path

import torch

from hiro_llm.config import ProjectConfig
from hiro_llm.training.checkpoint import HubPublisher, load_training_state, save_checkpoint
from hiro_llm.training.logging import ExperimentLogger
from hiro_llm.training.scheduler import learning_rate_at_step


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: Iterable,
        config: ProjectConfig,
        device: torch.device,
        publisher: HubPublisher,
        checkpoint_dir: str | Path = "models/checkpoints",
        validation_batches: Iterable | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.batches = batches
        self.config = config
        self.device = device
        self.publisher = publisher
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history: list[dict[str, float | int]] = []
        self.start_step = 0
        self.validation_batches = validation_batches
        self.logger = logger
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def resume(self, checkpoint_path: str | Path) -> None:
        payload = load_training_state(
            checkpoint_path,
            self.model,
            self.optimizer,
            map_location=self.device,
            expected_task=self.config.task,
        )
        self.start_step = int(payload["step"])
        if self.start_step >= self.config.training.total_steps:
            raise ValueError("resume step is not lower than training.total_steps")

    def _next_batch(self, iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(self.batches)
            return next(iterator), iterator

    def _train_step(self, batch) -> tuple[float, float, int]:
        tensors = [tensor.to(self.device, non_blocking=True) for tensor in batch]
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = self.config.training.amp and self.device.type in {"cuda", "cpu"}
        amp_context = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if amp_enabled
            else nullcontext()
        )
        with amp_context:
            if len(tensors) == 2:
                _, loss = self.model(tensors[0], tensors[1])
            elif len(tensors) == 3:
                _, loss = self.model(tensors[0], tensors[1], tensors[2])
            else:
                raise ValueError("training batches must contain two or three tensors")
        if loss is None:
            raise RuntimeError("model did not return a training loss")
        loss.backward()
        gradient_norm = (
            float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )
            )
            if self.config.training.gradient_clip_norm
            else float(
                torch.linalg.vector_norm(
                    torch.stack(
                        [
                            p.grad.detach().norm()
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
            )
        )
        self.optimizer.step()
        return float(loss.detach()), gradient_norm, int(tensors[0].numel())

    @torch.no_grad()
    def _validation(self) -> tuple[float, float] | None:
        if self.validation_batches is None:
            return None
        self.model.eval()
        losses = []
        iterator = iter(self.validation_batches)
        for _ in range(8):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            tensors = [tensor.to(self.device, non_blocking=True) for tensor in batch]
            _, loss = self.model(*tensors)
            if loss is not None:
                losses.append(float(loss))
        self.model.train()
        if not losses:
            return None
        mean = sum(losses) / len(losses)
        return mean, float(torch.exp(torch.tensor(mean)))

    def _save(self, step: int) -> Path:
        metrics = self.history[-1] if self.history else {}
        path = self.checkpoint_dir / f"checkpoint-{step:08d}.pt"
        save_checkpoint(path, self.model, self.optimizer, self.config, step, metrics)
        self.publisher.upload(path)
        return path

    def train(self) -> Path:
        self.publisher.prepare()
        self.model.train()
        batch_iterator = iter(self.batches)
        started = time.perf_counter()
        last_checkpoint: Path | None = None
        for step in range(self.start_step + 1, self.config.training.total_steps + 1):
            batch, batch_iterator = self._next_batch(batch_iterator)
            learning_rate = learning_rate_at_step(
                step,
                self.config.training.total_steps,
                self.config.training.warmup_steps,
                self.config.training.max_learning_rate,
                self.config.training.min_learning_rate,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
            loss, gradient_norm, token_count = self._train_step(batch)
            elapsed = time.perf_counter() - started
            record = {
                "step": step,
                "loss": loss,
                "learning_rate": learning_rate,
                "seconds": elapsed,
                "gradient_norm": gradient_norm,
                "tokens_per_second": token_count / max(elapsed, 1e-9),
                "samples_per_second": self.config.training.batch_size / max(elapsed, 1e-9),
            }
            validation = self._validation() if step % self.config.training.log_every == 0 else None
            if validation:
                record["validation_loss"], record["validation_perplexity"] = validation
            if self.device.type == "cuda":
                record.update(
                    {
                        "gpu_memory_allocated": float(torch.cuda.memory_allocated(self.device)),
                        "gpu_memory_reserved": float(torch.cuda.memory_reserved(self.device)),
                        "gpu_memory_peak": float(torch.cuda.max_memory_allocated(self.device)),
                    }
                )
            self.history.append(record)
            if self.logger and (step % self.config.training.log_every == 0 or step == 1):
                metrics = {
                    "train/loss": loss,
                    "train/perplexity": float(torch.exp(torch.tensor(loss))),
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": gradient_norm,
                    "train/tokens_per_second": record["tokens_per_second"],
                    "train/samples_per_second": record["samples_per_second"],
                    "train/elapsed_seconds": elapsed,
                    "model/parameter_count": self.parameter_count,
                }
                if validation:
                    metrics.update(
                        {"validation/loss": validation[0], "validation/perplexity": validation[1]}
                    )
                if self.device.type == "cuda":
                    metrics.update(
                        {
                            "system/gpu_memory_allocated": record["gpu_memory_allocated"],
                            "system/gpu_memory_reserved": record["gpu_memory_reserved"],
                            "system/gpu_memory_peak": record["gpu_memory_peak"],
                        }
                    )
                self.logger.log(metrics, step)
            if step % self.config.training.log_every == 0 or step == 1:
                print(
                    f"step={step} loss={loss:.4f} lr={learning_rate:.3e} elapsed={elapsed:.1f}s",
                    flush=True,
                )
            if step % self.config.training.checkpoint_every == 0:
                last_checkpoint = self._save(step)
                if self.logger:
                    self.logger.log(
                        {"checkpoint/step": step, "checkpoint/path": str(last_checkpoint)}, step
                    )
        needs_final_checkpoint = (
            self.config.training.total_steps % self.config.training.checkpoint_every != 0
        )
        if last_checkpoint is None or needs_final_checkpoint:
            last_checkpoint = self._save(self.config.training.total_steps)
        if self.logger:
            self.logger.finish()
        return last_checkpoint
