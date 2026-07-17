from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from hiro_llm.config import DataConfig, ModelConfig, VisionConfig
from hiro_llm.data.sources import acquire_source
from hiro_llm.data.text import _encode_instruction


def prepare_vision_data(
    config: DataConfig, model: ModelConfig, vision: VisionConfig
) -> tuple[Path, Path, Path]:
    source = acquire_source(config)
    source_root = Path(config.image_root).expanduser() if config.image_root else source.parent
    image_paths: list[str] = []
    inputs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if config.image_field not in sample:
                raise ValueError(f"missing JSONL field: {config.image_field}")
            image_path = (source_root / str(sample[config.image_field])).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"image does not exist: {image_path}")
            input_ids, target_ids = _encode_instruction(
                sample, config, model, image_tokens=vision.num_image_tokens
            )
            image_paths.append(str(image_path))
            inputs.append(input_ids)
            labels.append(target_ids)
            if config.max_samples is not None and len(inputs) >= config.max_samples:
                break
    if not inputs:
        raise ValueError("vision dataset contains no samples")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "image_paths.json"
    input_path = output_dir / "vision_input_ids.npy"
    label_path = output_dir / "vision_labels.npy"
    image_path.write_text(json.dumps(image_paths, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(input_path, np.stack(inputs))
    np.save(label_path, np.stack(labels))
    return image_path, input_path, label_path


class VisionDataset(Dataset):
    def __init__(
        self,
        image_path_file: str | Path,
        input_path: str | Path,
        label_path: str | Path,
        encoder_name: str,
        image_processor: Any | None = None,
    ) -> None:
        self.image_paths = json.loads(Path(image_path_file).read_text(encoding="utf-8"))
        self.inputs = np.load(input_path, mmap_mode="r")
        self.labels = np.load(label_path, mmap_mode="r")
        if len(self.image_paths) != len(self.inputs) or self.inputs.shape != self.labels.shape:
            raise ValueError("vision paths, inputs, and labels must contain the same samples")
        if image_processor is None:
            try:
                from transformers import CLIPImageProcessor
            except ImportError as exc:
                raise ImportError("Install hiro-llm[vision] to process images") from exc
            image_processor = CLIPImageProcessor.from_pretrained(encoder_name)
        self.image_processor = image_processor

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Install hiro-llm[vision] to process images") from exc
        with Image.open(self.image_paths[index]) as image:
            pixel_values = self.image_processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ].squeeze(0)
        inputs = torch.from_numpy(np.asarray(self.inputs[index], dtype=np.int64).copy())
        labels = torch.from_numpy(np.asarray(self.labels[index], dtype=np.int64).copy())
        return pixel_values, inputs, labels
