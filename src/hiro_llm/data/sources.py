from __future__ import annotations

import shutil
import urllib.parse
import urllib.request
from pathlib import Path

from hiro_llm.config import DataConfig


def acquire_source(config: DataConfig) -> Path:
    """Resolve a configured local, HTTP, or Hugging Face source into a local path."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.source_type == "local":
        path = Path(config.source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"data source does not exist: {path}")
        return path.resolve()
    if config.source_type == "http":
        filename = config.source_filename or Path(urllib.parse.urlparse(config.source).path).name
        if not filename:
            raise ValueError("cannot infer a filename from HTTP source")
        destination = output_dir / Path(filename).name
        with (
            urllib.request.urlopen(config.source, timeout=60) as response,
            destination.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        return destination.resolve()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface-hub is required for Hugging Face sources") from exc
    return Path(
        hf_hub_download(
            repo_id=config.source,
            repo_type="dataset",
            filename=config.source_filename or "data.jsonl",
            revision=config.revision,
            local_dir=output_dir,
        )
    ).resolve()
