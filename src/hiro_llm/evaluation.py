from __future__ import annotations

import math
from typing import Any

import torch

from hiro_llm.config import ProjectConfig
from hiro_llm.model.language_model import LanguageModel
from hiro_llm.training.checkpoint import read_checkpoint, resolve_checkpoint


def _dataset_text(
    dataset_id: str,
    config_name: str | None,
    split: str,
    revision: str | None,
    trust_remote_code: bool = False,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for benchmark evaluation") from exc
    kwargs = {"split": split, "revision": revision, "trust_remote_code": trust_remote_code}
    return (
        load_dataset(dataset_id, config_name, **kwargs)
        if config_name
        else load_dataset(dataset_id, **kwargs)
    )


@torch.no_grad()
def _perplexity(
    model: LanguageModel, tokens: list[int], device: torch.device, stride: int
) -> float:
    if len(tokens) < 2:
        raise ValueError("benchmark contains fewer than two tokens")
    total_nll = 0.0
    total_tokens = 0
    context = model.config.max_sequence_length
    for start in range(0, len(tokens) - 1, stride):
        end = min(start + context + 1, len(tokens))
        chunk = torch.tensor(tokens[start:end], dtype=torch.long, device=device).unsqueeze(0)
        _, loss = model(chunk[:, :-1], chunk[:, 1:])
        count = chunk.shape[1] - 1
        if loss is not None:
            total_nll += float(loss) * count
            total_tokens += count
        if end == len(tokens):
            break
    return math.exp(total_nll / max(total_tokens, 1))


def evaluate_checkpoint(
    checkpoint: str, benchmark: str, device: torch.device, config: ProjectConfig
) -> dict[str, Any]:
    payload = read_checkpoint(resolve_checkpoint(checkpoint), map_location=device)
    model = LanguageModel(config.model).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    stride = config.benchmark.context_stride or config.model.max_sequence_length
    if benchmark == "wikitext2":
        dataset = _dataset_text(
            "Salesforce/wikitext", "wikitext-2-raw-v1", "test", config.benchmark.datasets_revision
        )
        text = "\n".join(dataset["text"])
        tokens = (
            __import__("tiktoken")
            .get_encoding(config.data.tokenizer)
            .encode(text, allowed_special="all")
        )
        return {
            "wikitext2/perplexity": _perplexity(model, tokens, device, stride),
            "num_examples": len(dataset),
        }
    if benchmark == "ptb":
        dataset = _dataset_text(
            "ptb_text_only", "penn_treebank", "test", config.benchmark.datasets_revision,
            trust_remote_code=True,
        )
        text = "\n".join(dataset["sentence"])
        tokens = (
            __import__("tiktoken")
            .get_encoding(config.data.tokenizer)
            .encode(text, allowed_special="all")
        )
        return {
            "ptb/perplexity": _perplexity(model, tokens, device, stride),
            "num_examples": len(dataset),
        }
    if benchmark == "lambada":
        dataset = _dataset_text(
            "EleutherAI/lambada_openai", None, "test", config.benchmark.datasets_revision
        )
        encoder = __import__("tiktoken").get_encoding(config.data.tokenizer)
        correct = 0
        nll = 0.0
        count = 0
        for row in dataset:
            text = " ".join(str(row["text"]).split())
            words = text.rsplit(" ", 1)
            if len(words) != 2:
                continue
            context, answer = words
            context_ids = encoder.encode(context, allowed_special="all")
            answer_ids = encoder.encode(" " + answer, allowed_special="all")
            generated = []
            for token in answer_ids:
                window = context_ids[-model.config.max_sequence_length :]
                logits, _ = model(
                    torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
                )
                prediction = int(logits[:, -1].argmax(dim=-1))
                generated.append(prediction)
                nll -= float(torch.log_softmax(logits[:, -1], dim=-1)[0, token])
                context_ids.append(token)
            predicted = encoder.decode(generated).strip()
            correct += int(predicted == answer)
            count += 1
        return {
            "lambada/accuracy": correct / max(count, 1),
            "lambada/mean_nll": nll / max(count, 1),
            "lambada/num_examples": count,
        }
    raise ValueError(f"unknown benchmark: {benchmark}")


def evaluate_all(
    checkpoint: str, benchmarks: list[str], device: torch.device, config: ProjectConfig
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in benchmarks:
        results.update(evaluate_checkpoint(checkpoint, name, device, config))
    return results
