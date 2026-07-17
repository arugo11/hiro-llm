from __future__ import annotations

import torch

from hiro_llm.model.language_model import LanguageModel


def generate_text(
    model: LanguageModel,
    tokenizer_name: str,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int | None = None,
) -> str:
    import tiktoken

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    encoded = tokenizer.encode(prompt, allowed_special="all")
    if not encoded:
        encoded = [tokenizer.eot_token]
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(input_ids, max_new_tokens, temperature, top_k)
    return tokenizer.decode(generated[0].tolist())
