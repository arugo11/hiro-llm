from __future__ import annotations

from pathlib import Path

import torch

from hiro_llm.model.vision_language_model import VisionLanguageModel


def generate_vision_text(
    model: VisionLanguageModel,
    tokenizer_name: str,
    image_path: str | Path,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
) -> str:
    try:
        from PIL import Image
        from transformers import CLIPImageProcessor
    except ImportError as exc:
        raise ImportError("Install hiro-llm[vision] to generate from images") from exc
    import tiktoken

    tokenizer = tiktoken.get_encoding(tokenizer_name)
    processor = CLIPImageProcessor.from_pretrained(model.config.encoder_name)
    with Image.open(image_path) as image:
        pixel_values = processor(images=image.convert("RGB"), return_tensors="pt")[
            "pixel_values"
        ].to(device)
    prompt_ids = tokenizer.encode(prompt, allowed_special="all")
    token_ids = [tokenizer.eot_token] * model.config.num_image_tokens + prompt_ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    model.eval()
    generated = model.generate(pixel_values, input_ids, max_new_tokens, temperature)
    return tokenizer.decode(generated[0, input_ids.shape[1] :].tolist())
