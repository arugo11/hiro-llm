from pathlib import Path

import pytest

from hiro_llm.config import load_config
from hiro_llm.model.language_model import LanguageModel


@pytest.mark.parametrize(
    ("path", "variant", "layers", "embedding", "hidden", "heads"),
    [
        ("configs/shakespeare_pretrain.yaml", "baseline", 6, 256, 1024, 8),
        ("configs/layers12_shakespeare.yaml", "layers12", 12, 256, 1024, 8),
        ("configs/embed768_shakespeare.yaml", "embed768", 6, 768, 1024, 12),
        ("configs/hidden3072_shakespeare.yaml", "hidden3072", 6, 256, 3072, 8),
    ],
)
def test_variant_model_shapes(path, variant, layers, embedding, hidden, heads):
    config = load_config(path)
    assert (config.variant, config.model.num_layers, config.model.embedding_dim) == (
        variant,
        layers,
        embedding,
    )
    assert config.model.hidden_dim == hidden
    assert config.model.num_attention_heads == heads
    model = LanguageModel(config.model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count > 0


def test_all_variant_configs_exist():
    for variant in ("layers12", "embed768", "hidden3072"):
        for stage in ("shakespeare", "tinystories", "smoltalk"):
            assert Path(f"configs/{variant}_{stage}.yaml").is_file()
