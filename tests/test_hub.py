from pathlib import Path

import pytest

from hiro_llm.config import HubConfig
from hiro_llm.training.checkpoint import HubPublisher


def test_hub_authentication_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_HF_TOKEN", raising=False)
    publisher = HubPublisher(HubConfig(repo_id="owner/model", token_env="TEST_HF_TOKEN"))
    with pytest.raises(RuntimeError, match="TEST_HF_TOKEN"):
        publisher.prepare()


def test_hub_prepare_and_upload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setenv("TEST_HF_TOKEN", "secret")
    monkeypatch.setattr("huggingface_hub.HfApi.whoami", lambda self: calls.append("whoami"))
    monkeypatch.setattr(
        "huggingface_hub.HfApi.create_repo", lambda self, **kwargs: calls.append("create")
    )
    monkeypatch.setattr(
        "huggingface_hub.HfApi.upload_file", lambda self, **kwargs: calls.append("upload")
    )
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test")
    publisher = HubPublisher(HubConfig(repo_id="owner/model", token_env="TEST_HF_TOKEN"))
    publisher.prepare()
    publisher.upload(checkpoint)
    assert calls == ["whoami", "create", "upload"]


def test_hub_connection_error_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_HF_TOKEN", "secret")

    def fail(_self):
        raise ConnectionError("offline")

    monkeypatch.setattr("huggingface_hub.HfApi.whoami", fail)
    publisher = HubPublisher(HubConfig(repo_id="owner/model", token_env="TEST_HF_TOKEN"))
    with pytest.raises(ConnectionError, match="offline"):
        publisher.prepare()
