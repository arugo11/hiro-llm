from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / ".github" / "scripts" / "discord_notify.py"
SPEC = importlib.util.spec_from_file_location("discord_notify", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
discord_notify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(discord_notify)


def workflow_event(
    *,
    conclusion: str,
    event: str = "pull_request",
    branch: str = "feature",
    repository: str = "arugo11/hiro-llm",
    workflow: str = "Python CI",
) -> dict:
    return {
        "workflow_run": {
            "name": workflow,
            "conclusion": conclusion,
            "event": event,
            "head_branch": branch,
            "head_repository": {"full_name": repository},
            "head_sha": "1234567890abcdef",
            "actor": {"login": "contributor"},
            "run_number": 7,
            "run_attempt": 2,
            "html_url": "https://github.com/arugo11/hiro-llm/actions/runs/7",
        }
    }


@pytest.mark.parametrize("conclusion", sorted(discord_notify.FAILURE_CONCLUSIONS))
def test_failure_conclusions_are_notified(conclusion: str) -> None:
    payload = discord_notify.build_payload(
        "workflow_run", workflow_event(conclusion=conclusion), "arugo11/hiro-llm"
    )

    assert payload is not None
    assert payload["embeds"][0]["color"] == discord_notify.COLORS["failure"]


def test_main_push_success_is_notified() -> None:
    payload = discord_notify.build_payload(
        "workflow_run",
        workflow_event(conclusion="success", event="push", branch="main"),
        "arugo11/hiro-llm",
    )

    assert payload is not None
    assert payload["embeds"][0]["color"] == discord_notify.COLORS["success"]


def test_fork_branch_named_main_is_not_main_success() -> None:
    payload = discord_notify.build_payload(
        "workflow_run",
        workflow_event(
            conclusion="success", event="pull_request", branch="main", repository="fork/repo"
        ),
        "arugo11/hiro-llm",
    )

    assert payload is None


@pytest.mark.parametrize("conclusion", ["cancelled", "neutral", "skipped", "stale", "success"])
def test_non_alerting_conclusions_are_suppressed(conclusion: str) -> None:
    payload = discord_notify.build_payload(
        "workflow_run", workflow_event(conclusion=conclusion), "arugo11/hiro-llm"
    )

    assert payload is None


def test_unwatched_workflow_is_suppressed() -> None:
    payload = discord_notify.build_payload(
        "workflow_run",
        workflow_event(conclusion="failure", workflow="Discord notifications"),
        "arugo11/hiro-llm",
    )

    assert payload is None


@pytest.mark.parametrize(
    ("action", "merged", "expected"),
    [
        ("opened", False, "Pull request opened"),
        ("synchronize", False, "Pull request updated"),
        ("ready_for_review", False, "Pull request ready for review"),
        ("converted_to_draft", False, "Pull request converted to draft"),
        ("closed", False, "Pull request closed"),
        ("closed", True, "Pull request merged"),
    ],
)
def test_pull_request_lifecycle(action: str, merged: bool, expected: str) -> None:
    event = {
        "action": action,
        "pull_request": {
            "number": 12,
            "title": "Update docs",
            "html_url": "https://github.com/arugo11/hiro-llm/pull/12",
            "merged": merged,
            "draft": action == "converted_to_draft",
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "user": {"login": "author"},
        },
    }

    payload = discord_notify.build_payload("pull_request_target", event, "arugo11/hiro-llm")

    assert payload is not None
    assert payload["embeds"][0]["title"] == expected


def test_push_handles_missing_head_commit_and_disables_mentions() -> None:
    event = {
        "repository": {"full_name": "arugo11/hiro-llm", "html_url": "https://github.com/x/y"},
        "sender": {"login": "@everyone"},
        "head_commit": None,
        "commits": [],
        "after": "0" * 40,
        "compare": "https://github.com/x/y/compare/a...b",
    }

    payload = discord_notify.build_payload("push", event, "arugo11/hiro-llm")

    assert payload is not None
    assert payload["allowed_mentions"]["parse"] == []
    assert payload["embeds"][0]["fields"][1]["value"] == "@\u200beveryone"


def test_release_supports_prereleases() -> None:
    event = {
        "release": {
            "prerelease": True,
            "name": "Preview",
            "tag_name": "v1.0.0-rc1",
            "html_url": "https://github.com/x/y/releases/tag/v1.0.0-rc1",
            "author": {"login": "maintainer"},
        }
    }

    payload = discord_notify.build_payload("release", event, "arugo11/hiro-llm")

    assert payload is not None
    assert payload["embeds"][0]["title"] == "Prerelease published"


def test_safe_text_escapes_markdown_and_truncates() -> None:
    text = discord_notify.safe_text("@here [link](x) `code`" + "x" * 100, 30)

    assert "@\u200bhere" in text
    assert "\\[link\\]\\(x\\)" in text
    assert len(text) == 30
    assert text.endswith("…")


@pytest.mark.parametrize(
    "url",
    [
        "http://discord.com/api/webhooks/123/token",
        "https://example.com/api/webhooks/123/token",
        "https://discord.com/channels/123/456",
        "",
    ],
)
def test_invalid_webhook_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError):
        discord_notify.validate_webhook_url(url)


def test_valid_webhook_url_is_accepted() -> None:
    url = "https://discord.com" + "/api/webhooks/123/token_value"

    assert discord_notify.validate_webhook_url(url) == url
