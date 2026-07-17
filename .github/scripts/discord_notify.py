from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

FAILURE_CONCLUSIONS = {"action_required", "failure", "startup_failure", "timed_out"}
WATCHED_WORKFLOWS = {"Python CI", "Typst"}
COLORS = {"failure": 0xD73A49, "info": 0x2F81F7, "success": 0x2DA44E}
WEBHOOK_PATH = re.compile(r"^/api(?:/v\d+)?/webhooks/\d+/[A-Za-z0-9._-]+/?$")


def _value(mapping: Mapping[str, Any] | None, key: str, default: Any = "") -> Any:
    if not isinstance(mapping, Mapping):
        return default
    value = mapping.get(key, default)
    return default if value is None else value


def safe_text(value: Any, limit: int) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    text = " ".join(text.split())
    text = text.replace("@", "@\u200b")
    for character in ("\\", "`", "*", "_", "[", "]", "(", ")", "~", "|", ">"):
        text = text.replace(character, f"\\{character}")
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 1)]}…"


def _field(name: str, value: Any, *, inline: bool = True) -> dict[str, Any]:
    return {"name": name, "value": safe_text(value, 1024) or "-", "inline": inline}


def _embed(
    title: str,
    description: str,
    url: str,
    fields: list[dict[str, Any]],
    status: str = "info",
) -> dict[str, Any]:
    return {
        "title": safe_text(title, 256),
        "description": safe_text(description, 4096),
        "url": url,
        "color": COLORS[status],
        "fields": fields[:25],
    }


def _push_embed(payload: Mapping[str, Any]) -> dict[str, Any]:
    repository = _value(payload, "repository", {})
    head_commit = _value(payload, "head_commit", {})
    sender = _value(payload, "sender", {})
    commits = _value(payload, "commits", [])
    after = str(_value(payload, "after"))
    description = _value(head_commit, "message", "No head commit metadata")
    url = _value(payload, "compare") or _value(head_commit, "url") or _value(repository, "html_url")
    return _embed(
        "Push to main",
        description,
        str(url),
        [
            _field("Repository", _value(repository, "full_name")),
            _field("Actor", _value(sender, "login")),
            _field("Commits", len(commits) if isinstance(commits, list) else 0),
            _field("Commit", after[:7]),
        ],
    )


def _pull_request_embed(payload: Mapping[str, Any]) -> dict[str, Any]:
    action = str(_value(payload, "action"))
    pull_request = _value(payload, "pull_request", {})
    merged = bool(_value(pull_request, "merged", False))
    labels = {
        "closed": "Pull request merged" if merged else "Pull request closed",
        "converted_to_draft": "Pull request converted to draft",
        "opened": "Pull request opened",
        "ready_for_review": "Pull request ready for review",
        "reopened": "Pull request reopened",
        "synchronize": "Pull request updated",
    }
    head = _value(pull_request, "head", {})
    base = _value(pull_request, "base", {})
    user = _value(pull_request, "user", {})
    status = "success" if action == "closed" and merged else "info"
    return _embed(
        labels.get(action, "Pull request updated"),
        f"#{_value(pull_request, 'number')} {_value(pull_request, 'title')}",
        str(_value(pull_request, "html_url")),
        [
            _field("Author", _value(user, "login")),
            _field("Branches", f"{_value(head, 'ref')} → {_value(base, 'ref')}"),
            _field("Draft", str(bool(_value(pull_request, "draft", False))).lower()),
        ],
        status,
    )


def _release_embed(payload: Mapping[str, Any]) -> dict[str, Any]:
    release = _value(payload, "release", {})
    author = _value(release, "author", {})
    return _embed(
        "Prerelease published" if _value(release, "prerelease", False) else "Release published",
        _value(release, "name") or _value(release, "tag_name"),
        str(_value(release, "html_url")),
        [
            _field("Tag", _value(release, "tag_name")),
            _field("Author", _value(author, "login")),
        ],
        "success",
    )


def _workflow_run_embed(payload: Mapping[str, Any], repository_name: str) -> dict[str, Any] | None:
    workflow_run = _value(payload, "workflow_run", {})
    workflow_name = str(_value(workflow_run, "name"))
    conclusion = str(_value(workflow_run, "conclusion"))
    head_repository = _value(workflow_run, "head_repository", {})
    main_success = (
        conclusion == "success"
        and _value(workflow_run, "event") == "push"
        and _value(workflow_run, "head_branch") == "main"
        and _value(head_repository, "full_name") == repository_name
    )
    if workflow_name not in WATCHED_WORKFLOWS:
        return None
    if conclusion not in FAILURE_CONCLUSIONS and not main_success:
        return None
    head_sha = str(_value(workflow_run, "head_sha"))
    actor = _value(workflow_run, "actor", {})
    status = "success" if main_success else "failure"
    return _embed(
        f"{workflow_name}: {conclusion.replace('_', ' ')}",
        f"Workflow run #{_value(workflow_run, 'run_number')}",
        str(_value(workflow_run, "html_url")),
        [
            _field("Branch", _value(workflow_run, "head_branch")),
            _field("Commit", head_sha[:7]),
            _field("Actor", _value(actor, "login")),
            _field("Attempt", _value(workflow_run, "run_attempt", 1)),
        ],
        status,
    )


def build_payload(
    event_name: str, event: Mapping[str, Any], repository_name: str
) -> dict[str, Any] | None:
    if event_name == "push":
        embed = _push_embed(event)
    elif event_name == "pull_request_target":
        embed = _pull_request_embed(event)
    elif event_name == "release":
        embed = _release_embed(event)
    elif event_name == "workflow_run":
        embed = _workflow_run_embed(event, repository_name)
        if embed is None:
            return None
    elif event_name == "workflow_dispatch":
        repository = _value(event, "repository", {})
        repository_url = _value(repository, "html_url", f"https://github.com/{repository_name}")
        embed = _embed(
            "Discord notification test",
            "GitHub Actions can deliver notifications to this channel.",
            str(repository_url),
            [_field("Repository", repository_name)],
            "success",
        )
    else:
        raise ValueError(f"Unsupported event: {event_name}")
    return {
        "username": "hiro-llm GitHub",
        "allowed_mentions": {"parse": [], "roles": [], "users": [], "replied_user": False},
        "embeds": [embed],
    }


def validate_webhook_url(webhook_url: str) -> str:
    parsed = urllib.parse.urlsplit(webhook_url)
    if parsed.scheme != "https" or parsed.hostname not in {"discord.com", "discordapp.com"}:
        raise ValueError("DISCORD_WEBHOOK_URL must be an HTTPS Discord webhook URL")
    if not WEBHOOK_PATH.fullmatch(parsed.path):
        raise ValueError("DISCORD_WEBHOOK_URL has an unexpected path")
    return webhook_url


def post_webhook(webhook_url: str, payload: Mapping[str, Any]) -> None:
    parsed = urllib.parse.urlsplit(validate_webhook_url(webhook_url))
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("wait", "true"))
    request_url = urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))
    request = urllib.request.Request(
        request_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "hiro-llm-github-actions"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            if response.status not in {200, 204}:
                raise RuntimeError(f"Discord returned HTTP {response.status}")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Discord returned HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise RuntimeError("Discord webhook request failed") from error


def main() -> int:
    event_path = Path(os.environ["GITHUB_EVENT_PATH"])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    payload = build_payload(os.environ["GITHUB_EVENT_NAME"], event, os.environ["GITHUB_REPOSITORY"])
    if payload is None:
        print("Notification suppressed by policy")
        return 0
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not webhook_url:
        raise RuntimeError("DISCORD_WEBHOOK_URL is not configured")
    post_webhook(webhook_url, payload)
    print("Discord notification sent")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Notification failed: {error}", file=sys.stderr)
        sys.exit(1)
