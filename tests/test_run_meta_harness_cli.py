"""Offline safety coverage for the thin M6 background command wrapper."""

from __future__ import annotations

import json

import pytest

import scripts.run_meta_harness as cli_module
from meta_harness.server_run import ServerError


COMMON = ["--repository-root", "/safe/repository", "--run-id", "safe-run"]
COMMIT = "a" * 40


def _search_arguments(*extra: str) -> list[str]:
    return [
        "search",
        *COMMON,
        "--search-safe-manifest",
        "/safe/search/manifest.json",
        "--search-records",
        "/safe/search/records.json",
        "--source-commit",
        COMMIT,
        *extra,
    ]


def _smoke_arguments(*extra: str) -> list[str]:
    arguments = _search_arguments(*extra)
    arguments[0] = "smoke"
    return arguments


def test_argument_validation_rejects_missing_required_arguments_and_has_no_credentials():
    parser = cli_module.build_argument_parser()
    assert "api-key" not in parser.format_help().lower()
    assert "api-url" not in parser.format_help().lower()
    with pytest.raises(SystemExit) as exc_info:
        cli_module.cli(["search", "--repository-root", "/safe/repository"])
    assert exc_info.value.code == 2


def test_search_is_offline_by_default_and_does_not_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_module,
        "start_or_resume_search_run",
        lambda **kwargs: calls.append(kwargs),
    )
    messages = []

    result = cli_module.cli(_search_arguments(), output=messages.append)

    assert result == 2
    assert calls == []
    assert "offline by default" in messages[0]


def test_live_search_delegates_once_to_m6_without_credential_arguments(monkeypatch):
    calls = []

    def fake_start(**kwargs):
        calls.append(kwargs)
        return {"search": {"status": "running", "run_id": "safe-run"}}

    monkeypatch.setattr(cli_module, "start_or_resume_search_run", fake_start)
    messages = []

    assert cli_module.cli(_search_arguments("--live-search"), output=messages.append) == 0

    assert calls == [
        {
            "repository_root": "/safe/repository",
            "run_id": "safe-run",
            "search_safe_manifest_path": "/safe/search/manifest.json",
            "search_records_path": "/safe/search/records.json",
            "source_commit": COMMIT,
            "authorize_search_execution": True,
        }
    ]
    assert json.loads(messages[0])["search"]["status"] == "running"


def test_smoke_is_offline_by_default_and_live_flag_delegates_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli_module,
        "run_meta_harness_smoke",
        lambda **kwargs: calls.append(kwargs) or {"status": "complete"},
    )
    offline_messages = []
    assert cli_module.cli(_smoke_arguments(), output=offline_messages.append) == 2
    assert calls == []
    assert "offline by default" in offline_messages[0]

    live_messages = []
    assert (
        cli_module.cli(
            _smoke_arguments("--live-smoke"), output=live_messages.append
        )
        == 0
    )
    assert calls == [
        {
            "repository_root": "/safe/repository",
            "run_id": "safe-run",
            "search_safe_manifest_path": "/safe/search/manifest.json",
            "search_records_path": "/safe/search/records.json",
            "source_commit": COMMIT,
            "authorize_smoke_execution": True,
        }
    ]


def test_cli_logs_redact_sensitive_values_and_transport_text(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "inspect_server_run_status",
        lambda **_kwargs: {
            "api_key": "unmistakably-fake-secret",
            "endpoint": "https://runtime.invalid/private",
            "message": "payload data:image/png;base64," + "A" * 128,
            "status": "running",
        },
    )
    messages = []

    assert cli_module.cli(["search-status", *COMMON], output=messages.append) == 0

    rendered = messages[0]
    assert "unmistakably-fake-secret" not in rendered
    assert "runtime.invalid" not in rendered
    assert "A" * 128 not in rendered
    assert json.loads(rendered)["status"] == "running"


def test_process_lock_error_is_safe_and_activity_delegates_to_m6(monkeypatch):
    monkeypatch.setattr(
        cli_module,
        "start_or_resume_search_run",
        lambda **_kwargs: (_ for _ in ()).throw(
            ServerError("another process owns this run")
        ),
    )
    errors = []
    assert cli_module.cli(_search_arguments("--live-search"), output=errors.append) == 2
    assert errors == ["error: another process owns this run"]

    monkeypatch.setattr(
        cli_module,
        "inspect_server_run_activity",
        lambda **_kwargs: {"search_lock": "held", "final_lock": "available"},
    )
    status = []
    assert cli_module.cli(["activity", *COMMON], output=status.append) == 0
    assert json.loads(status[0])["search_lock"] == "held"


def test_interruption_resume_uses_same_run_and_never_redispatches_completed_work(monkeypatch):
    dispatched: list[str] = []
    interrupted = False

    def fake_start(**kwargs):
        nonlocal interrupted
        assert kwargs["run_id"] == "safe-run"
        if not interrupted:
            interrupted = True
            dispatched.append("request-001")
            raise ServerError("simulated infrastructure interruption")
        dispatched.append("request-002")
        return {"search": {"status": "running", "completed_request_hashes": list(dispatched)}}

    monkeypatch.setattr(cli_module, "start_or_resume_search_run", fake_start)
    first, second = [], []

    assert cli_module.cli(_search_arguments("--live-search"), output=first.append) == 2
    assert cli_module.cli(_search_arguments("--live-search"), output=second.append) == 0

    assert dispatched == ["request-001", "request-002"]
    assert "request-001" not in first[0]
    assert json.loads(second[0])["search"]["completed_request_hashes"] == [
        "<redacted>",
        "<redacted>",
    ]
