"""Offline coverage for shared zero-configuration operator helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_meta_harness as cli_module
from meta_harness.final_evaluation import FinalError
from meta_harness.preparation import PreparationError
from meta_harness.server_run import ServerError
from meta_harness.winner_freeze import FreezeError


# --- operator_render / operator_run --------------------------------------------------------


def test_operator_render_returns_sanitized_json():
    rendered = cli_module.operator_render(
        {
            "api_key": "UNMISTAKABLY_FAKE_SECRET",
            "endpoint": "https://runtime.invalid/private",
            "message": "data:image/png;base64," + "A" * 128,
            "status": "running",
        }
    )
    assert "UNMISTAKABLY_FAKE_SECRET" not in rendered
    assert "runtime.invalid" not in rendered
    assert "A" * 128 not in rendered
    assert json.loads(rendered)["status"] == "running"


def test_operator_run_success_returns_zero_and_renders_sanitized_output():
    out = []
    code = cli_module.operator_run(
        lambda: {"status": "ok", "api_key": "UNMISTAKABLY_FAKE"},
        output=out.append,
    )
    assert code == 0
    assert "UNMISTAKABLY_FAKE" not in out[0]
    assert json.loads(out[0])["status"] == "ok"


@pytest.mark.parametrize(
    "exc",
    [
        ServerError("safe server stop"),
        PreparationError("refusing to replace a different immutable artifact"),
        FinalError("FINAL requires a valid immutable frozen winner"),
        FreezeError("SEARCH run must reach a successful terminal state"),
    ],
)
def test_operator_run_maps_safe_errors_to_exit_two(exc):
    out = []
    code = cli_module.operator_run(lambda: (_ for _ in ()).throw(exc), output=out.append)
    assert code == 2
    assert out[0].startswith("error: ")
    assert str(exc) in out[0]


def test_operator_run_maps_unknown_exception_to_generic_exit_one():
    out = []
    code = cli_module.operator_run(
        lambda: (_ for _ in ()).throw(RuntimeError("boom")), output=out.append
    )
    assert code == 1
    assert out[0] == (
        "error: server operation failed; inspect the local sanitized status and retry"
    )


# --- resolve_operator_run_id -----------------------------------------------------------------


def test_default_run_id_is_official_v3(monkeypatch):
    monkeypatch.delenv("SCIVER_RUN_ID", raising=False)
    assert cli_module.resolve_operator_run_id() == "official_v3"


def test_sciver_run_id_overrides_default(monkeypatch):
    monkeypatch.setenv("SCIVER_RUN_ID", "my-run")
    assert cli_module.resolve_operator_run_id() == "my-run"


def test_override_beats_sciver_run_id(monkeypatch):
    monkeypatch.setenv("SCIVER_RUN_ID", "env-run")
    assert cli_module.resolve_operator_run_id("override-run") == "override-run"


def test_invalid_run_id_fails_closed():
    with pytest.raises(ServerError):
        cli_module.resolve_operator_run_id(env={"SCIVER_RUN_ID": "not ok!"})


# --- resolve_operator_dataset ---------------------------------------------------------------


def test_default_dataset_from_repository_layout(tmp_path):
    default = tmp_path / "data" / "sciver" / "testset.json"
    default.parent.mkdir(parents=True)
    default.write_text("[]", encoding="utf-8")
    assert cli_module.resolve_operator_dataset(tmp_path, env={}) == default.resolve()


def test_override_beats_environment(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text("[]", encoding="utf-8")
    b.write_text("[]", encoding="utf-8")
    env = {"SCIVER_DATASET_PATH": str(b)}
    assert cli_module.resolve_operator_dataset(tmp_path, str(a), env=env) == a.resolve()


def test_environment_used_when_no_override(tmp_path):
    a = tmp_path / "a.json"
    a.write_text("[]", encoding="utf-8")
    env = {"SCIVER_DATASET_PATH": str(a)}
    assert cli_module.resolve_operator_dataset(tmp_path, None, env=env) == a.resolve()


def test_missing_dataset_fails_closed_before_dispatch(tmp_path):
    with pytest.raises(ServerError, match="dataset not found"):
        cli_module.resolve_operator_dataset(tmp_path, env={})


# --- operator_preparation_paths -------------------------------------------------------------


def test_preparation_paths_derived_from_repository_layout(tmp_path):
    paths = cli_module.operator_preparation_paths(tmp_path, "runX")
    run_root = tmp_path / "workspace" / "meta_harness" / "full_search_v3" / "runX"
    assert paths["run_id"] == "runX"
    assert paths["run_root"] == run_root
    assert paths["preparation_root"] == run_root / "preparation"
    assert paths["search_safe_manifest"] == (
        run_root / "preparation" / "search" / "search_safe_manifest.json"
    )
    assert paths["search_records"] == (
        run_root / "preparation" / "search" / "search_records.json"
    )
    assert paths["private_manifest"] == (
        run_root / "preparation" / "private" / "private_split_manifest.json"
    )
