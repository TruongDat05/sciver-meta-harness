"""Offline safety + zero-configuration coverage for the root stage launchers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts.run_meta_harness as cli_module
from meta_harness.final_evaluation import FinalError
from meta_harness.server_run import ServerError


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

LAUNCHERS = [
    ("smoke.py", "RUN_LIVE_SMOKE", "--live-smoke", "run_meta_harness_smoke"),
    ("search.py", "RUN_FULL_SEARCH", "--live-search", "start_or_resume_search_run"),
    ("final.py", "RUN_FINAL_ONCE", "--live-final", "start_or_resume_server_run_final"),
]


def _load(monkeypatch, name):
    path = REPOSITORY_ROOT / name
    spec = importlib.util.spec_from_file_location(name[:-3], path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("scripts.run_meta_harness", cli_module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_cli_environment", lambda: None)
    return module


def _configure(monkeypatch, tmp_path, name, server_fn, *, run_id="safe-run"):
    module = _load(monkeypatch, name)
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")

    def fake_paths(repository_root, rid):
        run_root = tmp_path / "ws" / "full_search_v3" / rid
        prep = run_root / "preparation"
        paths = {
            "repository_root": Path(repository_root),
            "run_id": rid,
            "run_root": run_root,
            "preparation_root": prep,
            "search_safe_manifest": prep / "search" / "search_safe_manifest.json",
            "search_records": prep / "search" / "search_records.json",
            "private_manifest": prep / "private" / "private_split_manifest.json",
        }
        for artifact in (
            paths["search_safe_manifest"],
            paths["search_records"],
            paths["private_manifest"],
        ):
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("{}", encoding="utf-8")
        return paths

    monkeypatch.setattr(cli_module, "operator_preparation_paths", fake_paths)
    monkeypatch.setattr(
        cli_module,
        "resolve_operator_run_id",
        lambda override=None, env=None: override or run_id,
    )
    monkeypatch.setattr(
        cli_module,
        "resolve_operator_dataset",
        lambda root, override=None, env=None: dataset,
    )
    monkeypatch.setattr(cli_module, "prepare_run", lambda **k: {"status": "ok"})
    monkeypatch.setattr(cli_module, "freeze_server_run_winner", lambda **k: {})

    real = getattr(cli_module, server_fn)
    calls = []
    monkeypatch.setattr(
        cli_module,
        server_fn,
        lambda **k: calls.append(k) or {"status": "ok", "run_id": run_id},
    )
    return SimpleNamespace(
        module=module,
        dataset=dataset,
        calls=calls,
        server_fn=server_fn,
        run_id=run_id,
        real=real,
    )


# --- generic safety (all three launchers) ---------------------------------------------------


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_import_has_no_dispatch_or_remote_side_effect(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    spy = []
    monkeypatch.setattr(cli_module, "operator_run", lambda *a, **k: spy.append(1) or 0)
    module = _load(monkeypatch, name)
    assert callable(module.main)
    assert spy == []


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_no_authorization_fail_closed_no_dispatch(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    monkeypatch.delenv(envvar, raising=False)
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    monkeypatch.setattr(cli_module, server_fn, h.real)
    assert h.module.main([]) == 2
    assert h.calls == []


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_explicit_live_flag_delegates_exactly_once_with_no_credentials(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    assert h.module.main([flag]) == 0
    assert len(h.calls) == 1
    assert "api_key" not in h.calls[0]
    assert "api_url" not in h.calls[0]


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_env_var_exactly_one_authorizes(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    monkeypatch.setenv(envvar, "1")
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    assert h.module.main([]) == 0
    assert len(h.calls) == 1


@pytest.mark.parametrize("value", ["true", "yes", "on", "0", ""])
@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_other_env_values_do_not_authorize(
    monkeypatch, tmp_path, name, envvar, flag, server_fn, value,
):
    monkeypatch.setenv(envvar, value)
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    monkeypatch.setattr(cli_module, server_fn, h.real)
    assert h.module.main([]) == 2
    assert h.calls == []


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_server_error_retains_safe_exit_code(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    monkeypatch.setattr(
        cli_module,
        server_fn,
        lambda **k: (_ for _ in ()).throw(ServerError("another process owns this run")),
    )
    captured = []
    real_run = cli_module.operator_run
    monkeypatch.setattr(
        cli_module,
        "operator_run",
        lambda dispatch, output=None: real_run(dispatch, output=captured.append),
    )
    assert h.module.main([flag]) == 2
    assert captured == ["error: another process owns this run"]


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_zero_required_arguments(monkeypatch, tmp_path, name, envvar, flag, server_fn):
    """main([]) reaches dispatch (fail-closed), never an argparse required-arg error."""
    monkeypatch.delenv(envvar, raising=False)
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    monkeypatch.setattr(cli_module, server_fn, h.real)
    assert h.module.main([]) == 2
    assert h.calls == []


# --- shared run identity / repository-root / artifact derivation ----------------------------


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_all_launchers_use_the_shared_run_id_resolver(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    monkeypatch.setattr(cli_module, server_fn, h.real)
    monkeypatch.setattr(
        cli_module,
        "resolve_operator_run_id",
        lambda override=None, env=None: h.run_id,
    )
    h.module.main([])
    assert len(h.calls) == 0  # no auth -> no dispatch; resolver already ran without SystemExit


@pytest.mark.parametrize("name,envvar,flag,server_fn", LAUNCHERS)
def test_repository_root_from_launcher_not_cwd(
    monkeypatch, tmp_path, name, envvar, flag, server_fn,
):
    h = _configure(monkeypatch, tmp_path, name, server_fn)
    received = []
    monkeypatch.setattr(
        cli_module,
        "operator_preparation_paths",
        lambda root, rid: received.append((Path(root), rid)) or {
            "repository_root": Path(root),
            "run_id": rid,
            "run_root": tmp_path,
            "preparation_root": tmp_path / "p",
            "search_safe_manifest": tmp_path / "m" / "s.json",
            "search_records": tmp_path / "m" / "r.json",
            "private_manifest": tmp_path / "m" / "p.json",
        },
    )
    (tmp_path / "m").mkdir(parents=True, exist_ok=True)
    for f in ("s.json", "r.json", "p.json"):
        (tmp_path / "m" / f).write_text("{}", encoding="utf-8")
    previous = os.getcwd()
    try:
        os.chdir(tmp_path)
        h.module.main([])
    finally:
        os.chdir(previous)
    assert received and received[0][0] == REPOSITORY_ROOT


# --- stage-specific coordination ------------------------------------------------------------


def test_smoke_prepares_then_dispatches_reusing_the_same_run(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "smoke.py", "run_meta_harness_smoke")
    prepared = []
    monkeypatch.setattr(cli_module, "prepare_run", lambda **k: prepared.append(k) or {"status": "ok"})
    assert h.module.main(["--live-smoke"]) == 0
    assert len(prepared) == 1
    assert prepared[0]["run_id"] == h.run_id
    assert h.calls[0]["run_id"] == h.run_id
    assert h.calls[0]["source_commit"] is None


def test_search_reuses_smoke_run_without_preparing(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "search.py", "start_or_resume_search_run")
    prepared = []
    monkeypatch.setattr(cli_module, "prepare_run", lambda **k: prepared.append(k) or {"status": "ok"})
    monkeypatch.setattr(
        cli_module,
        "resolve_operator_run_id",
        lambda override=None, env=None: "official_v3",
    )
    assert h.module.main(["--live-search"]) == 0
    assert prepared == []
    assert h.calls[0]["run_id"] == "official_v3"
    assert h.calls[0]["source_commit"] is None


def test_final_freezes_then_dispatches_on_the_search_run(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "final.py", "start_or_resume_server_run_final")
    frozen = []
    monkeypatch.setattr(cli_module, "freeze_server_run_winner", lambda **k: frozen.append(k) or {})
    monkeypatch.setattr(
        cli_module,
        "resolve_operator_run_id",
        lambda override=None, env=None: "official_v3",
    )
    assert h.module.main(["--live-final"]) == 0
    assert len(frozen) == 1
    assert frozen[0]["run_id"] == "official_v3"
    assert h.calls[0]["run_id"] == "official_v3"


def test_final_does_not_overwrite_when_freezing_conflicts(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "final.py", "start_or_resume_server_run_final")
    monkeypatch.setattr(
        cli_module,
        "freeze_server_run_winner",
        lambda **k: (_ for _ in ()).throw(
            cli_module.FreezeError("a different immutable winner is already frozen")
        ),
    )
    captured = []
    real_run = cli_module.operator_run
    monkeypatch.setattr(
        cli_module, "operator_run", lambda dispatch, output=None: real_run(dispatch, output=captured.append)
    )
    assert h.module.main(["--live-final"]) == 2
    assert h.calls == []
    assert "different immutable winner" in captured[0]


def test_final_requires_terminal_search_via_freeze(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "final.py", "start_or_resume_server_run_final")
    monkeypatch.setattr(
        cli_module,
        "freeze_server_run_winner",
        lambda **k: (_ for _ in ()).throw(
            cli_module.FreezeError("SEARCH run must reach a successful terminal state")
        ),
    )
    assert h.module.main(["--live-final"]) == 2
    assert h.calls == []


def test_smoke_repeated_invocation_reuses_same_run(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "smoke.py", "run_meta_harness_smoke")
    assert h.module.main(["--live-smoke"]) == 0
    assert h.module.main(["--live-smoke"]) == 0
    assert len(h.calls) == 2
    assert h.calls[0]["run_id"] == h.calls[1]["run_id"] == h.run_id


def test_search_requires_smoke_receipt_through_engine(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "search.py", "start_or_resume_search_run")
    monkeypatch.setattr(
        cli_module,
        "start_or_resume_search_run",
        lambda **k: (_ for _ in ()).throw(
            ServerError("SEARCH requires a compatible completed SMOKE receipt; run the isolated SMOKE stage first")
        ),
    )
    captured = []
    real_run = cli_module.operator_run
    monkeypatch.setattr(
        cli_module, "operator_run", lambda dispatch, output=None: real_run(dispatch, output=captured.append)
    )
    assert h.module.main(["--live-search"]) == 2
    assert "SMOKE receipt" in captured[0]


def test_missing_dataset_fails_before_live_dispatch(monkeypatch, tmp_path):
    h = _configure(monkeypatch, tmp_path, "smoke.py", "run_meta_harness_smoke")
    monkeypatch.setattr(
        cli_module,
        "resolve_operator_dataset",
        lambda root, override=None, env=None: (_ for _ in ()).throw(
            ServerError("dataset not found at /nope; pass --dataset-path")
        ),
    )
    captured = []
    real_run = cli_module.operator_run
    monkeypatch.setattr(
        cli_module, "operator_run", lambda dispatch, output=None: real_run(dispatch, output=captured.append)
    )
    assert h.module.main(["--live-smoke"]) == 2
    assert h.calls == []
    assert "dataset not found" in captured[0]
