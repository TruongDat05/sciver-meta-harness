"""Offline delegation and safety coverage for the M6 server interface."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import MappingProxyType

import pytest
from PIL import Image

import meta_harness.server_run as server
from meta_harness.search_evaluator import SearchInput
from meta_harness.search_orchestrator import ResumeError
from meta_harness.preparation import PreparationArtifacts
from meta_harness.solver import SolverResult


RUN_ID = "server_offline"
COMMIT = "a" * 40
PINNED_COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOLVER_IDENTITY = "b" * 64


def _search_input() -> SearchInput:
    return SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": "c" * 64,
                "search_membership_sha256": "d" * 64,
            }
        ),
        _records=(),
    )


def _smoke_search_input(directory: Path) -> SearchInput:
    image_path = directory / "smoke.png"
    Image.new("RGB", (2, 2), color="blue").save(image_path, format="PNG")
    return SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": "c" * 64,
                "search_membership_sha256": "d" * 64,
            }
        ),
        _records=(
            MappingProxyType(
                {
                    "sample_id": "search-only-smoke-sample",
                    "claim_type": "direct",
                    "claim": "A scientific claim.",
                    "context": "SEARCH-only evidence.",
                    "caption": "A SEARCH-only figure caption.",
                    "image_path": str(image_path),
                    "gold_label": "GROUND_TRUTH_MUST_NOT_LEAK",
                }
            ),
        ),
    )


def _report(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "metrics": {"macro_f1": 0.75, "accuracy": 0.8, "parse_coverage": 1.0},
    }


def _state(status: str = "running") -> dict[str, object]:
    return {
        "status": status,
        "stop_reason": None,
        "p0": {"status": "complete", "report": _report("cot")},
        "iterations": [],
        "patience": {"best_macro_f1": 0.75, "best_accuracy": 0.8, "consecutive_non_improving": 0},
        "ranking": ["cot"],
        "winner_id": "cot",
        "identity": {"safe": "identity"},
    }


@pytest.fixture
def offline_search(monkeypatch, tmp_path):
    search_input = _smoke_search_input(tmp_path)
    monkeypatch.setattr(server, "load_experiment_search_input", lambda **_kwargs: search_input)
    monkeypatch.setattr(server.shutil, "which", lambda _command: "/fake/bin/proposer")
    monkeypatch.setattr(server, "_source_commit", lambda _root, _supplied: COMMIT)
    return tmp_path, search_input


def test_prepare_delegates_to_m1_with_explicit_run_directories(monkeypatch, tmp_path):
    called = {}
    artifacts = PreparationArtifacts(
        private_manifest_path=tmp_path / "private.json",
        search_safe_manifest_path=tmp_path / "safe.json",
        search_dataset_path=tmp_path / "records.json",
        summary={"protocol_id": "sciver_full_search_v3", "split_sha256": "e" * 64},
    )

    def fake_prepare(**kwargs):
        called.update(kwargs)
        return artifacts

    monkeypatch.setattr(server, "prepare_experiment", fake_prepare)
    status = server.prepare_run(
        dataset_path=tmp_path / "synthetic.json",
        repository_root=tmp_path,
        run_id=RUN_ID,
    )

    assert called["private_directory"].name == "private"
    assert called["search_directory"].name == "search"
    assert status["run_id"] == RUN_ID
    assert status["private_manifest_path"] == str(artifacts.private_manifest_path)


def test_search_preflight_is_dry_run_and_never_constructs_clients(
    monkeypatch, offline_search
):
    root, _search_input = offline_search
    monkeypatch.setattr(server, "create_live_solver_client", lambda **_kwargs: pytest.fail("dry run constructed solver"))
    monkeypatch.setattr(server, "Proposer", lambda **_kwargs: pytest.fail("dry run constructed proposer"))
    monkeypatch.setattr(server.subprocess, "run", lambda **_kwargs: pytest.fail("explicit source commit must avoid subprocess"))

    status = server.preflight_search_run(
        repository_root=root,
        run_id=RUN_ID,
        search_safe_manifest_path=root / "safe.json",
        search_records_path=root / "records.json",
        source_commit=COMMIT,
        solver_identity_sha256=SOLVER_IDENTITY,
    )

    assert status["operation"] == "search_preflight"
    assert status["resume"] is False
    assert status["workload"]["maximum_search_logical_calls"] == 51000
    assert status["solver"]["identity_sha256"] == SOLVER_IDENTITY
    assert status["solver"]["model"] == "gemma-4-26B-A4B-it"
    assert status["config_sha256"] == (
        "de66f778339d8dd19520fbbf258b7292c0378b8f47fa5fc613db7d33ef4a621f"
    )
    assert set(status["checkpoints"]) == {
        "smoke_receipt_path",
        "search_state_path",
        "search_cache_directory",
        "frozen_winner_path",
        "final_state_path",
        "final_receipt_path",
    }


def test_existing_search_resume_is_checked_by_m4_identity(monkeypatch, offline_search):
    root, _search_input = offline_search
    state_path = root / "existing.json"
    state_path.touch()
    monkeypatch.setattr(server, "experiment_orchestration_state_path", lambda *_args: state_path)

    class IncompatibleOrchestrator:
        def __init__(self, **_kwargs):
            pass

        def state(self):
            raise ResumeError("Meta-Harness orchestration run identity is incompatible")

    monkeypatch.setattr(server, "Orchestrator", IncompatibleOrchestrator)

    with pytest.raises(ResumeError, match="identity is incompatible"):
        server.preflight_search_run(
            repository_root=root,
            run_id=RUN_ID,
            search_safe_manifest_path=root / "safe.json",
            search_records_path=root / "records.json",
            source_commit=COMMIT,
            solver_identity_sha256=SOLVER_IDENTITY,
        )


def test_search_preflight_reports_missing_proposer_cli(monkeypatch, offline_search):
    root, _search_input = offline_search
    monkeypatch.setattr(server.shutil, "which", lambda _command: None)

    with pytest.raises(server.ServerError, match="proposer CLI"):
        server.preflight_search_run(
            repository_root=root,
            run_id=RUN_ID,
            search_safe_manifest_path=root / "safe.json",
            search_records_path=root / "records.json",
            source_commit=COMMIT,
            solver_identity_sha256=SOLVER_IDENTITY,
        )


def test_search_requires_explicit_authorization_before_runtime_credentials(monkeypatch, offline_search):
    root, _search_input = offline_search
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: pytest.fail("gate read credentials"))

    with pytest.raises(server.ServerAuthorizationError, match="SEARCH"):
        server.start_or_resume_search_run(
            repository_root=root,
            run_id=RUN_ID,
            search_safe_manifest_path=root / "safe.json",
            search_records_path=root / "records.json",
            authorize_search_execution=False,
        )


def test_search_start_delegates_without_leaking_runtime_values(monkeypatch, offline_search):
    root, search_input = offline_search
    calls = {}
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: ("https://runtime.invalid", "not-a-real-key"))
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(server, "_construct_live_solver", lambda *_args: object())
    monkeypatch.setattr(server, "SearchCache", lambda _path: object())
    monkeypatch.setattr(server, "RequestExecutor", lambda **_kwargs: object())
    monkeypatch.setattr(server, "Proposer", lambda **_kwargs: object())
    monkeypatch.setattr(
        server,
        "preflight_search_run",
        lambda **_kwargs: {"operation": "search_preflight"},
    )
    monkeypatch.setattr(server, "_require_compatible_smoke_receipt", lambda **_kwargs: None)

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def run(self):
            return _state("patience_stopped")

    monkeypatch.setattr(server, "Orchestrator", FakeOrchestrator)
    result = server.start_or_resume_search_run(
        repository_root=root,
        run_id=RUN_ID,
        search_safe_manifest_path=root / "safe.json",
        search_records_path=root / "records.json",
        authorize_search_execution=True,
        source_commit=COMMIT,
    )

    assert calls["search_input"] is search_input
    assert result["search"]["status"] == "patience_stopped"
    serialized = str(result)
    assert "runtime.invalid" not in serialized
    assert "not-a-real-key" not in serialized


def test_search_server_factory_provides_distinct_clients_to_concurrent_workers(
    monkeypatch, offline_search
):
    """Guard against sharing mutable HTTP-client state across SEARCH workers."""

    root, _search_input = offline_search
    constructed_clients = []
    executor_clients = []
    monkeypatch.setattr(
        server,
        "_runtime_credentials",
        lambda **_kwargs: ("https://runtime.invalid", "not-a-real-key"),
    )
    monkeypatch.setattr(
        server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY
    )
    monkeypatch.setattr(
        server,
        "_construct_live_solver",
        lambda *_args: constructed_clients.append(object()) or constructed_clients[-1],
    )
    monkeypatch.setattr(server, "SearchCache", lambda _path: object())
    monkeypatch.setattr(server, "Proposer", lambda **_kwargs: object())
    monkeypatch.setattr(
        server,
        "preflight_search_run",
        lambda **_kwargs: {"operation": "search_preflight"},
    )
    monkeypatch.setattr(server, "_require_compatible_smoke_receipt", lambda **_kwargs: None)

    class ConcurrentWorkerProbeExecutor:
        def __init__(self, *, client_factory, **_kwargs):
            with ThreadPoolExecutor(max_workers=2) as pool:
                executor_clients.extend(pool.map(lambda _index: client_factory(), range(2)))

    class FakeOrchestrator:
        def __init__(self, **_kwargs):
            pass

        def run(self):
            return _state("patience_stopped")

    monkeypatch.setattr(server, "RequestExecutor", ConcurrentWorkerProbeExecutor)
    monkeypatch.setattr(server, "Orchestrator", FakeOrchestrator)

    server.start_or_resume_search_run(
        repository_root=root,
        run_id=RUN_ID,
        search_safe_manifest_path=root / "safe.json",
        search_records_path=root / "records.json",
        authorize_search_execution=True,
        source_commit=COMMIT,
        maximum_in_flight_requests=2,
    )

    assert len(executor_clients) == 2
    assert executor_clients[0] is not executor_clients[1]


def test_smoke_requires_separate_authorization_before_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        server, "_runtime_credentials", lambda **_kwargs: pytest.fail("gate read credentials")
    )

    with pytest.raises(server.ServerAuthorizationError, match="SMOKE"):
        server.run_meta_harness_smoke(
            repository_root=tmp_path,
            run_id=RUN_ID,
            search_safe_manifest_path=tmp_path / "safe.json",
            search_records_path=tmp_path / "records.json",
            authorize_smoke_execution=False,
        )


def test_smoke_uses_one_search_p0_request_and_receipt_resume_is_idempotent(
    monkeypatch, tmp_path
):
    search_input = _smoke_search_input(tmp_path)
    dispatched = []

    class FakeSolver:
        def list_model_ids(self):
            return ("gemma-4-26B-A4B-it", "another-model")

        def complete(self, request):
            dispatched.append(request)
            return SolverResult(content="Therefore, the final answer is: Answer: yes")

    monkeypatch.setattr(server, "load_experiment_search_input", lambda **_kwargs: search_input)
    monkeypatch.setattr(server.shutil, "which", lambda _command: "/fake/bin/proposer")
    monkeypatch.setattr(server, "_source_commit", lambda _root, _supplied: COMMIT)
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: ("https://runtime.invalid", "unmistakably-fake-key"))
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(server, "_construct_live_solver", lambda *_args: FakeSolver())

    arguments = {
        "repository_root": tmp_path,
        "run_id": RUN_ID,
        "search_safe_manifest_path": tmp_path / "safe.json",
        "search_records_path": tmp_path / "records.json",
        "authorize_smoke_execution": True,
        "source_commit": COMMIT,
    }
    first = server.run_meta_harness_smoke(**arguments)
    second = server.run_meta_harness_smoke(**arguments)

    assert len(dispatched) == 1
    assert first["status"] == "complete" and first["logical_calls"] == 1
    assert first["model_list_preflight"]["status"] == "passed"
    assert first["model_list_preflight"]["model_count"] == 2
    assert first["reused"] is False and second["reused"] is True
    assert "GROUND_TRUTH_MUST_NOT_LEAK" not in str(dispatched[0].messages)
    receipt_text = server.server_run_smoke_receipt_path(
        tmp_path, RUN_ID
    ).read_text(encoding="utf-8")
    for forbidden in (
        "GROUND_TRUTH_MUST_NOT_LEAK",
        "SEARCH-only evidence",
        "runtime.invalid",
        "unmistakably-fake-key",
        "final answer",
    ):
        assert forbidden not in receipt_text

    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: "e" * 64)
    with pytest.raises(server.ServerError, match="SMOKE receipt is incompatible"):
        server.run_meta_harness_smoke(**arguments)
    assert len(dispatched) == 1

    monkeypatch.setattr(
        server,
        "solver_identity_from_api_url",
        lambda _value: SOLVER_IDENTITY,
    )
    receipt_path = server.server_run_smoke_receipt_path(tmp_path, RUN_ID)
    previous_model_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    previous_model_receipt["identity"]["solver"]["model"] = (
        "Qwen/Qwen3.5-35B-A3B"
    )
    previous_model_receipt["identity_sha256"] = server._sha256_json(
        previous_model_receipt["identity"]
    )
    receipt_path.write_text(json.dumps(previous_model_receipt), encoding="utf-8")

    assert previous_model_receipt["identity_sha256"] != first["identity_sha256"]
    with pytest.raises(server.ServerError, match="SMOKE receipt is incompatible"):
        server.run_meta_harness_smoke(**arguments)
    assert len(dispatched) == 1


def test_checkout_validation_rejects_placeholder_origin_head_and_dirty_state(
    monkeypatch, tmp_path
):
    (tmp_path / ".git").mkdir()
    (tmp_path / "requirements.txt").touch()
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "sciver_meta_harness.ipynb").touch()
    (tmp_path / "meta_harness").mkdir()
    (tmp_path / "meta_harness" / "server_run.py").touch()

    for placeholder in ("0" * 40, "a" * 40, "REPLACE_WITH_COMMIT"):
        with pytest.raises(server.ServerError, match="non-placeholder"):
            server.validate_server_run_checkout(
                repository_root=tmp_path,
                pinned_commit_sha=placeholder,
            )

    values = {
        ("rev-parse", "--show-toplevel"): str(tmp_path),
        ("remote", "get-url", "origin"): "https://invalid.example.test/wrong.git",
        ("rev-parse", "HEAD"): PINNED_COMMIT,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(
        server,
        "_git_output",
        lambda _root, *args, **_kwargs: values[args],
    )
    with pytest.raises(server.ServerError, match="origin"):
        server.validate_server_run_checkout(
            repository_root=tmp_path,
            pinned_commit_sha=PINNED_COMMIT,
        )

    values[("remote", "get-url", "origin")] = server.EXPECTED_REPOSITORY_ORIGIN
    values[("rev-parse", "HEAD")] = "89abcdef0123456789abcdef0123456789abcdef"
    with pytest.raises(server.ServerError, match="PINNED_COMMIT_SHA"):
        server.validate_server_run_checkout(
            repository_root=tmp_path,
            pinned_commit_sha=PINNED_COMMIT,
        )

    values[("rev-parse", "HEAD")] = PINNED_COMMIT
    values[("status", "--porcelain=v1", "--untracked-files=all")] = "?? local.txt"
    with pytest.raises(server.ServerError, match="not clean"):
        server.validate_server_run_checkout(
            repository_root=tmp_path,
            pinned_commit_sha=PINNED_COMMIT,
        )

    # The operator notebook self-saves output on every run; its own autogenerated
    # modifications (plus Jupyter checkpoints) must not trip the dirty guard.
    values[("status", "--porcelain=v1", "--untracked-files=all")] = (
        " M notebooks/sciver_meta_harness.ipynb\n"
        "?? notebooks/.ipynb_checkpoints/sciver_meta_harness-checkpoint.ipynb\n"
        "?? local.txt"
    )
    with pytest.raises(server.ServerError, match="not clean"):
        server.validate_server_run_checkout(
            repository_root=tmp_path,
            pinned_commit_sha=PINNED_COMMIT,
        )

    values[("status", "--porcelain=v1", "--untracked-files=all")] = (
        " M notebooks/sciver_meta_harness.ipynb\n"
        "?? notebooks/.ipynb_checkpoints/sciver_meta_harness-checkpoint.ipynb"
    )
    status = server.validate_server_run_checkout(
        repository_root=tmp_path,
        pinned_commit_sha=PINNED_COMMIT,
    )
    assert status["worktree_clean"] is True

    values[("status", "--porcelain=v1", "--untracked-files=all")] = ""
    status = server.validate_server_run_checkout(
        repository_root=tmp_path,
        pinned_commit_sha=PINNED_COMMIT,
    )
    assert status == {
        "operation": "checkout_validation",
        "status": "passed",
        "origin": server.EXPECTED_REPOSITORY_ORIGIN,
        "source_commit": PINNED_COMMIT,
        "worktree_clean": True,
        "dependency_metadata": "requirements.txt",
    }


def test_solver_identity_normalizes_base_and_binds_transport_paths():
    first = server.solver_identity_from_api_url(
        " https://invalid.example.test/api/// "
    )
    equivalent = server.solver_identity_from_api_url(
        "https://invalid.example.test/api"
    )
    changed_path = server.solver_identity_from_api_url(
        "https://invalid.example.test/api",
        chat_completions_path="/different/completions",
    )

    assert first == equivalent
    assert changed_path != first


def test_runtime_credentials_normalize_and_reject_unsafe_values(monkeypatch):
    url, key = server._runtime_credentials(
        api_url=" https://invalid.example.test/base/// ",
        api_key="  unmistakably-fake-key  ",
    )
    assert url == "https://invalid.example.test/base"
    assert key == "unmistakably-fake-key"

    with pytest.raises(server.ServerError, match="invalid characters"):
        server._runtime_credentials(
            api_url="https://invalid.example.test/base",
            api_key="unmistakably-fake-key\n",
        )
    with pytest.raises(server.ServerError, match="safe base URL"):
        server._runtime_credentials(
            api_url="https://invalid.example.test/base?query=value",
            api_key="unmistakably-fake-key",
        )


def test_model_list_preflight_missing_locked_model_fails_without_solver_post():
    class MissingModelClient:
        def list_model_ids(self):
            return ("another-model",)

        def complete(self, _request):
            pytest.fail("missing model dispatched a solver request")

    with pytest.raises(server.ServerError, match="missing_locked_model"):
        server._run_model_list_preflight(MissingModelClient())


def test_full_search_requires_compatible_smoke_receipt_before_client(
    monkeypatch, tmp_path
):
    search_input = _smoke_search_input(tmp_path)
    monkeypatch.setattr(server, "load_experiment_search_input", lambda **_kwargs: search_input)
    monkeypatch.setattr(server.shutil, "which", lambda _command: "/fake/bin/proposer")
    monkeypatch.setattr(server, "_source_commit", lambda _root, _supplied: COMMIT)
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: ("https://runtime.invalid", "unmistakably-fake-key"))
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(
        server, "_construct_live_solver", lambda *_args: pytest.fail("missing SMOKE dispatched")
    )

    with pytest.raises(server.ServerError, match="compatible completed SMOKE"):
        server.start_or_resume_search_run(
            repository_root=tmp_path,
            run_id=RUN_ID,
            search_safe_manifest_path=tmp_path / "safe.json",
            search_records_path=tmp_path / "records.json",
            authorize_search_execution=True,
            source_commit=COMMIT,
        )


def test_search_status_delegates_to_nonmutating_m4_loader(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    state_path.touch()
    monkeypatch.setattr(server, "experiment_orchestration_state_path", lambda *_args: state_path)
    monkeypatch.setattr(server, "load_experiment_orchestration_state", lambda _path: _state())

    status = server.inspect_server_run_status(repository_root=tmp_path, run_id=RUN_ID)

    assert status["operation"] == "search_status"
    assert status["winner_id"] == "cot"


def test_activity_inspection_is_read_only_and_reports_held_run_lock(tmp_path):
    run_directory = tmp_path / "workspace" / "meta_harness" / "full_search_v3" / RUN_ID
    run_directory.mkdir(parents=True)
    lock_path = run_directory / ".orchestration.lock"
    lock_path.touch()

    import fcntl
    import os

    descriptor = os.open(lock_path, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = server.inspect_server_run_activity(
            repository_root=tmp_path, run_id=RUN_ID
        )
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)

    assert status["smoke_lock"] == "not_created"
    assert status["search_lock"] == "held"
    assert status["final_lock"] == "not_created"


def test_freeze_delegates_to_m5(monkeypatch, tmp_path):
    artifact = {
        "schema_version": "sciver_full_search_v3_freeze_v2",
        "run_id": RUN_ID,
        "prompt_variant": "meta_cot",
        "top_k": [
            {"rank": 1, "candidate_id": "candidate_001", "prompt_sha256": "e" * 64},
            {"rank": 2, "candidate_id": "candidate_002", "prompt_sha256": "e" * 64},
        ],
        "artifact_sha256": "f" * 64,
    }
    monkeypatch.setattr(server, "freeze_experiment_winner", lambda **_kwargs: artifact)

    status = server.freeze_server_run_winner(repository_root=tmp_path, run_id=RUN_ID)

    assert status["operation"] == "freeze"
    assert status["prompt_variant"] == "meta_cot"
    assert status["top_k_count"] == 2
    assert status["top_k"][0]["candidate_id"] == "candidate_001"


def test_final_preflight_delegates_without_constructing_solver(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(server, "_trusted_final_records", lambda *_args: [{"sample_id": "private-only"}])
    monkeypatch.setattr(server, "canonical_final_evaluation_solver_contract", lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "create_live_solver_client", lambda **_kwargs: pytest.fail("FINAL preflight constructed solver"))

    def fake_preflight(**kwargs):
        calls.update(kwargs)
        return {"run_id": RUN_ID, "execution_identity_sha256": "a" * 64}

    monkeypatch.setattr(server, "preflight_final_evaluation", fake_preflight)
    result = server.preflight_server_run_final(
        repository_root=tmp_path,
        run_id=RUN_ID,
        dataset_path=tmp_path / "synthetic.json",
        private_manifest_path=tmp_path / "private.json",
        search_safe_manifest_path=tmp_path / "safe.json",
        solver_identity_sha256=SOLVER_IDENTITY,
    )

    assert result["run_id"] == RUN_ID
    assert calls["final_records"] == [{"sample_id": "private-only"}]


def test_final_requires_separate_authorization(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: pytest.fail("FINAL gate read credentials"))

    with pytest.raises(server.ServerAuthorizationError, match="separate"):
        server.start_or_resume_server_run_final(
            repository_root=tmp_path,
            run_id=RUN_ID,
            dataset_path=tmp_path / "synthetic.json",
            private_manifest_path=tmp_path / "private.json",
            search_safe_manifest_path=tmp_path / "safe.json",
            authorize_final_execution=False,
        )


def test_final_start_and_status_delegate_without_exposing_private_records(monkeypatch, tmp_path):
    private_records = [{"sample_id": "private-only"}]
    receipt = {
        "status": "complete",
        "logical_calls": 6000,
        "identity_sha256": "c" * 64,
        "variants": [
            {"prompt_variant": "cot", "candidate_id": "cot", "prompt_sha256": "d" * 64, "completed_request_count": 1000, "completed_request_hashes_sha256": "a" * 64},
            {"prompt_variant": "meta_cot", "candidate_id": "candidate_001", "prompt_sha256": "e" * 64, "completed_request_count": 1000, "completed_request_hashes_sha256": "b" * 64},
            {"prompt_variant": "meta_cot", "candidate_id": "candidate_002", "prompt_sha256": "e2" * 32, "completed_request_count": 1000, "completed_request_hashes_sha256": "b2" * 32},
            {"prompt_variant": "meta_cot", "candidate_id": "candidate_003", "prompt_sha256": "e3" * 32, "completed_request_count": 1000, "completed_request_hashes_sha256": "b3" * 32},
            {"prompt_variant": "meta_cot", "candidate_id": "candidate_004", "prompt_sha256": "e4" * 32, "completed_request_count": 1000, "completed_request_hashes_sha256": "b4" * 32},
            {"prompt_variant": "meta_cot", "candidate_id": "candidate_005", "prompt_sha256": "e5" * 32, "completed_request_count": 1000, "completed_request_hashes_sha256": "b5" * 32},
        ],
    }
    called = {}
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: ("https://runtime.invalid", "not-a-real-key"))
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(server, "_trusted_final_records", lambda *_args: private_records)
    monkeypatch.setattr(server, "canonical_final_evaluation_solver_contract", lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "preflight_final_evaluation", lambda **_kwargs: {"run_id": RUN_ID})
    monkeypatch.setattr(server, "_construct_live_solver", lambda *_args: object())

    def fake_execute(**kwargs):
        called.update(kwargs)
        return receipt

    monkeypatch.setattr(server, "execute_final_evaluation", fake_execute)
    result = server.start_or_resume_server_run_final(
        repository_root=tmp_path,
        run_id=RUN_ID,
        dataset_path=tmp_path / "synthetic.json",
        private_manifest_path=tmp_path / "private.json",
        search_safe_manifest_path=tmp_path / "safe.json",
        authorize_final_execution=True,
    )

    assert called["final_records"] is private_records
    assert result["final"]["logical_calls"] == 6000
    assert "private-only" not in str(result)
    assert "not-a-real-key" not in str(result)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.touch()
    state_path = tmp_path / "final_state.json"
    state_path.touch()
    monkeypatch.setattr(server, "final_evaluation_completion_receipt_path", lambda *_args: receipt_path)
    monkeypatch.setattr(server, "final_evaluation_state_path", lambda *_args: state_path)
    monkeypatch.setattr(server, "load_final_evaluation_state", lambda _path: {"status": "complete", "identity": {}})
    monkeypatch.setattr(server, "load_bound_final_evaluation_completion_receipt", lambda _path, _state: receipt)
    inspected = server.inspect_server_run_final_status(repository_root=tmp_path, run_id=RUN_ID)
    assert inspected == result["final"]
