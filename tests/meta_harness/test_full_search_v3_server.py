"""Offline delegation and safety coverage for the M6 server interface."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

import pytest
from PIL import Image

import meta_harness.full_search_v3_server as server
from meta_harness.full_search_v3_evaluator import FullSearchV3SearchInput
from meta_harness.full_search_v3_orchestrator import FullSearchV3ResumeError
from meta_harness.full_search_v3_preparation import FullSearchV3PreparationArtifacts
from meta_harness.full_search_v3_solver import SolverResult


RUN_ID = "server_offline"
COMMIT = "a" * 40
SOLVER_IDENTITY = "b" * 64


def _search_input() -> FullSearchV3SearchInput:
    return FullSearchV3SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": "c" * 64,
                "search_membership_sha256": "d" * 64,
            }
        ),
        _records=(),
    )


def _smoke_search_input(directory: Path) -> FullSearchV3SearchInput:
    image_path = directory / "smoke.png"
    Image.new("RGB", (2, 2), color="blue").save(image_path, format="PNG")
    return FullSearchV3SearchInput(
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
    search_input = _search_input()
    monkeypatch.setattr(server, "load_full_search_v3_search_input", lambda **_kwargs: search_input)
    monkeypatch.setattr(server.shutil, "which", lambda _command: "/fake/bin/proposer")
    monkeypatch.setattr(server, "_source_commit", lambda _root, _supplied: COMMIT)
    return tmp_path, search_input


def test_prepare_delegates_to_m1_with_explicit_run_directories(monkeypatch, tmp_path):
    called = {}
    artifacts = FullSearchV3PreparationArtifacts(
        private_manifest_path=tmp_path / "private.json",
        search_safe_manifest_path=tmp_path / "safe.json",
        search_dataset_path=tmp_path / "records.json",
        summary={"protocol_id": "sciver_full_search_v3", "split_sha256": "e" * 64},
    )

    def fake_prepare(**kwargs):
        called.update(kwargs)
        return artifacts

    monkeypatch.setattr(server, "prepare_full_search_v3", fake_prepare)
    status = server.prepare_full_search_v3_server_run(
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
    monkeypatch.setattr(server, "FullSearchV3Proposer", lambda **_kwargs: pytest.fail("dry run constructed proposer"))
    monkeypatch.setattr(server.subprocess, "run", lambda **_kwargs: pytest.fail("explicit source commit must avoid subprocess"))

    status = server.preflight_full_search_v3_server_run(
        repository_root=root,
        run_id=RUN_ID,
        search_safe_manifest_path=root / "safe.json",
        search_records_path=root / "records.json",
        source_commit=COMMIT,
        solver_identity_sha256=SOLVER_IDENTITY,
    )

    assert status["operation"] == "search_preflight"
    assert status["resume"] is False
    assert status["workload"]["maximum_search_logical_calls"] == 41000
    assert status["solver"]["identity_sha256"] == SOLVER_IDENTITY
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
    monkeypatch.setattr(server, "full_search_v3_orchestration_state_path", lambda *_args: state_path)

    class IncompatibleOrchestrator:
        def __init__(self, **_kwargs):
            pass

        def state(self):
            raise FullSearchV3ResumeError("V3 orchestration run identity is incompatible")

    monkeypatch.setattr(server, "FullSearchV3Orchestrator", IncompatibleOrchestrator)

    with pytest.raises(FullSearchV3ResumeError, match="identity is incompatible"):
        server.preflight_full_search_v3_server_run(
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

    with pytest.raises(server.FullSearchV3ServerError, match="proposer CLI"):
        server.preflight_full_search_v3_server_run(
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

    with pytest.raises(server.FullSearchV3ServerAuthorizationError, match="SEARCH"):
        server.start_or_resume_full_search_v3_server_run(
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
    monkeypatch.setattr(server, "FullSearchV3SearchCache", lambda _path: object())
    monkeypatch.setattr(server, "FullSearchV3RequestExecutor", lambda **_kwargs: object())
    monkeypatch.setattr(server, "FullSearchV3Proposer", lambda **_kwargs: object())
    monkeypatch.setattr(
        server,
        "preflight_full_search_v3_server_run",
        lambda **_kwargs: {"operation": "search_preflight"},
    )
    monkeypatch.setattr(server, "_require_compatible_smoke_receipt", lambda **_kwargs: None)

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def run(self):
            return _state("patience_stopped")

    monkeypatch.setattr(server, "FullSearchV3Orchestrator", FakeOrchestrator)
    result = server.start_or_resume_full_search_v3_server_run(
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


def test_smoke_requires_separate_authorization_before_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(
        server, "_runtime_credentials", lambda **_kwargs: pytest.fail("gate read credentials")
    )

    with pytest.raises(server.FullSearchV3ServerAuthorizationError, match="SMOKE"):
        server.run_full_search_v3_server_smoke(
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
        def complete(self, request):
            dispatched.append(request)
            return SolverResult(content="Therefore, the final answer is: Answer: yes")

    monkeypatch.setattr(server, "load_full_search_v3_search_input", lambda **_kwargs: search_input)
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
    first = server.run_full_search_v3_server_smoke(**arguments)
    second = server.run_full_search_v3_server_smoke(**arguments)

    assert len(dispatched) == 1
    assert first["status"] == "complete" and first["logical_calls"] == 1
    assert first["reused"] is False and second["reused"] is True
    assert "GROUND_TRUTH_MUST_NOT_LEAK" not in str(dispatched[0].messages)
    receipt_text = server.full_search_v3_server_smoke_receipt_path(
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
    with pytest.raises(server.FullSearchV3ServerError, match="SMOKE receipt is incompatible"):
        server.run_full_search_v3_server_smoke(**arguments)
    assert len(dispatched) == 1


def test_full_search_requires_compatible_smoke_receipt_before_client(
    monkeypatch, tmp_path
):
    search_input = _smoke_search_input(tmp_path)
    monkeypatch.setattr(server, "load_full_search_v3_search_input", lambda **_kwargs: search_input)
    monkeypatch.setattr(server.shutil, "which", lambda _command: "/fake/bin/proposer")
    monkeypatch.setattr(server, "_source_commit", lambda _root, _supplied: COMMIT)
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: ("https://runtime.invalid", "unmistakably-fake-key"))
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(
        server, "_construct_live_solver", lambda *_args: pytest.fail("missing SMOKE dispatched")
    )

    with pytest.raises(server.FullSearchV3ServerError, match="compatible completed SMOKE"):
        server.start_or_resume_full_search_v3_server_run(
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
    monkeypatch.setattr(server, "full_search_v3_orchestration_state_path", lambda *_args: state_path)
    monkeypatch.setattr(server, "load_full_search_v3_orchestration_state", lambda _path: _state())

    status = server.inspect_full_search_v3_server_status(repository_root=tmp_path, run_id=RUN_ID)

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
        status = server.inspect_full_search_v3_server_activity(
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
        "schema_version": "sciver_full_search_v3_freeze_v1",
        "run_id": RUN_ID,
        "prompt_variant": "meta_cot",
        "winner": {"candidate_id": "cot"},
        "artifact_sha256": "f" * 64,
        "hashes": {"prompt_sha256": "e" * 64},
    }
    monkeypatch.setattr(server, "freeze_full_search_v3_winner", lambda **_kwargs: artifact)

    status = server.freeze_full_search_v3_server_winner(repository_root=tmp_path, run_id=RUN_ID)

    assert status["operation"] == "freeze"
    assert status["prompt_variant"] == "meta_cot"


def test_final_preflight_delegates_without_constructing_solver(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(server, "_trusted_final_records", lambda *_args: [{"sample_id": "private-only"}])
    monkeypatch.setattr(server, "canonical_full_search_v3_final_solver_contract", lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "create_live_solver_client", lambda **_kwargs: pytest.fail("FINAL preflight constructed solver"))

    def fake_preflight(**kwargs):
        calls.update(kwargs)
        return {"run_id": RUN_ID, "execution_identity_sha256": "a" * 64}

    monkeypatch.setattr(server, "preflight_full_search_v3_final", fake_preflight)
    result = server.preflight_full_search_v3_server_final(
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

    with pytest.raises(server.FullSearchV3ServerAuthorizationError, match="separate"):
        server.start_or_resume_full_search_v3_server_final(
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
        "logical_calls": 2000,
        "identity_sha256": "c" * 64,
        "variants": [
            {"prompt_variant": "cot", "candidate_id": "cot", "prompt_sha256": "d" * 64, "completed_request_count": 1000, "completed_request_hashes_sha256": "a" * 64},
            {"prompt_variant": "meta_cot", "candidate_id": "candidate_001", "prompt_sha256": "e" * 64, "completed_request_count": 1000, "completed_request_hashes_sha256": "b" * 64},
        ],
    }
    called = {}
    monkeypatch.setattr(server, "_runtime_credentials", lambda **_kwargs: ("https://runtime.invalid", "not-a-real-key"))
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(server, "_trusted_final_records", lambda *_args: private_records)
    monkeypatch.setattr(server, "canonical_full_search_v3_final_solver_contract", lambda **kwargs: kwargs)
    monkeypatch.setattr(server, "preflight_full_search_v3_final", lambda **_kwargs: {"run_id": RUN_ID})
    monkeypatch.setattr(server, "_construct_live_solver", lambda *_args: object())

    def fake_execute(**kwargs):
        called.update(kwargs)
        return receipt

    monkeypatch.setattr(server, "execute_full_search_v3_final", fake_execute)
    result = server.start_or_resume_full_search_v3_server_final(
        repository_root=tmp_path,
        run_id=RUN_ID,
        dataset_path=tmp_path / "synthetic.json",
        private_manifest_path=tmp_path / "private.json",
        search_safe_manifest_path=tmp_path / "safe.json",
        authorize_final_execution=True,
    )

    assert called["final_records"] is private_records
    assert result["final"]["logical_calls"] == 2000
    assert "private-only" not in str(result)
    assert "not-a-real-key" not in str(result)

    receipt_path = tmp_path / "receipt.json"
    receipt_path.touch()
    monkeypatch.setattr(server, "full_search_v3_final_completion_receipt_path", lambda *_args: receipt_path)
    monkeypatch.setattr(server, "load_full_search_v3_final_completion_receipt", lambda _path: receipt)
    inspected = server.inspect_full_search_v3_server_final_status(repository_root=tmp_path, run_id=RUN_ID)
    assert inspected == result["final"]
