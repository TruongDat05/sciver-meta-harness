"""Offline coverage for immutable terminal full-SEARCH V3 winner freezing."""

from __future__ import annotations

import json
import hashlib
import subprocess
from types import MappingProxyType

import pytest

from meta_harness.config import FULL_SEARCH_V3_PROTOCOL_ID, FULL_SEARCH_V3_SEARCH_SIZE
from meta_harness.full_search_v3_evaluator import (
    FullSearchV3SearchInput,
    canonical_full_search_v3_p0_prompt_sha256,
)
from meta_harness.full_search_v3_freeze import (
    FullSearchV3FreezeError,
    FullSearchV3FrozenArtifactConflictError,
    freeze_full_search_v3_winner,
    full_search_v3_frozen_winner_path,
    load_full_search_v3_frozen_winner,
)
from meta_harness.full_search_v3_orchestrator import FullSearchV3Orchestrator
from meta_harness.full_search_v3_proposer import (
    FullSearchV3Candidate,
    FullSearchV3ProposalResult,
)
from meta_harness.prompt_family import (
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)


def _search_input():
    return FullSearchV3SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": "a" * 64,
                "search_membership_sha256": "b" * 64,
            }
        ),
        _records=(),
    )


def _report(candidate_id, prompt_sha256, macro_f1, accuracy):
    return {
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "stage": "SEARCH",
        "candidate_id": candidate_id,
        "prompt_sha256": prompt_sha256,
        "total_records": FULL_SEARCH_V3_SEARCH_SIZE,
        "completed_solver_responses": FULL_SEARCH_V3_SEARCH_SIZE,
        "parsed_predictions": FULL_SEARCH_V3_SEARCH_SIZE,
        "abstentions_or_parse_failures": 0,
        "infrastructure_failures": 0,
        "metrics": {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "parse_coverage": 1.0,
            "rankable": True,
        },
    }


class _FakeProposer:
    def propose(self, **kwargs):
        iteration = kwargs["iteration"]
        templates = {
            method: source + f"\nOffline freeze candidate {iteration}."
            for method, source in canonical_baseline_sources().items()
        }
        candidate = FullSearchV3Candidate(
            candidate_id=f"candidate_{iteration:03d}",
            parent_id=kwargs["parent_id"],
            hypothesis="Independent checks reduce SEARCH classification errors.",
            expected_tradeoff="The additional check may increase response length.",
            templates=MappingProxyType(templates),
            source_sha256=template_source_sha256(templates),
        )
        return FullSearchV3ProposalResult(
            candidate=candidate,
            receipt_path=kwargs["proposal_directory"] / "offline-receipt.json",
            attempt=1,
        )


class _FakeEvaluator:
    def __init__(self, *, p0_metrics, candidate_metrics):
        self.p0_metrics = p0_metrics
        self.candidate_metrics = candidate_metrics

    def p0(self, **_kwargs):
        return _report(
            "cot", canonical_full_search_v3_p0_prompt_sha256(), *self.p0_metrics
        )

    def candidate(self, **kwargs):
        iteration = int(kwargs["candidate_id"].rsplit("_", 1)[1])
        return _report(
            kwargs["candidate_id"],
            template_source_sha256(kwargs["prompt"]),
            *self.candidate_metrics(iteration),
        )


def _terminal_state(tmp_path, *, p0_metrics=(0.5, 0.5), candidate_metrics=None):
    evaluator = _FakeEvaluator(
        p0_metrics=p0_metrics,
        candidate_metrics=candidate_metrics or (lambda _iteration: (0.4, 0.4)),
    )
    state = FullSearchV3Orchestrator(
        repository_root=tmp_path,
        run_id="offline_freeze",
        search_input=_search_input(),
        solver_identity_sha256="c" * 64,
        cache=object(),
        executor=object(),
        proposer=_FakeProposer(),
        proposer_identity={"kind": "offline_fake", "version": 1},
        p0_evaluator=evaluator.p0,
        candidate_evaluator=evaluator.candidate,
    ).run()
    assert state["status"] == "patience_stopped"
    return state


def _state_path(tmp_path):
    return (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_freeze"
        / "orchestration_state.json"
    )


def _write_canonical(path, value):
    path.write_text(canonical_json(value), encoding="utf-8")


def _offline_subprocess_boundary(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("freeze must not start a subprocess"),
    )


def test_freeze_recomputes_and_freezes_a_candidate_winner_offline(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    _terminal_state(
        tmp_path,
        candidate_metrics=lambda iteration: (0.6, 0.6) if iteration == 1 else (0.5, 0.5),
    )

    artifact = freeze_full_search_v3_winner(
        repository_root=tmp_path, run_id="offline_freeze"
    )

    assert artifact["prompt_variant"] == "meta_cot"
    assert artifact["winner"] == {
        "candidate_id": "candidate_001",
        "source_kind": "search_candidate",
        "rank": 1,
    }
    assert set(artifact["templates"]) == {
        "direct", "analytical", "parallel", "sequential"
    }
    assert artifact["hashes"]["ranking_sha256"]
    assert artifact["search_metrics"]["macro_f1"] == 0.6
    assert full_search_v3_frozen_winner_path(tmp_path, "offline_freeze").is_file()


def test_freeze_allows_p0_winner_without_changing_canonical_cot(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    before = dict(canonical_baseline_sources())
    _terminal_state(tmp_path, p0_metrics=(0.9, 0.9))

    artifact = freeze_full_search_v3_winner(
        repository_root=tmp_path, run_id="offline_freeze"
    )

    assert artifact["prompt_variant"] == "meta_cot"
    assert artifact["winner"]["candidate_id"] == "cot"
    assert artifact["winner"]["source_kind"] == "canonical_cot"
    assert artifact["templates"] == before
    assert dict(canonical_baseline_sources()) == before


def test_freeze_rejects_incomplete_search(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    state = _terminal_state(tmp_path)
    state["status"] = "running"
    _write_canonical(_state_path(tmp_path), state)

    with pytest.raises(FullSearchV3FreezeError, match="terminal"):
        freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")


def test_freeze_rejects_tampered_candidate_prompt_or_hash(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    state = _terminal_state(tmp_path)
    state["iterations"][0]["candidate"]["templates"]["direct"] += " tampered"
    _write_canonical(_state_path(tmp_path), state)

    with pytest.raises(FullSearchV3FreezeError, match="prompt hash"):
        freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")


def test_identical_freeze_is_idempotent(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    _terminal_state(tmp_path)

    first = freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")
    second = freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")

    assert second == first


def test_freeze_rejects_conflicting_existing_artifact(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    _terminal_state(tmp_path)
    artifact = freeze_full_search_v3_winner(
        repository_root=tmp_path, run_id="offline_freeze"
    )
    artifact["search_metrics"]["ranking"][-1]["accuracy"] = 0.39
    # Keep the replacement internally valid while making its freeze identity different.
    artifact["hashes"]["ranking_sha256"] = hashlib.sha256(
        canonical_json(artifact["search_metrics"]["ranking"]).encode("utf-8")
    ).hexdigest()
    artifact["artifact_sha256"] = hashlib.sha256(
        canonical_json({key: value for key, value in artifact.items() if key != "artifact_sha256"}).encode("utf-8")
    ).hexdigest()
    _write_canonical(full_search_v3_frozen_winner_path(tmp_path, "offline_freeze"), artifact)

    with pytest.raises(FullSearchV3FrozenArtifactConflictError, match="different"):
        freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")


def test_freeze_cleans_up_after_atomic_write_failure(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    _terminal_state(tmp_path)
    destination = full_search_v3_frozen_winner_path(tmp_path, "offline_freeze")

    def fail_replace(*_args, **_kwargs):
        raise OSError("simulated atomic write failure")

    monkeypatch.setattr("meta_harness.full_search_v3_freeze.os.replace", fail_replace)
    with pytest.raises(FullSearchV3FreezeError, match="atomically"):
        freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")

    assert not destination.exists()
    assert list(destination.parent.glob(".freeze.*.tmp")) == []


def test_frozen_artifact_hash_tampering_is_rejected(tmp_path, monkeypatch):
    _offline_subprocess_boundary(monkeypatch)
    _terminal_state(tmp_path)
    destination = full_search_v3_frozen_winner_path(tmp_path, "offline_freeze")
    freeze_full_search_v3_winner(repository_root=tmp_path, run_id="offline_freeze")
    artifact = json.loads(destination.read_text(encoding="utf-8"))
    artifact["templates"]["direct"] += " tampered"
    _write_canonical(destination, artifact)

    with pytest.raises(FullSearchV3FreezeError, match="prompt hash"):
        load_full_search_v3_frozen_winner(destination)
