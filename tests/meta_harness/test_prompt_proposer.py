"""Offline contract tests for the one-candidate full SEARCH proposer."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from meta_harness.prompt_proposer import (
    CandidateValidationError,
    ProposerInfrastructureError,
    ProposalExhausted,
    Proposer,
    build_prompt_proposer_input,
)
from meta_harness.prompt_family import canonical_baseline_sources, template_source_sha256


def _templates(suffix: str = "") -> dict[str, str]:
    return {
        method: source + f"\nApply one independent support check{suffix}."
        for method, source in canonical_baseline_sources().items()
    }


def _payload(*, candidate_id: str = "v3_candidate_001", suffix: str = "") -> dict:
    return {
        "iteration": 1,
        "candidate": {
            "candidate_id": candidate_id,
            "parent_id": "baseline_cot",
            "hypothesis": "Independent support checks may reduce ambiguous decisions.",
            "expected_tradeoff": "The added check may make responses longer.",
            "templates": _templates(suffix),
        },
    }


class FakeRunner:
    def __init__(self, outputs: list[dict | str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        output = self.outputs.pop(0)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            output if isinstance(output, str) else json.dumps(output),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


def _propose(tmp_path, runner, **kwargs):
    return Proposer(runner=runner).propose(
        proposal_directory=tmp_path,
        run_id="offline_v3",
        iteration=1,
        aggregate_search_metrics={"macro_f1": 0.6, "accuracy": 0.65},
        lineage=[],
        representative_search_failures=[
            {
                "pattern": "ambiguous_support",
                "summary": "Ambiguous support relationships recur across methods.",
                "count": 4,
                "methods": ["direct", "parallel"],
            }
        ],
        **kwargs,
    )


def test_valid_one_candidate_output_is_accepted_and_receipted(tmp_path):
    runner = FakeRunner([_payload()])

    result = _propose(tmp_path, runner)

    assert result.attempt == 1
    assert result.candidate.candidate_id == "v3_candidate_001"
    assert set(result.candidate.templates) == {
        "direct",
        "analytical",
        "parallel",
        "sequential",
    }
    assert result.candidate.source_sha256 == template_source_sha256(_templates())
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "accepted"
    assert receipt["candidate_id"] == "v3_candidate_001"
    command, kwargs = runner.calls[0]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert kwargs["shell"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_template_keys_are_rejected_before_any_solver_call(
    tmp_path, mutation
):
    payload = _payload()
    if mutation == "missing":
        del payload["candidate"]["templates"]["direct"]
    else:
        payload["candidate"]["templates"]["extra"] = "not allowed"
    runner = FakeRunner([payload, payload, payload])
    solver_calls = 0

    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner)

    assert solver_calls == 0
    receipts = sorted(
        (tmp_path / "workspace" / "meta_harness" / "full_search_v3" / "offline_v3" / "proposals" / "iteration_0001").glob("attempt_*.json")
    )
    assert len(receipts) == 3
    assert all(json.loads(path.read_text())["status"] == "rejected" for path in receipts)


def test_placeholder_mutation_is_rejected_before_any_solver_call(tmp_path):
    payload = _payload()
    payload["candidate"]["templates"]["direct"] = payload["candidate"][
        "templates"
    ]["direct"].replace("$caption", "$caption1")
    runner = FakeRunner([payload, payload, payload])

    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner)

    assert len(runner.calls) == 3


def test_duplicate_proposal_is_rejected_before_any_solver_call(tmp_path):
    payload = _payload()
    runner = FakeRunner([payload, payload, payload])
    duplicate_hash = template_source_sha256(payload["candidate"]["templates"])
    solver_calls = 0

    with pytest.raises(ProposalExhausted):
        _propose(
            tmp_path,
            runner,
            existing_candidate_ids=["earlier_candidate"],
            existing_source_sha256=[duplicate_hash],
        )

    assert solver_calls == 0
    receipt_path = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_v3"
        / "proposals"
        / "iteration_0001"
        / "attempt_00003.json"
    )
    assert json.loads(receipt_path.read_text())["category"] == "duplicate"


def test_three_attempt_exhaustion_records_each_rejection(tmp_path):
    invalid = _payload()
    invalid["candidate"]["templates"] = {"direct": "invalid"}
    runner = FakeRunner([invalid, invalid, invalid])

    with pytest.raises(ProposalExhausted, match="three invalid"):
        _propose(tmp_path, runner)

    assert len(runner.calls) == 3


def test_rejected_attempt_receipts_resume_without_repeating_an_attempt(tmp_path):
    invalid = _payload()
    invalid["candidate"]["templates"] = {"direct": "invalid"}

    class InterruptingRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, command, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated proposer interruption")
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps(invalid), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    interrupted = InterruptingRunner()
    with pytest.raises(RuntimeError, match="simulated proposer interruption"):
        _propose(tmp_path, interrupted)

    resumed = FakeRunner([invalid, invalid])
    with pytest.raises(ProposalExhausted, match="three invalid"):
        _propose(tmp_path, resumed)

    receipts = sorted(
        (
            tmp_path
            / "workspace"
            / "meta_harness"
            / "full_search_v3"
            / "offline_v3"
            / "proposals"
            / "iteration_0001"
        ).glob("attempt_*.json")
    )
    assert [path.name for path in receipts] == [
        "attempt_00001.json",
        "attempt_00002.json",
        "attempt_00003.json",
    ]
    assert len(resumed.calls) == 2


def test_infrastructure_failure_receipt_retries_without_consuming_invalid_budget(tmp_path):
    class UnavailableRunner:
        def __call__(self, command, **kwargs):
            raise OSError("simulated unavailable proposer")

    with pytest.raises(ProposerInfrastructureError):
        _propose(tmp_path, UnavailableRunner())

    resumed = FakeRunner([_payload()])
    result = _propose(tmp_path, resumed)

    assert result.attempt == 2
    receipt_directory = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_v3"
        / "proposals"
        / "iteration_0001"
    )
    receipt = json.loads((receipt_directory / "attempt_00002.json").read_text())
    assert receipt["status"] == "accepted"
    assert len(resumed.calls) == 1


@pytest.mark.parametrize("run_id", [".", "..", "-leading", "space run"])
def test_proposer_rejects_run_ids_outside_the_shared_contract(tmp_path, run_id):
    with pytest.raises(CandidateValidationError, match="run_id"):
        Proposer(runner=FakeRunner([_payload()])).propose(
            proposal_directory=tmp_path,
            run_id=run_id,
            iteration=1,
        )


def test_proposer_input_is_sanitized_and_rejects_prohibited_failure_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("API_KEY", "FAKE_TEST_SECRET_DO_NOT_USE")
    runner = FakeRunner([_payload()])

    _propose(tmp_path, runner)

    serialized = runner.calls[0][1]["input"].decode("utf-8")
    assert "FAKE_TEST_SECRET_DO_NOT_USE" not in serialized
    assert "gold_label" not in serialized
    assert "PRIVATE_FINAL_RECORD" not in serialized
    assert "sample_id" not in serialized
    assert "raw_trace" not in serialized
    assert "aggregate_search_metrics" in serialized
    assert "representative_search_failures" in serialized

    with pytest.raises(ValueError, match="prohibited"):
        build_prompt_proposer_input(
            iteration=1,
            parent_id="baseline_cot",
            parent_templates=canonical_baseline_sources(),
            aggregate_search_metrics={},
            lineage=[],
            representative_search_failures=[
                {
                    "pattern": "private_final_record",
                    "summary": "This must never be visible.",
                    "count": 1,
                    "methods": ["direct"],
                }
            ],
        )
    assert len(runner.calls) == 1
