"""Fully mocked state-machine coverage for full SEARCH orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import MappingProxyType

import pytest

from meta_harness import prompt_proposer as pp
from meta_harness.config import EXPERIMENT_PROTOCOL_ID, EXPERIMENT_SEARCH_SIZE
from meta_harness.search_evaluator import (
    EXPERIMENT_P0_CANDIDATE_ID,
    SearchInput,
    canonical_experiment_p0_prompt_sha256,
)
from meta_harness.search_orchestrator import (
    Orchestrator,
    ResumeError,
)
from meta_harness.prompt_proposer import (
    Candidate,
    ProposalExhausted,
    ProposalResult,
    Proposer,
    build_prompt_proposer_input,
)
from meta_harness.prompt_family import (
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)


def _search_input():
    return SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": "a" * 64,
                "search_membership_sha256": "b" * 64,
            }
        ),
        _records=(),
    )


def _report(candidate_id, prompt_sha256, macro_f1, accuracy, *, rankable=True):
    return {
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "stage": "SEARCH",
        "candidate_id": candidate_id,
        "prompt_sha256": prompt_sha256,
        "total_records": EXPERIMENT_SEARCH_SIZE,
        "completed_solver_responses": EXPERIMENT_SEARCH_SIZE,
        "parsed_predictions": EXPERIMENT_SEARCH_SIZE,
        "abstentions_or_parse_failures": 0,
        "infrastructure_failures": 0,
        "metrics": {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "parse_coverage": 1.0,
            "rankable": rankable,
        },
    }


class FakeProposer:
    def __init__(self):
        self.calls = []

    def propose(self, **kwargs):
        self.calls.append(kwargs)
        iteration = kwargs["iteration"]
        templates = {
            method: source + f"\nIteration {iteration} independent check."
            for method, source in canonical_baseline_sources().items()
        }
        candidate = Candidate(
            candidate_id=f"candidate_{iteration:03d}",
            parent_id=kwargs["parent_id"],
            hypothesis="Independent checks may reduce ambiguous decisions.",
            expected_tradeoff="The check may add brief reasoning.",
            templates=MappingProxyType(templates),
            source_sha256=template_source_sha256(templates),
        )
        return ProposalResult(
            candidate=candidate,
            receipt_path=kwargs["proposal_directory"] / "fake-receipt.json",
            attempt=1,
        )


class FakeEvaluator:
    def __init__(self, *, p0_metrics=(0.9, 0.9), candidate_metrics=None, p0_hash=None):
        self.p0_metrics = p0_metrics
        self.candidate_metrics = candidate_metrics or (lambda _iteration: (0.8, 0.8))
        self.p0_hash = p0_hash or canonical_experiment_p0_prompt_sha256()
        self.p0_calls = []
        self.candidate_calls = []

    def p0(self, **kwargs):
        self.p0_calls.append(kwargs)
        return _report("cot", self.p0_hash, *self.p0_metrics)

    def candidate(self, **kwargs):
        self.candidate_calls.append(kwargs)
        iteration = int(kwargs["candidate_id"].rsplit("_", 1)[1])
        return _report(
            kwargs["candidate_id"],
            template_source_sha256(kwargs["prompt"]),
            *self.candidate_metrics(iteration),
        )


def _orchestrator(tmp_path, proposer, evaluator, **kwargs):
    return Orchestrator(
        repository_root=tmp_path,
        run_id="offline_run",
        search_input=_search_input(),
        solver_identity_sha256="c" * 64,
        cache=object(),
        executor=object(),
        proposer=proposer,
        proposer_identity={"kind": "offline_fake", "version": 1},
        p0_evaluator=evaluator.p0,
        candidate_evaluator=evaluator.candidate,
        **kwargs,
    )


def test_p0_completes_before_one_candidate_per_completed_iteration(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    original_propose = proposer.propose

    def p0_guarded_propose(**kwargs):
        assert len(evaluator.p0_calls) == 1
        return original_propose(**kwargs)

    proposer.propose = p0_guarded_propose
    orchestration = _orchestrator(tmp_path, proposer, evaluator)

    state = orchestration.run()

    assert evaluator.p0_calls and proposer.calls
    assert len(proposer.calls) == len(evaluator.candidate_calls) == 38
    assert state["status"] == "patience_stopped"
    assert [entry["iteration"] for entry in state["iterations"]] == list(range(1, 39))
    assert all(entry["candidate"] is not None for entry in state["iterations"])


def test_invalid_exhausted_proposal_has_zero_candidate_evaluations(tmp_path):
    class ExhaustedProposer:
        def __init__(self):
            self.calls = 0
            self.attempts = 0

        def propose(self, **_kwargs):
            self.calls += 1
            self.attempts += 3
            raise ProposalExhausted("three invalid attempts")

    proposer = ExhaustedProposer()
    evaluator = FakeEvaluator()

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert state["status"] == "proposal_exhausted"
    assert state["stop_reason"] == "proposal_attempts_exhausted"
    assert proposer.calls == 1
    assert proposer.attempts == 3
    assert len(evaluator.p0_calls) == 1
    assert evaluator.candidate_calls == []

    resumed_proposer = ExhaustedProposer()
    resumed_evaluator = FakeEvaluator()
    resumed = _orchestrator(tmp_path, resumed_proposer, resumed_evaluator).run()

    assert resumed["status"] == "proposal_exhausted"
    assert resumed["stop_reason"] == "proposal_attempts_exhausted"
    assert resumed_proposer.calls == 0
    assert resumed_proposer.attempts == 0
    assert resumed_evaluator.p0_calls == []
    assert resumed_evaluator.candidate_calls == []


@pytest.mark.parametrize(
    "stop_reason",
    [None, "unknown_reason"],
)
def test_proposal_exhausted_state_rejects_invalid_stop_reason(tmp_path, stop_reason):
    class ExhaustedProposer:
        def propose(self, **_kwargs):
            raise ProposalExhausted("three invalid attempts")

    _orchestrator(tmp_path, ExhaustedProposer(), FakeEvaluator()).run()

    state_path = (
        tmp_path / "workspace" / "meta_harness" / "full_search_v3" / "offline_run" / "orchestration_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "proposal_exhausted"
    state["stop_reason"] = stop_reason
    state_path.write_text(canonical_json(state), encoding="utf-8")

    with pytest.raises(ResumeError, match="inconsistent"):
        _orchestrator(tmp_path, ExhaustedProposer(), FakeEvaluator()).run()


def test_no_early_stop_before_38_and_patience_stops_at_8_after_minimum(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(p0_metrics=(0.9, 0.9))

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert len(state["iterations"]) == 38
    assert state["patience"]["consecutive_non_improving"] == 38
    assert state["status"] == "patience_stopped"


def test_metric_improvement_resets_patience(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(
        p0_metrics=(0.5, 0.5),
        candidate_metrics=lambda iteration: (0.6, 0.6)
        if iteration == 8
        else ((0.5, 0.5) if iteration < 8 else (0.6, 0.6)),
    )

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert len(state["iterations"]) == 38
    assert state["iterations"][7]["metric_improved"] is True
    assert state["status"] == "patience_stopped"


def test_hash_only_winner_change_does_not_reset_patience(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(
        p0_metrics=(0.5, 0.5),
        candidate_metrics=lambda _iteration: (0.5, 0.5),
        p0_hash="f" * 64,
    )

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert state["winner_id"] != EXPERIMENT_P0_CANDIDATE_ID
    assert state["iterations"][0]["metric_improved"] is False
    assert state["patience"]["consecutive_non_improving"] == 38


def test_hard_maximum_stops_at_50_completed_iterations(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(
        p0_metrics=(0.5, 0.5),
        candidate_metrics=lambda iteration: (0.5 + iteration / 1000, 0.5),
    )

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert state["status"] == "max_stopped"
    assert len(state["iterations"]) == 50
    assert state["patience"]["consecutive_non_improving"] == 0


def test_p0_can_remain_winner_when_all_candidates_are_worse(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(p0_metrics=(0.9, 0.9))

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert state["winner_id"] == EXPERIMENT_P0_CANDIDATE_ID


@pytest.mark.parametrize("interruption", ["proposal_accepted", "candidate_complete"])
def test_resume_after_durable_proposal_or_evaluation_does_not_repeat_work(
    tmp_path, interruption
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()

    def interrupt(transition, _state):
        if transition == interruption:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated"):
        _orchestrator(
            tmp_path, proposer, evaluator, transition_hook=interrupt
        ).run()

    resumed = _orchestrator(tmp_path, proposer, evaluator).run()

    assert resumed["status"] == "patience_stopped"
    assert len(evaluator.p0_calls) == 1
    assert [call["candidate_id"] for call in evaluator.candidate_calls].count(
        "candidate_001"
    ) == 1
    assert [call["iteration"] for call in proposer.calls].count(1) == 1


def test_incompatible_resume_identity_is_rejected(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    orchestration = _orchestrator(tmp_path, proposer, evaluator)
    orchestration.state()

    with pytest.raises(ResumeError, match="identity"):
        Orchestrator(
            repository_root=tmp_path,
            run_id="offline_run",
            search_input=_search_input(),
            solver_identity_sha256="d" * 64,
            cache=object(),
            executor=object(),
            proposer=proposer,
            proposer_identity={"kind": "offline_fake", "version": 1},
            p0_evaluator=evaluator.p0,
            candidate_evaluator=evaluator.candidate,
        ).state()


def test_previous_model_run_identity_is_not_resumable(tmp_path):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    orchestration = _orchestrator(tmp_path, proposer, evaluator)
    state = orchestration.state()

    assert state["identity"]["config_sha256"] == (
        "de66f778339d8dd19520fbbf258b7292c0378b8f47fa5fc613db7d33ef4a621f"
    )
    state["identity"]["config_sha256"] = (
        "9cd31a36b1763de32ed3e3878176aee5d7521c645d8ca4bfe4e4f91dc5019517"
    )
    orchestration.state_path.write_text(canonical_json(state), encoding="utf-8")

    with pytest.raises(ResumeError, match="run identity is incompatible"):
        _orchestrator(tmp_path, proposer, evaluator).state()


def _iteration1_recipient_envelope() -> dict:
    return build_prompt_proposer_input(
        iteration=1,
        parent_id="cot",
        parent_templates=canonical_baseline_sources(),
        aggregate_search_metrics={
            "macro_f1": 0.9,
            "accuracy": 0.9,
            "parse_coverage": 1.0,
            "total_records": EXPERIMENT_SEARCH_SIZE,
            "parsed_predictions": EXPERIMENT_SEARCH_SIZE,
            "abstentions_or_parse_failures": 0,
            "infrastructure_failures": 0,
        },
        lineage=[],
        representative_search_failures=[],
    )


def test_accepted_receipt_recovers_without_reproposing_after_interruption(tmp_path):
    templates = {
        method: source
        + "\nRecovered independent check. Respond with 'yes' or 'no'."
        for method, source in canonical_baseline_sources().items()
    }
    candidate = {
        "candidate_id": "candidate_001",
        "parent_id": "cot",
        "hypothesis": "Independent checks may reduce ambiguous decisions.",
        "expected_tradeoff": "The check may add brief reasoning.",
        "templates": templates,
        "source_sha256": template_source_sha256(templates),
    }
    envelope = _iteration1_recipient_envelope()
    base_prompt = pp._build_prompt(envelope)
    base_hash = hashlib.sha256(base_prompt.encode("utf-8")).hexdigest()
    receipt_path = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_run"
        / "proposals"
        / "iteration_0001"
        / "attempt_00001.json"
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        canonical_json(
            {
                "schema_version": pp.EXPERIMENT_PROPOSER_SCHEMA_VERSION,
                "protocol_id": EXPERIMENT_PROTOCOL_ID,
                "instruction_version": pp.EXPERIMENT_PROPOSER_INSTRUCTION_VERSION,
                "timestamp": "2026-01-01T00:00:00Z",
                "status": "accepted",
                "iteration": 1,
                "attempt": 1,
                "input_sha256": base_hash,
                "attempt_prompt_sha256": base_hash,
                "category": None,
                "candidate_id": "candidate_001",
                "candidate_source_sha256": template_source_sha256(templates),
                "candidate": candidate,
            }
        ),
        encoding="utf-8",
    )
    proposer = FakeProposer()
    evaluator = FakeEvaluator()

    state = _orchestrator(tmp_path, proposer, evaluator).run()

    assert state["iterations"][0]["candidate"]["candidate_id"] == "candidate_001"
    assert [call["iteration"] for call in proposer.calls].count(1) == 0
    assert [call["candidate_id"] for call in evaluator.candidate_calls].count(
        "candidate_001"
    ) == 1


def test_proposer_receives_only_sanitized_search_feedback_and_no_final_data(tmp_path):
    class InspectingProposer(FakeProposer):
        def propose(self, **kwargs):
            assert set(kwargs) == {
                "proposal_directory", "run_id", "iteration", "parent_id",
                "parent_templates", "aggregate_search_metrics", "lineage",
                "representative_search_failures", "existing_candidate_ids",
                "existing_source_sha256",
            }
            assert "FINAL" not in repr(kwargs)
            assert "gold_label" not in repr(kwargs)
            return super().propose(**kwargs)

    proposer = InspectingProposer()
    evaluator = FakeEvaluator()
    failures = [
        {
            "pattern": "ambiguous_support",
            "summary": "Ambiguous support relationships recur.",
            "count": 2,
            "methods": ["direct"],
        }
    ]

    _orchestrator(
        tmp_path,
        proposer,
        evaluator,
        representative_search_failures=failures,
    ).run()

    assert list(proposer.calls[0]["representative_search_failures"]) == failures
    with pytest.raises(ValueError, match="prohibited"):
        _orchestrator(
            tmp_path / "rejected",
            FakeProposer(),
            FakeEvaluator(),
            representative_search_failures=[
                {
                    "pattern": "private_final_record",
                    "summary": "must not be exposed",
                    "count": 1,
                    "methods": ["direct"],
                }
            ],
        )


def test_mocked_complete_search_flow_resumes_from_atomic_checkpoint(tmp_path):
    ordered_ids = tuple(f"search-{index:04d}" for index in range(1000))
    search_input = SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": "a" * 64,
                "search_membership_sha256": "b" * 64,
            }
        ),
        _records=tuple(
            MappingProxyType({"sample_id": sample_id}) for sample_id in ordered_ids
        ),
    )

    class ScriptedCodexRunner:
        def __init__(self):
            self.calls = []

        def __call__(self, command, **kwargs):
            serialized = kwargs["input"].decode("utf-8")
            envelope = json.loads(serialized.rsplit("SANITIZED_INPUT_JSON\n", 1)[1])
            assert set(envelope) == {
                "protocol_id",
                "iteration",
                "parent",
                "aggregate_search_metrics",
                "lineage",
                "representative_search_failures",
            }
            assert set(envelope["aggregate_search_metrics"]) <= {
                "macro_f1",
                "accuracy",
                "parse_coverage",
                "total_records",
                "parsed_predictions",
                "abstentions_or_parse_failures",
                "infrastructure_failures",
            }
            assert "FINAL" not in repr(envelope)
            assert "sample_id" not in repr(envelope)
            iteration = envelope["iteration"]
            templates = {
                method: source
        + f"\nMocked integration check {iteration}. Respond with 'yes' or 'no'."
                for method, source in envelope["parent"]["templates"].items()
            }
            payload = {
                "iteration": iteration,
                "candidate": {
                    "candidate_id": f"candidate_{iteration:03d}",
                    "parent_id": envelope["parent"]["candidate_id"],
                    "hypothesis": "Independent checks may reduce ambiguous decisions.",
                    "expected_tradeoff": "The added check may make responses longer.",
                    "templates": templates,
                },
            }
            self.calls.append((iteration, envelope))
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps(payload), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    class MembershipCheckingEvaluator(FakeEvaluator):
        def p0(self, **kwargs):
            assert tuple(kwargs["search_input"].sample_ids) == ordered_ids
            return super().p0(**kwargs)

        def candidate(self, **kwargs):
            assert tuple(kwargs["search_input"].sample_ids) == ordered_ids
            return super().candidate(**kwargs)

    runner = ScriptedCodexRunner()
    proposer = Proposer(runner=runner)
    evaluator = MembershipCheckingEvaluator()
    interrupted = {"value": False}

    def interrupt_once(transition, _state):
        if transition == "candidate_complete" and not interrupted["value"]:
            interrupted["value"] = True
            raise RuntimeError("simulated checkpoint interruption")

    first = Orchestrator(
        repository_root=tmp_path,
        run_id="mocked_integration",
        search_input=search_input,
        solver_identity_sha256="c" * 64,
        cache=object(),
        executor=object(),
        proposer=proposer,
        proposer_identity={"kind": "offline_fake", "version": 1},
        p0_evaluator=evaluator.p0,
        candidate_evaluator=evaluator.candidate,
        transition_hook=interrupt_once,
    )
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        first.run()

    resumed = Orchestrator(
        repository_root=tmp_path,
        run_id="mocked_integration",
        search_input=search_input,
        solver_identity_sha256="c" * 64,
        cache=object(),
        executor=object(),
        proposer=proposer,
        proposer_identity={"kind": "offline_fake", "version": 1},
        p0_evaluator=evaluator.p0,
        candidate_evaluator=evaluator.candidate,
    )
    state = resumed.run()

    assert state["status"] == "patience_stopped"
    assert len(state["iterations"]) == 38
    assert len(runner.calls) == len(evaluator.candidate_calls) == 38
    assert len(evaluator.p0_calls) == 1
    assert [iteration for iteration, _envelope in runner.calls] == list(range(1, 39))
    assert sum(
        call["candidate_id"] == "candidate_001"
        for call in evaluator.candidate_calls
    ) == 1
    assert json.loads(resumed.state_path.read_text(encoding="utf-8")) == state


def _capturing_proposer():
    class CapturingProposer(FakeProposer):
        def __init__(self):
            super().__init__()
            self.lineage_by_iteration: dict[int, list[dict]] = {}
            self.failure_summaries_by_iteration: dict[int, list[dict]] = {}

        def propose(self, **kwargs):
            self.lineage_by_iteration[kwargs["iteration"]] = list(
                kwargs.get("lineage", ())
            )
            self.failure_summaries_by_iteration[kwargs["iteration"]] = list(
                kwargs.get("representative_search_failures", ())
            )
            return super().propose(**kwargs)

    return CapturingProposer()


def test_history_delta_is_relative_to_own_parent_not_later_winner(tmp_path):
    proposer = _capturing_proposer()
    evaluator = FakeEvaluator(
        p0_metrics=(0.6, 0.6),
        candidate_metrics=lambda it: {1: (0.8, 0.8), 2: (0.9, 0.9)}.get(
            it, (0.9, 0.9)
        ),
    )

    _orchestrator(tmp_path, proposer, evaluator).run()

    lineage_3 = proposer.lineage_by_iteration[3]
    c1 = next(entry for entry in lineage_3 if entry["candidate_id"] == "candidate_001")
    c2 = next(entry for entry in lineage_3 if entry["candidate_id"] == "candidate_002")

    # candidate_001's parent is cot (0.6), so its delta is +0.2 and it improved
    # at creation, even though candidate_002 (0.9) later superseded it as winner.
    assert c1["parent_id"] == "cot"
    assert c1["delta_macro_f1"] == pytest.approx(0.2)
    assert c1["improved"] is True
    assert c1["hypothesis"] and c1["expected_tradeoff"]
    assert "source_sha256" in c1 and "parse_coverage" in c1
    # candidate_002's parent is candidate_001 (0.8), so delta is +0.1.
    assert c2["parent_id"] == "candidate_001"
    assert c2["delta_macro_f1"] == pytest.approx(0.1)
    assert c2["improved"] is True


def test_aggregate_failure_summaries_are_safe_and_derived(tmp_path):
    class ParsingFailureEvaluator(FakeEvaluator):
        def candidate(self, **kwargs):
            report = super().candidate(**kwargs)
            iteration = int(kwargs["candidate_id"].rsplit("_", 1)[1])
            if iteration <= 2:
                report["abstentions_or_parse_failures"] = 2
            return report

    proposer = _capturing_proposer()
    evaluator = ParsingFailureEvaluator()

    _orchestrator(tmp_path, proposer, evaluator).run()

    summaries = proposer.failure_summaries_by_iteration[3]
    parse_entries = [entry for entry in summaries if entry["pattern"] == "parse_failure"]
    assert parse_entries
    assert parse_entries[0]["count"] == 4
    assert parse_entries[0]["methods"] == []
    assert "per-method" in parse_entries[0]["summary"] or "unknown" in parse_entries[0][
        "summary"
    ]
    serialized = repr(summaries)
    assert "FINAL" not in serialized
    assert "sample_id" not in serialized
    assert "gold_label" not in serialized
    assert "raw_trace" not in serialized
