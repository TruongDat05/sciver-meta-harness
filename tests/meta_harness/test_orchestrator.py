from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import scripts.run_meta_harness as runner_cli
from meta_harness.config import MetaHarnessConfig
from meta_harness.orchestrator import (
    BudgetLimits,
    EarlyStoppingConfig,
    FrontierEntry,
    MetaHarnessOrchestrator,
    OrchestratorError,
    ResumeConfigurationError,
    RunLockedError,
    pareto_frontier,
    run_lock,
    winner_rank_key,
)
from meta_harness.schemas import Candidate, CandidateBatch, template_source_sha256
from meta_harness.split_manager import build_split_manifest


def _templates(suffix: str) -> dict[str, str]:
    return {
        "direct": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            f"Check support {suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "analytical": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            f"Check contradiction {suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "parallel": (
            "Claim: $claim\nContext: $context\nCaption 1: $caption1\n"
            f"Caption 2: $caption2\nCompare evidence {suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "sequential": (
            "Claim: $claim\nContext: $context\nCaption 1: $caption1\n"
            f"Caption 2: $caption2\nFollow evidence {suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
    }


def _candidate(
    candidate_id: str,
    parent_id: str,
    iteration: int,
    index: int,
) -> Candidate:
    templates = _templates(f"iteration {iteration} candidate {index}")
    return Candidate(
        candidate_id=candidate_id,
        parent_id=parent_id,
        search_axis="exploitation" if index == 1 else "exploration",
        hypothesis=(
            "Explicit evidence checks will reduce unsupported positive answers."
        ),
        templates=templates,
        expected_tradeoff="The reasoning may require more output text.",
        source_sha256=template_source_sha256(templates),
    )


class FakeProposer:
    def __init__(self, *, failures: set[int] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[dict] = []

    def propose(self, candidate_store, **kwargs):
        self.calls.append(kwargs)
        iteration = kwargs["iteration"]
        if iteration in self.failures:
            raise RuntimeError("simulated proposer failure")
        parent_id = kwargs["parent_candidate_ids"][0]
        candidates = tuple(
            _candidate(
                f"iter_{iteration:03d}_candidate_{index:02d}",
                parent_id,
                iteration,
                index,
            )
            for index in (1, 2)
        )
        return ProposalResult(
            CandidateBatch(iteration, candidates),
            {
                "status": "success",
                "iteration": iteration,
                "candidate_ids": [
                    candidate.candidate_id for candidate in candidates
                ],
            },
        )


@dataclass
class ProposalResult:
    batch: CandidateBatch
    metadata: dict


class FakeEvaluator:
    def __init__(
        self,
        scores: dict[str, float] | None = None,
        *,
        failures: set[str] | None = None,
        tokens: int = 4,
        latency: float = 1.0,
    ) -> None:
        self.scores = scores or {}
        self.failures = failures or set()
        self.tokens = tokens
        self.latency = latency
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        candidate_id = kwargs["candidate_id"]
        if candidate_id in self.failures:
            raise RuntimeError("simulated evaluator failure")
        score = self.scores.get(candidate_id, 0.5)
        return {
            "candidate_id": candidate_id,
            "metrics": {
                "Macro-F1": score,
                "accuracy": score,
                "coverage": 1.0,
                "parse_coverage": 1.0,
                "request_coverage": 1.0,
            },
            "unresolved_api_failures": 0,
            "rankable": True,
            "resource_usage": {
                "solver_calls": len(kwargs["sample_ids"]),
                "tokens": self.tokens,
                "latency_seconds": self.latency,
            },
        }


def _manifest_and_samples(config: MetaHarnessConfig | None = None):
    records = [
        {"sample_id": f"sample-{index}", "paper_id": f"paper-{index}"}
        for index in range(5)
    ]
    manifest = build_split_manifest(records, config)
    validation_ids = manifest["splits"]["validation"]["sample_ids"]
    samples = tuple({"sample_id": sample_id} for sample_id in validation_ids)
    return manifest, samples


def _orchestrator(
    tmp_path: Path,
    *,
    run_id: str = "run_001",
    proposer: FakeProposer | None = None,
    evaluator: FakeEvaluator | None = None,
    config: MetaHarnessConfig | None = None,
    manifest=None,
    samples=None,
    limits: BudgetLimits | None = None,
    early_stopping: EarlyStoppingConfig | None = None,
    transition_hook=None,
):
    config = config or MetaHarnessConfig()
    if manifest is None or samples is None:
        manifest, samples = _manifest_and_samples(config)
    return MetaHarnessOrchestrator(
        repository_root=tmp_path,
        run_id=run_id,
        config=config,
        split_manifest=manifest,
        validation_samples=samples,
        solver=object(),
        proposer=proposer or FakeProposer(),
        evaluator=evaluator or FakeEvaluator(),
        limits=limits or BudgetLimits(max_iterations=1, max_candidates=2),
        early_stopping=early_stopping or EarlyStoppingConfig(),
        transition_hook=transition_hook,
    )


def test_frontier_dominance_and_deterministic_exact_ties():
    entries = [
        FrontierEntry("z_equal", 0.8, 2, 10, 2.0),
        FrontierEntry("a_equal", 0.8, 2, 10, 2.0),
        FrontierEntry("dominated", 0.7, 3, 11, 3.0),
        FrontierEntry("efficient", 0.75, 1, 5, 1.0),
    ]

    frontier = pareto_frontier(reversed(entries))

    assert [entry.candidate_id for entry in frontier] == [
        "a_equal",
        "efficient",
    ]
    assert min(
        [
            FrontierEntry("lower_accuracy", 0.8, 2, 2, 0.1, accuracy=0.6),
            FrontierEntry("higher_accuracy", 0.8, 2, 8, 0.2, accuracy=0.7),
        ],
        key=winner_rank_key,
    ).candidate_id == "higher_accuracy"


def test_tampered_rankable_flag_cannot_override_coverage_gate(tmp_path):
    evaluator = FakeEvaluator()

    def unsafe_report(**kwargs):
        report = evaluator(**kwargs)
        report["metrics"]["coverage"] = 0.5
        report["rankable"] = True
        return report

    state = _orchestrator(
        tmp_path,
        evaluator=unsafe_report,
    ).run()

    assert state["frontier"] == []
    assert state["best_candidate_id"] is None


@pytest.mark.parametrize(
    ("request_coverage", "parse_coverage"),
    [(0.99, 1.0), (1.0, 0.99)],
)
def test_proposer_is_refused_until_baseline_has_full_request_and_parse_coverage(
    tmp_path,
    request_coverage,
    parse_coverage,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()

    def incomplete_baseline(**kwargs):
        report = evaluator(**kwargs)
        report["metrics"]["request_coverage"] = request_coverage
        report["metrics"]["parse_coverage"] = parse_coverage
        report["metrics"]["coverage"] = parse_coverage
        report["rankable"] = False
        return report

    state = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=incomplete_baseline,
    ).run()

    assert state["status"] == "baseline_incomplete"
    assert state["stop_reason"] == "baseline_coverage"
    assert proposer.calls == []
    assert [call["candidate_id"] for call in evaluator.calls] == [
        "baseline_cot"
    ]


def test_loop_evaluates_baseline_and_fixed_validation_then_stops_at_limits(
    tmp_path,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(
        {
            "baseline_cot": 0.5,
            "iter_001_candidate_01": 0.7,
            "iter_001_candidate_02": 0.6,
            "iter_002_candidate_01": 0.65,
            "iter_002_candidate_02": 0.64,
        }
    )
    orchestrator = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        limits=BudgetLimits(
            max_iterations=2,
            max_candidates=4,
            max_solver_calls=5,
            max_tokens=100,
        ),
    )

    state = orchestrator.run()

    assert state["status"] == "completed"
    assert state["stop_reason"] == "max_iterations"
    assert state["budgets"]["consumed"] == {
        "iterations": 2,
        "candidates": 4,
        "proposer_calls": 2,
        "solver_calls": 5,
        "tokens": 20,
        "wall_time_seconds": pytest.approx(
            state["budgets"]["consumed"]["wall_time_seconds"]
        ),
    }
    assert state["best_candidate_id"] == "iter_001_candidate_01"
    assert len(proposer.calls) == 2
    assert all(
        call["search_feedback"]
        == {"schema_version": 1, "histories": []}
        for call in proposer.calls
    )
    assert [call["candidate_id"] for call in evaluator.calls] == [
        "baseline_cot",
        "iter_001_candidate_01",
        "iter_001_candidate_02",
        "iter_002_candidate_01",
        "iter_002_candidate_02",
    ]
    assert all(call["split_name"] == "validation" for call in evaluator.calls)
    assert all(
        tuple(call["sample_ids"]) == orchestrator.validation_sample_ids
        for call in evaluator.calls
    )


@pytest.mark.parametrize(
    ("limits", "expected_reason", "expected_proposals", "expected_evaluations"),
    [
        (
            BudgetLimits(max_iterations=3, max_candidates=1),
            "candidate_budget",
            0,
            1,
        ),
        (
            BudgetLimits(
                max_iterations=3,
                max_candidates=6,
                max_solver_calls=1,
            ),
            "solver_call_budget",
            0,
            1,
        ),
        (
            BudgetLimits(
                max_iterations=3,
                max_candidates=6,
                max_tokens=4,
            ),
            "token_budget",
            0,
            1,
        ),
    ],
)
def test_candidate_solver_and_token_budgets_stop_before_extra_calls(
    tmp_path,
    limits,
    expected_reason,
    expected_proposals,
    expected_evaluations,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(tokens=4)
    state = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        limits=limits,
    ).run()

    assert state["status"] == "budget_exhausted"
    assert state["stop_reason"] == expected_reason
    assert len(proposer.calls) == expected_proposals
    assert len(evaluator.calls) == expected_evaluations


def test_wall_time_budget_stops_without_calling_evaluator(tmp_path):
    class StepClock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            self.value += 0.25
            return self.value

    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    orchestrator = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        limits=BudgetLimits(
            max_iterations=2,
            max_candidates=4,
            max_wall_time_seconds=0.5,
        ),
    )
    orchestrator.clock = StepClock()

    state = orchestrator.run()

    assert state["status"] == "budget_exhausted"
    assert state["stop_reason"] == "baseline_budget"
    assert proposer.calls == []
    assert evaluator.calls == []


def test_early_stopping_uses_global_macro_f1_min_delta_and_patience(tmp_path):
    evaluator = FakeEvaluator(
        {
            "baseline_cot": 0.5,
            "iter_001_candidate_01": 0.504,
            "iter_001_candidate_02": 0.49,
            "iter_002_candidate_01": 0.503,
            "iter_002_candidate_02": 0.48,
        }
    )
    state = _orchestrator(
        tmp_path,
        evaluator=evaluator,
        limits=BudgetLimits(max_iterations=5, max_candidates=10),
        early_stopping=EarlyStoppingConfig(
            patience=2,
            min_delta=0.005,
            min_iterations=2,
        ),
    ).run()

    assert state["status"] == "early_stopped"
    assert state["next_iteration"] == 3
    assert state["early_stopping"]["completed_iterations"] == 2
    assert state["early_stopping"]["non_improving_iterations"] == 2


def test_interruption_resume_and_terminal_idempotency_do_not_duplicate_work(
    tmp_path,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator(
        {
            "baseline_cot": 0.5,
            "iter_001_candidate_01": 0.6,
            "iter_001_candidate_02": 0.7,
        }
    )
    interrupted = False

    def interrupt_after_first_candidate(name, state):
        nonlocal interrupted
        candidate = state["candidates"].get("iter_001_candidate_01")
        if (
            not interrupted
            and name == "evaluation_completed"
            and candidate
            and candidate["status"] == "evaluated"
        ):
            interrupted = True
            raise KeyboardInterrupt

    first = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        transition_hook=interrupt_after_first_candidate,
    )
    with pytest.raises(KeyboardInterrupt):
        first.run()

    calls_after_interrupt = [
        call["candidate_id"] for call in evaluator.calls
    ]
    assert calls_after_interrupt == ["baseline_cot", "iter_001_candidate_01"]

    resumed = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
    )
    state = resumed.run(resume=True)
    assert state["status"] == "completed"
    assert [call["candidate_id"] for call in evaluator.calls] == [
        "baseline_cot",
        "iter_001_candidate_01",
        "iter_001_candidate_02",
    ]
    assert len(proposer.calls) == 1

    calls_before_idempotent_resume = len(evaluator.calls)
    proposals_before_idempotent_resume = len(proposer.calls)
    assert resumed.run(resume=True)["status"] == "completed"
    assert len(evaluator.calls) == calls_before_idempotent_resume
    assert len(proposer.calls) == proposals_before_idempotent_resume


@pytest.mark.parametrize("transition_name", ["proposal_returned", "candidate_validated"])
def test_resume_recovers_durable_proposal_without_reproposing(
    tmp_path,
    transition_name,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    interrupted = False

    def interrupt(name, state):
        nonlocal interrupted
        if not interrupted and name == transition_name:
            interrupted = True
            raise KeyboardInterrupt

    first = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        transition_hook=interrupt,
    )
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert len(proposer.calls) == 1

    state = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
    ).run(resume=True)

    assert state["status"] == "completed"
    assert len(proposer.calls) == 1
    assert state["budgets"]["consumed"]["proposer_calls"] == 1
    assert set(state["iterations"]["1"]["candidate_ids"]) == {
        "iter_001_candidate_01",
        "iter_001_candidate_02",
    }


def test_candidate_failure_is_isolated_and_iteration_recovers(tmp_path):
    evaluator = FakeEvaluator(
        {
            "baseline_cot": 0.5,
            "iter_001_candidate_02": 0.7,
        },
        failures={"iter_001_candidate_01"},
    )
    state = _orchestrator(
        tmp_path,
        evaluator=evaluator,
        limits=BudgetLimits(
            max_iterations=1,
            max_candidates=2,
            max_consecutive_failures=2,
        ),
    ).run()

    assert state["status"] == "completed"
    assert state["iterations"]["1"]["status"] == "partial_failure"
    assert state["candidates"]["iter_001_candidate_01"]["status"] == "failed"
    assert state["candidates"]["iter_001_candidate_02"]["status"] == "evaluated"
    assert state["best_candidate_id"] == "iter_001_candidate_02"
    assert state["failures"]["total"] == 1


def test_consecutive_proposer_failures_stop_at_limit_without_early_stop(
    tmp_path,
):
    proposer = FakeProposer(failures={1, 2, 3})
    state = _orchestrator(
        tmp_path,
        proposer=proposer,
        limits=BudgetLimits(
            max_iterations=5,
            max_candidates=10,
            max_consecutive_failures=2,
        ),
    ).run()

    assert state["status"] == "failure_limit"
    assert state["stop_reason"] == "consecutive_failures"
    assert len(proposer.calls) == 2
    assert state["early_stopping"]["failed_iterations"] == 2
    assert state["early_stopping"]["completed_iterations"] == 0
    assert state["early_stopping"]["non_improving_iterations"] == 0


def test_resume_rejects_mismatch_and_requires_safe_non_decreasing_mark(
    tmp_path,
):
    original = _orchestrator(tmp_path)
    assert original.run()["status"] == "completed"

    changed_limits = _orchestrator(
        tmp_path,
        limits=BudgetLimits(max_iterations=2, max_candidates=4),
    )
    with pytest.raises(ResumeConfigurationError, match="safe marking"):
        changed_limits.run(resume=True)

    state = changed_limits.run(
        resume=True,
        safe_resume_changes={"max_iterations", "max_candidates"},
    )
    assert state["configuration"]["limits"]["max_iterations"] == 2

    changed_config = MetaHarnessConfig(seed=43)
    manifest, samples = _manifest_and_samples(changed_config)
    mismatched = _orchestrator(
        tmp_path,
        config=changed_config,
        manifest=manifest,
        samples=samples,
        limits=BudgetLimits(max_iterations=2, max_candidates=4),
    )
    with pytest.raises(ResumeConfigurationError, match="immutable"):
        mismatched.run(
            resume=True,
            safe_resume_changes={"max_iterations", "max_candidates"},
        )


def test_safe_budget_extension_reopens_stopped_run_without_duplicate_work(
    tmp_path,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    first = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        limits=BudgetLimits(max_iterations=1, max_candidates=1),
    ).run()

    assert first["status"] == "budget_exhausted"
    assert first["stop_reason"] == "candidate_budget"
    assert [call["candidate_id"] for call in evaluator.calls] == [
        "baseline_cot"
    ]

    resumed = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        limits=BudgetLimits(max_iterations=1, max_candidates=2),
    ).run(
        resume=True,
        safe_resume_changes={"max_candidates"},
    )

    assert resumed["status"] == "completed"
    assert resumed["stop_reason"] == "max_iterations"
    assert len(proposer.calls) == 1
    assert [call["candidate_id"] for call in evaluator.calls] == [
        "baseline_cot",
        "iter_001_candidate_01",
        "iter_001_candidate_02",
    ]
    assert resumed["budgets"]["consumed"]["solver_calls"] == 3


def test_resume_rejects_tampered_configuration_snapshot_hash(tmp_path):
    orchestrator = _orchestrator(tmp_path)
    orchestrator.run()
    state = json.loads(orchestrator.state_path.read_text(encoding="utf-8"))
    state["configuration"]["config"]["seed"] = 999
    orchestrator.state_path.write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    with pytest.raises(
        OrchestratorError,
        match="configuration snapshot hash mismatch",
    ):
        _orchestrator(tmp_path).run(resume=True)


def test_run_lock_rejects_concurrent_writer(tmp_path):
    lock_path = tmp_path / "run" / ".run.lock"

    with run_lock(lock_path):
        with pytest.raises(RunLockedError, match="already writing"):
            with run_lock(lock_path):
                pass


def test_final_test_ids_and_samples_never_reach_proposer_or_evaluation(
    tmp_path,
):
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    orchestrator = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
    )
    final_ids = {
        json.dumps(value, sort_keys=True)
        for value in orchestrator.split_manifest["splits"]["final_test"][
            "sample_ids"
        ]
    }

    orchestrator.run()

    proposer_payload = json.dumps(proposer.calls, sort_keys=True)
    assert not any(final_id in proposer_payload for final_id in final_ids)
    for call in evaluator.calls:
        evaluated = {
            json.dumps(value, sort_keys=True) for value in call["sample_ids"]
        }
        loaded = {
            json.dumps(sample["sample_id"], sort_keys=True)
            for sample in call["samples"]
        }
        assert evaluated.isdisjoint(final_ids)
        assert loaded.isdisjoint(final_ids)


def test_search_rejects_loaded_samples_outside_validation_split(tmp_path):
    manifest, samples = _manifest_and_samples()
    final_id = manifest["splits"]["final_test"]["sample_ids"][0]

    with pytest.raises(ValueError, match="no search or final-test samples"):
        _orchestrator(
            tmp_path,
            manifest=manifest,
            samples=(*samples, {"sample_id": final_id}),
        )


def test_search_rejects_unsupported_generation_settings_before_state_creation(
    tmp_path,
):
    config = MetaHarnessConfig(temperature=0.1)
    manifest, samples = _manifest_and_samples(config)

    with pytest.raises(ValueError, match="must remain null"):
        _orchestrator(
            tmp_path,
            config=config,
            manifest=manifest,
            samples=samples,
        )

    assert not (tmp_path / "workspace" / "meta_harness").exists()


def test_search_rejects_split_built_from_different_configuration(tmp_path):
    manifest, samples = _manifest_and_samples(MetaHarnessConfig(seed=7))

    with pytest.raises(ValueError, match="split manifest configuration"):
        _orchestrator(
            tmp_path,
            config=MetaHarnessConfig(seed=42),
            manifest=manifest,
            samples=samples,
        )


def test_cli_dry_run_validates_and_prints_plan_without_boundaries_or_state(
    tmp_path,
    monkeypatch,
    capsys,
):
    config = MetaHarnessConfig()
    manifest, samples = _manifest_and_samples(config)
    monkeypatch.setattr(
        runner_cli,
        "load_meta_harness_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        runner_cli,
        "load_split_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        runner_cli,
        "_load_validation_samples",
        lambda *args: samples,
    )
    dotenv_boundary = Mock(
        side_effect=AssertionError("dry-run loaded .env")
    )
    monkeypatch.setattr(
        runner_cli,
        "load_cli_environment",
        dotenv_boundary,
    )
    monkeypatch.setattr(runner_cli, "REPOSITORY_ROOT", tmp_path)

    result = runner_cli.cli(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--dataset-path",
            str(tmp_path / "dataset.json"),
            "--run-id",
            "dry_run_001",
            "--dry-run",
            "--max-iterations",
            "2",
            "--max-candidates",
            "4",
        ]
    )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["planned_iterations"] == 2
    assert output["planned_candidates"] == 4
    dotenv_boundary.assert_not_called()
    assert not (
        runner_cli.REPOSITORY_ROOT
        / "workspace"
        / "meta_harness"
        / "runs"
        / "dry_run_001"
    ).exists()


def test_search_live_mode_requires_credentials_before_loading_run_data(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setattr(
        runner_cli,
        "load_cli_environment",
        lambda: False,
    )
    config_loader = Mock(
        side_effect=AssertionError("credentials must be checked first")
    )
    monkeypatch.setattr(
        runner_cli,
        "load_meta_harness_config",
        config_loader,
    )

    result = runner_cli.cli(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--dataset-path",
            str(tmp_path / "validation.json"),
            "--run-id",
            "missing_credentials",
            "--live-api",
        ]
    )

    assert result == 1
    config_loader.assert_not_called()
    assert "API_URL and API_KEY" in capsys.readouterr().err
