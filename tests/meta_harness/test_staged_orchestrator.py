from __future__ import annotations

import json
from itertools import chain, repeat
from pathlib import Path

import pytest
import scripts.run_meta_harness as runner_cli

from meta_harness.baseline import canonical_baseline_sources
from meta_harness.config import MetaHarnessConfig, SearchProtocol
from meta_harness.hard_search import build_hard_search_manifest
from meta_harness.orchestrator import (
    BudgetLimits,
    EarlyStoppingConfig,
    FinalizedRunError,
    ResumeConfigurationError,
    load_run_state,
)
from meta_harness.finalize import freeze_winner, load_frozen_winner
from meta_harness.schemas import Candidate, CandidateBatch, template_source_sha256
from meta_harness.split_manager import build_split_manifest
from meta_harness.staged_orchestrator import StagedMetaHarnessOrchestrator


class FakeProposer:
    def __init__(self):
        self.calls = []

    def propose(self, store, **kwargs):
        self.calls.append(kwargs)
        iteration = kwargs["iteration"]
        parent = kwargs["parent_candidate_ids"][0]
        candidates = []
        for index, axis in ((1, "exploitation"), (2, "exploration")):
            templates = {
                method: (
                    source
                    + f"\nCheck staged rule {iteration}-{index}. "
                    "Choose yes or no. Answer: yes or Answer: no."
                )
                for method, source in canonical_baseline_sources().items()
            }
            candidates.append(
                Candidate(
                    candidate_id=f"iter_{iteration:03d}_candidate_{index:02d}",
                    parent_id=parent,
                    search_axis=axis,
                    hypothesis=(
                        "A staged check is expected to reduce incorrect "
                        "decisions on the search set."
                    ),
                    templates=templates,
                    expected_tradeoff="The check may shift recall.",
                    source_sha256=template_source_sha256(templates),
                )
            )
        return CandidateBatch(iteration, tuple(candidates))


class FakeEvaluator:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "candidate_id": kwargs["candidate_id"],
                "split_name": kwargs["split_name"],
                "sample_count": len(kwargs["sample_ids"]),
            }
        )
        path = Path(kwargs["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if path.exists():
            existing = {
                json.dumps(row["sample_id"]): row
                for row in (
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            }
        for sample_id in kwargs["sample_ids"]:
            marker = json.dumps(sample_id)
            existing.setdefault(
                marker,
                {
                    "run_id": kwargs["run_id"],
                    "candidate_id": kwargs["candidate_id"],
                    "prompt_variant": kwargs["candidate_id"],
                    "sample_id": sample_id,
                    "split_sha256": kwargs["split_manifest"]["split_sha256"],
                    "attempt_count": 1,
                    "request_status": "success",
                    "parse_status": "parsed",
                    "reasoning_method": "direct",
                    "method": "direct",
                    "raw_response": "Reasoning only. Answer: yes",
                    "prediction": "yes",
                    "gold_label": "yes",
                    "latency": 0.1,
                    "usage": {"total_tokens": 7},
                },
            )
        rows = list(existing.values())
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        candidate_id = kwargs["candidate_id"]
        score = (
            0.9
            if candidate_id.endswith("01")
            else 0.7
            if candidate_id != "baseline_cot"
            else 0.5
        )
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
                "solver_calls": len(rows),
                "tokens": len(rows) * 7,
                "latency_seconds": len(rows) / 10,
            },
        }


def _fixture(
    tmp_path,
    *,
    iterations=1,
    protocol_max=None,
    patience=None,
    model=None,
    estimated_tokens_per_solver_call=4096,
    limits=None,
    early_stopping=None,
    clock=None,
):
    protocol = SearchProtocol(
        min_iterations=iterations,
        target_iterations=iterations,
        max_iterations=protocol_max or max(iterations, 2),
        promotion_top_k=1,
        search_examples=50,
        smoke_examples=10,
        early_stopping_patience=patience,
        estimated_tokens_per_solver_call=estimated_tokens_per_solver_call,
    )
    config = MetaHarnessConfig(
        model=model or "gemma-4-26B-A4B-it",
        search_protocol=protocol,
        search_protocol_explicit=True,
    )
    records = [
        {
            "sample_id": f"sample-{index:03d}",
            "paper_id": f"paper-{index:03d}",
            "gold_label": "yes",
            "type": "direct",
            "claim": f"claim {index}",
            "section": f"evidence {index}",
        }
        for index in range(300)
    ]
    split = build_split_manifest(records, config)
    search_ids = set(split["splits"]["search"]["sample_ids"])
    reserved = [record for record in records if record["sample_id"] in search_ids]
    baseline = [
        {
            "sample_id": record["sample_id"],
            "gold_label": "yes",
            "prediction": "yes" if index % 5 == 0 else "no",
            "request_status": "success",
            "reasoning_method": "direct",
        }
        for index, record in enumerate(reserved)
    ]
    hard = build_hard_search_manifest(
        reserved,
        baseline,
        split,
        target_size=50,
    )
    selected_ids = {item["sample_id"] for item in hard["items"]}
    search_samples = tuple(
        record for record in reserved if record["sample_id"] in selected_ids
    )
    validation_ids = set(split["splits"]["validation"]["sample_ids"])
    validation_samples = tuple(
        record for record in records if record["sample_id"] in validation_ids
    )
    proposer = FakeProposer()
    evaluator = FakeEvaluator()
    orchestrator = StagedMetaHarnessOrchestrator(
        repository_root=tmp_path,
        run_id="staged_run",
        config=config,
        split_manifest=split,
        hard_search_manifest=hard,
        search_samples=search_samples,
        validation_samples=validation_samples,
        solver=object(),
        proposer=proposer,
        evaluator=evaluator,
        limits=limits,
        early_stopping=early_stopping,
        **({"clock": clock} if clock is not None else {}),
    )
    return orchestrator, proposer, evaluator, config, split, hard


def test_staged_search_promotes_only_top_k_and_resume_is_idempotent(tmp_path):
    orchestrator, proposer, evaluator, _, split, _ = _fixture(tmp_path)

    plan = orchestrator.planned_workload()
    state = orchestrator.run()

    assert state["status"] == "completed"
    assert state["promotion"]["candidate_ids"] == ["iter_001_candidate_01"]
    validation_calls = [
        call for call in evaluator.calls if call["split_name"] == "validation"
    ]
    assert validation_calls == [
        {
            "candidate_id": "iter_001_candidate_01",
            "split_name": "validation",
            "sample_count": split["splits"]["validation"]["sample_count"],
        }
    ]
    assert plan["stages"]["search"]["smoke_calls"] == 20
    assert plan["stages"]["protected_validation"]["promoted_candidates"] == 1
    calls = (len(proposer.calls), len(evaluator.calls))
    assert orchestrator.run(resume=True) == state
    assert (len(proposer.calls), len(evaluator.calls)) == calls


def test_proposer_visible_experience_excludes_protected_and_gold_data(tmp_path):
    orchestrator, _, _, _, split, _ = _fixture(tmp_path)
    orchestrator.run()

    experience = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "runs"
        / "staged_run"
        / "experience"
    )
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in experience.rglob("*")
        if path.is_file()
    )
    assert "gold_label" not in serialized
    assert '"prediction"' not in serialized
    assert "final_test" not in serialized
    assert all(
        sample_id not in serialized
        for sample_id in split["splits"]["validation"]["sample_ids"]
    )
    assert all(
        sample_id not in serialized
        for sample_id in split["splits"]["final_test"]["sample_ids"]
    )


def test_no_performance_stop_before_min_iterations(tmp_path):
    orchestrator, proposer, _, _, _, _ = _fixture(
        tmp_path,
        iterations=2,
        patience=1,
    )

    state = orchestrator.run()

    assert len(proposer.calls) == 2
    assert state["next_iteration"] == 3


def test_model_change_requires_new_run_and_finalized_run_is_immutable(tmp_path):
    orchestrator, _, _, _, split, hard = _fixture(tmp_path)
    orchestrator.run()
    changed = MetaHarnessConfig(
        model="gemma-4-31B-it",
        search_protocol=orchestrator.protocol,
        search_protocol_explicit=True,
    )
    changed_split = dict(split)
    changed_split["config_sha256"] = changed.sha256()
    payload = {
        key: value
        for key, value in changed_split.items()
        if key != "split_sha256"
    }
    import hashlib

    changed_split["split_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    changed_hard = dict(hard)
    changed_hard["source_split_sha256"] = changed_split["split_sha256"]
    hard_payload = {
        key: value
        for key, value in changed_hard.items()
        if key != "hard_search_sha256"
    }
    changed_hard["hard_search_sha256"] = hashlib.sha256(
        json.dumps(
            hard_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(ResumeConfigurationError):
        StagedMetaHarnessOrchestrator(
            repository_root=tmp_path,
            run_id="staged_run",
            config=changed,
            split_manifest=changed_split,
            hard_search_manifest=changed_hard,
            search_samples=orchestrator.search_samples,
            validation_samples=orchestrator.validation_samples,
            solver=object(),
            proposer=FakeProposer(),
            evaluator=FakeEvaluator(),
        ).run(resume=True)

    frozen = orchestrator.run_directory / "finalization" / "frozen_winner.json"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("{}", encoding="utf-8")
    with pytest.raises(FinalizedRunError):
        orchestrator.run(resume=True)


def test_archived_main_v1_v2_artifacts_remain_readable():
    root = Path(__file__).resolve().parents[2]

    v1 = load_run_state(root, "sciver-meta-gemma-main-v1")
    v2 = load_run_state(root, "sciver-meta-gemma-main-v2")
    frozen = load_frozen_winner(
        root
        / "workspace"
        / "meta_harness"
        / "runs"
        / "sciver-meta-gemma-main-v2"
        / "finalization"
        / "frozen_winner.json"
    )

    assert v1["schema_version"] == v2["schema_version"] == 1
    assert v1["configuration"]["evaluation_procedure"] == (
        "fixed_validation_split_v1"
    )
    assert frozen["run_id"] == "sciver-meta-gemma-main-v2"


def test_staged_resume_targets_retry_pending_api_stage(tmp_path):
    orchestrator, proposer, evaluator, _, _, _ = _fixture(tmp_path)
    original = evaluator.__call__
    failed_once = False

    def fail_first_candidate_smoke(**kwargs):
        nonlocal failed_once
        report = original(**kwargs)
        if (
            not failed_once
            and kwargs["candidate_id"] == "iter_001_candidate_01"
            and len(kwargs["sample_ids"]) == 10
        ):
            failed_once = True
            report["metrics"]["request_coverage"] = 0.9
            report["unresolved_api_failures"] = 1
            report["rankable"] = False
        return report

    orchestrator.evaluator = fail_first_candidate_smoke
    first = orchestrator.run()

    assert first["status"] == "retry_required"
    assert first["candidates"]["iter_001_candidate_01"]["stages"]["smoke"][
        "status"
    ] == "retry_pending"
    proposal_calls = len(proposer.calls)
    completed = orchestrator.run(resume=True)
    assert completed["status"] == "completed"
    assert len(proposer.calls) == proposal_calls


def test_new_31b_staged_run_freezes_its_own_model_identity(tmp_path):
    orchestrator, _, _, _, split, _ = _fixture(
        tmp_path,
        model="gemma-4-31B-it",
    )
    state = orchestrator.run()

    winner = freeze_winner(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=split,
        code_revision="0" * 40,
    )

    assert winner["search_solver_configuration"]["model"] == (
        "gemma-4-31B-it"
    )
    assert winner["candidate_id"] == state["best_candidate_id"]


def test_staged_cli_dry_run_reports_stage_calls_and_tokens_offline(
    tmp_path,
    monkeypatch,
    capsys,
):
    orchestrator, _, _, config, split, hard = _fixture(tmp_path / "fixture")
    config_path = tmp_path / "config.json"
    split_path = tmp_path / "split.json"
    hard_path = tmp_path / "hard.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    hard_path.write_text(json.dumps(hard), encoding="utf-8")
    monkeypatch.setattr(runner_cli, "REPOSITORY_ROOT", tmp_path / "cli-root")
    monkeypatch.setattr(
        runner_cli,
        "_load_validation_samples",
        lambda *args: orchestrator.validation_samples,
    )
    monkeypatch.setattr(
        runner_cli,
        "_load_exact_samples",
        lambda *args: orchestrator.search_samples,
    )

    result = runner_cli.cli(
        [
            "--config",
            str(config_path),
            "--split-manifest",
            str(split_path),
            "--hard-search-manifest",
            str(hard_path),
            "--search-dataset-path",
            str(tmp_path / "search.json"),
            "--dataset-path",
            str(tmp_path / "validation.json"),
            "--run-id",
            "dry_staged",
            "--dry-run",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_proposer_calls"] == 1
    assert plan["stages"]["search"]["solver_calls"] == 150
    assert plan["stages"]["protected_validation"]["solver_calls"] > 0
    assert plan["stages"]["final_test"]["included_in_search_execution"] is False
    assert plan["total_estimated_tokens_before_final"] > 0
    assert not (tmp_path / "cli-root" / "workspace").exists()


@pytest.mark.parametrize(
    ("limits", "expected_reason"),
    [
        (
            BudgetLimits(
                max_iterations=2,
                max_candidates=4,
                max_solver_calls=210,
            ),
            "solver_call_budget",
        ),
        (
            BudgetLimits(
                max_iterations=2,
                max_candidates=4,
                max_tokens=210 * 7,
            ),
            "token_budget",
        ),
        (
            BudgetLimits(max_iterations=2, max_candidates=2),
            "candidate_budget",
        ),
    ],
)
def test_staged_resource_ceilings_stop_before_extra_iteration(
    tmp_path,
    limits,
    expected_reason,
):
    orchestrator, proposer, evaluator, _, _, _ = _fixture(
        tmp_path,
        protocol_max=2,
        estimated_tokens_per_solver_call=7,
        limits=limits,
    )

    state = orchestrator.run()

    assert state["status"] == "completed"
    assert state["search_stop_reason"] == expected_reason
    assert len(proposer.calls) == 1
    assert {
        call["candidate_id"]
        for call in evaluator.calls
        if call["split_name"] == "search"
    } == {
        "baseline_cot",
        "iter_001_candidate_01",
        "iter_001_candidate_02",
    }
    assert state["budgets"]["consumed"]["solver_calls"] == 210
    assert state["budgets"]["consumed"]["tokens"] == 210 * 7


def test_staged_wall_time_ceiling_stops_before_evaluation(tmp_path):
    ticks = chain((0.0,), repeat(1.0))
    orchestrator, proposer, evaluator, _, _, _ = _fixture(
        tmp_path,
        limits=BudgetLimits(
            max_iterations=1,
            max_candidates=2,
            max_wall_time_seconds=0.5,
        ),
        clock=lambda: next(ticks),
    )

    state = orchestrator.run()

    assert state["status"] == "budget_exhausted"
    assert state["stop_reason"] == "wall_time_budget"
    assert proposer.calls == []
    assert evaluator.calls == []
    assert state["budgets"]["consumed"]["wall_time_seconds"] >= 0.5


def test_staged_failure_and_early_stopping_limits_are_enforced(tmp_path):
    failing, proposer, _, _, _, _ = _fixture(
        tmp_path / "failure",
        limits=BudgetLimits(
            max_iterations=1,
            max_candidates=2,
            max_consecutive_failures=1,
        ),
    )

    def fail_proposal(*_args, **_kwargs):
        proposer.calls.append("failed")
        raise RuntimeError("offline proposer failure")

    proposer.propose = fail_proposal
    failed_state = failing.run()
    assert failed_state["status"] == "failure_limit"
    assert failed_state["stop_reason"] == "consecutive_failures"
    assert failed_state["failures"]["consecutive"] == 1
    assert proposer.calls == ["failed"]

    stopping, stopping_proposer, _, _, _, _ = _fixture(
        tmp_path / "stopping",
        protocol_max=3,
        limits=BudgetLimits(max_iterations=3, max_candidates=6),
        early_stopping=EarlyStoppingConfig(
            patience=1,
            min_delta=0.5,
            min_iterations=2,
        ),
    )
    stopped_state = stopping.run()

    assert stopped_state["search_stop_reason"] == "performance_patience"
    assert len(stopping_proposer.calls) == 2


def test_staged_interruption_resume_does_not_duplicate_completed_work(
    tmp_path,
):
    orchestrator, proposer, evaluator, _, _, _ = _fixture(
        tmp_path,
        protocol_max=2,
    )

    def interrupt_after_iteration(transition, _state):
        if transition == "iteration_completed":
            raise RuntimeError("simulated interruption")

    orchestrator.transition_hook = interrupt_after_iteration
    with pytest.raises(RuntimeError, match="simulated interruption"):
        orchestrator.run()

    calls_before_resume = (len(proposer.calls), len(evaluator.calls))
    orchestrator.transition_hook = None
    state = orchestrator.run(resume=True)

    assert state["status"] == "completed"
    assert len(proposer.calls) == calls_before_resume[0]
    assert len(evaluator.calls) == calls_before_resume[1] + 1
    assert evaluator.calls[-1]["split_name"] == "validation"


def test_staged_resume_requires_explicit_non_decreasing_limit_changes(
    tmp_path,
):
    orchestrator, proposer, evaluator, config, split, hard = _fixture(
        tmp_path,
        protocol_max=2,
    )

    def interrupt_after_iteration(transition, _state):
        if transition == "iteration_completed":
            raise RuntimeError("simulated interruption")

    orchestrator.transition_hook = interrupt_after_iteration
    with pytest.raises(RuntimeError):
        orchestrator.run()

    extended = StagedMetaHarnessOrchestrator(
        repository_root=tmp_path,
        run_id="staged_run",
        config=config,
        split_manifest=split,
        hard_search_manifest=hard,
        search_samples=orchestrator.search_samples,
        validation_samples=orchestrator.validation_samples,
        solver=object(),
        proposer=proposer,
        evaluator=evaluator,
        limits=BudgetLimits(max_iterations=2, max_candidates=4),
        early_stopping=orchestrator.early_stopping,
    )
    with pytest.raises(
        ResumeConfigurationError,
        match="without explicit safe marking",
    ):
        extended.run(resume=True)

    state = extended.run(
        resume=True,
        safe_resume_changes=("max_iterations", "max_candidates"),
    )

    assert state["status"] == "completed"
    assert state["configuration"]["limits"]["max_iterations"] == 2
    assert state["configuration"]["limits"]["max_candidates"] == 4
    assert len(proposer.calls) == 2


def test_staged_fails_closed_when_cli_ceilings_cannot_fund_minimum(
    tmp_path,
):
    with pytest.raises(
        runner_cli.OrchestratorError,
        match="cannot fund the configured minimum 10-iteration protocol",
    ):
        _fixture(
            tmp_path,
            iterations=10,
            protocol_max=15,
            limits=BudgetLimits(max_iterations=2, max_candidates=4),
        )


def test_staged_cli_flags_reach_effective_limits_and_report_10_to_15_plan(
    tmp_path,
    monkeypatch,
    capsys,
):
    orchestrator, _, _, config, split, hard = _fixture(
        tmp_path / "fixture",
        iterations=10,
        protocol_max=15,
    )
    config_path = tmp_path / "config.json"
    split_path = tmp_path / "split.json"
    hard_path = tmp_path / "hard.json"
    config_path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
    split_path.write_text(json.dumps(split), encoding="utf-8")
    hard_path.write_text(json.dumps(hard), encoding="utf-8")
    monkeypatch.setattr(runner_cli, "REPOSITORY_ROOT", tmp_path / "cli-root")
    monkeypatch.setattr(
        runner_cli,
        "_load_validation_samples",
        lambda *args: orchestrator.validation_samples,
    )
    monkeypatch.setattr(
        runner_cli,
        "_load_exact_samples",
        lambda *args: orchestrator.search_samples,
    )

    result = runner_cli.cli(
        [
            "--config",
            str(config_path),
            "--split-manifest",
            str(split_path),
            "--hard-search-manifest",
            str(hard_path),
            "--search-dataset-path",
            str(tmp_path / "search.json"),
            "--dataset-path",
            str(tmp_path / "validation.json"),
            "--run-id",
            "dry_staged",
            "--dry-run",
            "--max-iterations",
            "15",
            "--max-candidates",
            "30",
            "--max-solver-calls",
            "2000",
            "--max-tokens",
            "10000000",
            "--max-wall-time",
            "500",
            "--max-consecutive-failures",
            "5",
            "--patience",
            "3",
            "--min-delta",
            "0.01",
            "--min-iterations",
            "10",
        ]
    )

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["requested_limits"] == {
        "max_candidates": 30,
        "max_consecutive_failures": 5,
        "max_iterations": 15,
        "max_solver_calls": 2000,
        "max_tokens": 10000000,
        "max_wall_time_seconds": 500.0,
    }
    assert plan["effective_limits"] == plan["requested_limits"]
    assert plan["effective_early_stopping"] == {
        "min_delta": 0.01,
        "min_iterations": 10,
        "patience": 3,
    }
    assert plan["iteration_plan"] == {
        "configured_maximum": 15,
        "configured_minimum": 10,
        "configured_target": 10,
        "effective_maximum": 15,
        "effective_minimum": 10,
    }
    assert plan["plans"]["minimum"]["iterations"] == 10
    assert plan["plans"]["minimum"]["candidates"] == 20
    assert plan["plans"]["maximum"]["iterations"] == 15
    assert plan["plans"]["maximum"]["candidates"] == 30
    assert plan["stages"]["final_test"]["included_in_search_execution"] is False
    assert not (tmp_path / "cli-root" / "workspace").exists()
