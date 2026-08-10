import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest

from meta_harness.candidate_store import CandidateStore
from meta_harness.config import MetaHarnessConfig
from meta_harness.evaluator import build_evaluation_report
from meta_harness.orchestrator import (
    EVALUATION_PROCEDURE,
    score_evaluation_report,
)
from meta_harness.retry import RetryError, retry_api_failures
from meta_harness.schemas import template_source_sha256
from meta_harness.split_manager import build_split_manifest, save_split_manifest
from model_inference.remote_client import InvalidRequestError
from utils.result_writer import iter_result_records


RUN_ID = "retry-run"
CANDIDATE_ID = "retry_candidate"
SAMPLE_COUNT = 395


class FakeSolver:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []
        self.last_usage = None

    def create_chat_completion(self, model, messages):
        self.calls.append((model, messages))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        self.last_usage = {"total_tokens": 3}
        return self.outcome


def _templates():
    suffix = "Conclude with exactly Answer: yes or Answer: no."
    return {
        "direct": (
            "Check the claim against the evidence.\nClaim: $claim\n"
            f"Context: $context\nCaption: $caption\n{suffix}"
        ),
        "analytical": (
            "Analyze the claim against the evidence.\nClaim: $claim\n"
            f"Context: $context\nCaption: $caption\n{suffix}"
        ),
        "parallel": (
            "Compare both evidence items.\nClaim: $claim\nContext: $context\n"
            f"Caption 1: $caption1\nCaption 2: $caption2\n{suffix}"
        ),
        "sequential": (
            "Review both evidence items in order.\nClaim: $claim\n"
            f"Context: $context\nCaption 1: $caption1\n"
            f"Caption 2: $caption2\n{suffix}"
        ),
    }


def _sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _create_run(root):
    image_path = root / "fixture.png"
    Image.new("RGB", (2, 2), "blue").save(image_path)
    config = MetaHarnessConfig()
    records = [
        {
            "sample_id": f"sample-{index:04d}",
            "paper_id": f"paper-{index:04d}",
            "claim_type": "direct",
            "claim": f"claim-{index:04d}",
            "context": f"context-{index:04d}",
            "caption": f"caption-{index:04d}",
            "image_path": str(image_path),
            "gold_label": "yes" if index % 2 == 0 else "no",
        }
        for index in range(1975)
    ]
    manifest = build_split_manifest(records, config)
    validation_ids = tuple(
        manifest["splits"]["validation"]["sample_ids"]
    )
    assert len(validation_ids) == SAMPLE_COUNT
    by_id = {record["sample_id"]: record for record in records}
    validation_records = [by_id[sample_id] for sample_id in validation_ids]
    data_directory = root / "workspace" / "meta_harness" / "data"
    save_split_manifest(data_directory / "sciver_split.json", manifest)
    _write_json(
        data_directory / "sciver_validation.json",
        validation_records,
    )

    store = CandidateStore(root, RUN_ID)
    templates = _templates()
    candidate = store.create(
        {
            "candidate_id": CANDIDATE_ID,
            "parent_id": "baseline_cot",
            "search_axis": "exploration",
            "hypothesis": (
                "The evidence check will reduce unsupported yes predictions."
            ),
            "templates": templates,
            "expected_tradeoff": (
                "The additional check may reduce recall while improving precision."
            ),
            "source_sha256": template_source_sha256(templates),
        },
        status="evaluated",
    )
    run_directory = store.run_directory
    evaluation = run_directory / "evaluations" / CANDIDATE_ID
    result_path = evaluation / "validation.results.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_records = []
    failed_sample_id = validation_ids[0]
    for sample_id in validation_ids:
        sample = by_id[sample_id]
        failed = sample_id == failed_sample_id
        result_records.append(
            {
                "run_id": RUN_ID,
                "sample_id": sample_id,
                "dataset": "SciVer",
                "model": config.model,
                "prompt_variant": CANDIDATE_ID,
                "method": "direct",
                "prediction": None if failed else sample["gold_label"],
                "parse_status": "not_attempted" if failed else "parsed",
                "raw_response": (
                    None
                    if failed
                    else f"Answer: {sample['gold_label']}"
                ),
                "request_status": "api_failure" if failed else "success",
                "error_type": "InvalidRequestError" if failed else None,
                "error_message": (
                    "Remote API rejected the request (HTTP 400)."
                    if failed
                    else None
                ),
                "timestamp": "2026-01-01T00:00:00Z",
                "attempt_count": 1,
                "gold_label": sample["gold_label"],
                "candidate_id": CANDIDATE_ID,
                "split_sha256": manifest["split_sha256"],
                "reasoning_method": "direct",
                "latency": 0.1,
                "usage": None,
            }
        )
    result_path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":")) + "\n"
            for record in result_records
        ),
        encoding="utf-8",
    )
    report = build_evaluation_report(
        run_id=RUN_ID,
        candidate=candidate,
        split_manifest=manifest,
        split_name="validation",
        selected_ids=validation_ids,
        model=config.model,
        prompt_variant=CANDIDATE_ID,
        config_sha256=config.sha256(),
        records=result_records,
    )
    metrics_path = evaluation / "validation.metrics.json"
    _write_json(metrics_path, report)
    score = score_evaluation_report(
        report,
        result_path=result_path,
        candidate_id=CANDIDATE_ID,
    )
    snapshot = {
        "config": config.as_dict(),
        "config_sha256": config.sha256(),
        "split_sha256": manifest["split_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "validation_sample_count": len(validation_ids),
        "validation_sample_ids_sha256": _sha256_json(validation_ids),
        "evaluation_procedure": EVALUATION_PROCEDURE,
        "candidates_per_iteration": 2,
        "limits": {
            "max_iterations": 10,
            "max_candidates": 20,
            "max_solver_calls": None,
            "max_tokens": None,
            "max_wall_time_seconds": None,
            "max_consecutive_failures": 3,
        },
        "early_stopping": {
            "patience": 3,
            "min_delta": 0.005,
            "min_iterations": 0,
        },
        "prior_search_feedback_sha256": _sha256_json(
            {"schema_version": 1, "histories": []}
        ),
    }
    state = {
        "schema_version": 1,
        "orchestrator_version": "meta_harness_orchestrator_v1",
        "run_id": RUN_ID,
        "status": "completed",
        "stop_reason": "max_iterations",
        "transition": 1,
        "last_transition": "completed",
        "next_iteration": 2,
        "configuration": snapshot,
        "configuration_sha256": _sha256_json(snapshot),
        "budgets": {
            "limits": snapshot["limits"],
            "consumed": {
                "iterations": 1,
                "candidates": 1,
                "proposer_calls": 1,
                "solver_calls": score["solver_calls"],
                "tokens": score["tokens"],
                "wall_time_seconds": 1.0,
            },
        },
        "candidates": {
            CANDIDATE_ID: {
                "candidate_id": CANDIDATE_ID,
                "candidate_sha256": candidate.sha256(),
                "prompt_sha256": candidate.source_sha256,
                "role": "candidate",
                "iteration": 1,
                "status": "evaluated",
                "score": score,
                "usage": {
                    field: score[field]
                    for field in (
                        "solver_calls",
                        "tokens",
                        "latency_seconds",
                    )
                },
                "result_path": (
                    f"evaluations/{CANDIDATE_ID}/validation.results.jsonl"
                ),
                "metrics_path": (
                    f"evaluations/{CANDIDATE_ID}/validation.metrics.json"
                ),
                "failure": None,
            }
        },
        "scores": {CANDIDATE_ID: score},
        "frontier": [],
        "best_candidate_id": None,
        "iterations": {},
        "early_stopping": {
            **snapshot["early_stopping"],
            "completed_iterations": 1,
            "failed_iterations": 0,
            "non_improving_iterations": 0,
        },
        "failures": {"total": 0, "consecutive": 0, "records": []},
        "proposer_metadata": {},
    }
    _write_json(run_directory / "run_state.json", state)
    return failed_sample_id


@pytest.fixture(scope="module")
def retry_run_factory(tmp_path_factory):
    source = tmp_path_factory.mktemp("retry-source")
    failed_sample_id = _create_run(source)

    def create(destination):
        root = destination / "repository"
        shutil.copytree(source, root)
        return root, failed_sample_id

    return create


def _retry(root, solver):
    return retry_api_failures(
        repository_root=root,
        run_id=RUN_ID,
        candidate_id=CANDIDATE_ID,
        max_attempts=2,
        live_api=True,
        solver=solver,
    )


def _paths(root):
    evaluation = (
        root
        / "workspace"
        / "meta_harness"
        / "runs"
        / RUN_ID
        / "evaluations"
        / CANDIDATE_ID
    )
    return (
        evaluation / "validation.results.jsonl",
        evaluation / "validation.metrics.json",
        evaluation.parents[1] / "run_state.json",
    )


def test_retries_only_one_api_failure(retry_run_factory, tmp_path):
    root, failed_sample_id = retry_run_factory(tmp_path)
    solver = FakeSolver("Answer: yes")

    summary = _retry(root, solver)

    result_path, _, _ = _paths(root)
    attempts = [
        record
        for record in iter_result_records(result_path)
        if record["sample_id"] == failed_sample_id
    ]
    assert len(solver.calls) == summary["retried_sample_count"] == 1
    assert [record["attempt_count"] for record in attempts] == [1, 2]
    assert attempts[0]["request_status"] == "api_failure"
    assert attempts[1]["request_status"] == "success"


def test_skips_394_successful_samples(retry_run_factory, tmp_path):
    root, failed_sample_id = retry_run_factory(tmp_path)
    solver = FakeSolver("Answer: yes")

    summary = _retry(root, solver)

    assert summary["skipped_successful_samples"] == 394
    assert len(solver.calls) == 1
    prompt = solver.calls[0][1][0]["content"][-1]["text"]
    assert failed_sample_id.split("-")[-1] in prompt
    assert "gold_label" not in prompt.casefold()


def test_successful_retry_yields_395_effective_results(
    retry_run_factory,
    tmp_path,
):
    root, _ = retry_run_factory(tmp_path)

    summary = _retry(root, FakeSolver("Answer: yes"))

    _, metrics_path, state_path = _paths(root)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert summary["effective_result_count"] == SAMPLE_COUNT
    assert metrics["sample_count"] == SAMPLE_COUNT
    assert metrics["metrics"]["coverage"] == 1.0
    assert metrics["unresolved_api_failures"] == 0
    assert summary["eligible"] is True
    assert state["best_candidate_id"] == CANDIDATE_ID


def test_repeated_http_400_remains_unresolved_and_is_safely_diagnosable(
    retry_run_factory,
    tmp_path,
    monkeypatch,
):
    root, failed_sample_id = retry_run_factory(tmp_path)
    fake_key = "obviously-fake-placeholder"
    monkeypatch.setenv("API_KEY", fake_key)
    solver = FakeSolver(
        InvalidRequestError(
            "Remote API rejected the request (HTTP 400).",
            http_status_code=400,
            response_error_code="invalid_request",
            response_error_message=(
                f"bad image Authorization: Bearer {fake_key} "
                + "A" * 160
            ),
        )
    )

    summary = _retry(root, solver)

    result_path, metrics_path, state_path = _paths(root)
    latest = [
        record
        for record in iter_result_records(result_path)
        if record["sample_id"] == failed_sample_id
    ][-1]
    serialized = result_path.read_text(encoding="utf-8")
    assert len(solver.calls) == 1
    assert summary["unresolved_api_failures"] == 1
    assert latest["request_status"] == "api_failure"
    assert latest["http_status_code"] == 400
    assert latest["response_error_code"] == "invalid_request"
    assert "[REDACTED]" in latest["response_error_message"]
    assert fake_key not in serialized
    assert json.loads(metrics_path.read_text())["unresolved_api_failures"] == 1
    assert json.loads(state_path.read_text())["best_candidate_id"] is None


def test_rerun_after_success_is_idempotent(retry_run_factory, tmp_path):
    root, _ = retry_run_factory(tmp_path)
    _retry(root, FakeSolver("Answer: yes"))
    paths = _paths(root)
    before = [path.read_bytes() for path in paths]
    solver = FakeSolver(AssertionError("successful samples must not rerun"))

    summary = _retry(root, solver)

    assert solver.calls == []
    assert summary["attempts_appended"] == 0
    assert summary["skipped_successful_samples"] == SAMPLE_COUNT
    assert [path.read_bytes() for path in paths] == before


def test_immutable_run_identity_is_validated_before_append(
    retry_run_factory,
    tmp_path,
):
    root, _ = retry_run_factory(tmp_path)
    result_path, _, state_path = _paths(root)
    original_results = result_path.read_bytes()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["configuration"]["dataset_sha256"] = "0" * 64
    state["configuration_sha256"] = _sha256_json(state["configuration"])
    _write_json(state_path, state)
    solver = FakeSolver("Answer: yes")

    with pytest.raises(RetryError, match="frozen run identity"):
        _retry(root, solver)

    assert solver.calls == []
    assert result_path.read_bytes() == original_results


def test_retry_requires_live_opt_in_before_any_request(
    retry_run_factory,
    tmp_path,
):
    root, _ = retry_run_factory(tmp_path)
    solver = FakeSolver("Answer: yes")

    with pytest.raises(RetryError, match="live-api"):
        retry_api_failures(
            repository_root=root,
            run_id=RUN_ID,
            candidate_id=CANDIDATE_ID,
            max_attempts=2,
            live_api=False,
            solver=solver,
        )

    assert solver.calls == []


def test_frozen_run_is_refused_without_modifying_results(
    retry_run_factory,
    tmp_path,
):
    root, _ = retry_run_factory(tmp_path)
    result_path, _, state_path = _paths(root)
    finalization = state_path.parent / "finalization"
    _write_json(finalization / "frozen_winner.json", {"frozen": True})
    original_results = result_path.read_bytes()
    solver = FakeSolver("Answer: yes")

    with pytest.raises(RetryError, match="frozen or final-tested"):
        _retry(root, solver)

    assert solver.calls == []
    assert result_path.read_bytes() == original_results
