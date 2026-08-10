from __future__ import annotations

import json

import pytest

from meta_harness.baseline import canonical_baseline_sources
from meta_harness.candidate_store import CandidateStore
from meta_harness.proposer.feedback import (
    ProposerFeedbackError,
    load_reparsed_search_feedback,
    search_feedback_candidate_count,
)
from meta_harness.schemas import Candidate, template_source_sha256


def _candidate() -> Candidate:
    templates = {
        method: (
            source
            + "\nCheck only the stated relation before deciding. "
            "Choose yes or no and use exactly Answer: yes or Answer: no."
        )
        for method, source in canonical_baseline_sources().items()
    }
    return Candidate(
        candidate_id="localized_relation_v1",
        parent_id="baseline_cot",
        search_axis="exploration",
        hypothesis=(
            "A localized relation check is expected to increase validation "
            "Macro-F1 relative to baseline_cot."
        ),
        templates=templates,
        expected_tradeoff="The rule may shift class-specific recall.",
        source_sha256=template_source_sha256(templates),
    )


def _metrics(candidate_id: str, candidate: Candidate | None) -> dict:
    prompt_sha256 = (
        template_source_sha256(canonical_baseline_sources())
        if candidate is None
        else candidate.source_sha256
    )
    candidate_sha256 = "b" * 64 if candidate is None else candidate.sha256()
    confusion = (
        {
            "actual_no": {"predicted_no": 1, "predicted_yes": 0},
            "actual_yes": {"predicted_no": 1, "predicted_yes": 0},
        }
        if candidate is None
        else {
            "actual_no": {"predicted_no": 0, "predicted_yes": 1},
            "actual_yes": {"predicted_no": 0, "predicted_yes": 1},
        }
    )
    return {
        "evaluator_version": "meta_harness_evaluator_v1",
        "parser_version": "answer_parser_v2",
        "run_id": "history_v1",
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "prompt_sha256": prompt_sha256,
        "dataset": "SciVer",
        "model": "gemma-4-26B-A4B-it",
        "split": "validation",
        "split_sha256": "a" * 64,
        "sample_count": 2,
        "metrics": {
            "accuracy": 0.5,
            "Macro-F1": 0.5,
            "coverage": 1.0,
            "parse_coverage": 1.0,
            "request_coverage": 1.0,
            "precision": {"yes": 0.5, "no": 0.5},
            "recall": {"yes": 0.5, "no": 0.5},
            "F1": {"yes": 0.5, "no": 0.5},
            "confusion_matrix": confusion,
        },
        "failure_counts": {
            "invalid_input": 0,
            "api_failure": 0,
            "parse_failure": 0,
        },
        "unresolved_api_failures": 0,
        "rankable": True,
    }


def _records(candidate_id: str) -> list[dict]:
    predictions = (
        ("no", "no")
        if candidate_id == "baseline_cot"
        else ("yes", "yes")
    )
    return [
        {
            "candidate_id": candidate_id,
            "prompt_variant": candidate_id,
            "sample_id": "PRIVATE_CASE_ALPHA",
            "reasoning_method": "direct",
            "gold_label": "yes",
            "prediction": predictions[0],
            "parse_status": "parsed",
            "request_status": "success",
        },
        {
            "candidate_id": candidate_id,
            "prompt_variant": candidate_id,
            "sample_id": "PRIVATE_CASE_BETA",
            "reasoning_method": "parallel",
            "gold_label": "no",
            "prediction": predictions[1],
            "parse_status": "parsed",
            "request_status": "success",
        },
    ]


def _write_evaluation(directory, report, records) -> None:
    directory.mkdir(parents=True)
    (directory / "validation.metrics.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (directory / "validation.results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _history_fixture(tmp_path):
    store = CandidateStore(tmp_path, "history_v1")
    candidate = store.create(_candidate(), status="evaluated")
    reparse = tmp_path / "parser_v2_reparse"
    _write_evaluation(
        reparse / "baseline_cot",
        _metrics("baseline_cot", None),
        _records("baseline_cot"),
    )
    _write_evaluation(
        reparse / candidate.candidate_id,
        _metrics(candidate.candidate_id, candidate),
        _records(candidate.candidate_id),
    )
    (reparse / "summary.json").write_text(
        json.dumps(
            {
                "parser_version": "answer_parser_v2",
                "source_run_id": "history_v1",
                "evaluations": [
                    {"candidate_id": candidate.candidate_id},
                    {"candidate_id": "baseline_cot"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return reparse


def test_reparsed_history_exposes_anonymous_paired_errors_and_deltas(tmp_path):
    feedback = load_reparsed_search_feedback(
        tmp_path,
        _history_fixture(tmp_path),
    )

    assert search_feedback_candidate_count(feedback) == 1
    history = feedback["histories"][0]
    assert history["baseline"]["incorrect_case_keys"] == ["case_0001"]
    candidate = history["tried_candidates"][0]
    assert candidate["paired_outcome_counts"]["corrected"] == 1
    assert candidate["paired_outcome_counts"]["regressed"] == 1
    assert candidate["delta_vs_baseline"]["metrics"]["accuracy"] == 0.0
    assert candidate["delta_vs_baseline"]["confusion_matrix"] == {
        "actual_no": {"predicted_no": -1, "predicted_yes": 1},
        "actual_yes": {"predicted_no": -1, "predicted_yes": 1},
    }

    serialized = json.dumps(feedback, sort_keys=True)
    assert "PRIVATE_CASE_ALPHA" not in serialized
    assert "PRIVATE_CASE_BETA" not in serialized
    assert "gold_label" not in serialized
    assert "prediction" not in serialized
    assert "raw_response" not in serialized


def test_reparsed_history_rejects_noncurrent_parser(tmp_path):
    reparse = _history_fixture(tmp_path)
    summary_path = reparse / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parser_version"] = "answer_parser_v1"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ProposerFeedbackError, match="current parser"):
        load_reparsed_search_feedback(tmp_path, reparse)


def test_reparsed_history_rejects_nonvalidation_artifacts(tmp_path):
    reparse = _history_fixture(tmp_path)
    metrics_path = reparse / "baseline_cot" / "validation.metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["split"] = "final_test"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(ProposerFeedbackError, match="only validation"):
        load_reparsed_search_feedback(tmp_path, reparse)
