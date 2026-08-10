import csv
import json
from unittest.mock import patch

import pytest

from evaluation.metrics import (
    EvaluationError,
    evaluate_dataset_records,
    evaluate_records,
    load_jsonl,
    macro_average_accuracy,
)
from scripts import evaluate_results


@pytest.fixture(autouse=True)
def block_http_requests():
    """Evaluation must remain entirely local."""

    with patch(
        "requests.sessions.Session.request",
        side_effect=AssertionError("evaluation tests must not access the network"),
    ):
        yield


def _record(sample_id, status, *, prediction=None, gold_label=None, **overrides):
    parse_status = "parsed" if status == "success" else "not_attempted"
    if status == "parse_failure":
        parse_status = "invalid"
        prediction = "invalid"
    record = {
        "run_id": "run-a",
        "sample_id": sample_id,
        "dataset": "SciVer",
        "model": "synthetic-model",
        "method": "direct",
        "prediction": prediction,
        "parse_status": parse_status,
        "request_status": status,
        "attempt_count": 1,
    }
    if gold_label is not None:
        record["gold_label"] = gold_label
    record.update(overrides)
    return record


def _synthetic_records():
    # Manually verified run-a expectations after the sample-1 retry:
    # outcomes = 3 success, 1 API failure, 1 invalid input, 1 parse failure;
    # parse coverage = 3/6; all-label accuracy = 2/5;
    # parsed-label accuracy = 2/3; confusion = TP 1, FP 1, FN 0, TN 1.
    return [
        _record("sample-1", "api_failure", gold_label="yes"),
        _record(
            "sample-1",
            "success",
            prediction="yes",
            gold_label="yes",
            attempt_count=2,
        ),
        _record("sample-2", "success", prediction="yes", gold_label="no"),
        _record("sample-3", "parse_failure", gold_label=False),
        _record("sample-4", "api_failure", gold_label=True),
        _record("sample-5", "invalid_input"),
        _record("sample-6", "success", prediction="no", gold_label=False),
    ]


def test_metrics_use_final_attempt_and_keep_failures_out_of_predictions():
    summaries = evaluate_records(_synthetic_records())

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["total_samples"] == 6
    assert summary["successful_requests"] == 3
    assert summary["api_failures"] == 1
    assert summary["invalid_inputs"] == 1
    assert summary["parse_failures"] == 1
    assert summary["parse_coverage"] == {
        "value": 0.5,
        "numerator": 3,
        "denominator": 6,
    }
    assert summary["accuracy_all_labeled"] == {
        "value": 0.4,
        "numerator": 2,
        "denominator": 5,
    }
    assert summary["accuracy_parsed_labeled"] == {
        "value": pytest.approx(2 / 3),
        "numerator": 2,
        "denominator": 3,
    }
    assert summary["confusion_matrix"] == {
        "actual_yes": {"predicted_yes": 1, "predicted_no": 0},
        "actual_no": {"predicted_yes": 1, "predicted_no": 1},
    }
    assert summary["per_class"] == {
        "yes": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "support": 2,
            "predicted": 2,
            "true_positive": 1,
        },
        "no": {
            "precision": 1.0,
            "recall": pytest.approx(1 / 3),
            "f1": 0.5,
            "support": 3,
            "predicted": 1,
            "true_positive": 1,
        },
    }
    assert summary["macro_f1"] == 0.5
    assert summary["unresolved_api_failures"] == 1
    assert summary["eligible"] is False


def test_groups_preserve_prompt_variant_and_reasoning_method_boundaries():
    records = [_record("base", "success", prediction="yes", gold_label="yes")]
    records.extend(
        [
            _record(
                "other-run",
                "success",
                prediction="no",
                gold_label="no",
                run_id="run-b",
            ),
            _record(
                "other-dataset",
                "success",
                prediction="no",
                gold_label="no",
                dataset="MuSciClaims",
            ),
            _record(
                "other-model",
                "success",
                prediction="no",
                gold_label="no",
                model="another-synthetic-model",
            ),
            _record(
                "other-method",
                "success",
                prediction="no",
                gold_label="no",
                method="analytical",
            ),
            _record(
                "other-prompt",
                "success",
                prediction="no",
                gold_label="no",
                prompt_variant="candidate-1",
            ),
        ]
    )

    summaries = evaluate_records(records)

    assert len(summaries) == 6
    identities = {
        (
            item["run_id"],
            item["dataset"],
            item["model"],
            item["prompt_variant"],
            item["method"],
        )
        for item in summaries
    }
    assert identities == {
        ("run-a", "SciVer", "synthetic-model", "cot", "direct"),
        ("run-b", "SciVer", "synthetic-model", "cot", "direct"),
        ("run-a", "MuSciClaims", "synthetic-model", "cot", "direct"),
        ("run-a", "SciVer", "another-synthetic-model", "cot", "direct"),
        ("run-a", "SciVer", "synthetic-model", "cot", "analytical"),
        ("run-a", "SciVer", "synthetic-model", "candidate-1", "direct"),
    }
    assert all(item["total_samples"] == 1 for item in summaries)


def test_unlabeled_group_reports_explicit_zero_denominators():
    summary = evaluate_records(
        [_record("sample", "api_failure", gold_label=None)]
    )[0]

    assert summary["parse_coverage"] == {
        "value": 0.0,
        "numerator": 0,
        "denominator": 1,
    }
    assert summary["accuracy_all_labeled"] == {
        "value": None,
        "numerator": 0,
        "denominator": 0,
    }
    assert summary["accuracy_parsed_labeled"] == {
        "value": None,
        "numerator": 0,
        "denominator": 0,
    }
    assert summary["per_class"]["yes"]["f1"] == 0.0
    assert summary["per_class"]["no"]["f1"] == 0.0
    assert summary["macro_f1"] == 0.0


def test_hand_calculated_imbalanced_class_metrics():
    records = [
        _record("yes-1", "success", prediction="yes", gold_label="yes"),
        _record("no-1", "success", prediction="yes", gold_label="no"),
        _record("no-2", "success", prediction="no", gold_label="no"),
        _record("no-3", "success", prediction="no", gold_label="no"),
        _record("no-4", "success", prediction="no", gold_label="no"),
    ]

    summary = evaluate_records(records)[0]

    assert summary["per_class"]["yes"] == {
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2 / 3),
        "support": 1,
        "predicted": 2,
        "true_positive": 1,
    }
    assert summary["per_class"]["no"] == {
        "precision": 1.0,
        "recall": 0.75,
        "f1": pytest.approx(6 / 7),
        "support": 4,
        "predicted": 3,
        "true_positive": 3,
    }
    assert summary["macro_f1"] == pytest.approx(16 / 21)
    assert summary["eligible"] is True


def test_zero_denominator_class_metrics_are_frozen_at_zero():
    summary = evaluate_records(
        [_record("yes-only", "success", prediction="yes", gold_label="yes")]
    )[0]

    assert summary["per_class"]["yes"]["precision"] == 1.0
    assert summary["per_class"]["yes"]["recall"] == 1.0
    assert summary["per_class"]["yes"]["f1"] == 1.0
    assert summary["per_class"]["no"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
        "predicted": 0,
        "true_positive": 0,
    }
    assert summary["macro_f1"] == 0.5


@pytest.mark.parametrize(
    ("record", "eligible", "unresolved_api_failures"),
    [
        (
            _record("success", "success", prediction="yes", gold_label="yes"),
            True,
            0,
        ),
        (_record("api", "api_failure", gold_label="yes"), False, 1),
        (_record("parse", "parse_failure", gold_label="yes"), False, 0),
    ],
)
def test_candidate_eligibility_requires_full_parse_coverage_and_no_api_failures(
    record, eligible, unresolved_api_failures
):
    summary = evaluate_records([record])[0]

    assert summary["eligible"] is eligible
    assert summary["unresolved_api_failures"] == unresolved_api_failures


def test_resolved_api_failure_does_not_make_latest_attempt_ineligible():
    records = [
        _record("retried", "api_failure", gold_label="yes"),
        _record(
            "retried",
            "success",
            prediction="yes",
            gold_label="yes",
            attempt_count=2,
        ),
    ]

    summary = evaluate_records(records)[0]

    assert summary["total_samples"] == 1
    assert summary["unresolved_api_failures"] == 0
    assert summary["parse_coverage"]["value"] == 1.0
    assert summary["eligible"] is True


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([_record("sample", "success", prediction="maybe")], "predict yes or no"),
        ([_record("sample", "success", prediction="yes", gold_label="unknown")], "gold_label"),
        (
            [
                _record("sample", "api_failure"),
                _record("sample", "api_failure"),
            ],
            "duplicate attempt_count",
        ),
    ],
)
def test_ambiguous_or_invalid_result_records_are_rejected(records, message):
    with pytest.raises(EvaluationError, match=message):
        evaluate_records(records)


def test_load_jsonl_reads_records_and_ignores_only_partial_final_line(tmp_path):
    path = tmp_path / "results.jsonl"
    record = _record("sample", "success", prediction="yes", gold_label="yes")
    path.write_text(json.dumps(record) + "\n" + '{"partial":', encoding="utf-8")

    assert load_jsonl(path) == [{**record, "prompt_variant": "cot"}]


def test_cli_exports_json_and_tabular_csv_with_denominators(tmp_path):
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(record) + "\n" for record in _synthetic_records()),
        encoding="utf-8",
    )
    json_path = tmp_path / "reports" / "summary.json"
    csv_path = tmp_path / "reports" / "summary.csv"

    exit_code = evaluate_results.cli(
        [
            str(results_path),
            "--summary-json",
            str(json_path),
            "--csv",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["grouping"] == [
        "run_id",
        "dataset",
        "model",
        "prompt_variant",
        "method",
    ]
    assert payload["groups"][0]["prompt_variant"] == "cot"
    assert payload["groups"][0]["accuracy_all_labeled"]["denominator"] == 5
    assert payload["groups"][0]["macro_f1"] == 0.5
    assert payload["datasets"][0]["accuracy"] == {
        "value": 0.4,
        "numerator": 2,
        "denominator": 5,
    }
    assert payload["datasets"][0]["failures"] == {
        "total": 3,
        "api_failure": 1,
        "invalid_input": 1,
        "parse_failure": 1,
    }
    assert payload["macro_average_accuracy"] == [
        {
            "model": "synthetic-model",
            "prompt_variant": "cot",
            "value": 0.4,
            "dataset_count": 1,
            "datasets": ["SciVer"],
        }
    ]

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["total_samples"] == "6"
    assert rows[0]["parse_coverage"] == "0.5"
    assert rows[0]["parse_coverage_denominator"] == "6"
    assert rows[0]["accuracy_all_labeled_denominator"] == "5"
    assert rows[0]["accuracy_parsed_labeled_denominator"] == "3"
    assert rows[0]["prompt_variant"] == "cot"
    assert rows[0]["yes_precision"] == "0.5"
    assert rows[0]["no_recall"] == str(1 / 3)
    assert rows[0]["macro_f1"] == "0.5"
    assert rows[0]["eligible"] == "False"
    assert rows[0]["actual_no_predicted_yes"] == "1"


def test_dataset_reports_combine_methods_and_macro_average_is_unweighted():
    records = [
        _record("s1", "success", prediction="yes", gold_label="yes"),
        _record(
            "s2",
            "success",
            prediction="no",
            gold_label="yes",
            method="analytical",
        ),
        _record(
            "a1",
            "success",
            prediction="no",
            gold_label="no",
            dataset="SciAtomicBench",
            run_id="run-atomic",
        ),
        _record(
            "a2",
            "success",
            prediction="yes",
            gold_label="yes",
            dataset="SciAtomicBench",
            run_id="run-atomic",
        ),
        _record(
            "a3",
            "success",
            prediction="no",
            gold_label="no",
            dataset="SciAtomicBench",
            run_id="run-atomic",
        ),
    ]

    summaries = evaluate_dataset_records(records)

    assert len(summaries) == 2
    sciver = next(summary for summary in summaries if summary["dataset"] == "SciVer")
    atomic = next(
        summary for summary in summaries if summary["dataset"] == "SciAtomicBench"
    )
    assert sciver["total_samples"] == 2
    assert sciver["accuracy"]["value"] == 0.5
    assert atomic["accuracy"]["value"] == 1.0
    assert macro_average_accuracy(summaries) == [
        {
            "model": "synthetic-model",
            "prompt_variant": "cot",
            "value": 0.75,
            "dataset_count": 2,
            "datasets": ["SciAtomicBench", "SciVer"],
        }
    ]


def test_macro_average_rejects_duplicate_dataset_runs():
    summaries = evaluate_dataset_records(
        [
            _record("first", "success", prediction="yes", gold_label="yes"),
            _record(
                "second",
                "success",
                prediction="no",
                gold_label="no",
                run_id="run-b",
            ),
        ]
    )

    with pytest.raises(
        EvaluationError,
        match="at most one run per model, prompt variant, and dataset",
    ):
        macro_average_accuracy(summaries)
