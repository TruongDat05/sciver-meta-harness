"""Offline metrics for JSONL benchmark inference results.

Metrics are computed per exact run, dataset, model, prompt variant, and
reasoning-method combination. When a sample has multiple attempts, only its
highest numbered attempt is evaluated so retries do not inflate sample counts.

Per-class precision and recall use ``0.0`` when their denominator is zero.
Per-class F1 is also ``0.0`` when precision plus recall is zero. Failed labeled
samples remain in the actual-class support used by recall and therefore in
Macro-F1, while the historical 2x2 parsed-prediction confusion matrix remains
unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

from utils.result_writer import (
    API_FAILURE,
    INVALID_INPUT,
    PARSE_FAILURE,
    SUCCESS,
    ResultWriterError,
    iter_result_records,
)


GROUP_FIELDS = ("run_id", "dataset", "model", "prompt_variant", "method")
DATASET_GROUP_FIELDS = ("run_id", "dataset", "model", "prompt_variant")
_REQUIRED_FIELDS = frozenset(
    {
        "run_id",
        "dataset",
        "model",
        "method",
        "sample_id",
        "prediction",
        "parse_status",
        "request_status",
        "attempt_count",
    }
)
_REQUEST_STATUSES = frozenset(
    {SUCCESS, API_FAILURE, INVALID_INPUT, PARSE_FAILURE}
)
_YES_LABELS = frozenset({"yes", "true"})
_NO_LABELS = frozenset({"no", "false"})


class EvaluationError(ValueError):
    """Raised when result data is ambiguous or violates the JSONL format."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load complete result records from a local JSONL file.

    A partial final line is ignored consistently with the durable result
    writer. Malformed records anywhere else are rejected.
    """

    result_path = Path(path)
    if not result_path.exists():
        raise EvaluationError(f"result file does not exist: {result_path}")
    if not result_path.is_file():
        raise EvaluationError(f"result path is not a file: {result_path}")
    try:
        return list(iter_result_records(result_path))
    except (OSError, UnicodeError, ResultWriterError) as exc:
        raise EvaluationError(f"could not read result file: {result_path}") from exc


def _json_identity(value: Any, field_name: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"{field_name} must be JSON serializable") from exc


def _validate_record(record: Mapping[str, Any], index: int) -> None:
    missing = _REQUIRED_FIELDS.difference(record)
    if missing:
        raise EvaluationError(
            f"result record {index} is missing required fields: "
            + ", ".join(sorted(missing))
        )

    for field in ("dataset", "model", "prompt_variant", "method"):
        value = _record_field(record, field)
        if not isinstance(value, str) or not value:
            raise EvaluationError(
                f"result record {index} field {field!r} must be a non-empty string"
            )
    _json_identity(record["run_id"], "run_id")
    _json_identity(record["sample_id"], "sample_id")

    status = record["request_status"]
    if status not in _REQUEST_STATUSES:
        raise EvaluationError(
            f"result record {index} has invalid request_status: {status!r}"
        )
    attempt_count = record["attempt_count"]
    if (
        not isinstance(attempt_count, int)
        or isinstance(attempt_count, bool)
        or attempt_count < 1
    ):
        raise EvaluationError(
            f"result record {index} attempt_count must be a positive integer"
        )

    if status == SUCCESS:
        if record["parse_status"] != "parsed":
            raise EvaluationError(
                f"result record {index} with success status must be parsed"
            )
        if _normalise_prediction(record["prediction"]) is None:
            raise EvaluationError(
                f"result record {index} with success status must predict yes or no"
            )
    elif status == PARSE_FAILURE:
        if record["parse_status"] == "parsed":
            raise EvaluationError(
                f"result record {index} with parse_failure status cannot be parsed"
            )
    elif record["parse_status"] == "parsed":
        raise EvaluationError(
            f"result record {index} with {status} status cannot be parsed"
        )


def _normalise_prediction(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    prediction = value.strip().casefold()
    return prediction if prediction in {"yes", "no"} else None


def _normalise_gold_label(
    value: Any,
    record_index: int,
    *,
    allow_unscored: bool = False,
) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        label = value.strip().casefold()
        if label in _YES_LABELS:
            return "yes"
        if label in _NO_LABELS:
            return "no"
    if allow_unscored:
        return None
    raise EvaluationError(
        f"result record {record_index} gold_label must be yes, no, true, false, "
        "a boolean, or null"
    )


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _record_field(record: Mapping[str, Any], field: str) -> Any:
    if field == "prompt_variant":
        return record.get(field, "cot")
    return record[field]


def _safe_class_ratio(numerator: int, denominator: int) -> float:
    """Use the frozen zero-denominator convention for class metrics."""

    return numerator / denominator if denominator else 0.0


def _per_class_metrics(
    confusion: Mapping[str, Mapping[str, int]],
    label_support: Mapping[str, int],
) -> dict[str, dict[str, int | float]]:
    metrics: dict[str, dict[str, int | float]] = {}
    for label in ("yes", "no"):
        other = "no" if label == "yes" else "yes"
        true_positive = confusion[f"actual_{label}"][f"predicted_{label}"]
        false_positive = confusion[f"actual_{other}"][f"predicted_{label}"]
        predicted_total = true_positive + false_positive
        actual_total = label_support[label]
        precision = _safe_class_ratio(true_positive, predicted_total)
        recall = _safe_class_ratio(true_positive, actual_total)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual_total,
            "predicted": predicted_total,
            "true_positive": true_positive,
        }
    return metrics


def _latest_attempts(
    records: Iterable[Mapping[str, Any]],
    *,
    group_fields: tuple[str, ...] = GROUP_FIELDS,
) -> dict[tuple[str, ...], list[tuple[int, Mapping[str, Any]]]]:
    latest: dict[
        tuple[str, ...],
        dict[str, tuple[int, int, Mapping[str, Any]]],
    ] = {}

    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise EvaluationError(f"result record {index} must be an object")
        _validate_record(record, index)
        group_key = tuple(
            _json_identity(_record_field(record, field), field)
            for field in group_fields
        )
        sample_key = _json_identity(record["sample_id"], "sample_id")
        attempt_count = record["attempt_count"]
        group = latest.setdefault(group_key, {})
        previous = group.get(sample_key)
        if previous is not None and previous[0] == attempt_count:
            raise EvaluationError(
                "duplicate attempt_count for the same run and sample at result "
                f"record {index}"
            )
        if previous is None or attempt_count > previous[0]:
            group[sample_key] = (attempt_count, index, record)

    return {
        group_key: sorted(
            ((index, record) for _, index, record in samples.values()),
            key=lambda item: item[0],
        )
        for group_key, samples in latest.items()
    }


def _evaluate_group(
    indexed_records: list[tuple[int, Mapping[str, Any]]],
    *,
    allow_unscored_gold_labels: bool = False,
    group_fields: tuple[str, ...] = GROUP_FIELDS,
) -> dict[str, Any]:
    first = indexed_records[0][1]
    summary: dict[str, Any] = {
        field: _record_field(first, field) for field in group_fields
    }
    status_counts = {
        SUCCESS: 0,
        API_FAILURE: 0,
        INVALID_INPUT: 0,
        PARSE_FAILURE: 0,
    }
    labeled_total = 0
    parsed_labeled_total = 0
    correct = 0
    confusion = {
        "actual_yes": {"predicted_yes": 0, "predicted_no": 0},
        "actual_no": {"predicted_yes": 0, "predicted_no": 0},
    }
    label_support = {"yes": 0, "no": 0}

    for record_index, record in indexed_records:
        status = record["request_status"]
        status_counts[status] += 1
        gold_label = _normalise_gold_label(
            record.get("gold_label"),
            record_index,
            allow_unscored=allow_unscored_gold_labels,
        )
        if gold_label is None:
            continue

        labeled_total += 1
        label_support[gold_label] += 1
        prediction = (
            _normalise_prediction(record["prediction"])
            if status == SUCCESS and record["parse_status"] == "parsed"
            else None
        )
        if prediction is None:
            continue

        parsed_labeled_total += 1
        if prediction == gold_label:
            correct += 1
        confusion[f"actual_{gold_label}"][f"predicted_{prediction}"] += 1

    total = len(indexed_records)
    parsed_total = status_counts[SUCCESS]
    accuracy = _ratio(correct, labeled_total)
    per_class = _per_class_metrics(confusion, label_support)
    macro_f1 = (per_class["yes"]["f1"] + per_class["no"]["f1"]) / 2
    unresolved_api_failures = status_counts[API_FAILURE]
    eligible = parsed_total == total and unresolved_api_failures == 0
    summary.update(
        {
            "total_samples": total,
            "successful_requests": status_counts[SUCCESS],
            "api_failures": status_counts[API_FAILURE],
            "invalid_inputs": status_counts[INVALID_INPUT],
            "parse_failures": status_counts[PARSE_FAILURE],
            "parse_coverage": _ratio(parsed_total, total),
            "accuracy": accuracy,
            "accuracy_all_labeled": accuracy,
            "accuracy_parsed_labeled": _ratio(correct, parsed_labeled_total),
            "confusion_matrix": confusion,
            "per_class": per_class,
            "macro_f1": macro_f1,
            "unresolved_api_failures": unresolved_api_failures,
            "eligible": eligible,
            "failures": {
                "total": total - status_counts[SUCCESS],
                "api_failure": status_counts[API_FAILURE],
                "invalid_input": status_counts[INVALID_INPUT],
                "parse_failure": status_counts[PARSE_FAILURE],
            },
        }
    )
    return summary


def evaluate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    allow_unscored_gold_labels: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate records without combining distinct run configurations.

    By default, unsupported gold labels remain an error.  Callers that need
    operational request and parse metrics for a dataset with native non-binary
    labels may explicitly leave those labels unscored; the stored records are
    never rewritten or coerced to yes/no.
    """

    groups = _latest_attempts(records)
    summaries = [
        _evaluate_group(
            group,
            allow_unscored_gold_labels=allow_unscored_gold_labels,
        )
        for group in groups.values()
    ]
    return sorted(
        summaries,
        key=lambda summary: tuple(
            _json_identity(summary[field], field) for field in GROUP_FIELDS
        ),
    )


def evaluate_dataset_records(
    records: Iterable[Mapping[str, Any]],
    *,
    allow_unscored_gold_labels: bool = False,
) -> list[dict[str, Any]]:
    """Aggregate final attempts across reasoning methods for each dataset run."""

    groups = _latest_attempts(records, group_fields=DATASET_GROUP_FIELDS)
    summaries = [
        _evaluate_group(
            group,
            allow_unscored_gold_labels=allow_unscored_gold_labels,
            group_fields=DATASET_GROUP_FIELDS,
        )
        for group in groups.values()
    ]
    return sorted(
        summaries,
        key=lambda summary: tuple(
            _json_identity(summary[field], field)
            for field in DATASET_GROUP_FIELDS
        ),
    )


def macro_average_accuracy(
    dataset_summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute the unweighted mean of dataset Accuracy values per model."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    seen_dataset_runs: set[tuple[str, str, str]] = set()
    for summary in dataset_summaries:
        model = str(summary["model"])
        prompt_variant = str(summary.get("prompt_variant", "cot"))
        dataset = str(summary["dataset"])
        key = (model, prompt_variant, dataset)
        if key in seen_dataset_runs:
            raise EvaluationError(
                "macro Accuracy requires at most one run per model, prompt "
                f"variant, and dataset; duplicate summaries found for model "
                f"{model!r}, prompt variant {prompt_variant!r}, "
                f"dataset {dataset!r}"
            )
        seen_dataset_runs.add(key)
        grouped.setdefault((model, prompt_variant), []).append(summary)
    results: list[dict[str, Any]] = []
    for (model, prompt_variant), summaries in sorted(grouped.items()):
        scored = [
            summary
            for summary in summaries
            if summary["accuracy"]["value"] is not None
        ]
        datasets = sorted({str(summary["dataset"]) for summary in scored})
        value = (
            sum(float(summary["accuracy"]["value"]) for summary in scored)
            / len(scored)
            if scored
            else None
        )
        results.append(
            {
                "model": model,
                "prompt_variant": prompt_variant,
                "value": value,
                "dataset_count": len(datasets),
                "datasets": datasets,
            }
        )
    return results


compute_metrics = evaluate_records


__all__ = [
    "EvaluationError",
    "DATASET_GROUP_FIELDS",
    "GROUP_FIELDS",
    "compute_metrics",
    "evaluate_dataset_records",
    "evaluate_records",
    "load_jsonl",
    "macro_average_accuracy",
]
