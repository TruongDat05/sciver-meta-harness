"""Build label-free, per-case feedback for prompt-family proposals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from meta_harness.candidate_store import CandidateStore, CandidateStoreError
from meta_harness.evaluator import DATASET_NAME, EVALUATOR_VERSION
from meta_harness.prompt_family import TEMPLATE_KEYS
from meta_harness.schemas import Candidate, canonical_json
from utils.answer_parser import PARSER_VERSION
from utils.result_writer import SUCCESS


SEARCH_FEEDBACK_SCHEMA_VERSION = 1
BASELINE_CANDIDATE_ID = "baseline_cot"
_RESULTS_NAME = "validation.results.jsonl"
_METRICS_NAME = "validation.metrics.json"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HISTORY_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_CASE_KEY = re.compile(r"^case_[0-9]{4,}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RATE_FIELDS = (
    "accuracy",
    "macro_f1",
    "parse_coverage",
    "request_coverage",
    "yes_precision",
    "yes_recall",
    "yes_f1",
    "no_precision",
    "no_recall",
    "no_f1",
)
_CONFUSION_ROWS = ("actual_no", "actual_yes")
_CONFUSION_COLUMNS = ("predicted_no", "predicted_yes")
_OUTCOMES = ("correct", "incorrect", "unresolved")
_PAIRED_BUCKETS = (
    "both_correct",
    "corrected",
    "regressed",
    "still_incorrect",
    "resolved_to_correct",
    "resolved_to_incorrect",
    "became_unresolved_from_correct",
    "became_unresolved_from_incorrect",
    "still_unresolved",
)
_SENSITIVE_TEXT = (
    re.compile(r"(?i)\b(?:final[_ -]?test|gold[_ -]?label|ground[_ -]?truth)\b"),
    re.compile(r"(?i)\b(?:raw[_ -]?response|authorization|bearer)\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|api[_ -]?url)\b"),
    re.compile(r"(?i)https?://"),
    re.compile(r"(?i)data:image/|\bbase64\b"),
)


class ProposerFeedbackError(ValueError):
    """Raised when proposer history is incomplete, unsafe, or inconsistent."""


def empty_search_feedback() -> dict[str, Any]:
    """Return the canonical empty feedback envelope."""

    return {
        "schema_version": SEARCH_FEEDBACK_SCHEMA_VERSION,
        "histories": [],
    }


def load_reparsed_search_feedback(
    repository_root: str | Path,
    reparse_directory: str | Path,
) -> dict[str, Any]:
    """Load one parser-v2 validation history without exposing record labels.

    The re-parse directory is read only. Its ``source_run_id`` selects the
    immutable candidate registry under the supplied repository root.
    """

    directory = Path(reparse_directory)
    summary = _load_json(directory / "summary.json", "re-parse summary")
    if summary.get("parser_version") != PARSER_VERSION:
        raise ProposerFeedbackError(
            "proposer history must use the current parser version "
            f"{PARSER_VERSION}"
        )
    source_run_id = _identifier(
        summary.get("source_run_id"),
        "re-parse source_run_id",
    )
    raw_evaluations = summary.get("evaluations")
    if (
        isinstance(raw_evaluations, (str, bytes))
        or not isinstance(raw_evaluations, Sequence)
        or not raw_evaluations
    ):
        raise ProposerFeedbackError(
            "re-parse summary must list evaluated candidates"
        )
    candidate_ids: list[str] = []
    for entry in raw_evaluations:
        if not isinstance(entry, Mapping):
            raise ProposerFeedbackError(
                "re-parse evaluation entries must be objects"
            )
        candidate_ids.append(
            _identifier(
                entry.get("candidate_id"),
                "re-parse candidate_id",
            )
        )
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ProposerFeedbackError(
            "re-parse summary contains duplicate candidate IDs"
        )
    if BASELINE_CANDIDATE_ID not in candidate_ids:
        raise ProposerFeedbackError(
            "re-parse summary is missing baseline_cot"
        )
    candidate_ids = [
        BASELINE_CANDIDATE_ID,
        *sorted(
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id != BASELINE_CANDIDATE_ID
        ),
    ]

    store = CandidateStore(repository_root, source_run_id)
    expected_candidates = set(candidate_ids) - {BASELINE_CANDIDATE_ID}
    try:
        registered_candidates = set(store.read_registry()["candidates"])
    except CandidateStoreError as exc:
        raise ProposerFeedbackError(
            "source candidate registry is unavailable or invalid"
        ) from exc
    if registered_candidates != expected_candidates:
        raise ProposerFeedbackError(
            "re-parse candidates do not match the immutable source registry"
        )

    history = build_search_history(
        store,
        directory,
        candidate_ids=candidate_ids,
        history_key="prior_search",
    )
    return normalize_search_feedback(
        {
            "schema_version": SEARCH_FEEDBACK_SCHEMA_VERSION,
            "histories": [history],
        }
    )


def build_search_history(
    candidate_store: CandidateStore,
    evaluation_root: str | Path,
    *,
    candidate_ids: Sequence[str],
    history_key: str,
) -> dict[str, Any]:
    """Build one compact history from complete validation evaluations."""

    if not isinstance(candidate_store, CandidateStore):
        raise TypeError("candidate_store must be a CandidateStore")
    key = _history_key(history_key)
    ordered_ids = tuple(
        _identifier(candidate_id, "candidate_id")
        for candidate_id in candidate_ids
    )
    if (
        not ordered_ids
        or ordered_ids[0] != BASELINE_CANDIDATE_ID
        or len(ordered_ids) != len(set(ordered_ids))
    ):
        raise ProposerFeedbackError(
            "candidate_ids must start with baseline_cot and contain no duplicates"
        )

    root = Path(evaluation_root)
    artifacts = {
        candidate_id: _load_evaluation(root, candidate_id)
        for candidate_id in ordered_ids
    }
    baseline_report, baseline_records = artifacts[BASELINE_CANDIDATE_ID]
    baseline_identity = _report_identity(baseline_report)
    baseline_by_marker, ordered_markers = _records_by_identity(
        baseline_records,
        candidate_id=BASELINE_CANDIDATE_ID,
    )
    if len(ordered_markers) != baseline_identity["case_count"]:
        raise ProposerFeedbackError(
            "baseline result count does not match its metrics report"
        )

    case_keys = {
        marker: f"case_{index:04d}"
        for index, marker in enumerate(ordered_markers, start=1)
    }
    baseline_outcomes = {
        marker: _record_outcome(baseline_by_marker[marker])
        for marker in ordered_markers
    }
    case_catalog = [
        {
            "case_key": case_keys[marker],
            "reasoning_method": _reasoning_method(
                baseline_by_marker[marker]
            ),
        }
        for marker in ordered_markers
    ]
    baseline = {
        "metrics": _metric_summary(baseline_report),
        "confusion_matrix": _confusion_matrix(baseline_report),
        "failure_counts": _failure_counts(baseline_report),
        "incorrect_case_keys": [
            case_keys[marker]
            for marker in ordered_markers
            if baseline_outcomes[marker] == "incorrect"
        ],
        "unresolved_case_keys": [
            case_keys[marker]
            for marker in ordered_markers
            if baseline_outcomes[marker] == "unresolved"
        ],
    }

    tried_candidates: list[dict[str, Any]] = []
    for candidate_id in ordered_ids[1:]:
        report, records = artifacts[candidate_id]
        identity = _report_identity(report)
        if {
            field: identity[field]
            for field in ("parser_version", "evaluator_version", "dataset", "model",
                          "split_sha256", "case_count")
        } != {
            field: baseline_identity[field]
            for field in ("parser_version", "evaluator_version", "dataset", "model",
                          "split_sha256", "case_count")
        }:
            raise ProposerFeedbackError(
                "all history evaluations must share parser, evaluator, dataset, "
                "model, split, and case count"
            )
        candidate = _load_candidate(candidate_store, candidate_id)
        _verify_candidate_report(candidate, report)
        records_by_marker, markers = _records_by_identity(
            records,
            candidate_id=candidate_id,
        )
        if set(markers) != set(ordered_markers):
            raise ProposerFeedbackError(
                "candidate results do not match baseline case identities"
            )

        paired_keys = {bucket: [] for bucket in _PAIRED_BUCKETS}
        for marker in ordered_markers:
            baseline_record = baseline_by_marker[marker]
            candidate_record = records_by_marker[marker]
            if (
                _reasoning_method(candidate_record)
                != _reasoning_method(baseline_record)
            ):
                raise ProposerFeedbackError(
                    "reasoning method changed for a paired validation case"
                )
            bucket = _paired_bucket(
                baseline_outcomes[marker],
                _record_outcome(candidate_record),
            )
            paired_keys[bucket].append(case_keys[marker])

        candidate_metrics = _metric_summary(report)
        candidate_confusion = _confusion_matrix(report)
        tried_candidates.append(
            {
                "strategy": _strategy(candidate),
                "metrics": candidate_metrics,
                "confusion_matrix": candidate_confusion,
                "failure_counts": _failure_counts(report),
                "delta_vs_baseline": {
                    "metrics": {
                        field: candidate_metrics[field]
                        - baseline["metrics"][field]
                        for field in _RATE_FIELDS
                    },
                    "confusion_matrix": {
                        row: {
                            column: (
                                candidate_confusion[row][column]
                                - baseline["confusion_matrix"][row][column]
                            )
                            for column in _CONFUSION_COLUMNS
                        }
                        for row in _CONFUSION_ROWS
                    },
                },
                "paired_outcome_counts": {
                    bucket: len(keys)
                    for bucket, keys in paired_keys.items()
                },
                "paired_error_case_keys": {
                    bucket: keys
                    for bucket, keys in paired_keys.items()
                    if bucket != "both_correct"
                },
            }
        )

    history = {
        "history_key": key,
        "parser_version": baseline_identity["parser_version"],
        "evaluator_version": baseline_identity["evaluator_version"],
        "dataset": baseline_identity["dataset"],
        "model": baseline_identity["model"],
        "split_sha256": baseline_identity["split_sha256"],
        "case_count": baseline_identity["case_count"],
        "case_catalog": case_catalog,
        "baseline": baseline,
        "tried_candidates": tried_candidates,
    }
    return _normalize_history(history)


def current_search_feedback(
    candidate_store: CandidateStore,
    evaluation_root: str | Path,
    *,
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    """Build feedback for currently complete evaluations, if available."""

    root = Path(evaluation_root)
    complete = [
        candidate_id
        for candidate_id in candidate_ids
        if (root / candidate_id / _RESULTS_NAME).is_file()
        and (root / candidate_id / _METRICS_NAME).is_file()
    ]
    if not complete or complete[0] != BASELINE_CANDIDATE_ID:
        return empty_search_feedback()
    return normalize_search_feedback(
        {
            "schema_version": SEARCH_FEEDBACK_SCHEMA_VERSION,
            "histories": [
                build_search_history(
                    candidate_store,
                    root,
                    candidate_ids=complete,
                    history_key="current_search",
                )
            ],
        }
    )


def merge_search_feedback(
    *values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Merge validated histories while preserving their order."""

    histories: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for value in values:
        normalized = normalize_search_feedback(value)
        for history in normalized["histories"]:
            key = history["history_key"]
            if key in seen_keys:
                raise ProposerFeedbackError(
                    "search feedback history keys must be unique"
                )
            seen_keys.add(key)
            histories.append(history)
    return {
        "schema_version": SEARCH_FEEDBACK_SCHEMA_VERSION,
        "histories": histories,
    }


def normalize_search_feedback(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate and detach the only feedback shape allowed into the proposer."""

    if value is None:
        return empty_search_feedback()
    if not isinstance(value, Mapping):
        raise ProposerFeedbackError("search feedback must be an object")
    if set(value) != {"schema_version", "histories"}:
        raise ProposerFeedbackError(
            "search feedback has unexpected or missing fields"
        )
    if value.get("schema_version") != SEARCH_FEEDBACK_SCHEMA_VERSION:
        raise ProposerFeedbackError(
            "search feedback schema_version is unsupported"
        )
    raw_histories = value.get("histories")
    if (
        isinstance(raw_histories, (str, bytes))
        or not isinstance(raw_histories, Sequence)
    ):
        raise ProposerFeedbackError("search feedback histories must be an array")
    histories = [_normalize_history(history) for history in raw_histories]
    keys = [history["history_key"] for history in histories]
    if len(keys) != len(set(keys)):
        raise ProposerFeedbackError(
            "search feedback history keys must be unique"
        )
    normalized = {
        "schema_version": SEARCH_FEEDBACK_SCHEMA_VERSION,
        "histories": histories,
    }
    encoded = canonical_json(normalized)
    if any(pattern.search(encoded) for pattern in _SENSITIVE_TEXT):
        raise ProposerFeedbackError(
            "search feedback contains forbidden sensitive content"
        )
    return normalized


def search_feedback_sha256(value: Mapping[str, Any] | None) -> str:
    """Hash normalized prior feedback for immutable resume identity."""

    normalized = normalize_search_feedback(value)
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def search_feedback_candidate_count(value: Mapping[str, Any] | None) -> int:
    """Count tried candidates in normalized feedback."""

    normalized = normalize_search_feedback(value)
    return sum(
        len(history["tried_candidates"])
        for history in normalized["histories"]
    )


def search_feedback_identities(
    value: Mapping[str, Any] | None,
) -> tuple[set[str], set[str], set[str]]:
    """Return tried IDs, prompt hashes, and normalized hypotheses."""

    normalized = normalize_search_feedback(value)
    candidate_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    hypotheses: set[str] = set()
    for history in normalized["histories"]:
        for entry in history["tried_candidates"]:
            strategy = entry["strategy"]
            candidate_ids.add(strategy["candidate_id"])
            prompt_hashes.add(strategy["source_sha256"])
            hypotheses.add(_normalized_text(strategy["hypothesis"]))
    return candidate_ids, prompt_hashes, hypotheses


def _normalize_history(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProposerFeedbackError("each search history must be an object")
    expected = {
        "history_key",
        "parser_version",
        "evaluator_version",
        "dataset",
        "model",
        "split_sha256",
        "case_count",
        "case_catalog",
        "baseline",
        "tried_candidates",
    }
    if set(value) != expected:
        raise ProposerFeedbackError(
            "search history has unexpected or missing fields"
        )
    history_key = _history_key(value["history_key"])
    parser_version = _fixed_text(
        value["parser_version"],
        PARSER_VERSION,
        "parser_version",
    )
    evaluator_version = _fixed_text(
        value["evaluator_version"],
        EVALUATOR_VERSION,
        "evaluator_version",
    )
    dataset = _fixed_text(value["dataset"], DATASET_NAME, "dataset")
    model = _safe_text(value["model"], "model", max_chars=128)
    split_sha256 = _sha256(value["split_sha256"], "split_sha256")
    case_count = _non_negative_integer(value["case_count"], "case_count")
    if case_count < 1:
        raise ProposerFeedbackError("case_count must be positive")

    raw_catalog = value["case_catalog"]
    if (
        isinstance(raw_catalog, (str, bytes))
        or not isinstance(raw_catalog, Sequence)
        or len(raw_catalog) != case_count
    ):
        raise ProposerFeedbackError(
            "case_catalog must contain exactly case_count entries"
        )
    case_catalog: list[dict[str, str]] = []
    for raw_case in raw_catalog:
        if not isinstance(raw_case, Mapping) or set(raw_case) != {
            "case_key",
            "reasoning_method",
        }:
            raise ProposerFeedbackError("case_catalog entry is invalid")
        case_catalog.append(
            {
                "case_key": _case_key(raw_case["case_key"]),
                "reasoning_method": _method(raw_case["reasoning_method"]),
            }
        )
    case_keys = [entry["case_key"] for entry in case_catalog]
    if len(case_keys) != len(set(case_keys)):
        raise ProposerFeedbackError("case_catalog keys must be unique")
    allowed_case_keys = set(case_keys)

    baseline = _normalize_baseline(value["baseline"], allowed_case_keys)
    raw_candidates = value["tried_candidates"]
    if (
        isinstance(raw_candidates, (str, bytes))
        or not isinstance(raw_candidates, Sequence)
    ):
        raise ProposerFeedbackError("tried_candidates must be an array")
    tried_candidates = [
        _normalize_tried_candidate(entry, allowed_case_keys, case_count)
        for entry in raw_candidates
    ]
    candidate_ids = [
        entry["strategy"]["candidate_id"] for entry in tried_candidates
    ]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ProposerFeedbackError(
            "tried candidate IDs must be unique within a history"
        )

    return {
        "history_key": history_key,
        "parser_version": parser_version,
        "evaluator_version": evaluator_version,
        "dataset": dataset,
        "model": model,
        "split_sha256": split_sha256,
        "case_count": case_count,
        "case_catalog": case_catalog,
        "baseline": baseline,
        "tried_candidates": tried_candidates,
    }


def _normalize_baseline(
    value: Any,
    allowed_case_keys: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "metrics",
        "confusion_matrix",
        "failure_counts",
        "incorrect_case_keys",
        "unresolved_case_keys",
    }:
        raise ProposerFeedbackError("baseline feedback is invalid")
    incorrect = _case_key_list(
        value["incorrect_case_keys"],
        allowed_case_keys,
        "incorrect_case_keys",
    )
    unresolved = _case_key_list(
        value["unresolved_case_keys"],
        allowed_case_keys,
        "unresolved_case_keys",
    )
    if set(incorrect).intersection(unresolved):
        raise ProposerFeedbackError(
            "baseline incorrect and unresolved cases must be disjoint"
        )
    return {
        "metrics": _normalize_metrics(value["metrics"]),
        "confusion_matrix": _normalize_confusion(
            value["confusion_matrix"],
            allow_negative=False,
        ),
        "failure_counts": _normalize_failure_counts(
            value["failure_counts"]
        ),
        "incorrect_case_keys": incorrect,
        "unresolved_case_keys": unresolved,
    }


def _normalize_tried_candidate(
    value: Any,
    allowed_case_keys: set[str],
    case_count: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "strategy",
        "metrics",
        "confusion_matrix",
        "failure_counts",
        "delta_vs_baseline",
        "paired_outcome_counts",
        "paired_error_case_keys",
    }:
        raise ProposerFeedbackError("tried candidate feedback is invalid")
    strategy = _normalize_strategy(value["strategy"])
    counts = _normalize_paired_counts(
        value["paired_outcome_counts"],
        case_count,
    )
    paired = _normalize_paired_keys(
        value["paired_error_case_keys"],
        allowed_case_keys,
    )
    for bucket, keys in paired.items():
        if len(keys) != counts[bucket]:
            raise ProposerFeedbackError(
                "paired case keys do not match paired outcome counts"
            )
    if sum(counts.values()) != case_count:
        raise ProposerFeedbackError(
            "paired outcome counts must sum to case_count"
        )
    delta = value["delta_vs_baseline"]
    if not isinstance(delta, Mapping) or set(delta) != {
        "metrics",
        "confusion_matrix",
    }:
        raise ProposerFeedbackError("delta_vs_baseline is invalid")
    return {
        "strategy": strategy,
        "metrics": _normalize_metrics(value["metrics"]),
        "confusion_matrix": _normalize_confusion(
            value["confusion_matrix"],
            allow_negative=False,
        ),
        "failure_counts": _normalize_failure_counts(
            value["failure_counts"]
        ),
        "delta_vs_baseline": {
            "metrics": _normalize_metrics(
                delta["metrics"],
                allow_negative=True,
            ),
            "confusion_matrix": _normalize_confusion(
                delta["confusion_matrix"],
                allow_negative=True,
            ),
        },
        "paired_outcome_counts": counts,
        "paired_error_case_keys": paired,
    }


def _normalize_strategy(value: Any) -> dict[str, str]:
    expected = {
        "candidate_id",
        "parent_id",
        "search_axis",
        "hypothesis",
        "expected_tradeoff",
        "source_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProposerFeedbackError("tried strategy is invalid")
    axis = value["search_axis"]
    if axis not in {"exploitation", "exploration"}:
        raise ProposerFeedbackError("tried strategy search_axis is invalid")
    return {
        "candidate_id": _identifier(
            value["candidate_id"],
            "strategy candidate_id",
        ),
        "parent_id": _identifier(value["parent_id"], "strategy parent_id"),
        "search_axis": axis,
        "hypothesis": _safe_text(
            value["hypothesis"],
            "strategy hypothesis",
            max_chars=2_000,
        ),
        "expected_tradeoff": _safe_text(
            value["expected_tradeoff"],
            "strategy expected_tradeoff",
            max_chars=2_000,
        ),
        "source_sha256": _sha256(
            value["source_sha256"],
            "strategy source_sha256",
        ),
    }


def _normalize_metrics(
    value: Any,
    *,
    allow_negative: bool = False,
) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_RATE_FIELDS):
        raise ProposerFeedbackError("feedback metrics are invalid")
    normalized = {
        field: _finite_number(value[field], f"metrics.{field}")
        for field in _RATE_FIELDS
    }
    lower = -1.0 if allow_negative else 0.0
    if any(not lower <= number <= 1.0 for number in normalized.values()):
        raise ProposerFeedbackError("feedback metric is outside its valid range")
    return normalized


def _normalize_confusion(
    value: Any,
    *,
    allow_negative: bool,
) -> dict[str, dict[str, int]]:
    if not isinstance(value, Mapping) or set(value) != set(_CONFUSION_ROWS):
        raise ProposerFeedbackError("confusion matrix is invalid")
    normalized: dict[str, dict[str, int]] = {}
    for row in _CONFUSION_ROWS:
        raw_row = value[row]
        if (
            not isinstance(raw_row, Mapping)
            or set(raw_row) != set(_CONFUSION_COLUMNS)
        ):
            raise ProposerFeedbackError("confusion matrix row is invalid")
        normalized[row] = {}
        for column in _CONFUSION_COLUMNS:
            number = raw_row[column]
            if isinstance(number, bool) or not isinstance(number, int):
                raise ProposerFeedbackError(
                    "confusion matrix cells must be integers"
                )
            if not allow_negative and number < 0:
                raise ProposerFeedbackError(
                    "confusion matrix counts must be non-negative"
                )
            normalized[row][column] = number
    return normalized


def _normalize_failure_counts(value: Any) -> dict[str, int]:
    expected = {"invalid_input", "api_failure", "parse_failure"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProposerFeedbackError("failure_counts are invalid")
    return {
        field: _non_negative_integer(value[field], f"failure_counts.{field}")
        for field in sorted(expected)
    }


def _normalize_paired_counts(value: Any, case_count: int) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_PAIRED_BUCKETS):
        raise ProposerFeedbackError("paired_outcome_counts are invalid")
    return {
        bucket: _bounded_count(value[bucket], case_count, bucket)
        for bucket in _PAIRED_BUCKETS
    }


def _normalize_paired_keys(
    value: Any,
    allowed_case_keys: set[str],
) -> dict[str, list[str]]:
    expected = set(_PAIRED_BUCKETS) - {"both_correct"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ProposerFeedbackError("paired_error_case_keys are invalid")
    normalized = {
        bucket: _case_key_list(value[bucket], allowed_case_keys, bucket)
        for bucket in _PAIRED_BUCKETS
        if bucket != "both_correct"
    }
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)):
        raise ProposerFeedbackError(
            "paired error case buckets must be disjoint"
        )
    return normalized


def _load_evaluation(
    root: Path,
    candidate_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = root / candidate_id
    report = _load_json(directory / _METRICS_NAME, "validation metrics")
    if report.get("candidate_id") != candidate_id:
        raise ProposerFeedbackError(
            "metrics candidate_id does not match its directory"
        )
    records = _load_jsonl(directory / _RESULTS_NAME)
    return report, records


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProposerFeedbackError(
            f"{description} must be readable valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise ProposerFeedbackError(f"{description} must contain an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProposerFeedbackError(
                        "validation result lines must contain objects"
                    )
                records.append(value)
    except ProposerFeedbackError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProposerFeedbackError(
            "validation results must be readable valid JSONL"
        ) from exc
    if not records:
        raise ProposerFeedbackError("validation results must not be empty")
    return records


def _report_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    if report.get("split") != "validation":
        raise ProposerFeedbackError(
            "only validation evaluations may enter proposer history"
        )
    parser_version = _fixed_text(
        report.get("parser_version"),
        PARSER_VERSION,
        "parser_version",
    )
    evaluator_version = _fixed_text(
        report.get("evaluator_version"),
        EVALUATOR_VERSION,
        "evaluator_version",
    )
    dataset = _fixed_text(report.get("dataset"), DATASET_NAME, "dataset")
    model = _safe_text(report.get("model"), "model", max_chars=128)
    split_sha256 = _sha256(report.get("split_sha256"), "split_sha256")
    case_count = _non_negative_integer(
        report.get("sample_count"),
        "sample_count",
    )
    if case_count < 1:
        raise ProposerFeedbackError("sample_count must be positive")
    return {
        "parser_version": parser_version,
        "evaluator_version": evaluator_version,
        "dataset": dataset,
        "model": model,
        "split_sha256": split_sha256,
        "case_count": case_count,
    }


def _records_by_identity(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
) -> tuple[dict[str, Mapping[str, Any]], list[str]]:
    by_marker: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []
    for record in records:
        if record.get("candidate_id", record.get("prompt_variant")) != candidate_id:
            raise ProposerFeedbackError(
                "result candidate identity does not match its evaluation"
            )
        marker = _record_identity(record)
        if marker in by_marker:
            raise ProposerFeedbackError(
                "validation results contain a duplicate case identity"
            )
        by_marker[marker] = record
        order.append(marker)
    return by_marker, order


def _record_identity(record: Mapping[str, Any]) -> str:
    if "sample_id" not in record:
        raise ProposerFeedbackError("validation result is missing sample identity")
    try:
        return canonical_json(record["sample_id"])
    except (TypeError, ValueError) as exc:
        raise ProposerFeedbackError(
            "validation result sample identity is invalid"
        ) from exc


def _record_outcome(record: Mapping[str, Any]) -> str:
    if (
        record.get("request_status") != SUCCESS
        or record.get("parse_status") != "parsed"
    ):
        return "unresolved"
    prediction = _binary_label(record.get("prediction"), "prediction")
    actual = _binary_label(record.get("gold_label"), "stored evaluation label")
    return "correct" if prediction == actual else "incorrect"


def _binary_label(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ProposerFeedbackError(f"{field} must be binary text")
    normalized = value.strip().lower()
    if normalized in {"yes", "true"}:
        return "yes"
    if normalized in {"no", "false"}:
        return "no"
    raise ProposerFeedbackError(f"{field} must be binary text")


def _reasoning_method(record: Mapping[str, Any]) -> str:
    return _method(record.get("reasoning_method", record.get("method")))


def _method(value: Any) -> str:
    if value not in TEMPLATE_KEYS:
        raise ProposerFeedbackError("reasoning_method is invalid")
    return value


def _paired_bucket(baseline: str, candidate: str) -> str:
    if baseline not in _OUTCOMES or candidate not in _OUTCOMES:
        raise ProposerFeedbackError("paired outcome is invalid")
    mapping = {
        ("correct", "correct"): "both_correct",
        ("incorrect", "correct"): "corrected",
        ("correct", "incorrect"): "regressed",
        ("incorrect", "incorrect"): "still_incorrect",
        ("unresolved", "correct"): "resolved_to_correct",
        ("unresolved", "incorrect"): "resolved_to_incorrect",
        ("correct", "unresolved"): "became_unresolved_from_correct",
        ("incorrect", "unresolved"): "became_unresolved_from_incorrect",
        ("unresolved", "unresolved"): "still_unresolved",
    }
    return mapping[(baseline, candidate)]


def _metric_summary(report: Mapping[str, Any]) -> dict[str, float]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ProposerFeedbackError("metrics report is missing metrics")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("F1")
    if not all(isinstance(value, Mapping) for value in (precision, recall, f1)):
        raise ProposerFeedbackError("metrics report is missing per-class metrics")
    value = {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("Macro-F1"),
        "parse_coverage": metrics.get(
            "parse_coverage",
            metrics.get("coverage"),
        ),
        "request_coverage": metrics.get("request_coverage"),
        "yes_precision": precision.get("yes"),
        "yes_recall": recall.get("yes"),
        "yes_f1": f1.get("yes"),
        "no_precision": precision.get("no"),
        "no_recall": recall.get("no"),
        "no_f1": f1.get("no"),
    }
    return _normalize_metrics(value)


def _confusion_matrix(report: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ProposerFeedbackError("metrics report is missing metrics")
    return _normalize_confusion(
        metrics.get("confusion_matrix"),
        allow_negative=False,
    )


def _failure_counts(report: Mapping[str, Any]) -> dict[str, int]:
    return _normalize_failure_counts(report.get("failure_counts"))


def _load_candidate(
    candidate_store: CandidateStore,
    candidate_id: str,
) -> Candidate:
    try:
        return candidate_store.load(candidate_id)
    except CandidateStoreError as exc:
        raise ProposerFeedbackError(
            "history candidate is unavailable or invalid"
        ) from exc


def _verify_candidate_report(
    candidate: Candidate,
    report: Mapping[str, Any],
) -> None:
    if (
        report.get("candidate_sha256") != candidate.sha256()
        or report.get("prompt_sha256") != candidate.source_sha256
    ):
        raise ProposerFeedbackError(
            "history metrics do not match the immutable candidate"
        )


def _strategy(candidate: Candidate) -> dict[str, str]:
    return {
        "candidate_id": candidate.candidate_id,
        "parent_id": candidate.parent_id,
        "search_axis": candidate.search_axis,
        "hypothesis": candidate.hypothesis,
        "expected_tradeoff": candidate.expected_tradeoff,
        "source_sha256": candidate.source_sha256,
    }


def _case_key_list(
    value: Any,
    allowed: set[str],
    field: str,
) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ProposerFeedbackError(f"{field} must be an array")
    normalized = [_case_key(item) for item in value]
    if len(normalized) != len(set(normalized)) or not set(normalized) <= allowed:
        raise ProposerFeedbackError(
            f"{field} contains duplicate or unknown case keys"
        )
    return normalized


def _case_key(value: Any) -> str:
    if not isinstance(value, str) or not _CASE_KEY.fullmatch(value):
        raise ProposerFeedbackError("case_key is invalid")
    return value


def _history_key(value: Any) -> str:
    if not isinstance(value, str) or not _HISTORY_KEY.fullmatch(value):
        raise ProposerFeedbackError("history_key is invalid")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ProposerFeedbackError(f"{field} is invalid")
    return value


def _safe_text(value: Any, field: str, *, max_chars: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > max_chars
        or any(pattern.search(value) for pattern in _SENSITIVE_TEXT)
    ):
        raise ProposerFeedbackError(f"{field} contains invalid or unsafe text")
    return value


def _fixed_text(value: Any, expected: str, field: str) -> str:
    if value != expected:
        raise ProposerFeedbackError(f"{field} must remain fixed to {expected}")
    return expected


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ProposerFeedbackError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProposerFeedbackError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ProposerFeedbackError(f"{field} must be finite")
    return number


def _non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProposerFeedbackError(f"{field} must be a non-negative integer")
    return value


def _bounded_count(value: Any, maximum: int, field: str) -> int:
    number = _non_negative_integer(value, field)
    if number > maximum:
        raise ProposerFeedbackError(f"{field} exceeds case_count")
    return number


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "ProposerFeedbackError",
    "SEARCH_FEEDBACK_SCHEMA_VERSION",
    "build_search_history",
    "current_search_feedback",
    "empty_search_feedback",
    "load_reparsed_search_feedback",
    "merge_search_feedback",
    "normalize_search_feedback",
    "search_feedback_candidate_count",
    "search_feedback_identities",
    "search_feedback_sha256",
]
