"""Offline re-parsing of immutable Meta-Harness evaluation responses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from evaluation.metrics import evaluate_dataset_records, load_jsonl
from utils.answer_parser import PARSER_VERSION, parse_answer
from utils.result_writer import (
    API_FAILURE,
    INVALID_INPUT,
    PARSE_FAILURE,
    SUCCESS,
    sanitize_result_data,
)


REPARSE_SCHEMA_VERSION = 1
REPARSE_VERSION = "offline_raw_response_reparse_v1"
_RESULTS_NAME = "validation.results.jsonl"
_METRICS_NAME = "validation.metrics.json"


class ReparseError(ValueError):
    """Raised when source artifacts cannot be safely re-parsed offline."""


def reparse_run(
    source_run_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_sample_count: int | None = None,
    expected_evaluation_count: int | None = None,
) -> dict[str, Any]:
    """Re-parse every evaluation in a run into a new, non-overwriting directory.

    Only stored ``raw_response`` values are parsed. Records without a response
    retain their original API/input failure status. The source tree is read
    only, and the destination must not already exist or be inside that tree.
    """

    source = Path(source_run_dir).resolve()
    output = Path(output_dir).resolve()
    _validate_paths(source, output)
    _validate_expected_count(expected_sample_count, "expected_sample_count")
    _validate_expected_count(
        expected_evaluation_count,
        "expected_evaluation_count",
    )

    evaluations_dir = source / "evaluations"
    evaluations = _discover_evaluations(evaluations_dir)
    if (
        expected_evaluation_count is not None
        and len(evaluations) != expected_evaluation_count
    ):
        raise ReparseError(
            "source evaluation count does not match expected_evaluation_count: "
            f"{len(evaluations)} != {expected_evaluation_count}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    try:
        reports: list[dict[str, Any]] = []
        for evaluation_dir in evaluations:
            reports.append(
                _reparse_evaluation(
                    source,
                    evaluation_dir,
                    temporary,
                    expected_sample_count=expected_sample_count,
                )
            )

        source_run_ids = sorted(
            {
                report["source_run_id"]
                for report in reports
                if isinstance(report["source_run_id"], str)
            }
        )
        if len(source_run_ids) != 1:
            raise ReparseError(
                "all evaluations must have the same non-empty source run_id"
            )
        summary = {
            "schema_version": REPARSE_SCHEMA_VERSION,
            "reparse_version": REPARSE_VERSION,
            "parser_version": PARSER_VERSION,
            "source_run_id": source_run_ids[0],
            "evaluation_count": len(reports),
            "expected_sample_count": expected_sample_count,
            "evaluations": reports,
        }
        _write_json(temporary / "summary.json", summary)

        try:
            temporary.rename(output)
        except FileExistsError as exc:
            raise ReparseError(f"output directory already exists: {output}") from exc
        return summary
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _validate_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise ReparseError(f"source run directory does not exist: {source}")
    if output.exists():
        raise ReparseError(f"output directory already exists: {output}")
    if output == source or source in output.parents:
        raise ReparseError(
            "output directory must be outside the immutable source run directory"
        )


def _validate_expected_count(value: int | None, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReparseError(f"{field} must be a positive integer")


def _discover_evaluations(evaluations_dir: Path) -> list[Path]:
    if not evaluations_dir.is_dir():
        raise ReparseError(
            f"source evaluations directory does not exist: {evaluations_dir}"
        )
    evaluations = sorted(
        path
        for path in evaluations_dir.iterdir()
        if path.is_dir()
        and (path / _RESULTS_NAME).is_file()
        and (path / _METRICS_NAME).is_file()
    )
    if not evaluations:
        raise ReparseError("source run contains no complete evaluation artifacts")
    return evaluations


def _reparse_evaluation(
    source: Path,
    evaluation_dir: Path,
    temporary_output: Path,
    *,
    expected_sample_count: int | None,
) -> dict[str, Any]:
    source_results = evaluation_dir / _RESULTS_NAME
    source_metrics = evaluation_dir / _METRICS_NAME
    records = load_jsonl(source_results)
    metrics = _load_metrics(source_metrics)
    candidate_id = _candidate_id(metrics, evaluation_dir.name)

    if expected_sample_count is not None and len(records) != expected_sample_count:
        raise ReparseError(
            f"{candidate_id} record count does not match expected_sample_count: "
            f"{len(records)} != {expected_sample_count}"
        )
    declared_count = metrics.get("sample_count")
    if declared_count != len(records):
        raise ReparseError(
            f"{candidate_id} metrics sample_count does not match result records"
        )

    reparsed_records = [_reparse_record(record) for record in records]
    summaries = evaluate_dataset_records(reparsed_records)
    if len(summaries) != 1:
        raise ReparseError(
            f"{candidate_id} must produce exactly one dataset-level summary"
        )
    summary = summaries[0]
    if summary["total_samples"] != len(records):
        raise ReparseError(
            f"{candidate_id} contains duplicate or inconsistent sample identities"
        )

    destination = temporary_output / candidate_id
    _write_jsonl(destination / _RESULTS_NAME, reparsed_records)
    report = _updated_metrics_report(
        metrics,
        summary,
        reparsed_records,
        source=source,
        source_results=source_results,
        source_metrics=source_metrics,
    )
    _write_json(destination / _METRICS_NAME, report)
    return _summary_entry(candidate_id, metrics, report)


def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReparseError(f"could not read metrics artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ReparseError(f"metrics artifact must contain an object: {path}")
    return value


def _candidate_id(metrics: Mapping[str, Any], directory_name: str) -> str:
    candidate_id = metrics.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or not candidate_id
        or candidate_id != directory_name
        or Path(candidate_id).name != candidate_id
    ):
        raise ReparseError(
            "metrics candidate_id must match its evaluation directory name"
        )
    return candidate_id


def _reparse_record(record: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    raw_response = record.get("raw_response")
    if not isinstance(raw_response, str):
        if record.get("request_status") not in {API_FAILURE, INVALID_INPUT, PARSE_FAILURE}:
            raise ReparseError(
                "a response-less record must retain an API, input, or parse failure"
            )
        return updated

    parsed = parse_answer(raw_response)
    if parsed["parse_status"] == "parsed":
        updated.update(
            {
                "prediction": parsed["prediction"],
                "parse_status": "parsed",
                "request_status": SUCCESS,
                "error_type": None,
                "error_message": None,
            }
        )
    else:
        updated.update(
            {
                "prediction": parsed["prediction"],
                "parse_status": "invalid",
                "request_status": PARSE_FAILURE,
                "error_type": "AnswerParseError",
                "error_message": parsed["parse_reason"],
            }
        )
    return updated


def _updated_metrics_report(
    source_report: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    source: Path,
    source_results: Path,
    source_metrics: Path,
) -> dict[str, Any]:
    report = deepcopy(dict(source_report))
    coverage = summary["parse_coverage"]["value"]
    request_coverage = sum(
        record.get("request_status") in {SUCCESS, PARSE_FAILURE}
        for record in records
    ) / len(records)
    report.update(
        {
            "parser_version": PARSER_VERSION,
            "metrics": {
                "accuracy": summary["accuracy"]["value"],
                "precision": {
                    label: summary["per_class"][label]["precision"]
                    for label in ("yes", "no")
                },
                "recall": {
                    label: summary["per_class"][label]["recall"]
                    for label in ("yes", "no")
                },
                "F1": {
                    label: summary["per_class"][label]["f1"]
                    for label in ("yes", "no")
                },
                "Macro-F1": summary["macro_f1"],
                "coverage": coverage,
                "parse_coverage": coverage,
                "request_coverage": request_coverage,
                "confusion_matrix": summary["confusion_matrix"],
            },
            "failure_counts": {
                "invalid_input": summary["invalid_inputs"],
                "api_failure": summary["api_failures"],
                "parse_failure": summary["parse_failures"],
            },
            "unresolved_api_failures": summary["unresolved_api_failures"],
            "rankable": summary["eligible"],
            "reparse": {
                "schema_version": REPARSE_SCHEMA_VERSION,
                "reparse_version": REPARSE_VERSION,
                "source_results": str(source_results.relative_to(source)),
                "source_results_sha256": _sha256(source_results),
                "source_metrics": str(source_metrics.relative_to(source)),
                "source_metrics_sha256": _sha256(source_metrics),
            },
        }
    )
    return report


def _summary_entry(
    candidate_id: str,
    source_report: Mapping[str, Any],
    reparsed_report: Mapping[str, Any],
) -> dict[str, Any]:
    original = _compact_metrics(source_report)
    reparsed = _compact_metrics(reparsed_report)
    return {
        "candidate_id": candidate_id,
        "source_run_id": reparsed_report.get("run_id"),
        "sample_count": reparsed_report.get("sample_count"),
        "original": original,
        "reparsed": reparsed,
        "delta": {
            field: reparsed[field] - original[field]
            for field in ("coverage", "accuracy", "macro_f1")
        },
    }


def _compact_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics")
    failures = report.get("failure_counts")
    if not isinstance(metrics, Mapping) or not isinstance(failures, Mapping):
        raise ReparseError("metrics report is missing metrics or failure_counts")
    values = {
        "coverage": metrics.get("parse_coverage", metrics.get("coverage")),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("Macro-F1"),
        "parse_failures": failures.get("parse_failure"),
        "api_failures": failures.get("api_failure"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values.values()
    ):
        raise ReparseError("metrics report contains invalid summary values")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReparseError(f"could not hash source artifact: {path}") from exc
    return digest.hexdigest()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as output:
            for record in records:
                clean_record = sanitize_result_data(record)
                output.write(
                    json.dumps(
                        clean_record,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    except (OSError, TypeError, ValueError) as exc:
        raise ReparseError(f"could not write re-parsed results: {path}") from exc


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        path.write_text(encoded + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise ReparseError(f"could not write re-parse artifact: {path}") from exc


__all__ = [
    "PARSER_VERSION",
    "REPARSE_SCHEMA_VERSION",
    "REPARSE_VERSION",
    "ReparseError",
    "reparse_run",
]
