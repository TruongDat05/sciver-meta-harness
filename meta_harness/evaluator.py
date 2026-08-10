"""Offline-safe evaluation of immutable Meta-Harness prompt candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Protocol

from evaluation.metrics import evaluate_dataset_records
from meta_harness.candidate_store import CandidateStore
from meta_harness.config import MetaHarnessConfig, SUPPORTED_SEARCH_MODELS
from meta_harness.prompt_family import PromptFamily, TEMPLATE_KEYS
from meta_harness.schemas import Candidate
from meta_harness.split_manager import (
    SPLIT_NAMES,
    verify_split_manifest,
)
from model_inference.remote_api import prepare_remote_requests
from utils.answer_parser import PARSER_VERSION, parse_answer
from utils.dataset_adapters import AdaptedSample
from utils.result_writer import (
    API_FAILURE,
    INVALID_INPUT,
    PARSE_FAILURE,
    SUCCESS,
    ResultWriter,
    iter_result_records,
)


DATASET_NAME = "SciVer"
EVALUATOR_SCHEMA_VERSION = 1
EVALUATOR_VERSION = "meta_harness_evaluator_v1"
_MISSING = object()
_YES_LABELS = frozenset({"yes", "true"})
_NO_LABELS = frozenset({"no", "false"})


class EvaluatorError(ValueError):
    """Raised when an evaluation request violates the frozen contract."""


class Solver(Protocol):
    """Existing remote-client request boundary used by the evaluator."""

    def create_chat_completion(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
    ) -> str: ...


def evaluate_candidate(
    *,
    run_id: str,
    candidate_id: str,
    candidate_store: CandidateStore,
    split_manifest: Mapping[str, Any],
    split_name: str,
    sample_ids: Sequence[Any],
    samples: Sequence[AdaptedSample | Mapping[str, Any]],
    solver: Solver,
    output_path: str | Path,
    metrics_path: str | Path | None = None,
    config: MetaHarnessConfig | None = None,
    model: str | None = None,
    prompt_variant: str | None = None,
    model_config_sha256: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    before_solver_call: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Evaluate an explicit, ordered subset of one declared SciVer split.

    Candidate prompt text is loaded only through ``CandidateStore``. Every
    locally valid sample that is not already successful makes one call through
    the injected remote-client interface. No live client is constructed here.
    """

    _validate_run_id(run_id)
    if not isinstance(candidate_store, CandidateStore):
        raise EvaluatorError(
            "candidate_store must be an immutable CandidateStore"
        )
    candidate = candidate_store.load(candidate_id)
    prompt_family = PromptFamily(candidate.templates)

    resolved_config = config or MetaHarnessConfig()
    if not isinstance(resolved_config, MetaHarnessConfig):
        raise EvaluatorError("config must be a MetaHarnessConfig")
    if resolved_config.model not in SUPPORTED_SEARCH_MODELS:
        raise EvaluatorError(
            "solver model is not supported by the staged search protocol"
        )
    if resolved_config.generation_request_options():
        raise EvaluatorError(
            "the existing remote-client interface does not support generation "
            "request options; use the frozen omitted configuration"
        )
    resolved_model = resolved_config.model if model is None else model
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise EvaluatorError("model must be non-empty text")
    resolved_prompt_variant = (
        candidate.candidate_id if prompt_variant is None else prompt_variant
    )
    if (
        not isinstance(resolved_prompt_variant, str)
        or not resolved_prompt_variant.strip()
    ):
        raise EvaluatorError("prompt_variant must be non-empty text")
    resolved_config_sha256 = (
        resolved_config.sha256()
        if model_config_sha256 is None
        else model_config_sha256
    )
    if (
        not isinstance(resolved_config_sha256, str)
        or len(resolved_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in resolved_config_sha256
        )
    ):
        raise EvaluatorError(
            "model_config_sha256 must be a lowercase hexadecimal SHA-256"
        )

    verify_split_manifest(split_manifest)
    selected_ids = _validate_selection(
        split_manifest,
        split_name,
        sample_ids,
    )
    selected_samples = _select_samples(samples, selected_ids)

    create_completion = getattr(solver, "create_chat_completion", None)
    if not callable(create_completion):
        raise EvaluatorError(
            "solver must provide a create_chat_completion method"
        )
    if not callable(clock):
        raise EvaluatorError("clock must be callable")
    if before_solver_call is not None and not callable(before_solver_call):
        raise EvaluatorError("before_solver_call must be callable or null")

    result_path = Path(output_path)
    writer = ResultWriter(result_path)
    split_sha256 = split_manifest["split_sha256"]
    completed_sample_ids = _completed_result_sample_ids(
        result_path,
        candidate_id=candidate.candidate_id,
        split_sha256=split_sha256,
    )
    for sample_id, record in selected_samples:
        method = _reasoning_method(record)
        sample_marker = _identity(sample_id, "sample_id")
        identity = {
            "run_id": run_id,
            "sample_id": sample_id,
            "dataset": DATASET_NAME,
            "model": resolved_model,
            "method": method,
            "prompt_variant": resolved_prompt_variant,
        }
        if sample_marker in completed_sample_ids:
            continue

        gold_label = _gold_label(record)
        if gold_label is None:
            _write_outcome(
                writer,
                identity,
                candidate_id=candidate.candidate_id,
                reasoning_method=method,
                gold_label=None,
                prediction=None,
                parse_status="not_attempted",
                raw_response=None,
                request_status=INVALID_INPUT,
                latency=None,
                usage=None,
                error=EvaluatorError(
                    "sample requires a normalized binary gold_label"
                ),
                split_sha256=split_sha256,
            )
            completed_sample_ids.add(sample_marker)
            continue

        try:
            messages = prepare_remote_requests(
                [_model_visible_record(record)],
                prompt_family,
            )[0]
        except Exception as exc:
            _write_outcome(
                writer,
                identity,
                candidate_id=candidate.candidate_id,
                reasoning_method=method,
                gold_label=gold_label,
                prediction=None,
                parse_status="not_attempted",
                raw_response=None,
                request_status=INVALID_INPUT,
                latency=None,
                usage=None,
                error=exc,
                split_sha256=split_sha256,
            )
            completed_sample_ids.add(sample_marker)
            continue

        if before_solver_call is not None:
            before_solver_call()
        started = clock()
        raw_response: Any = None
        try:
            raw_response = create_completion(resolved_model, messages)
        except Exception as exc:
            latency = _elapsed(started, clock())
            _write_outcome(
                writer,
                identity,
                candidate_id=candidate.candidate_id,
                reasoning_method=method,
                gold_label=gold_label,
                prediction=None,
                parse_status="not_attempted",
                raw_response=None,
                request_status=API_FAILURE,
                latency=latency,
                usage=None,
                error=exc,
                split_sha256=split_sha256,
            )
            completed_sample_ids.add(sample_marker)
            continue

        latency = _elapsed(started, clock())
        usage = _available_usage(solver)
        parsed = parse_answer(raw_response)
        if parsed["parse_status"] == "parsed":
            _write_outcome(
                writer,
                identity,
                candidate_id=candidate.candidate_id,
                reasoning_method=method,
                gold_label=gold_label,
                prediction=parsed["prediction"],
                parse_status="parsed",
                raw_response=raw_response,
                request_status=SUCCESS,
                latency=latency,
                usage=usage,
                split_sha256=split_sha256,
            )
        else:
            _write_outcome(
                writer,
                identity,
                candidate_id=candidate.candidate_id,
                reasoning_method=method,
                gold_label=gold_label,
                prediction=parsed["prediction"],
                parse_status="invalid",
                raw_response=raw_response,
                request_status=PARSE_FAILURE,
                latency=latency,
                usage=usage,
                error_type="AnswerParseError",
                error_message=parsed["parse_reason"],
                split_sha256=split_sha256,
            )
        completed_sample_ids.add(sample_marker)

    report = build_evaluation_report(
        run_id=run_id,
        candidate=candidate,
        split_manifest=split_manifest,
        split_name=split_name,
        selected_ids=selected_ids,
        model=resolved_model,
        prompt_variant=resolved_prompt_variant,
        config_sha256=resolved_config_sha256,
        records=_selected_result_records(
            result_path,
            run_id=run_id,
            candidate_id=candidate.candidate_id,
            prompt_variant=resolved_prompt_variant,
            model=resolved_model,
            selected_ids=selected_ids,
            split_sha256=split_sha256,
        ),
    )
    destination = (
        Path(metrics_path)
        if metrics_path is not None
        else metrics_path_for_results(result_path)
    )
    _write_deterministic_json(destination, report)
    return report


def metrics_path_for_results(output_path: str | Path) -> Path:
    """Return the default deterministic metrics path for a JSONL result file."""

    path = Path(output_path)
    return path.with_name(f"{path.stem}.metrics.json")


def _validate_run_id(run_id: Any) -> None:
    if not isinstance(run_id, str) or not run_id.strip():
        raise EvaluatorError("run_id must be non-empty text")


def _validate_selection(
    manifest: Mapping[str, Any],
    split_name: str,
    sample_ids: Sequence[Any],
) -> tuple[Any, ...]:
    if split_name not in SPLIT_NAMES:
        raise EvaluatorError(
            "split_name must be one of: " + ", ".join(SPLIT_NAMES)
        )
    if isinstance(sample_ids, (str, bytes)) or not isinstance(
        sample_ids,
        Sequence,
    ):
        raise EvaluatorError("sample_ids must be an explicit sequence")
    if not sample_ids:
        raise EvaluatorError("sample_ids must not be empty")

    selected = tuple(sample_ids)
    selected_markers = [_identity(value, "sample_id") for value in selected]
    if len(selected_markers) != len(set(selected_markers)):
        raise EvaluatorError("sample_ids must not contain duplicates")
    declared_markers = {
        _identity(value, "split sample_id")
        for value in manifest["splits"][split_name]["sample_ids"]
    }
    outside = [
        value
        for value, marker in zip(selected, selected_markers)
        if marker not in declared_markers
    ]
    if outside:
        raise EvaluatorError(
            "every requested sample_id must belong to the declared split"
        )
    return selected


def _select_samples(
    samples: Sequence[AdaptedSample | Mapping[str, Any]],
    selected_ids: Sequence[Any],
) -> list[tuple[Any, Mapping[str, Any]]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise EvaluatorError("samples must be a sequence of adapted samples")

    by_id: dict[str, tuple[Any, Mapping[str, Any]]] = {}
    for index, value in enumerate(samples):
        if isinstance(value, AdaptedSample):
            sample_id = value.sample_id
            record = value.record
        elif isinstance(value, Mapping) and isinstance(
            value.get("record"),
            Mapping,
        ):
            sample_id = value.get("sample_id", _MISSING)
            record = value["record"]
        elif isinstance(value, Mapping):
            sample_id = value.get("sample_id", _MISSING)
            record = value
        else:
            raise EvaluatorError(
                f"sample at index {index} must be adapted sample data"
            )
        if sample_id is _MISSING:
            raise EvaluatorError(f"sample at index {index} requires sample_id")
        if not isinstance(record, Mapping):
            raise EvaluatorError(
                f"sample at index {index} record must be a mapping"
            )
        marker = _identity(sample_id, "sample_id")
        if (
            "sample_id" in record
            and _identity(record["sample_id"], "record sample_id") != marker
        ):
            raise EvaluatorError(
                f"sample at index {index} has inconsistent sample_id values"
            )
        if marker in by_id:
            raise EvaluatorError("samples contain duplicate sample IDs")
        by_id[marker] = (sample_id, record)

    selected: list[tuple[Any, Mapping[str, Any]]] = []
    for sample_id in selected_ids:
        marker = _identity(sample_id, "sample_id")
        try:
            selected.append(by_id[marker])
        except KeyError as exc:
            raise EvaluatorError(
                "every requested sample_id must have loaded sample data"
            ) from exc
    return selected


def _reasoning_method(record: Mapping[str, Any]) -> str:
    value = record.get("claim_type")
    if isinstance(value, str) and value.casefold() in TEMPLATE_KEYS:
        return value.casefold()
    return "invalid"


def _gold_label(record: Mapping[str, Any]) -> str | None:
    value = record.get("gold_label", _MISSING)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _YES_LABELS:
            return "yes"
        if normalized in _NO_LABELS:
            return "no"
    return None


def _model_visible_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Detach gold-only fields before production request preparation."""

    gold_only_fields = {
        "gold_label",
        "label",
        "label_2class",
        "gold_answer",
        "gold_explanation",
        "ground_truth",
        "rationale",
    }
    return {
        key: value
        for key, value in record.items()
        if str(key).casefold() not in gold_only_fields
    }


def _elapsed(started: Any, finished: Any) -> float:
    if (
        isinstance(started, bool)
        or isinstance(finished, bool)
        or not isinstance(started, (int, float))
        or not isinstance(finished, (int, float))
        or not math.isfinite(float(started))
        or not math.isfinite(float(finished))
    ):
        raise EvaluatorError("clock must return finite numbers")
    return max(0.0, float(finished) - float(started))


def _available_usage(solver: Solver) -> Mapping[str, Any] | None:
    try:
        value = getattr(solver, "last_usage", None)
    except Exception:
        return None
    if value is None:
        return None
    if not isinstance(value, Mapping):
        return None
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    return decoded


def _write_outcome(
    writer: ResultWriter,
    identity: Mapping[str, Any],
    *,
    candidate_id: str,
    reasoning_method: str,
    gold_label: str | None,
    prediction: Any,
    parse_status: str,
    raw_response: Any,
    request_status: str,
    latency: float | None,
    usage: Mapping[str, Any] | None,
    error: BaseException | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    split_sha256: str,
) -> None:
    response_details = {}
    if error is not None:
        for field in (
            "http_status_code",
            "response_error_code",
            "response_error_message",
        ):
            value = getattr(error, field, None)
            if value is not None:
                response_details[field] = value
    writer.write_result(
        **identity,
        candidate_id=candidate_id,
        reasoning_method=reasoning_method,
        gold_label=gold_label,
        prediction=prediction,
        parse_status=parse_status,
        raw_response=raw_response,
        request_status=request_status,
        latency=latency,
        usage=usage,
        error=error,
        error_type=error_type,
        error_message=error_message,
        split_sha256=split_sha256,
        **response_details,
    )


def _selected_result_records(
    path: Path,
    *,
    run_id: str,
    candidate_id: str,
    prompt_variant: str,
    model: str,
    selected_ids: Sequence[Any],
    split_sha256: str,
) -> list[dict[str, Any]]:
    selected_markers = {_identity(value, "sample_id") for value in selected_ids}
    records = [
        record
        for record in iter_result_records(path)
        if record["run_id"] == run_id
        and record["dataset"] == DATASET_NAME
        and record["model"] == model
        and record.get("prompt_variant", "cot") == prompt_variant
        and record.get("candidate_id", candidate_id) == candidate_id
        and record.get("split_sha256") == split_sha256
        and _identity(record["sample_id"], "sample_id") in selected_markers
    ]
    latest: dict[str, dict[str, Any]] = {}
    seen_attempts: set[tuple[str, int]] = set()
    for record in records:
        marker = _identity(record["sample_id"], "sample_id")
        attempt = record.get("attempt_count")
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
        ):
            raise EvaluatorError(
                "evaluation result attempt_count must be a positive integer"
            )
        key = (marker, attempt)
        if key in seen_attempts:
            raise EvaluatorError(
                "evaluation results contain a duplicate sample attempt"
            )
        seen_attempts.add(key)
        current = latest.get(marker)
        if current is None or attempt > current["attempt_count"]:
            latest[marker] = record
    if set(latest) != selected_markers:
        raise EvaluatorError(
            "evaluation results must contain an effective attempt for every "
            "(candidate_id, sample_id, split_sha256)"
        )
    return [latest[_identity(sample_id, "sample_id")] for sample_id in selected_ids]


def _completed_result_sample_ids(
    path: Path,
    *,
    candidate_id: str,
    split_sha256: str,
) -> set[str]:
    completed: set[str] = set()
    for record in iter_result_records(path):
        if (
            record.get("candidate_id", record.get("prompt_variant"))
            != candidate_id
            or record.get("split_sha256") != split_sha256
            or record.get("request_status") == API_FAILURE
        ):
            continue
        marker = _identity(record.get("sample_id"), "sample_id")
        if marker in completed:
            raise EvaluatorError(
                "duplicate (candidate_id, sample_id, split_sha256) result"
            )
        completed.add(marker)
    return completed


def build_evaluation_report(
    *,
    run_id: str,
    candidate: Candidate,
    split_manifest: Mapping[str, Any],
    split_name: str,
    selected_ids: Sequence[Any],
    model: str,
    prompt_variant: str,
    config_sha256: str,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build deterministic validation metrics from each sample's latest attempt."""

    summaries = evaluate_dataset_records(records)
    if len(summaries) != 1:
        raise EvaluatorError(
            "evaluation results must produce exactly one candidate summary"
        )
    summary = summaries[0]
    coverage = summary["parse_coverage"]["value"]
    if summary["total_samples"] != len(selected_ids):
        raise EvaluatorError(
            "evaluation results must contain one effective result per selected sample"
        )
    request_coverage = (
        summary["successful_requests"] + summary["parse_failures"]
    ) / summary["total_samples"]
    metrics = {
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
    }
    rankable = summary["eligible"]
    return {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "evaluator_version": EVALUATOR_VERSION,
        "parser_version": PARSER_VERSION,
        "run_id": run_id,
        "candidate_id": candidate.candidate_id,
        "prompt_variant": prompt_variant,
        "candidate_sha256": candidate.sha256(),
        "prompt_sha256": candidate.source_sha256,
        "dataset": DATASET_NAME,
        "model": model,
        "config_sha256": config_sha256,
        "split": split_name,
        "split_sha256": split_manifest["split_sha256"],
        "sample_ids": list(selected_ids),
        "sample_count": len(selected_ids),
        "metrics": metrics,
        "failure_counts": {
            "invalid_input": summary["invalid_inputs"],
            "api_failure": summary["api_failures"],
            "parse_failure": summary["parse_failures"],
        },
        "unresolved_api_failures": summary["unresolved_api_failures"],
        "rankable": rankable,
    }


def _identity(value: Any, field: str) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise EvaluatorError(f"{field} must be JSON serializable") from exc


def _write_deterministic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DATASET_NAME",
    "EVALUATOR_SCHEMA_VERSION",
    "EVALUATOR_VERSION",
    "EvaluatorError",
    "PARSER_VERSION",
    "Solver",
    "build_evaluation_report",
    "evaluate_candidate",
    "metrics_path_for_results",
]
