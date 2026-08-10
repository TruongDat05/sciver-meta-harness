"""Safe, targeted retries for unresolved Meta-Harness API failures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from meta_harness.config import MetaHarnessConfig
from meta_harness.evaluator import DATASET_NAME, build_evaluation_report
from meta_harness.orchestrator import (
    EVALUATION_PROCEDURE,
    OrchestratorError,
    load_run_candidate,
    load_run_state,
    recompute_selection_state,
    run_lock,
    score_evaluation_report,
)
from meta_harness.prompt_family import PromptFamily, TEMPLATE_KEYS
from meta_harness.split_manager import load_split_manifest, verify_split_manifest
from model_inference.remote_api import prepare_remote_requests
from utils.answer_parser import parse_answer
from utils.dataset_adapters import AdaptedSample, get_dataset_adapter
from utils.result_writer import (
    API_FAILURE,
    INVALID_INPUT,
    PARSE_FAILURE,
    SUCCESS,
    ResultWriter,
    iter_result_records,
)


_RESULTS_NAME = "validation.results.jsonl"
_METRICS_NAME = "validation.metrics.json"
_SPLIT_SUFFIX = "_split.json"
_VALIDATION_SUFFIX = "_validation.json"
_FINALIZATION_ARTIFACTS = (
    "frozen_winner.json",
    "results_manifest.json",
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RetryError(ValueError):
    """Raised before a targeted retry can mutate run artifacts."""


def retry_api_failures(
    *,
    repository_root: str | Path,
    run_id: str,
    candidate_id: str,
    max_attempts: int,
    live_api: bool,
    solver: Any | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    validation_samples: Sequence[AdaptedSample | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Retry only latest ``api_failure`` outcomes up to a cumulative ceiling.

    ``max_attempts`` includes the original attempt. An injected solver is the
    offline test boundary; the explicit live opt-in remains mandatory.
    """

    try:
        return _retry_api_failures(
            repository_root=repository_root,
            run_id=run_id,
            candidate_id=candidate_id,
            max_attempts=max_attempts,
            live_api=live_api,
            solver=solver,
            split_manifest=split_manifest,
            validation_samples=validation_samples,
        )
    except RetryError:
        raise
    except (OSError, OrchestratorError, ValueError) as exc:
        raise RetryError(str(exc)) from exc


def _retry_api_failures(
    *,
    repository_root: str | Path,
    run_id: str,
    candidate_id: str,
    max_attempts: int,
    live_api: bool,
    solver: Any | None,
    split_manifest: Mapping[str, Any] | None,
    validation_samples: Sequence[AdaptedSample | Mapping[str, Any]] | None,
) -> dict[str, Any]:
    root = Path(repository_root)
    _validate_identifier(run_id, "run_id")
    _validate_identifier(candidate_id, "candidate_id")
    if (
        isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or max_attempts < 1
    ):
        raise RetryError("max_attempts must be a positive integer")
    if live_api is not True:
        raise RetryError("targeted retries require explicit --live-api opt-in")

    run_directory = (
        root / "workspace" / "meta_harness" / "runs" / run_id
    )
    with run_lock(run_directory / ".run.lock"):
        _refuse_finalized_run(run_directory)
        context = _load_and_verify_context(
            root=root,
            run_id=run_id,
            candidate_id=candidate_id,
            split_manifest=split_manifest,
            validation_samples=validation_samples,
        )
        records = context["records"]
        latest = _latest_attempts(records)
        initial_statuses = [
            latest[_identity(sample_id)]["request_status"]
            for sample_id in context["sample_ids"]
        ]
        retryable = [
            sample_id
            for sample_id in context["sample_ids"]
            if latest[_identity(sample_id)]["request_status"] == API_FAILURE
            and latest[_identity(sample_id)]["attempt_count"] < max_attempts
        ]

        prepared = {
            _identity(sample_id): prepare_remote_requests(
                [_model_visible_record(context["samples"][_identity(sample_id)])],
                context["prompt_family"],
            )[0]
            for sample_id in retryable
        }
        resolved_solver = solver
        if retryable and resolved_solver is None:
            resolved_solver = _create_live_solver()
        if retryable and not callable(
            getattr(resolved_solver, "create_chat_completion", None)
        ):
            raise RetryError(
                "solver must provide a create_chat_completion method"
            )

        writer = ResultWriter(context["result_path"])
        appended = 0
        retried_samples = 0
        for sample_id in retryable:
            marker = _identity(sample_id)
            source = latest[marker]
            identity = _result_identity(source)
            retried_samples += 1
            while True:
                attempt_count = writer.next_attempt_count(**identity)
                if attempt_count > max_attempts:
                    break
                outcome = _request_once(
                    writer=writer,
                    identity=identity,
                    source=source,
                    messages=prepared[marker],
                    solver=resolved_solver,
                    attempt_count=attempt_count,
                )
                appended += 1
                if outcome != API_FAILURE:
                    break

        updated_records = list(iter_result_records(context["result_path"]))
        _verify_result_identities(
            updated_records,
            run_id=run_id,
            candidate_id=candidate_id,
            candidate=context["candidate"],
            config=context["config"],
            split_manifest=context["split_manifest"],
            sample_ids=context["sample_ids"],
            samples=context["samples"],
            report=context["source_report"],
        )
        report = build_evaluation_report(
            run_id=run_id,
            candidate=context["candidate"],
            split_manifest=context["split_manifest"],
            split_name="validation",
            selected_ids=context["sample_ids"],
            model=context["config"].model,
            prompt_variant=candidate_id,
            config_sha256=context["config"].sha256(),
            records=updated_records,
        )
        _atomic_write_json_if_changed(context["metrics_path"], report)

        state = context["state"]
        score = score_evaluation_report(
            report,
            result_path=context["result_path"],
            candidate_id=candidate_id,
        )
        state_changed = _update_run_state(
            state,
            candidate_id=candidate_id,
            score=score,
            full_coverage=(
                report["sample_count"] == len(context["sample_ids"])
                and score["request_coverage"] == 1.0
                and score["parse_coverage"] == 1.0
                and score["unresolved_api_failures"] == 0
            ),
        )
        if state_changed:
            _atomic_write_json(run_directory / "run_state.json", state)

        effective = _latest_attempts(updated_records)
        return {
            "run_id": run_id,
            "candidate_id": candidate_id,
            "max_attempts": max_attempts,
            "retried_sample_count": retried_samples,
            "attempts_appended": appended,
            "skipped_successful_samples": initial_statuses.count(SUCCESS),
            "skipped_invalid_input_samples": initial_statuses.count(INVALID_INPUT),
            "skipped_parse_failure_samples": initial_statuses.count(PARSE_FAILURE),
            "effective_result_count": len(effective),
            "unresolved_api_failures": report["unresolved_api_failures"],
            "eligible": score["eligible"],
            "best_candidate_id": state.get("best_candidate_id"),
        }


def _load_and_verify_context(
    *,
    root: Path,
    run_id: str,
    candidate_id: str,
    split_manifest: Mapping[str, Any] | None,
    validation_samples: Sequence[AdaptedSample | Mapping[str, Any]] | None,
) -> dict[str, Any]:
    state = load_run_state(root, run_id)
    if state.get("run_id") != run_id:
        raise RetryError("run_id does not match the immutable run state")
    configuration = state.get("configuration")
    if not isinstance(configuration, Mapping):
        raise RetryError("run configuration snapshot is missing")
    if configuration.get("evaluation_procedure") != EVALUATION_PROCEDURE:
        raise RetryError("run evaluation procedure is not the frozen procedure")
    raw_config = configuration.get("config")
    if not isinstance(raw_config, Mapping):
        raise RetryError("frozen solver configuration is missing")
    config = MetaHarnessConfig.from_mapping(raw_config)
    if config.sha256() != configuration.get("config_sha256"):
        raise RetryError("frozen solver configuration hash mismatch")

    manifest = (
        dict(split_manifest)
        if split_manifest is not None
        else _load_matching_split_manifest(root, configuration)
    )
    verify_split_manifest(manifest)
    for field in ("config_sha256", "dataset_sha256", "split_sha256"):
        if manifest.get(field) != configuration.get(field):
            raise RetryError(f"split manifest {field} does not match the run")
    sample_ids = tuple(manifest["splits"]["validation"]["sample_ids"])
    if len(sample_ids) != configuration.get("validation_sample_count"):
        raise RetryError("validation sample count does not match the run")
    if _sha256_json(sample_ids) != configuration.get(
        "validation_sample_ids_sha256"
    ):
        raise RetryError("validation sample identity hash does not match the run")

    samples = _load_validation_samples(
        root,
        validation_samples,
        sample_ids,
    )
    candidate_states = state.get("candidates")
    if not isinstance(candidate_states, Mapping):
        raise RetryError("run state candidates are missing")
    candidate_state = candidate_states.get(candidate_id)
    if not isinstance(candidate_state, Mapping):
        raise RetryError("candidate is not registered in the run state")
    candidate = load_run_candidate(root, run_id, candidate_id)
    if candidate.sha256() != candidate_state.get("candidate_sha256"):
        raise RetryError("candidate hash does not match the immutable run state")
    if candidate.source_sha256 != candidate_state.get("prompt_sha256"):
        raise RetryError("prompt hash does not match the immutable run state")

    evaluation_directory = (
        root
        / "workspace"
        / "meta_harness"
        / "runs"
        / run_id
        / "evaluations"
        / candidate_id
    )
    result_path = evaluation_directory / _RESULTS_NAME
    metrics_path = evaluation_directory / _METRICS_NAME
    run_root = evaluation_directory.parents[1]
    expected_result_path = str(result_path.relative_to(run_root))
    expected_metrics_path = str(metrics_path.relative_to(run_root))
    if candidate_state.get("result_path") != expected_result_path:
        raise RetryError("candidate result path does not match the run state")
    if candidate_state.get("metrics_path") != expected_metrics_path:
        raise RetryError("candidate metrics path does not match the run state")

    report = _load_json(metrics_path, "validation metrics")
    records = list(iter_result_records(result_path))
    _verify_report_identity(
        report,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate=candidate,
        config=config,
        split_manifest=manifest,
        sample_ids=sample_ids,
    )
    _verify_result_identities(
        records,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate=candidate,
        config=config,
        split_manifest=manifest,
        sample_ids=sample_ids,
        samples=samples,
        report=report,
    )
    build_evaluation_report(
        run_id=run_id,
        candidate=candidate,
        split_manifest=manifest,
        split_name="validation",
        selected_ids=sample_ids,
        model=config.model,
        prompt_variant=candidate_id,
        config_sha256=config.sha256(),
        records=records,
    )
    return {
        "state": state,
        "config": config,
        "split_manifest": manifest,
        "sample_ids": sample_ids,
        "samples": samples,
        "candidate": candidate,
        "prompt_family": PromptFamily(candidate.templates),
        "result_path": result_path,
        "metrics_path": metrics_path,
        "source_report": report,
        "records": records,
    }


def _load_validation_samples(
    root: Path,
    supplied: Sequence[AdaptedSample | Mapping[str, Any]] | None,
    sample_ids: Sequence[Any],
) -> dict[str, Mapping[str, Any]]:
    values: Sequence[AdaptedSample | Mapping[str, Any]]
    if supplied is None:
        values = _load_matching_validation_data(root, sample_ids)
    else:
        if isinstance(supplied, (str, bytes)) or not isinstance(
            supplied, Sequence
        ):
            raise RetryError("validation_samples must be a sequence")
        values = supplied

    by_id: dict[str, Mapping[str, Any]] = {}
    for index, value in enumerate(values):
        if isinstance(value, AdaptedSample):
            sample_id = value.sample_id
            record = value.record
        elif isinstance(value, Mapping) and isinstance(
            value.get("record"), Mapping
        ):
            sample_id = value.get("sample_id")
            record = value["record"]
        elif isinstance(value, Mapping):
            sample_id = value.get("sample_id")
            record = value
        else:
            raise RetryError(
                f"validation sample at index {index} is not adapted sample data"
            )
        marker = _identity(sample_id)
        if marker in by_id:
            raise RetryError("validation samples contain duplicate identities")
        by_id[marker] = record

    expected = {_identity(sample_id) for sample_id in sample_ids}
    if set(by_id) != expected:
        raise RetryError(
            "validation data must contain exactly the frozen sample identities"
        )
    return by_id


def _load_matching_split_manifest(
    root: Path,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    data_directory = root / "workspace" / "meta_harness" / "data"
    matches = []
    for path in sorted(data_directory.glob(f"*{_SPLIT_SUFFIX}")):
        try:
            manifest = load_split_manifest(path)
        except ValueError:
            continue
        if all(
            manifest.get(field) == configuration.get(field)
            for field in ("config_sha256", "dataset_sha256", "split_sha256")
        ):
            matches.append(manifest)
    if not matches:
        raise RetryError(
            "no prepared split manifest matches the frozen run identity"
        )
    first = matches[0]
    if any(match != first for match in matches[1:]):
        raise RetryError(
            "multiple different split manifests match the frozen run identity"
        )
    return first


def _load_matching_validation_data(
    root: Path,
    sample_ids: Sequence[Any],
) -> Sequence[AdaptedSample]:
    data_directory = root / "workspace" / "meta_harness" / "data"
    expected = {_identity(sample_id) for sample_id in sample_ids}
    matches: list[Sequence[AdaptedSample]] = []
    adapter = get_dataset_adapter("SciVer")
    for path in sorted(data_directory.glob(f"*{_VALIDATION_SUFFIX}")):
        try:
            values = adapter.load(path)
            markers = [_identity(value.sample_id) for value in values]
        except (OSError, ValueError):
            continue
        if len(markers) == len(expected) and set(markers) == expected:
            matches.append(values)
    if not matches:
        raise RetryError(
            "no validation-only data matches the frozen sample identities"
        )
    signatures = {
        _sha256_json(
            [
                {
                    "sample_id": value.sample_id,
                    "record": dict(value.record),
                }
                for value in values
            ]
        )
        for values in matches
    }
    if len(signatures) != 1:
        raise RetryError(
            "multiple different validation inputs match the frozen identities"
        )
    return matches[0]


def _verify_report_identity(
    report: Mapping[str, Any],
    *,
    run_id: str,
    candidate_id: str,
    candidate: Any,
    config: MetaHarnessConfig,
    split_manifest: Mapping[str, Any],
    sample_ids: Sequence[Any],
) -> None:
    expected = {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "prompt_variant": candidate_id,
        "candidate_sha256": candidate.sha256(),
        "prompt_sha256": candidate.source_sha256,
        "dataset": DATASET_NAME,
        "model": config.model,
        "config_sha256": config.sha256(),
        "split": "validation",
        "split_sha256": split_manifest["split_sha256"],
        "sample_count": len(sample_ids),
        "sample_ids": list(sample_ids),
    }
    changed = [
        field for field, value in expected.items() if report.get(field) != value
    ]
    if changed:
        raise RetryError(
            "validation metrics changed immutable identity fields: "
            + ", ".join(changed)
        )


def _verify_result_identities(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    candidate_id: str,
    candidate: Any,
    config: MetaHarnessConfig,
    split_manifest: Mapping[str, Any],
    sample_ids: Sequence[Any],
    samples: Mapping[str, Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    if not records:
        raise RetryError("candidate validation results are missing")
    expected_markers = {_identity(sample_id) for sample_id in sample_ids}
    for index, record in enumerate(records, start=1):
        marker = _identity(record.get("sample_id"))
        method = record.get("method")
        expected_method = _reasoning_method(samples.get(marker, {}))
        expected = {
            "run_id": run_id,
            "dataset": DATASET_NAME,
            "model": config.model,
            "prompt_variant": candidate_id,
            "candidate_id": candidate_id,
            "split_sha256": split_manifest["split_sha256"],
            "method": expected_method,
            "reasoning_method": expected_method,
        }
        if marker not in expected_markers or any(
            record.get(field) != value for field, value in expected.items()
        ):
            raise RetryError(
                f"validation result {index} does not match the frozen identity"
            )
        if method not in TEMPLATE_KEYS:
            raise RetryError(
                f"validation result {index} has an invalid reasoning method"
            )
    latest = _latest_attempts(records)
    if set(latest) != expected_markers:
        raise RetryError(
            "validation results must contain one effective record for every "
            "frozen sample identity"
        )
    _verify_report_identity(
        report,
        run_id=run_id,
        candidate_id=candidate_id,
        candidate=candidate,
        config=config,
        split_manifest=split_manifest,
        sample_ids=sample_ids,
    )


def _latest_attempts(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    seen: set[tuple[str, int]] = set()
    successful_attempts: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        attempt = record.get("attempt_count")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise RetryError(
                f"validation result {index} has invalid attempt_count"
            )
        marker = _identity(record.get("sample_id"))
        key = (marker, attempt)
        if key in seen:
            raise RetryError(
                "duplicate attempt_count for one validation sample identity"
            )
        seen.add(key)
        if record.get("request_status") == SUCCESS:
            previous_success = successful_attempts.get(marker)
            if previous_success is not None:
                raise RetryError(
                    "validation results contain duplicate successful attempts"
                )
            successful_attempts[marker] = attempt
        current = latest.get(marker)
        if current is None or attempt > current["attempt_count"]:
            latest[marker] = record
    if any(
        latest[marker]["attempt_count"] != attempt
        for marker, attempt in successful_attempts.items()
    ):
        raise RetryError("validation results contain an attempt after success")
    return latest


def _request_once(
    *,
    writer: ResultWriter,
    identity: Mapping[str, Any],
    source: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    solver: Any,
    attempt_count: int,
) -> str:
    started = time.perf_counter()
    try:
        raw_response = solver.create_chat_completion(identity["model"], messages)
    except Exception as exc:
        latency = _elapsed(started, time.perf_counter())
        details = _response_error_details(exc)
        written = writer.write_result(
            **identity,
            candidate_id=source["candidate_id"],
            split_sha256=source["split_sha256"],
            reasoning_method=source["reasoning_method"],
            gold_label=source.get("gold_label"),
            prediction=None,
            parse_status="not_attempted",
            raw_response=None,
            request_status=API_FAILURE,
            latency=latency,
            usage=None,
            error=exc,
            attempt_count=attempt_count,
            **details,
        )
        if not written:
            raise RetryError("result writer refused a targeted API attempt")
        return API_FAILURE

    latency = _elapsed(started, time.perf_counter())
    usage = _available_usage(solver)
    parsed = parse_answer(raw_response)
    if parsed["parse_status"] == "parsed":
        status = SUCCESS
        error_type = None
        error_message = None
        parse_status = "parsed"
    else:
        status = PARSE_FAILURE
        error_type = "AnswerParseError"
        error_message = parsed["parse_reason"]
        parse_status = "invalid"
    written = writer.write_result(
        **identity,
        candidate_id=source["candidate_id"],
        split_sha256=source["split_sha256"],
        reasoning_method=source["reasoning_method"],
        gold_label=source.get("gold_label"),
        prediction=parsed["prediction"],
        parse_status=parse_status,
        raw_response=raw_response,
        request_status=status,
        latency=latency,
        usage=usage,
        error_type=error_type,
        error_message=error_message,
        attempt_count=attempt_count,
    )
    if not written:
        raise RetryError("result writer refused a targeted API attempt")
    return status


def _result_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: record[field]
        for field in (
            "run_id",
            "sample_id",
            "dataset",
            "model",
            "method",
            "prompt_variant",
        )
    }


def _reasoning_method(record: Mapping[str, Any]) -> str:
    value = record.get("claim_type")
    if isinstance(value, str) and value.casefold() in TEMPLATE_KEYS:
        return value.casefold()
    raise RetryError("validation sample has an invalid reasoning method")


def _model_visible_record(record: Mapping[str, Any]) -> dict[str, Any]:
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


def _available_usage(solver: Any) -> Mapping[str, Any] | None:
    try:
        value = getattr(solver, "last_usage", None)
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        decoded = json.loads(encoded)
    except (AttributeError, TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _response_error_details(error: BaseException) -> dict[str, Any]:
    details = {}
    for field in (
        "http_status_code",
        "response_error_code",
        "response_error_message",
    ):
        value = getattr(error, field, None)
        if value is not None:
            details[field] = value
    return details


def _update_run_state(
    state: dict[str, Any],
    *,
    candidate_id: str,
    score: Mapping[str, Any],
    full_coverage: bool,
) -> bool:
    updated = deepcopy(state)
    candidate = updated["candidates"][candidate_id]
    candidate["score"] = dict(score)
    candidate["usage"] = {
        field: score[field]
        for field in ("solver_calls", "tokens", "latency_seconds")
    }
    candidate["failure"] = None
    scores = updated.setdefault("scores", {})
    scores[candidate_id] = dict(score)

    consumed = updated.get("budgets", {}).get("consumed")
    if not isinstance(consumed, dict):
        raise RetryError("run budget accounting is invalid")
    consumed["solver_calls"] = sum(
        entry["usage"]["solver_calls"]
        for entry in updated["candidates"].values()
    )
    consumed["tokens"] = sum(
        entry["usage"]["tokens"]
        for entry in updated["candidates"].values()
    )
    if full_coverage:
        recompute_selection_state(updated)

    updated["transition"] = state.get("transition")
    updated["last_transition"] = state.get("last_transition")
    if updated == state:
        return False

    transition = state.get("transition")
    if isinstance(transition, bool) or not isinstance(transition, int):
        raise RetryError("run transition counter is invalid")
    updated["transition"] = transition + 1
    updated["last_transition"] = "api_failures_retried"
    state.clear()
    state.update(updated)
    return True


def _create_live_solver() -> Any:
    from model_inference.remote_client import (
        RemoteChatCompletionsClient,
        RetrySettings,
    )
    from model_inference.remote_config import validate_config_for_live_request
    from utils.cli_environment import load_cli_environment

    load_cli_environment()
    config = validate_config_for_live_request()
    return RemoteChatCompletionsClient(
        api_key=config.api_key,
        api_url=config.api_url,
        timeout=config.timeout_seconds,
        retry_settings=RetrySettings(max_retries=config.max_retries),
    )


def _refuse_finalized_run(run_directory: Path) -> None:
    finalization = run_directory / "finalization"
    if any((finalization / name).exists() for name in _FINALIZATION_ARTIFACTS):
        raise RetryError("frozen or final-tested runs cannot be retried")
    executions = finalization / "executions"
    if executions.exists():
        raise RetryError("frozen or final-tested runs cannot be retried")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RetryError(f"{label} must be readable valid JSON") from exc
    if not isinstance(value, dict):
        raise RetryError(f"{label} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _atomic_write_json_if_changed(
    path: Path,
    value: Mapping[str, Any],
) -> bool:
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        existing = None
    if existing == value:
        return False
    _atomic_write_json(path, value)
    return True


def _identity(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RetryError("sample identities must be JSON serializable") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_identity(value).encode("utf-8")).hexdigest()


def _elapsed(started: float, finished: float) -> float:
    if not math.isfinite(started) or not math.isfinite(finished):
        raise RetryError("retry clock returned a non-finite value")
    return max(0.0, finished - started)


def _validate_identifier(value: Any, field: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise RetryError(f"{field} is invalid")


__all__ = ["RetryError", "retry_api_failures"]
