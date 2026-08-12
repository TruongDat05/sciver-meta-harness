"""Immutable input boundary for the full SEARCH V3 evaluator.

This module deliberately owns no solver client, candidate selection, or
orchestration. It admits only the one prepared SEARCH artifact required by
the locked protocol, so later evaluation code cannot receive a subset, a
legacy split, or FINAL material.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from string import Template
from types import MappingProxyType
from typing import Any
import tempfile

from meta_harness.config import (
    FULL_SEARCH_V3_PROTOCOL_ID,
    canonical_full_search_v3_config,
)
from meta_harness.full_search_v3_preparation import (
    FullSearchV3PreparationError,
    load_full_search_v3_search_dataset,
    load_full_search_v3_search_safe_manifest,
    verify_full_search_v3_search_dataset,
    verify_full_search_v3_search_safe_manifest,
)
from meta_harness.full_search_v3_retry import (
    SolverExecutionFailure,
    SolverFailureMetadata,
)
from meta_harness.full_search_v3_cache import (
    FullSearchV3SearchCache,
    SearchCacheError,
    contains_sensitive_value,
)
from meta_harness.full_search_v3_concurrency import FullSearchV3RequestExecutor
from meta_harness.full_search_v3_solver import (
    SolverResult,
    build_solver_request,
    build_solver_request_identity,
)
from meta_harness.prompt_family import PromptFamily, TEMPLATE_KEYS, template_source_sha256
from evaluation.metrics import EvaluationError, evaluate_dataset_records
from utils.answer_parser import PARSER_VERSION, parse_answer
from utils.constant import COT_PROMPT
from utils.result_writer import API_FAILURE, PARSE_FAILURE, SUCCESS


FULL_SEARCH_V3_STAGE = "SEARCH"
FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256 = (
    "8e5f28db7669026a4c419c972da4bb1caacf4ece0e2f1b8b7b8ab1dca204ec8c"
)
FULL_SEARCH_V3_METRICS_SCHEMA_VERSION = (
    "sciver_full_search_v3_prediction_metrics_v1"
)
FULL_SEARCH_V3_CHECKPOINT_SCHEMA_VERSION = (
    "sciver_full_search_v3_candidate_checkpoint_v1"
)
FULL_SEARCH_V3_P0_CANDIDATE_ID = "cot"
_METRICS_RUN_ID = "sciver_full_search_v3"
_DATASET_NAME = "SciVer"
_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_UNSAFE_CANDIDATE_ID = re.compile(
    r"(?:api[._-]?(?:key|url)|authorization|bearer|credential|secret|"
    r"token|endpoint|https?|base64|data[._-]?image)",
    re.IGNORECASE,
)
_BASE64_LIKE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9+/]{48,}={0,2}\Z")


class FullSearchV3EvaluatorInputError(ValueError):
    """Raised before evaluation when SEARCH inputs are not the locked artifact."""


class FullSearchV3EvaluationResumeError(RuntimeError):
    """Raised when a checkpoint cannot safely resume the immutable evaluation."""


class FullSearchV3EvaluationIncomplete(RuntimeError):
    """Raised after all possible work when infrastructure failures remain."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__("full SEARCH evaluation has unresolved infrastructure failures")


@dataclass(frozen=True)
class FullSearchV3SearchInput:
    """An immutable, complete, manifest-ordered SEARCH input snapshot."""

    _manifest: Mapping[str, Any]
    _records: tuple[Mapping[str, Any], ...]

    @property
    def stage(self) -> str:
        """Return the only stage admitted by this API."""

        return FULL_SEARCH_V3_STAGE

    @property
    def manifest(self) -> Mapping[str, Any]:
        """Return the frozen SEARCH-safe manifest snapshot."""

        return self._manifest

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        """Return all 1,000 frozen SEARCH records in manifest order."""

        return self._records

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return complete immutable SEARCH membership in dispatch order."""

        return tuple(record["sample_id"] for record in self._records)


@dataclass(frozen=True)
class FullSearchV3PredictionOutcome:
    """One parser-owned response or sanitized infrastructure failure.

    ``raw_response`` remains in memory only. Aggregate artifacts never persist
    it, so model text cannot accidentally become a diagnostic payload.
    """

    sample_id: str
    raw_response: object | None = None
    infrastructure_failure: SolverFailureMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise FullSearchV3EvaluatorInputError(
                "prediction outcome sample_id must be non-empty text"
            )
        if self.infrastructure_failure is not None:
            if not isinstance(self.infrastructure_failure, SolverFailureMetadata):
                raise FullSearchV3EvaluatorInputError(
                    "infrastructure_failure must use sanitized solver metadata"
                )
            if self.raw_response is not None:
                raise FullSearchV3EvaluatorInputError(
                    "infrastructure failures cannot also supply a model response"
                )

    @classmethod
    def from_solver_result(
        cls, sample_id: str, result: SolverResult
    ) -> "FullSearchV3PredictionOutcome":
        if not isinstance(result, SolverResult):
            raise FullSearchV3EvaluatorInputError("result must be a SolverResult")
        return cls(sample_id=sample_id, raw_response=result.content)

    @classmethod
    def from_infrastructure_failure(
        cls, sample_id: str, failure: SolverExecutionFailure
    ) -> "FullSearchV3PredictionOutcome":
        if not isinstance(failure, SolverExecutionFailure):
            raise FullSearchV3EvaluatorInputError(
                "failure must be a SolverExecutionFailure"
            )
        return cls(sample_id=sample_id, infrastructure_failure=failure.metadata)


def load_full_search_v3_search_input(
    *,
    search_safe_manifest_path: str | Path,
    search_records_path: str | Path,
) -> FullSearchV3SearchInput:
    """Load and validate the sole admissible full-SEARCH evaluator input.

    There are intentionally no parameters for a sample subset, limit, hard
    set, smoke set, validation set, FINAL set, or caller-chosen stage.
    """

    try:
        manifest = load_full_search_v3_search_safe_manifest(search_safe_manifest_path)
        records = load_full_search_v3_search_dataset(search_records_path)
    except FullSearchV3PreparationError as exc:
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH evaluator inputs must be valid prepared SEARCH artifacts"
        ) from exc
    return validate_full_search_v3_search_input(manifest=manifest, records=records)


def validate_full_search_v3_search_input(
    *,
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> FullSearchV3SearchInput:
    """Validate and freeze one complete V3 SEARCH manifest/artifact pair."""

    if not isinstance(manifest, Mapping):
        raise FullSearchV3EvaluatorInputError("SEARCH manifest must be an object")
    if manifest.get("protocol_id") != FULL_SEARCH_V3_PROTOCOL_ID:
        raise FullSearchV3EvaluatorInputError(
            "SEARCH manifest does not identify the locked V3 protocol"
        )
    if manifest.get("split_sha256") != FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256:
        raise FullSearchV3EvaluatorInputError(
            "SEARCH manifest does not use the required split SHA-256"
        )
    if "FINAL" in manifest or manifest.get("stage", FULL_SEARCH_V3_STAGE) != (
        FULL_SEARCH_V3_STAGE
    ):
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH evaluator accepts only the SEARCH stage"
        )

    expected_ids = _manifest_sample_ids(manifest)
    observed_ids = _record_sample_ids(records)
    expected_count = canonical_full_search_v3_config().search_size
    if len(expected_ids) != expected_count or len(observed_ids) != expected_count:
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH evaluator requires exactly 1,000 SEARCH records"
        )
    if len(expected_ids) != len(set(expected_ids)):
        raise FullSearchV3EvaluatorInputError(
            "SEARCH manifest contains duplicate sample IDs"
        )
    if len(observed_ids) != len(set(observed_ids)):
        raise FullSearchV3EvaluatorInputError("SEARCH records contain duplicate sample IDs")
    if set(observed_ids) != set(expected_ids):
        raise FullSearchV3EvaluatorInputError(
            "SEARCH records do not provide complete manifest ID coverage"
        )
    if observed_ids != expected_ids:
        raise FullSearchV3EvaluatorInputError(
            "SEARCH records are not in immutable manifest order"
        )

    try:
        verify_full_search_v3_search_safe_manifest(manifest)
        verify_full_search_v3_search_dataset(records, manifest)
    except FullSearchV3PreparationError as exc:
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH evaluator inputs do not satisfy the prepared artifact contract"
        ) from exc

    return FullSearchV3SearchInput(
        _manifest=_freeze_json(manifest),
        _records=tuple(_freeze_json(record) for record in records),
    )


def _require_complete_search_input(
    search_input: FullSearchV3SearchInput,
) -> FullSearchV3SearchInput:
    """Revalidate the only input eligible for a V3 protocol artifact."""

    if not isinstance(search_input, FullSearchV3SearchInput):
        raise FullSearchV3EvaluatorInputError(
            "search_input must be a validated FullSearchV3SearchInput"
        )
    # The frozen carrier is public for inspection, so every public protocol
    # result boundary reuses the authoritative prepared-artifact validator.
    return validate_full_search_v3_search_input(
        manifest=_thaw_json(search_input.manifest),
        records=_thaw_json(search_input.records),
    )


def _validate_candidate_id(value: Any) -> str:
    """Accept only a bounded opaque identifier safe to persist verbatim."""

    if (
        not isinstance(value, str)
        or not _CANDIDATE_ID.fullmatch(value)
        or ".." in value
        or _UNSAFE_CANDIDATE_ID.search(value)
        or _BASE64_LIKE_CANDIDATE_ID.fullmatch(value)
        or contains_sensitive_value(value)
    ):
        raise FullSearchV3EvaluatorInputError(
            "candidate_id must be a safe opaque identifier"
        )
    return value


def _compute_prediction_metrics(
    search_input: FullSearchV3SearchInput,
    outcomes: Sequence[FullSearchV3PredictionOutcome],
) -> dict[str, Any]:
    """Pure, non-protocol metric accounting for testable record collections.

    This helper never persists data and intentionally returns no protocol ID,
    stage, candidate ID, or eligibility/rankability field. Public V3 reports
    must instead enter through :func:`account_full_search_v3_predictions`.
    """

    if not isinstance(search_input, FullSearchV3SearchInput):
        raise FullSearchV3EvaluatorInputError(
            "search_input must be FullSearchV3SearchInput"
        )
    ordered_outcomes = _validate_outcomes(search_input, outcomes)
    metric_records = _metric_records(
        search_input,
        "_metric_fixture_",
        ordered_outcomes,
    )
    try:
        summaries = evaluate_dataset_records(metric_records)
    except EvaluationError as exc:
        raise FullSearchV3EvaluatorInputError(
            "SEARCH records do not provide valid binary labels for trusted metrics"
        ) from exc
    if len(summaries) != 1:
        raise FullSearchV3EvaluatorInputError(
            "prediction accounting must produce exactly one metric summary"
        )
    summary = summaries[0]
    return {
        "total_records": summary["total_samples"],
        "completed_solver_responses": summary["successful_requests"]
        + summary["parse_failures"],
        "parsed_predictions": summary["successful_requests"],
        "abstentions_or_parse_failures": summary["parse_failures"],
        "infrastructure_failures": summary["api_failures"],
        "infrastructure_failure_categories": _failure_categories(ordered_outcomes),
        "macro_f1": summary["macro_f1"],
        "accuracy": summary["accuracy"]["value"],
        "parse_coverage": summary["parse_coverage"]["value"],
    }


def account_full_search_v3_predictions(
    *,
    search_input: FullSearchV3SearchInput,
    candidate_id: str,
    outcomes: Sequence[FullSearchV3PredictionOutcome],
    metrics_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute and optionally persist trusted metrics for complete SEARCH only.

    Outcomes must be supplied once, in the input's manifest order. A parser
    rejection becomes ``parse_failure`` and remains in the metric denominator;
    a retry-exhausted transport failure becomes ``api_failure`` and never a
    model prediction. No raw model response is persisted in the report.
    """

    search_input = _require_complete_search_input(search_input)
    candidate_id = _validate_candidate_id(candidate_id)
    metrics = _compute_prediction_metrics(search_input, outcomes)
    report = {
        "schema_version": FULL_SEARCH_V3_METRICS_SCHEMA_VERSION,
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "stage": FULL_SEARCH_V3_STAGE,
        "split_sha256": search_input.manifest["split_sha256"],
        "search_membership_sha256": search_input.manifest[
            "search_membership_sha256"
        ],
        "candidate_id": candidate_id,
        "model": canonical_full_search_v3_config().solver_model,
        "parser_version": PARSER_VERSION,
        "sample_ids_sha256": _sample_ids_sha256(search_input.sample_ids),
        "total_records": metrics["total_records"],
        "completed_solver_responses": metrics["completed_solver_responses"],
        "parsed_predictions": metrics["parsed_predictions"],
        "abstentions_or_parse_failures": metrics["abstentions_or_parse_failures"],
        "infrastructure_failures": metrics["infrastructure_failures"],
        "infrastructure_failure_categories": metrics[
            "infrastructure_failure_categories"
        ],
        "metrics": {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "parse_coverage": metrics["parse_coverage"],
            "rankable": (
                metrics["parsed_predictions"] == metrics["total_records"]
                and metrics["infrastructure_failures"] == 0
            ),
        },
    }
    if report["total_records"] != len(search_input.records):
        raise FullSearchV3EvaluatorInputError(
            "trusted metrics did not account for every SEARCH record"
        )
    if metrics_path is not None:
        _persist_full_search_v3_prediction_metrics(metrics_path, report)
    return report


def evaluate_full_search_v3_candidate(
    *,
    search_input: FullSearchV3SearchInput,
    candidate_id: str,
    prompt: Mapping[str, Template | str],
    solver_identity_sha256: str,
    cache: FullSearchV3SearchCache,
    executor: FullSearchV3RequestExecutor,
    checkpoint_path: str | Path,
    result_path: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate one prompt over the immutable full SEARCH membership.

    The cache is the durable completion store. The checkpoint contains only
    identity hashes and parser outcomes, never prompts, requests, responses,
    images, credentials, endpoints, or FINAL/legacy-stage material.
    """

    search_input = _require_complete_search_input(search_input)
    if not isinstance(cache, FullSearchV3SearchCache):
        raise FullSearchV3EvaluatorInputError("cache must be FullSearchV3SearchCache")
    if not isinstance(executor, FullSearchV3RequestExecutor):
        raise FullSearchV3EvaluatorInputError(
            "executor must be FullSearchV3RequestExecutor"
        )
    if executor.cache is not cache:
        raise FullSearchV3EvaluatorInputError(
            "executor and evaluator must use the same SEARCH cache"
        )
    candidate_id = _validate_candidate_id(candidate_id)
    _require_sha256(solver_identity_sha256, "solver_identity_sha256")
    prompt_family = PromptFamily(prompt)
    prompt_sha256 = _prompt_sha256(prompt_family)
    # P0 arrives as the canonical mapping itself. Keep that mapping at the
    # model-visible request boundary; validation/hashing above does not permit
    # an equivalent serialized or rebuilt prompt to replace it.
    request_prompt: Mapping[str, Template] = (
        COT_PROMPT if prompt is COT_PROMPT else prompt_family
    )
    requests = _candidate_requests(
        search_input=search_input,
        candidate_id=candidate_id,
        prompt=request_prompt,
        prompt_sha256=prompt_sha256,
        solver_identity_sha256=solver_identity_sha256,
    )
    identity = _checkpoint_identity(
        search_input=search_input,
        candidate_id=candidate_id,
        prompt_sha256=prompt_sha256,
        solver_identity_sha256=solver_identity_sha256,
        requests=requests,
    )
    checkpoint_destination = Path(checkpoint_path)
    result_destination = Path(result_path)
    checkpoint = _load_or_initialize_checkpoint(
        checkpoint_destination,
        identity=identity,
        resume=resume,
    )
    completed = {
        entry["sample_id"]: entry for entry in checkpoint["completed_samples"]
    }
    outcomes: dict[str, FullSearchV3PredictionOutcome] = {}
    failures: dict[str, SolverFailureMetadata] = {}

    for item in requests:
        sample_id = item["sample_id"]
        request_identity = item["identity"]
        request = item["request"]
        saved = completed.get(sample_id)
        if saved is not None:
            result = _rehydrate_completed_result(cache, request_identity, saved)
            outcomes[sample_id] = FullSearchV3PredictionOutcome.from_solver_result(
                sample_id, result
            )
            continue
        try:
            result = executor.complete(request_identity, request)
        except SolverExecutionFailure as exc:
            failures[sample_id] = exc.metadata
            checkpoint["last_infrastructure_failures"] = _failure_entries(
                failures, requests
            )
            checkpoint["status"] = "incomplete"
            _write_checkpoint(checkpoint_destination, checkpoint)
            continue
        outcome = FullSearchV3PredictionOutcome.from_solver_result(sample_id, result)
        outcomes[sample_id] = outcome
        completed[sample_id] = _completed_entry(request_identity, outcome)
        checkpoint["completed_samples"] = _ordered_completed_entries(completed, requests)
        checkpoint["last_infrastructure_failures"] = _failure_entries(
            failures, requests
        )
        checkpoint["status"] = "running"
        _write_checkpoint(checkpoint_destination, checkpoint)

    ordered_outcomes = tuple(
        outcomes.get(item["sample_id"])
        or FullSearchV3PredictionOutcome(
            sample_id=item["sample_id"],
            infrastructure_failure=failures[item["sample_id"]],
        )
        for item in requests
    )
    report = account_full_search_v3_predictions(
        search_input=search_input,
        candidate_id=candidate_id,
        outcomes=ordered_outcomes,
    )
    report["prompt_sha256"] = prompt_sha256
    _validate_final_report(search_input, report, prompt_sha256=prompt_sha256)
    if report["infrastructure_failures"]:
        checkpoint["status"] = "incomplete"
        checkpoint["last_infrastructure_failures"] = _failure_entries(failures, requests)
        _write_checkpoint(checkpoint_destination, checkpoint)
        raise FullSearchV3EvaluationIncomplete(report)
    if report["metrics"]["rankable"] is not True:
        checkpoint["status"] = "ineligible"
        checkpoint["last_infrastructure_failures"] = []
        _write_checkpoint(checkpoint_destination, checkpoint)
        return report

    checkpoint["status"] = "eligible"
    checkpoint["last_infrastructure_failures"] = []
    _write_checkpoint(checkpoint_destination, checkpoint)
    _persist_full_search_v3_prediction_metrics(result_destination, report)
    return report


def canonical_full_search_v3_p0_prompt_sha256() -> str:
    """Return the immutable source hash of the canonical ``cot`` P0 prompt."""

    return template_source_sha256(COT_PROMPT)


def evaluate_full_search_v3_p0(
    *,
    search_input: FullSearchV3SearchInput,
    solver_identity_sha256: str,
    cache: FullSearchV3SearchCache,
    executor: FullSearchV3RequestExecutor,
    checkpoint_path: str | Path,
    result_path: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Evaluate frozen canonical P0 without accepting caller prompt text.

    This deliberately passes the original ``COT_PROMPT`` mapping, rather than
    a serialized snapshot or reconstructed template family. Its source hash is
    included in the same immutable request/checkpoint/result identity used for
    every candidate.
    """

    report = evaluate_full_search_v3_candidate(
        search_input=search_input,
        candidate_id=FULL_SEARCH_V3_P0_CANDIDATE_ID,
        prompt=COT_PROMPT,
        solver_identity_sha256=solver_identity_sha256,
        cache=cache,
        executor=executor,
        checkpoint_path=checkpoint_path,
        result_path=result_path,
        resume=resume,
    )
    if report.get("prompt_sha256") != canonical_full_search_v3_p0_prompt_sha256():
        raise FullSearchV3EvaluatorInputError(
            "canonical P0 result does not carry the frozen COT prompt hash"
        )
    return report


def _candidate_requests(
    *,
    search_input: FullSearchV3SearchInput,
    candidate_id: str,
    prompt: Mapping[str, Template],
    prompt_sha256: str,
    solver_identity_sha256: str,
) -> tuple[dict[str, Any], ...]:
    config = canonical_full_search_v3_config()
    requests: list[dict[str, Any]] = []
    for record in search_input.records:
        sample_id = record["sample_id"]
        request = build_solver_request(record, prompt, config=config)
        request_identity = build_solver_request_identity(
            request,
            sample_id=sample_id,
            candidate_id=candidate_id,
            prompt_sha256=prompt_sha256,
            split_sha256=search_input.manifest["split_sha256"],
            search_membership_sha256=search_input.manifest[
                "search_membership_sha256"
            ],
            solver_identity_sha256=solver_identity_sha256,
            config=config,
            parser_version=PARSER_VERSION,
        )
        requests.append(
            {
                "sample_id": sample_id,
                "request": request,
                "identity": request_identity,
            }
        )
    return tuple(requests)


def _checkpoint_identity(
    *,
    search_input: FullSearchV3SearchInput,
    candidate_id: str,
    prompt_sha256: str,
    solver_identity_sha256: str,
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    config = canonical_full_search_v3_config()
    return {
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "stage": FULL_SEARCH_V3_STAGE,
        "config_sha256": config.sha256(),
        "split_sha256": search_input.manifest["split_sha256"],
        "search_membership_sha256": search_input.manifest[
            "search_membership_sha256"
        ],
        "sample_ids_sha256": _sample_ids_sha256(search_input.sample_ids),
        # IDs are already present in the SEARCH-safe manifest. Keeping this
        # ordered list lets resume reject an entry whose request digest was
        # copied onto the wrong sample without retaining model-visible input.
        "sample_ids": list(search_input.sample_ids),
        "sample_count": len(search_input.sample_ids),
        "candidate_id": candidate_id,
        "prompt_sha256": prompt_sha256,
        "solver_identity_sha256": solver_identity_sha256,
        "solver_model": config.solver_model,
        "generation": {
            "temperature": config.solver_temperature,
            "top_p": config.solver_top_p,
            "seed": config.solver_seed,
            "n": config.solver_n,
            "stream": config.solver_stream,
            "max_tokens": config.solver_max_tokens,
        },
        "parser_version": PARSER_VERSION,
        "request_identities_sha256": [
            item["identity"].sha256() for item in requests
        ],
    }


def _load_or_initialize_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if not path.exists():
        if resume:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint is missing; start a new evaluation"
            )
        return {
            "schema_version": FULL_SEARCH_V3_CHECKPOINT_SCHEMA_VERSION,
            "artifact_type": "full_search_candidate_checkpoint",
            "identity": dict(identity),
            "status": "running",
            "completed_samples": [],
            "last_infrastructure_failures": [],
        }
    if not resume:
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint already exists; use resume with identical inputs"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint is corrupt or unreadable"
        ) from exc
    _validate_checkpoint(payload, identity)
    return payload


def _validate_checkpoint(payload: Any, identity: Mapping[str, Any]) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "artifact_type",
        "identity",
        "status",
        "completed_samples",
        "last_infrastructure_failures",
    }:
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint has an invalid schema"
        )
    if (
        payload["schema_version"] != FULL_SEARCH_V3_CHECKPOINT_SCHEMA_VERSION
        or payload["artifact_type"] != "full_search_candidate_checkpoint"
        or payload["identity"] != dict(identity)
    ):
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint identity is incompatible with this evaluation"
        )
    if payload["status"] not in {"running", "incomplete", "ineligible", "eligible"}:
        raise FullSearchV3EvaluationResumeError("full SEARCH checkpoint status is invalid")
    completed = payload["completed_samples"]
    if not isinstance(completed, list):
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint completed samples are invalid"
        )
    sample_ids = identity.get("sample_ids")
    request_digests = identity.get("request_identities_sha256")
    if (
        not isinstance(sample_ids, list)
        or not isinstance(request_digests, list)
        or len(sample_ids) != identity["sample_count"]
        or len(request_digests) != identity["sample_count"]
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
        or any(not isinstance(digest, str) for digest in request_digests)
        or len(set(request_digests)) != len(request_digests)
    ):
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint identity has invalid ordered request membership"
        )
    expected = {
        request_digest: (position, sample_ids[position])
        for position, request_digest in enumerate(request_digests)
    }
    prior_position = -1
    completed_request_digests: set[str] = set()
    for entry in completed:
        if not isinstance(entry, dict) or set(entry) != {
            "sample_id",
            "request_identity_sha256",
            "request_status",
            "parse_status",
            "prediction",
        }:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint completed sample is invalid"
            )
        request_sha256 = entry["request_identity_sha256"]
        if request_sha256 not in expected:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint contains a foreign request identity"
            )
        position, expected_sample_id = expected[request_sha256]
        if entry["sample_id"] != expected_sample_id:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint completion sample does not match its request"
            )
        if position <= prior_position:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint completed samples are not ordered uniquely"
            )
        prior_position = position
        completed_request_digests.add(request_sha256)
        if entry["request_status"] == SUCCESS:
            if entry["parse_status"] != "parsed" or entry["prediction"] not in {
                "yes",
                "no",
            }:
                raise FullSearchV3EvaluationResumeError(
                    "full SEARCH checkpoint parsed completion is invalid"
                )
        elif entry["request_status"] == PARSE_FAILURE:
            if entry["parse_status"] != "invalid" or entry["prediction"] != "invalid":
                raise FullSearchV3EvaluationResumeError(
                    "full SEARCH checkpoint parse failure is invalid"
                )
        else:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint completion status is invalid"
            )
    failures = payload["last_infrastructure_failures"]
    if not isinstance(failures, list):
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint infrastructure failures are invalid"
        )
    prior_failure_position = -1
    failure_request_digests: set[str] = set()
    for entry in failures:
        if not isinstance(entry, dict) or set(entry) != {
            "sample_id",
            "request_identity_sha256",
            "failure",
        }:
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint infrastructure failure is invalid"
            )
        request_sha256 = entry["request_identity_sha256"]
        if (
            request_sha256 not in expected
            or entry["sample_id"] != expected[request_sha256][1]
            or not isinstance(entry["failure"], dict)
        ):
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint infrastructure failure is invalid"
            )
        position = expected[request_sha256][0]
        if (
            position <= prior_failure_position
            or request_sha256 in completed_request_digests
        ):
            raise FullSearchV3EvaluationResumeError(
                "full SEARCH checkpoint failure membership is inconsistent"
            )
        prior_failure_position = position
        failure_request_digests.add(request_sha256)
    if payload["status"] in {"eligible", "ineligible"} and (
        len(completed_request_digests) != identity["sample_count"]
        or failure_request_digests
    ):
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint terminal status is incomplete"
        )


def _rehydrate_completed_result(
    cache: FullSearchV3SearchCache,
    identity: Any,
    saved: Mapping[str, Any],
) -> SolverResult:
    try:
        result = cache.get(identity)
    except SearchCacheError as exc:
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH completed cache entry is unavailable or corrupt"
        ) from exc
    if result is None:
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint completion is absent from the compatible cache"
        )
    outcome = FullSearchV3PredictionOutcome.from_solver_result(
        saved["sample_id"], result
    )
    if _completed_entry(identity, outcome) != dict(saved):
        raise FullSearchV3EvaluationResumeError(
            "full SEARCH checkpoint completion differs from the compatible cache"
        )
    return result


def _completed_entry(identity: Any, outcome: FullSearchV3PredictionOutcome) -> dict[str, Any]:
    parsed = parse_answer(outcome.raw_response)
    if parsed["parse_status"] == "parsed":
        request_status = SUCCESS
        parse_status = "parsed"
    else:
        request_status = PARSE_FAILURE
        parse_status = "invalid"
    return {
        "sample_id": outcome.sample_id,
        "request_identity_sha256": identity.sha256(),
        "request_status": request_status,
        "parse_status": parse_status,
        "prediction": parsed["prediction"],
    }


def _ordered_completed_entries(
    completed: Mapping[str, Mapping[str, Any]], requests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [dict(completed[item["sample_id"]]) for item in requests if item["sample_id"] in completed]


def _failure_entries(
    failures: Mapping[str, SolverFailureMetadata], requests: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "sample_id": item["sample_id"],
            "request_identity_sha256": item["identity"].sha256(),
            "failure": failures[item["sample_id"]].as_dict(),
        }
        for item in requests
        if item["sample_id"] in failures
    ]


def _write_checkpoint(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_replace_bytes(path, _canonical_json_bytes(value), "full SEARCH checkpoint")


def _validate_final_report(
    search_input: FullSearchV3SearchInput,
    report: Mapping[str, Any],
    *,
    prompt_sha256: str,
) -> None:
    total = len(search_input.sample_ids)
    if (
        report.get("stage") != FULL_SEARCH_V3_STAGE
        or report.get("split_sha256") != search_input.manifest["split_sha256"]
        or report.get("search_membership_sha256")
        != search_input.manifest["search_membership_sha256"]
        or report.get("sample_ids_sha256") != _sample_ids_sha256(search_input.sample_ids)
        or report.get("prompt_sha256") != prompt_sha256
        or report.get("total_records") != total
        or report.get("completed_solver_responses", -1)
        + report.get("infrastructure_failures", -1)
        != total
        or report.get("parsed_predictions", -1)
        + report.get("abstentions_or_parse_failures", -1)
        != report.get("completed_solver_responses")
        or not isinstance(report.get("metrics"), Mapping)
        or report["metrics"].get("macro_f1") is None
        or report["metrics"].get("accuracy") is None
    ):
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH candidate report is incomplete or incompatible"
        )


def _prompt_sha256(prompt: PromptFamily) -> str:
    return template_source_sha256(
        {method: prompt[method].template for method in TEMPLATE_KEYS}
    )


def _sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return _sha256_json(list(sample_ids))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise FullSearchV3EvaluatorInputError(
            f"{field} must be a lowercase hexadecimal SHA-256"
        )


def _atomic_replace_bytes(path: Path, encoded: bytes, context: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
    except OSError as exc:
        raise FullSearchV3EvaluatorInputError(f"{context} cannot be created") from exc
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise FullSearchV3EvaluatorInputError(f"{context} cannot be persisted") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _persist_full_search_v3_prediction_metrics(
    path: str | Path, report: Mapping[str, Any]
) -> Path:
    """Atomically create a report built by the validated evaluator boundary."""

    encoded = _canonical_json_bytes(report)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise FullSearchV3EvaluatorInputError(
                "full SEARCH metrics artifact is not readable"
            ) from exc
        if existing != encoded:
            raise FullSearchV3EvaluatorInputError(
                "refusing to replace a different full SEARCH metrics artifact"
            )
        return destination

    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    except OSError as exc:
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH metrics artifact cannot be created"
        ) from exc
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return _persist_full_search_v3_prediction_metrics(destination, report)
        except OSError as exc:
            raise FullSearchV3EvaluatorInputError(
                "full SEARCH metrics artifact could not be created atomically"
            ) from exc
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _fsync_directory(directory: Path) -> None:
    """Persist a completed atomic rename/link before reporting success."""

    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_outcomes(
    search_input: FullSearchV3SearchInput,
    outcomes: Sequence[FullSearchV3PredictionOutcome],
) -> tuple[FullSearchV3PredictionOutcome, ...]:
    if isinstance(outcomes, (str, bytes)) or not isinstance(outcomes, Sequence):
        raise FullSearchV3EvaluatorInputError("prediction outcomes must be a sequence")
    ordered = tuple(outcomes)
    expected_ids = search_input.sample_ids
    observed_ids = tuple(
        outcome.sample_id
        if isinstance(outcome, FullSearchV3PredictionOutcome)
        else None
        for outcome in ordered
    )
    if any(sample_id is None for sample_id in observed_ids):
        raise FullSearchV3EvaluatorInputError(
            "prediction outcomes must use FullSearchV3PredictionOutcome"
        )
    if len(ordered) != len(expected_ids):
        raise FullSearchV3EvaluatorInputError(
            "prediction outcomes must cover every immutable SEARCH record"
        )
    if len(set(observed_ids)) != len(observed_ids):
        raise FullSearchV3EvaluatorInputError("prediction outcomes contain duplicates")
    if set(observed_ids) != set(expected_ids):
        raise FullSearchV3EvaluatorInputError(
            "prediction outcomes do not match immutable SEARCH membership"
        )
    if observed_ids != expected_ids:
        raise FullSearchV3EvaluatorInputError(
            "prediction outcomes are not in immutable manifest order"
        )
    return ordered


def _metric_records(
    search_input: FullSearchV3SearchInput,
    candidate_id: str,
    outcomes: Sequence[FullSearchV3PredictionOutcome],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, (source, outcome) in enumerate(zip(search_input.records, outcomes)):
        if "gold_label" not in source or source["gold_label"] is None:
            raise FullSearchV3EvaluatorInputError(
                f"SEARCH record {index} has no binary gold label for trusted metrics"
            )
        method = source.get("claim_type")
        if not isinstance(method, str) or not method:
            raise FullSearchV3EvaluatorInputError(
                f"SEARCH record {index} has no reasoning method"
            )
        common = {
            "run_id": _METRICS_RUN_ID,
            "dataset": _DATASET_NAME,
            "model": canonical_full_search_v3_config().solver_model,
            "prompt_variant": candidate_id,
            "method": method,
            "sample_id": outcome.sample_id,
            "gold_label": source["gold_label"],
            "attempt_count": 1,
        }
        if outcome.infrastructure_failure is not None:
            records.append(
                {
                    **common,
                    "prediction": None,
                    "parse_status": "not_attempted",
                    "request_status": API_FAILURE,
                }
            )
            continue
        parsed = parse_answer(outcome.raw_response)
        if parsed["parse_status"] == "parsed":
            records.append(
                {
                    **common,
                    "prediction": parsed["prediction"],
                    "parse_status": "parsed",
                    "request_status": SUCCESS,
                }
            )
        else:
            records.append(
                {
                    **common,
                    "prediction": parsed["prediction"],
                    "parse_status": "invalid",
                    "request_status": PARSE_FAILURE,
                }
            )
    return records


def _failure_categories(
    outcomes: Sequence[FullSearchV3PredictionOutcome],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        failure = outcome.infrastructure_failure
        if failure is not None:
            category = failure.category.value
            counts[category] = counts.get(category, 0) + 1
    return {category: counts[category] for category in sorted(counts)}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FullSearchV3EvaluatorInputError(
            "full SEARCH metrics artifact must contain JSON values"
        ) from exc


def _manifest_sample_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    search = manifest.get("SEARCH")
    if not isinstance(search, Mapping):
        raise FullSearchV3EvaluatorInputError("SEARCH manifest has no SEARCH stage")
    sample_ids = search.get("sample_ids")
    if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
        raise FullSearchV3EvaluatorInputError("SEARCH manifest sample IDs must be a list")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise FullSearchV3EvaluatorInputError(
            "SEARCH manifest sample IDs must be non-empty text"
        )
    return tuple(sample_ids)


def _record_sample_ids(records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FullSearchV3EvaluatorInputError("SEARCH records must be a sequence")
    sample_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise FullSearchV3EvaluatorInputError(
                f"SEARCH record {index} must be an object"
            )
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FullSearchV3EvaluatorInputError(
                f"SEARCH record {index} has an invalid sample ID"
            )
        sample_ids.append(sample_id)
    return tuple(sample_ids)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a plain JSON-compatible copy of an immutable input carrier."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "FULL_SEARCH_V3_CHECKPOINT_SCHEMA_VERSION",
    "FULL_SEARCH_V3_P0_CANDIDATE_ID",
    "FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256",
    "FULL_SEARCH_V3_STAGE",
    "FULL_SEARCH_V3_METRICS_SCHEMA_VERSION",
    "FullSearchV3EvaluationIncomplete",
    "FullSearchV3EvaluationResumeError",
    "FullSearchV3EvaluatorInputError",
    "FullSearchV3PredictionOutcome",
    "FullSearchV3SearchInput",
    "account_full_search_v3_predictions",
    "canonical_full_search_v3_p0_prompt_sha256",
    "evaluate_full_search_v3_candidate",
    "evaluate_full_search_v3_p0",
    "load_full_search_v3_search_input",
    "validate_full_search_v3_search_input",
]
