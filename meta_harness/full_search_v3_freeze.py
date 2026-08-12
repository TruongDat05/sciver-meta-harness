"""Offline, create-once freezing of a terminal full-SEARCH V3 winner.

This module intentionally consumes only the durable SEARCH orchestration
state.  It has no FINAL inputs, paths, evaluator, solver, or proposer
dependencies, so freezing is a strict boundary before paired FINAL work.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from meta_harness.config import (
    FULL_SEARCH_V3_MAX_ITERATIONS,
    FULL_SEARCH_V3_MIN_ITERATIONS,
    FULL_SEARCH_V3_PATIENCE,
    FULL_SEARCH_V3_PROTOCOL_ID,
    canonical_full_search_v3_config,
)
from meta_harness.full_search_v3_evaluator import (
    FULL_SEARCH_V3_P0_CANDIDATE_ID,
    canonical_full_search_v3_p0_prompt_sha256,
)
from meta_harness.full_search_v3_orchestrator import (
    FULL_SEARCH_V3_ORCHESTRATOR_SCHEMA_VERSION,
    FULL_SEARCH_V3_ORCHESTRATOR_VERSION,
)
from meta_harness.full_search_v3_ranking import (
    FullSearchV3PatienceState,
    FullSearchV3RankedCandidate,
    advance_full_search_v3_patience,
    eligible_full_search_v3_candidate,
    rank_eligible_full_search_v3_reports,
)
from meta_harness.prompt_family import (
    TEMPLATE_KEYS,
    PromptFamily,
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)


FULL_SEARCH_V3_FREEZE_SCHEMA_VERSION = "sciver_full_search_v3_freeze_v1"
FULL_SEARCH_V3_FREEZE_VERSION = "sciver_full_search_v3_offline_freeze_v1"
FULL_SEARCH_V3_FREEZE_VARIANT = "meta_cot"
_STATE_FILENAME = "orchestration_state.json"
_FREEZE_FILENAME = "frozen_winner.json"
_LOCK_FILENAME = ".freeze.lock"
_FREEZABLE_STATUSES = frozenset({"patience_stopped", "max_stopped"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FullSearchV3FreezeError(RuntimeError):
    """Raised when a SEARCH winner cannot safely become immutable P*."""


class FullSearchV3FrozenArtifactConflictError(FullSearchV3FreezeError):
    """Raised when a create-once frozen artifact has different content."""


def full_search_v3_frozen_winner_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the V3-only location for one run's immutable winner artifact."""

    return _run_directory(repository_root, run_id) / "freeze" / _FREEZE_FILENAME


def freeze_full_search_v3_winner(
    *, repository_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Verify a terminal SEARCH run and atomically freeze its ranked P*.

    The stored winner and ranking are never trusted on their own: this entry
    point reconstructs eligible reports, applies the locked rank key, and
    checks the durable patience/schedule state before serializing a create-once
    ``meta_cot`` artifact.  It neither accepts nor reads FINAL material.
    """

    run_directory = _run_directory(repository_root, run_id)
    state = _load_canonical_object(run_directory / _STATE_FILENAME, "SEARCH state")
    artifact = _build_frozen_artifact(run_id=_safe_identifier(run_id), state=state)
    destination = full_search_v3_frozen_winner_path(repository_root, run_id)
    with _freeze_lock(destination.parent / _LOCK_FILENAME):
        if destination.exists():
            existing = load_full_search_v3_frozen_winner(destination)
            if canonical_json(existing) != canonical_json(artifact):
                raise FullSearchV3FrozenArtifactConflictError(
                    "a different immutable V3 winner is already frozen"
                )
            return existing
        _atomic_create_json(destination, artifact)
    return load_full_search_v3_frozen_winner(destination)


def load_full_search_v3_frozen_winner(path: str | Path) -> dict[str, Any]:
    """Load a frozen artifact and reject malformed or tampered content."""

    artifact = _load_canonical_object(Path(path), "frozen V3 winner")
    required = {
        "schema_version",
        "freeze_version",
        "run_id",
        "prompt_variant",
        "winner",
        "templates",
        "search_metrics",
        "hashes",
        "artifact_sha256",
    }
    if set(artifact) != required:
        raise FullSearchV3FreezeError("frozen V3 winner fields are invalid")
    if artifact["schema_version"] != FULL_SEARCH_V3_FREEZE_SCHEMA_VERSION:
        raise FullSearchV3FreezeError("unsupported frozen V3 winner schema")
    if artifact["freeze_version"] != FULL_SEARCH_V3_FREEZE_VERSION:
        raise FullSearchV3FreezeError("unsupported frozen V3 winner version")
    if artifact["prompt_variant"] != FULL_SEARCH_V3_FREEZE_VARIANT:
        raise FullSearchV3FreezeError("frozen V3 winner must register meta_cot")
    _safe_identifier(artifact["run_id"])
    templates = _template_sources(artifact["templates"], "frozen templates")
    hashes = artifact["hashes"]
    if not isinstance(hashes, Mapping) or set(hashes) != {
        "prompt_sha256",
        "source_sha256",
        "config_sha256",
        "split_sha256",
        "search_membership_sha256",
        "sample_ids_sha256",
        "ranking_sha256",
        "run_identity_sha256",
    }:
        raise FullSearchV3FreezeError("frozen V3 winner hashes are invalid")
    for name, value in hashes.items():
        _sha256(value, f"hashes.{name}")
    prompt_sha256 = template_source_sha256(templates)
    if hashes["prompt_sha256"] != prompt_sha256 or hashes["source_sha256"] != prompt_sha256:
        raise FullSearchV3FreezeError("frozen V3 winner prompt hash is inconsistent")
    _validate_frozen_winner(artifact["winner"], prompt_sha256)
    _validate_search_metrics(artifact["search_metrics"])
    if hashes["ranking_sha256"] != _sha256_json(artifact["search_metrics"]["ranking"]):
        raise FullSearchV3FreezeError("frozen V3 ranking hash is inconsistent")
    _sha256(artifact["artifact_sha256"], "artifact_sha256")
    expected = _sha256_json({key: value for key, value in artifact.items() if key != "artifact_sha256"})
    if artifact["artifact_sha256"] != expected:
        raise FullSearchV3FreezeError("frozen V3 winner artifact hash is inconsistent")
    return _json_copy(artifact)


def _build_frozen_artifact(*, run_id: str, state: Mapping[str, Any]) -> dict[str, Any]:
    identity, ranking, reports_by_id, candidate_templates = _verify_terminal_state(state)
    winner = ranking[0]
    templates = (
        canonical_baseline_sources()
        if winner.candidate_id == FULL_SEARCH_V3_P0_CANDIDATE_ID
        else candidate_templates[winner.candidate_id]
    )
    sources = _template_sources(templates, "winner templates")
    prompt_sha256 = template_source_sha256(sources)
    if prompt_sha256 != winner.prompt_sha256:
        raise FullSearchV3FreezeError("recomputed winner prompt hash is inconsistent")
    report = reports_by_id[winner.candidate_id]
    ranking_payload = [
        {
            "candidate_id": item.candidate_id,
            "prompt_sha256": item.prompt_sha256,
            "macro_f1": item.macro_f1,
            "accuracy": item.accuracy,
        }
        for item in ranking
    ]
    payload = {
        "schema_version": FULL_SEARCH_V3_FREEZE_SCHEMA_VERSION,
        "freeze_version": FULL_SEARCH_V3_FREEZE_VERSION,
        "run_id": run_id,
        "prompt_variant": FULL_SEARCH_V3_FREEZE_VARIANT,
        "winner": {
            "candidate_id": winner.candidate_id,
            "source_kind": (
                "canonical_cot"
                if winner.candidate_id == FULL_SEARCH_V3_P0_CANDIDATE_ID
                else "search_candidate"
            ),
            "rank": 1,
        },
        "templates": sources,
        "search_metrics": {
            "macro_f1": winner.macro_f1,
            "accuracy": winner.accuracy,
            "parse_coverage": report["metrics"]["parse_coverage"],
            "total_records": report["total_records"],
            "completed_solver_responses": report["completed_solver_responses"],
            "parsed_predictions": report["parsed_predictions"],
            "abstentions_or_parse_failures": report["abstentions_or_parse_failures"],
            "infrastructure_failures": report["infrastructure_failures"],
            "ranking": ranking_payload,
        },
        "hashes": {
            "prompt_sha256": prompt_sha256,
            "source_sha256": prompt_sha256,
            "config_sha256": identity["config_sha256"],
            "split_sha256": identity["split_sha256"],
            "search_membership_sha256": identity["search_membership_sha256"],
            "sample_ids_sha256": identity["sample_ids_sha256"],
            "ranking_sha256": _sha256_json(ranking_payload),
            "run_identity_sha256": _sha256_json(identity),
        },
    }
    return {**payload, "artifact_sha256": _sha256_json(payload)}


def _verify_terminal_state(
    state: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    tuple[FullSearchV3RankedCandidate, ...],
    dict[str, dict[str, Any]],
    dict[str, dict[str, str]],
]:
    required = {
        "schema_version", "orchestrator_version", "identity", "status", "stop_reason",
        "p0", "iterations", "patience", "ranking", "winner_id",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise FullSearchV3FreezeError("SEARCH state fields are invalid")
    if state["schema_version"] != FULL_SEARCH_V3_ORCHESTRATOR_SCHEMA_VERSION or state[
        "orchestrator_version"
    ] != FULL_SEARCH_V3_ORCHESTRATOR_VERSION:
        raise FullSearchV3FreezeError("SEARCH state schema is incompatible")
    if state["status"] not in _FREEZABLE_STATUSES:
        raise FullSearchV3FreezeError("SEARCH run must reach a successful terminal state")
    identity = _validate_identity(state["identity"])
    p0 = state["p0"]
    if not isinstance(p0, Mapping) or set(p0) != {"status", "report"} or p0["status"] != "complete":
        raise FullSearchV3FreezeError("SEARCH P0 is not complete")
    p0_report = _validated_report(p0["report"], FULL_SEARCH_V3_P0_CANDIDATE_ID)
    if p0_report["prompt_sha256"] != identity["canonical_p0_prompt_sha256"]:
        raise FullSearchV3FreezeError("SEARCH P0 prompt hash does not match run identity")

    iterations = state["iterations"]
    if isinstance(iterations, (str, bytes)) or not isinstance(iterations, Sequence):
        raise FullSearchV3FreezeError("SEARCH iterations are invalid")
    if not FULL_SEARCH_V3_MIN_ITERATIONS <= len(iterations) <= FULL_SEARCH_V3_MAX_ITERATIONS:
        raise FullSearchV3FreezeError("SEARCH run did not complete the locked schedule")
    if state["status"] == "max_stopped":
        if len(iterations) != FULL_SEARCH_V3_MAX_ITERATIONS or state["stop_reason"] != "maximum_completed_iterations":
            raise FullSearchV3FreezeError("maximum SEARCH terminal state is inconsistent")
    elif state["stop_reason"] != "metric_patience":
        raise FullSearchV3FreezeError("patience SEARCH terminal state is inconsistent")

    reports_by_id = {FULL_SEARCH_V3_P0_CANDIDATE_ID: p0_report}
    candidate_templates: dict[str, dict[str, str]] = {}
    expected_iterations = list(range(1, len(iterations) + 1))
    observed_iterations: list[int] = []
    prior = FullSearchV3PatienceState(
        best_macro_f1=eligible_full_search_v3_candidate(p0_report).macro_f1,
        best_accuracy=eligible_full_search_v3_candidate(p0_report).accuracy,
    )
    for entry in iterations:
        if not isinstance(entry, Mapping) or set(entry) != {
            "iteration", "status", "candidate", "report", "metric_improved"
        }:
            raise FullSearchV3FreezeError("SEARCH iteration state is invalid")
        observed_iterations.append(entry["iteration"])
        if entry["status"] != "complete":
            raise FullSearchV3FreezeError("SEARCH run has an incomplete candidate iteration")
        candidate = entry["candidate"]
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "candidate_id", "parent_id", "hypothesis", "expected_tradeoff", "templates", "source_sha256"
        }:
            raise FullSearchV3FreezeError("SEARCH candidate state is invalid")
        candidate_id = _safe_identifier(candidate["candidate_id"])
        if candidate_id == FULL_SEARCH_V3_P0_CANDIDATE_ID or candidate_id in reports_by_id:
            raise FullSearchV3FreezeError("SEARCH candidate IDs are not unique")
        _safe_identifier(candidate["parent_id"])
        templates = _template_sources(candidate["templates"], "SEARCH candidate templates")
        source_sha256 = _sha256(candidate["source_sha256"], "SEARCH candidate source_sha256")
        if template_source_sha256(templates) != source_sha256:
            raise FullSearchV3FreezeError("SEARCH candidate prompt hash is inconsistent")
        report = _validated_report(entry["report"], candidate_id)
        if report["prompt_sha256"] != source_sha256:
            raise FullSearchV3FreezeError("SEARCH report prompt hash does not match candidate")
        reports_by_id[candidate_id] = report
        candidate_templates[candidate_id] = templates
        current_ranking = rank_eligible_full_search_v3_reports(reports_by_id.values())
        update = advance_full_search_v3_patience(prior, current_ranking)
        if entry["metric_improved"] is not update.metric_improved:
            raise FullSearchV3FreezeError("SEARCH candidate patience record is inconsistent")
        prior = update.state
    if observed_iterations != expected_iterations:
        raise FullSearchV3FreezeError("SEARCH iteration ordering is inconsistent")
    ranking = rank_eligible_full_search_v3_reports(reports_by_id.values())
    if not ranking:
        raise FullSearchV3FreezeError("SEARCH run has no rankable winner")
    if state["ranking"] != [candidate.candidate_id for candidate in ranking] or state["winner_id"] != ranking[0].candidate_id:
        raise FullSearchV3FreezeError("stored SEARCH ranking or winner is inconsistent")
    if _patience_mapping(prior) != state["patience"]:
        raise FullSearchV3FreezeError("stored SEARCH patience is inconsistent")
    if state["status"] == "patience_stopped" and prior.consecutive_non_improving < FULL_SEARCH_V3_PATIENCE:
        raise FullSearchV3FreezeError("SEARCH patience terminal threshold was not reached")
    return identity, ranking, reports_by_id, candidate_templates


def _validate_identity(value: Any) -> dict[str, Any]:
    required = {
        "protocol_id", "config_sha256", "split_sha256", "search_membership_sha256",
        "sample_ids_sha256", "canonical_p0_prompt_sha256", "parser_version",
        "solver_identity_sha256", "proposer_identity",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FullSearchV3FreezeError("SEARCH run identity is invalid")
    if value["protocol_id"] != FULL_SEARCH_V3_PROTOCOL_ID:
        raise FullSearchV3FreezeError("SEARCH run protocol is incompatible")
    if value["config_sha256"] != canonical_full_search_v3_config().sha256():
        raise FullSearchV3FreezeError("SEARCH run configuration is incompatible")
    if value["canonical_p0_prompt_sha256"] != canonical_full_search_v3_p0_prompt_sha256():
        raise FullSearchV3FreezeError("SEARCH canonical cot prompt has changed")
    for field in required - {"protocol_id", "parser_version", "proposer_identity"}:
        _sha256(value[field], f"SEARCH identity {field}")
    if not isinstance(value["parser_version"], str) or not value["parser_version"]:
        raise FullSearchV3FreezeError("SEARCH parser identity is invalid")
    if not isinstance(value["proposer_identity"], Mapping):
        raise FullSearchV3FreezeError("SEARCH proposer identity is invalid")
    try:
        return _json_copy(value)
    except (TypeError, ValueError) as exc:
        raise FullSearchV3FreezeError("SEARCH run identity is not canonical JSON data") from exc


def _validated_report(value: Any, candidate_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullSearchV3FreezeError("SEARCH candidate report is missing")
    report = _json_copy(value)
    ranked = eligible_full_search_v3_candidate(report)
    if ranked is None or ranked.candidate_id != candidate_id:
        raise FullSearchV3FreezeError("SEARCH candidate report is not fully rankable")
    return report


def _template_sources(value: Any, context: str) -> dict[str, str]:
    try:
        family = PromptFamily(value)
    except Exception as exc:
        raise FullSearchV3FreezeError(f"{context} are invalid") from exc
    return {method: family[method].template for method in TEMPLATE_KEYS}


def _validate_frozen_winner(value: Any, prompt_sha256: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"candidate_id", "source_kind", "rank"}:
        raise FullSearchV3FreezeError("frozen V3 winner fields are invalid")
    _safe_identifier(value["candidate_id"])
    if value["source_kind"] not in {"canonical_cot", "search_candidate"} or value["rank"] != 1:
        raise FullSearchV3FreezeError("frozen V3 winner source is invalid")
    if value["candidate_id"] == FULL_SEARCH_V3_P0_CANDIDATE_ID:
        if value["source_kind"] != "canonical_cot" or prompt_sha256 != canonical_full_search_v3_p0_prompt_sha256():
            raise FullSearchV3FreezeError("frozen V3 cot winner is inconsistent")
    elif value["source_kind"] != "search_candidate":
        raise FullSearchV3FreezeError("frozen V3 candidate winner source is invalid")


def _validate_search_metrics(value: Any) -> None:
    required = {
        "macro_f1", "accuracy", "parse_coverage", "total_records", "completed_solver_responses",
        "parsed_predictions", "abstentions_or_parse_failures", "infrastructure_failures", "ranking",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FullSearchV3FreezeError("frozen V3 SEARCH metrics are invalid")
    ranked = []
    for item in value["ranking"] if isinstance(value["ranking"], list) else ():
        if not isinstance(item, Mapping) or set(item) != {"candidate_id", "prompt_sha256", "macro_f1", "accuracy"}:
            raise FullSearchV3FreezeError("frozen V3 ranking is invalid")
        ranked.append(FullSearchV3RankedCandidate(**item))
    if not ranked or tuple(ranked) != tuple(sorted(ranked, key=lambda item: (-item.macro_f1, -item.accuracy, item.prompt_sha256, item.candidate_id))):
        raise FullSearchV3FreezeError("frozen V3 ranking order is invalid")
    winner = ranked[0]
    if value["macro_f1"] != winner.macro_f1 or value["accuracy"] != winner.accuracy:
        raise FullSearchV3FreezeError("frozen V3 winner metrics are inconsistent")
    if value["parse_coverage"] != 1.0 or any(value[field] != 1000 for field in (
        "total_records", "completed_solver_responses", "parsed_predictions"
    )) or any(value[field] != 0 for field in ("abstentions_or_parse_failures", "infrastructure_failures")):
        raise FullSearchV3FreezeError("frozen V3 SEARCH completeness metrics are invalid")


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, FullSearchV3FreezeError) as exc:
        raise FullSearchV3FreezeError(f"{label} is unreadable") from exc
    if not isinstance(value, dict) or encoded != canonical_json(value).encode("utf-8"):
        raise FullSearchV3FreezeError(f"{label} is not canonical")
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(value).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".freeze.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise FullSearchV3FrozenArtifactConflictError(
                "a V3 winner was frozen concurrently"
            )
        os.replace(temporary, path)
    except FullSearchV3FreezeError:
        raise
    except OSError as exc:
        raise FullSearchV3FreezeError("unable to atomically freeze V3 winner") from exc
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _freeze_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FullSearchV3FreezeError("another process owns this V3 freeze") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _run_directory(repository_root: str | Path, run_id: str) -> Path:
    return Path(repository_root) / "workspace" / "meta_harness" / "full_search_v3" / _safe_identifier(run_id)


def _safe_identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FullSearchV3FreezeError("run or candidate identifier is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FullSearchV3FreezeError(f"{field} must be a lowercase SHA-256")
    return value


def _patience_mapping(value: FullSearchV3PatienceState) -> dict[str, Any]:
    return {
        "best_macro_f1": value.best_macro_f1,
        "best_accuracy": value.best_accuracy,
        "consecutive_non_improving": value.consecutive_non_improving,
    }


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> dict[str, Any]:
    return json.loads(canonical_json(value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullSearchV3FreezeError("duplicate JSON key")
        result[key] = value
    return result


__all__ = [
    "FULL_SEARCH_V3_FREEZE_SCHEMA_VERSION",
    "FULL_SEARCH_V3_FREEZE_VARIANT",
    "FULL_SEARCH_V3_FREEZE_VERSION",
    "FullSearchV3FreezeError",
    "FullSearchV3FrozenArtifactConflictError",
    "freeze_full_search_v3_winner",
    "full_search_v3_frozen_winner_path",
    "load_full_search_v3_frozen_winner",
]
