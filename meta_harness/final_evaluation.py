"""Isolated, offline planning and resumable state for full-SEARCH FINAL.

Only this FINAL boundary imports the trusted private-manifest loader.  It
validates private membership and ordered records without constructing a solver
client, reading credentials, or retaining records, labels, or sample IDs in
the resulting preflight summary or durable state.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from meta_harness.config import (
    EXPERIMENT_FINAL_SIZE,
    EXPERIMENT_PROTOCOL_ID,
    canonical_experiment_config,
)
from meta_harness.search_evaluator import (
    EXPERIMENT_P0_CANDIDATE_ID,
    canonical_experiment_p0_prompt_sha256,
)
from meta_harness.winner_freeze import (
    FreezeError,
    experiment_frozen_winner_path,
    load_experiment_frozen_winner,
)
from meta_harness.run_identity import validate_run_identity
from meta_harness.preparation import (
    PreparationError,
    load_experiment_search_safe_manifest,
    load_trusted_experiment_private_manifest,
    verify_experiment_search_safe_manifest,
)
from meta_harness.solver import SolverGenerationSettings
from meta_harness.solver import (
    SolverClient,
    build_solver_request,
    solver_request_payload_sha256,
)
from meta_harness.retry import (
    SolverExecutionFailure,
    SolverRetryPolicy,
    execute_solver_request_with_retry,
)
from meta_harness.prompt_family import (
    TEMPLATE_KEYS,
    PromptFamily,
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)
from utils.answer_parser import PARSER_VERSION
from utils.answer_parser import parse_answer
from utils.constant import COT_PROMPT


EXPERIMENT_FINAL_PLAN_SCHEMA_VERSION = "sciver_full_search_v3_final_plan_v1"
EXPERIMENT_FINAL_STATE_SCHEMA_VERSION = "sciver_full_search_v3_final_state_v1"
EXPERIMENT_FINAL_STATE_VERSION = "sciver_full_search_v3_final_state_v2"
_STATE_FILENAME = "final_state.json"
_RECEIPT_FILENAME = "completion.json"
_LOCK_FILENAME = ".final.lock"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FinalError(RuntimeError):
    """Raised when FINAL planning cannot safely cross the frozen boundary."""


class FinalAuthorizationError(FinalError):
    """Raised when durable FINAL work lacks its own explicit authorization."""


class FinalIdentityError(FinalError):
    """Raised when a FINAL state cannot resume under the supplied identity."""


def account_final_evaluation_outcomes(
    *,
    gold_labels: Sequence[str],
    parsed_predictions: Sequence[str | None],
    infrastructure_failures: Sequence[bool],
) -> dict[str, Any]:
    """Return aggregate-only binary FINAL metrics for aligned in-memory outcomes.

    This pure helper deliberately accepts no sample identifiers or model text;
    callers can checkpoint its aggregate result without retaining FINAL rows.
    """

    if not (
        len(gold_labels) == len(parsed_predictions) == len(infrastructure_failures)
    ):
        raise FinalError("FINAL outcome counts must be aligned")
    confusion = {"yes": {"yes": 0, "no": 0}, "no": {"yes": 0, "no": 0}}
    label_support = {"yes": 0, "no": 0}
    parsed = 0
    failed = 0
    for gold, prediction, infrastructure_failure in zip(
        gold_labels, parsed_predictions, infrastructure_failures
    ):
        if gold not in {"yes", "no"} or not isinstance(infrastructure_failure, bool):
            raise FinalError("FINAL outcomes must use binary labels")
        label_support[gold] += 1
        if infrastructure_failure:
            if prediction is not None:
                raise FinalError("failed FINAL outcome cannot have a prediction")
            failed += 1
        elif prediction is not None:
            if prediction not in {"yes", "no"}:
                raise FinalError("FINAL prediction must be binary")
            confusion[gold][prediction] += 1
            parsed += 1
    total = len(gold_labels)
    correct = sum(confusion[label][label] for label in ("yes", "no"))
    f1_scores = []
    for label in ("yes", "no"):
        true_positive = confusion[label][label]
        false_positive = confusion["no" if label == "yes" else "yes"][label]
        false_negative = label_support[label] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "confusion": confusion,
        "parsed_predictions": parsed,
        "abstentions_or_parse_failures": total - parsed - failed,
        "infrastructure_failures": failed,
        "metrics": {
            "accuracy": None if total == 0 else correct / total,
            "macro_f1": sum(f1_scores) / len(f1_scores),
            "parse_coverage": None if total == 0 else parsed / total,
        },
    }


@dataclass(frozen=True)
class FinalSolverContract:
    """Non-secret solver/parser contract required before FINAL state creation."""

    solver_identity_sha256: str
    model: str
    generation: Mapping[str, Any]
    parser_version: str

    def __post_init__(self) -> None:
        _sha256(self.solver_identity_sha256, "solver_identity_sha256")
        expected = canonical_experiment_config()
        if self.model != expected.solver_model:
            raise FinalError("FINAL solver model is incompatible")
        if dict(self.generation) != SolverGenerationSettings.from_config(
            expected
        ).as_request_options():
            raise FinalError("FINAL solver generation is incompatible")
        if self.parser_version != PARSER_VERSION:
            raise FinalError("FINAL parser version is incompatible")

    def as_dict(self) -> dict[str, Any]:
        return {
            "solver_identity_sha256": self.solver_identity_sha256,
            "model": self.model,
            "generation": dict(self.generation),
            "parser_version": self.parser_version,
        }


def canonical_final_evaluation_solver_contract(
    *, solver_identity_sha256: str
) -> FinalSolverContract:
    """Build the fixed non-secret contract without constructing a live client."""

    config = canonical_experiment_config()
    return FinalSolverContract(
        solver_identity_sha256=solver_identity_sha256,
        model=config.solver_model,
        generation=SolverGenerationSettings.from_config(config).as_request_options(),
        parser_version=PARSER_VERSION,
    )


def preflight_final_evaluation(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FinalSolverContract,
) -> dict[str, Any]:
    """Validate FINAL inputs with no credentials, state mutation, or dispatch.

    The returned summary has only public/hashed identities.  It deliberately
    omits private paths, FINAL membership, labels, records, availability, and
    metrics.
    """

    identity = _build_execution_identity(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=frozen_winner_path,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=solver_contract,
    )
    return {
        "schema_version": EXPERIMENT_FINAL_PLAN_SCHEMA_VERSION,
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "run_id": _run_id(run_id),
        "execution_identity_sha256": _sha256_json(identity),
        "p0_prompt_sha256": identity["p0"]["prompt_sha256"],
        "p_star_prompt_sha256": identity["p_star"]["prompt_sha256"],
    }


def initialize_final_evaluation(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FinalSolverContract,
    authorize_final_execution: bool,
) -> dict[str, Any]:
    """Create or resume a FINAL-only planned state after explicit authorization.

    This authorization is intentionally unrelated to SEARCH and does not
    enable any solver work.  Dispatch is not implemented by this module.
    """

    if authorize_final_execution is not True:
        raise FinalAuthorizationError(
            "FINAL state creation requires explicit FINAL authorization"
        )
    identity = _build_execution_identity(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=frozen_winner_path,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=solver_contract,
    )
    state = _planned_state(identity)
    destination = _final_state_path(repository_root, run_id)
    with _final_lock(destination.parent / _LOCK_FILENAME):
        if destination.exists():
            existing = load_final_evaluation_state(destination)
            if existing["identity"] != identity:
                raise FinalIdentityError(
                    "existing FINAL state has an incompatible execution identity"
                )
            return existing
        _atomic_create_json(destination, state)
    return load_final_evaluation_state(destination)


def execute_final_evaluation(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FinalSolverContract,
    solver: SolverClient,
    authorize_final_execution: bool,
    retry_policy: SolverRetryPolicy | None = None,
) -> dict[str, Any]:
    """Run the frozen paired FINAL plan through an explicitly injected client.

    This function never constructs a client or reads credentials.  It records
    only deterministic request hashes, so interrupted work resumes without
    redispatching completed calls and without persisting FINAL records, IDs,
    labels, prompts, or model text.
    """

    if authorize_final_execution is not True:
        raise FinalAuthorizationError(
            "FINAL execution requires explicit FINAL authorization"
        )
    active_retry_policy = retry_policy or SolverRetryPolicy()
    if not isinstance(active_retry_policy, SolverRetryPolicy):
        raise TypeError("retry_policy must be SolverRetryPolicy")
    if not callable(getattr(solver, "complete", None)):
        raise TypeError("solver must provide a complete method")
    identity = _build_execution_identity(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=frozen_winner_path,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=solver_contract,
    )
    initialize_final_evaluation(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=frozen_winner_path,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=solver_contract,
        authorize_final_execution=True,
    )
    destination = _final_state_path(repository_root, run_id)
    receipt_path = destination.parent / _RECEIPT_FILENAME
    with _final_lock(destination.parent / _LOCK_FILENAME):
        state = load_final_evaluation_state(destination)
        if state["identity"] != identity:
            raise FinalIdentityError(
                "FINAL state has an incompatible execution identity"
            )
        if receipt_path.exists():
            receipt = load_final_evaluation_completion_receipt(receipt_path)
            _validate_receipt_identity(receipt, identity)
            if state["status"] != "complete":
                raise FinalIdentityError(
                    "FINAL completion receipt conflicts with incomplete state"
                )
            return receipt
        if _migrate_zero_completion_legacy_state(state):
            _atomic_replace_json(destination, state)
        frozen = load_experiment_frozen_winner(
            frozen_winner_path
            if frozen_winner_path is not None
            else experiment_frozen_winner_path(repository_root, run_id)
        )
        prompts = {
            "cot": COT_PROMPT,
            "meta_cot": PromptFamily(frozen["templates"]),
        }
        for variant in state["variants"]:
            prompt_variant = variant["prompt_variant"]
            expected_hashes = _final_request_hashes(
                identity=identity,
                records=final_records,
                prompt=prompts[prompt_variant],
                candidate_id=variant["candidate_id"],
            )
            completed_hashes = variant["completed_request_sha256"]
            if completed_hashes != expected_hashes[: len(completed_hashes)]:
                raise FinalIdentityError(
                    "FINAL checkpoint does not match immutable request order"
                )
            if len(completed_hashes) and "outcomes" not in variant:
                raise FinalIdentityError(
                    "legacy FINAL state cannot resume without aggregate outcomes"
                )
            for index in range(len(completed_hashes), len(expected_hashes)):
                request = build_solver_request(final_records[index], prompts[prompt_variant])
                try:
                    result = execute_solver_request_with_retry(
                        solver, request, policy=active_retry_policy
                    )
                    parsed = parse_answer(result.content)
                    prediction = (
                        parsed["prediction"]
                        if parsed["parse_status"] == "parsed"
                        else None
                    )
                    infrastructure_failure = False
                except SolverExecutionFailure:
                    prediction = None
                    infrastructure_failure = True
                _account_final_outcome(
                    variant["outcomes"],
                    gold_label=final_records[index].get("gold_label"),
                    prediction=prediction,
                    infrastructure_failure=infrastructure_failure,
                )
                variant["metrics"] = _metrics_from_final_accounting(variant["outcomes"])
                completed_hashes.append(expected_hashes[index])
                state["status"] = "running"
                _refresh_state_hash(state)
                _atomic_replace_json(destination, state)
        state["status"] = "complete"
        _refresh_state_hash(state)
        _atomic_replace_json(destination, state)
        receipt = _completion_receipt(identity, state["variants"])
        _atomic_create_json(receipt_path, receipt)
    return load_final_evaluation_completion_receipt(receipt_path)


def final_evaluation_state_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the FINAL-only state location for a run."""

    return _final_state_path(repository_root, run_id)


def load_final_evaluation_state(path: str | Path) -> dict[str, Any]:
    """Load a safe FINAL planning state without exposing private membership."""

    state = _load_canonical_object(Path(path), "FINAL state")
    required = {
        "schema_version", "state_version", "identity", "status", "variants", "state_sha256"
    }
    if set(state) != required:
        raise FinalIdentityError("FINAL state fields are invalid")
    if state["schema_version"] != EXPERIMENT_FINAL_STATE_SCHEMA_VERSION or state[
        "state_version"
    ] != EXPERIMENT_FINAL_STATE_VERSION:
        raise FinalIdentityError("FINAL state schema is incompatible")
    if state["status"] not in {"planned", "running", "complete"}:
        raise FinalIdentityError("FINAL state status is invalid")
    _validate_execution_identity(state["identity"])
    _validate_variants(state["variants"], state["identity"])
    expected = _sha256_json({key: value for key, value in state.items() if key != "state_sha256"})
    if state["state_sha256"] != expected:
        raise FinalIdentityError("FINAL state hash is inconsistent")
    return _json_copy(state)


def final_evaluation_completion_receipt_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the create-once paired FINAL completion receipt path."""

    return _final_state_path(repository_root, run_id).parent / _RECEIPT_FILENAME


def load_final_evaluation_completion_receipt(path: str | Path) -> dict[str, Any]:
    """Load a receipt containing only safe paired-execution identities."""

    receipt = _load_canonical_object(Path(path), "FINAL completion receipt")
    required = {
        "schema_version", "identity_sha256", "status", "variants", "logical_calls", "receipt_sha256"
    }
    if set(receipt) != required or receipt["schema_version"] != EXPERIMENT_FINAL_STATE_SCHEMA_VERSION:
        raise FinalIdentityError("FINAL completion receipt fields are invalid")
    if receipt["status"] != "complete" or receipt["logical_calls"] != 2 * EXPERIMENT_FINAL_SIZE:
        raise FinalIdentityError("FINAL completion receipt is incomplete")
    _sha256(receipt["identity_sha256"], "FINAL receipt identity_sha256")
    _validate_receipt_variants(receipt["variants"])
    expected = _sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    if receipt["receipt_sha256"] != expected:
        raise FinalIdentityError("FINAL completion receipt hash is inconsistent")
    return _json_copy(receipt)


def _build_execution_identity(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FinalSolverContract,
) -> dict[str, Any]:
    safe_run_id = _run_id(run_id)
    if not isinstance(solver_contract, FinalSolverContract):
        raise TypeError("solver_contract must be FinalSolverContract")
    artifact_path = (
        Path(frozen_winner_path)
        if frozen_winner_path is not None
        else experiment_frozen_winner_path(repository_root, safe_run_id)
    )
    try:
        frozen = load_experiment_frozen_winner(artifact_path)
    except FreezeError as exc:
        raise FinalError("FINAL requires a valid immutable frozen winner") from exc
    if frozen["run_id"] != safe_run_id:
        raise FinalError("frozen winner belongs to a different run")
    private_manifest, safe_manifest = _load_matching_manifests(
        private_manifest_path, search_safe_manifest_path
    )
    ordered_membership_sha256 = _validate_final_records(final_records, private_manifest)
    search_identity = _load_search_identity(repository_root, safe_run_id)
    _validate_cross_stage_identity(
        frozen=frozen,
        private_manifest=private_manifest,
        search_identity=search_identity,
        solver_contract=solver_contract,
    )
    identity = {
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "run_id": safe_run_id,
        "config_sha256": private_manifest["config_sha256"],
        "split_sha256": private_manifest["split"]["split_sha256"],
        "final_membership_commitment": private_manifest["final_membership_commitment"],
        "final_record_order_sha256": ordered_membership_sha256,
        "final_size": EXPERIMENT_FINAL_SIZE,
        "frozen_winner_artifact_sha256": frozen["artifact_sha256"],
        "search_run_identity_sha256": frozen["hashes"]["run_identity_sha256"],
        "solver": solver_contract.as_dict(),
        "p0": {
            "candidate_id": EXPERIMENT_P0_CANDIDATE_ID,
            "prompt_sha256": canonical_experiment_p0_prompt_sha256(),
        },
        "p_star": {
            "candidate_id": frozen["winner"]["candidate_id"],
            "prompt_sha256": frozen["hashes"]["prompt_sha256"],
            "prompt_variant": frozen["prompt_variant"],
        },
    }
    _validate_execution_identity(identity)
    return identity


def _load_matching_manifests(
    private_manifest_path: str | Path, search_safe_manifest_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        private_manifest = load_trusted_experiment_private_manifest(
            private_manifest_path
        )
        safe_manifest = load_experiment_search_safe_manifest(search_safe_manifest_path)
        verify_experiment_search_safe_manifest(safe_manifest, private_manifest)
    except PreparationError as exc:
        raise FinalError(
            "FINAL private and SEARCH-safe manifests are incompatible"
        ) from exc
    if safe_manifest["final_membership_commitment"] != private_manifest[
        "final_membership_commitment"
    ]:
        raise FinalError("FINAL membership commitment is incompatible")
    return private_manifest, safe_manifest


def _validate_final_records(
    records: Sequence[Mapping[str, Any]], private_manifest: Mapping[str, Any]
) -> str:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FinalError("FINAL records must be a sequence")
    expected_ids = tuple(private_manifest["split"]["FINAL"]["sample_ids"])
    if len(records) != EXPERIMENT_FINAL_SIZE:
        raise FinalError("FINAL records must contain exactly 1,000 rows")
    observed_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise FinalError("FINAL record must be an object")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FinalError("FINAL record has an invalid sample identity")
        observed_ids.append(sample_id)
    if len(observed_ids) != len(set(observed_ids)):
        raise FinalError("FINAL records contain duplicate sample identities")
    if tuple(observed_ids) != expected_ids:
        raise FinalError("FINAL records are not in immutable manifest order")
    return _sha256_json(list(observed_ids))


def _load_search_identity(repository_root: str | Path, run_id: str) -> dict[str, Any]:
    path = (
        Path(repository_root)
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / _run_id(run_id)
        / "orchestration_state.json"
    )
    state = _load_canonical_object(path, "SEARCH state")
    identity = state.get("identity")
    if not isinstance(identity, Mapping):
        raise FinalError("SEARCH run identity is missing")
    required = {
        "protocol_id", "config_sha256", "split_sha256", "search_membership_sha256",
        "sample_ids_sha256", "canonical_p0_prompt_sha256", "parser_version",
        "solver_identity_sha256", "proposer_identity",
    }
    if set(identity) != required or identity["protocol_id"] != EXPERIMENT_PROTOCOL_ID:
        raise FinalError("SEARCH run identity is incompatible")
    for field in required - {"protocol_id", "parser_version", "proposer_identity"}:
        _sha256(identity[field], f"SEARCH identity {field}")
    if identity["parser_version"] != PARSER_VERSION:
        raise FinalError("SEARCH parser identity is incompatible")
    return _json_copy(identity)


def _validate_cross_stage_identity(
    *,
    frozen: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    search_identity: Mapping[str, Any],
    solver_contract: FinalSolverContract,
) -> None:
    hashes = frozen["hashes"]
    if hashes["config_sha256"] != private_manifest["config_sha256"] or hashes[
        "split_sha256"
    ] != private_manifest["split"]["split_sha256"]:
        raise FinalError("frozen winner and private split are incompatible")
    if hashes["run_identity_sha256"] != _sha256_json(search_identity):
        raise FinalError("frozen winner and SEARCH run identity are incompatible")
    if search_identity["config_sha256"] != private_manifest["config_sha256"] or search_identity[
        "split_sha256"
    ] != private_manifest["split"]["split_sha256"]:
        raise FinalError("SEARCH run and private split are incompatible")
    if search_identity["solver_identity_sha256"] != solver_contract.solver_identity_sha256:
        raise FinalError("FINAL solver identity is incompatible with SEARCH")
    if search_identity["canonical_p0_prompt_sha256"] != canonical_experiment_p0_prompt_sha256():
        raise FinalError("SEARCH P0 identity is incompatible")
    templates = {method: canonical_baseline_sources()[method] for method in TEMPLATE_KEYS}
    if template_source_sha256(templates) != canonical_experiment_p0_prompt_sha256():
        raise FinalError("canonical P0 prompt identity is inconsistent")
    if frozen["winner"]["candidate_id"] == EXPERIMENT_P0_CANDIDATE_ID:
        if frozen["hashes"]["prompt_sha256"] != canonical_experiment_p0_prompt_sha256():
            raise FinalError("frozen P0 winner identity is inconsistent")
    elif template_source_sha256(PromptFamily(frozen["templates"])) != frozen["hashes"]["prompt_sha256"]:
        raise FinalError("frozen P* prompt identity is inconsistent")


def _planned_state(identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": EXPERIMENT_FINAL_STATE_SCHEMA_VERSION,
        "state_version": EXPERIMENT_FINAL_STATE_VERSION,
        "identity": _json_copy(identity),
        "status": "planned",
        "variants": [
            {
                "prompt_variant": "cot",
                "candidate_id": identity["p0"]["candidate_id"],
                "prompt_sha256": identity["p0"]["prompt_sha256"],
                "completed_request_sha256": [],
                "outcomes": _empty_final_accounting(),
                "metrics": _metrics_from_final_accounting(_empty_final_accounting()),
            },
            {
                "prompt_variant": "meta_cot",
                "candidate_id": identity["p_star"]["candidate_id"],
                "prompt_sha256": identity["p_star"]["prompt_sha256"],
                "completed_request_sha256": [],
                "outcomes": _empty_final_accounting(),
                "metrics": _metrics_from_final_accounting(_empty_final_accounting()),
            },
        ],
    }
    return {**payload, "state_sha256": _sha256_json(payload)}


def _migrate_zero_completion_legacy_state(state: dict[str, Any]) -> bool:
    """Add aggregate accounting only to an untouched legacy FINAL plan.

    FINAL records and outcomes were intentionally absent from the former state
    format, so any completed legacy request cannot be reconstructed safely.
    A zero-completion planned state has no such ambiguity and can receive the
    empty aggregate counters before its first dispatch.
    """

    legacy_variants = [
        variant
        for variant in state["variants"]
        if "outcomes" not in variant and "metrics" not in variant
    ]
    if not legacy_variants:
        return False
    if (
        len(legacy_variants) != len(state["variants"])
        or state["status"] != "planned"
        or any(variant["completed_request_sha256"] for variant in state["variants"])
    ):
        raise FinalIdentityError(
            "legacy FINAL state cannot resume without aggregate outcomes"
        )
    for variant in state["variants"]:
        accounting = _empty_final_accounting()
        variant["outcomes"] = accounting
        variant["metrics"] = _metrics_from_final_accounting(accounting)
    _refresh_state_hash(state)
    return True


def _validate_execution_identity(value: Any) -> None:
    required = {
        "protocol_id", "run_id", "config_sha256", "split_sha256", "final_membership_commitment",
        "final_record_order_sha256", "final_size", "frozen_winner_artifact_sha256",
        "search_run_identity_sha256", "solver", "p0", "p_star",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FinalIdentityError("FINAL execution identity fields are invalid")
    if value["protocol_id"] != EXPERIMENT_PROTOCOL_ID or value["final_size"] != EXPERIMENT_FINAL_SIZE:
        raise FinalIdentityError("FINAL execution identity is incompatible")
    _run_id(value["run_id"])
    for field in required - {"protocol_id", "run_id", "final_size", "solver", "p0", "p_star"}:
        _sha256(value[field], f"FINAL identity {field}")
    contract = FinalSolverContract(**value["solver"])
    _validate_prompt_identity(value["p0"], EXPERIMENT_P0_CANDIDATE_ID)
    _validate_prompt_identity(value["p_star"], None)
    if value["p0"]["prompt_sha256"] != canonical_experiment_p0_prompt_sha256() or value[
        "p_star"
    ]["prompt_variant"] != "meta_cot":
        raise FinalIdentityError("FINAL prompt identities are incompatible")
    if not isinstance(contract, FinalSolverContract):
        raise AssertionError("validated contract unexpectedly has the wrong type")


def _validate_prompt_identity(value: Any, candidate_id: str | None) -> None:
    if not isinstance(value, Mapping):
        raise FinalIdentityError("FINAL prompt identity is invalid")
    expected = {"candidate_id", "prompt_sha256"}
    if candidate_id is None:
        expected = {"candidate_id", "prompt_sha256", "prompt_variant"}
    if set(value) != expected or (candidate_id is not None and value["candidate_id"] != candidate_id):
        raise FinalIdentityError("FINAL prompt identity is invalid")
    _identifier(value["candidate_id"])
    _sha256(value["prompt_sha256"], "FINAL prompt_sha256")


def _validate_variants(value: Any, identity: Mapping[str, Any]) -> None:
    expected = [
        {
            "prompt_variant": "cot",
            **identity["p0"],
            "completed_request_sha256": None,
        },
        {
            "prompt_variant": "meta_cot",
            "candidate_id": identity["p_star"]["candidate_id"],
            "prompt_sha256": identity["p_star"]["prompt_sha256"],
            "completed_request_sha256": None,
        },
    ]
    if not isinstance(value, list) or len(value) != 2:
        raise FinalIdentityError("FINAL paired variants are incompatible")
    for actual, template in zip(value, expected):
        completed = actual.get("completed_request_sha256") if isinstance(actual, Mapping) else None
        extra = (
            {key: item for key, item in actual.items() if key != "completed_request_sha256"}
            if isinstance(actual, Mapping)
            else None
        )
        expected_extra = {key: item for key, item in template.items() if key != "completed_request_sha256"}
        has_accounting = isinstance(actual, Mapping) and {"outcomes", "metrics"} <= set(actual)
        if (
            not isinstance(actual, Mapping)
            or (extra != expected_extra and not (
                has_accounting
                and {key: item for key, item in actual.items() if key not in {"completed_request_sha256", "outcomes", "metrics"}}
                == expected_extra
            ))
            or not isinstance(completed, list)
            or len(completed) > EXPERIMENT_FINAL_SIZE
            or len(completed) != len(set(completed))
        ):
            raise FinalIdentityError("FINAL paired variants are incompatible")
        for item in completed:
            _sha256(item, "FINAL completed request hash")
        if has_accounting:
            _validate_final_accounting(actual["outcomes"], len(completed))
            if actual["metrics"] != _metrics_from_final_accounting(actual["outcomes"]):
                raise FinalIdentityError("FINAL aggregate metrics are inconsistent")


def _final_request_hashes(
    *,
    identity: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    prompt: Mapping[str, Any],
    candidate_id: str,
) -> list[str]:
    identity_sha256 = _sha256_json(identity)
    result: list[str] = []
    for index, record in enumerate(records):
        request = build_solver_request(record, prompt)
        result.append(
            _sha256_json(
                {
                    "execution_identity_sha256": identity_sha256,
                    "candidate_id": candidate_id,
                    "order": index,
                    "request_payload_sha256": solver_request_payload_sha256(request),
                }
            )
        )
    return result


def _empty_final_accounting() -> dict[str, Any]:
    return {
        "confusion": {"yes": {"yes": 0, "no": 0}, "no": {"yes": 0, "no": 0}},
        "label_support": {"yes": 0, "no": 0},
        "parsed_predictions": 0,
        "abstentions_or_parse_failures": 0,
        "infrastructure_failures": 0,
    }


def _account_final_outcome(
    accounting: dict[str, Any],
    *,
    gold_label: Any,
    prediction: str | None,
    infrastructure_failure: bool,
) -> None:
    _validate_final_accounting(accounting, sum(accounting["label_support"].values()))
    if gold_label not in {"yes", "no"}:
        raise FinalError("FINAL record has no binary gold label for metrics")
    if not isinstance(infrastructure_failure, bool):
        raise FinalError("FINAL infrastructure outcome is invalid")
    accounting["label_support"][gold_label] += 1
    if infrastructure_failure:
        if prediction is not None:
            raise FinalError("failed FINAL outcome cannot have a prediction")
        accounting["infrastructure_failures"] += 1
        return
    if prediction is None:
        accounting["abstentions_or_parse_failures"] += 1
        return
    if prediction not in {"yes", "no"}:
        raise FinalError("FINAL prediction must be binary")
    accounting["confusion"][gold_label][prediction] += 1
    accounting["parsed_predictions"] += 1


def _metrics_from_final_accounting(accounting: Mapping[str, Any]) -> dict[str, Any]:
    _validate_final_accounting(accounting, sum(accounting.get("label_support", {}).values()))
    total = sum(accounting["label_support"].values())
    confusion = accounting["confusion"]
    correct = sum(confusion[label][label] for label in ("yes", "no"))
    f1_scores = []
    for label in ("yes", "no"):
        true_positive = confusion[label][label]
        false_positive = confusion["no" if label == "yes" else "yes"][label]
        false_negative = accounting["label_support"][label] - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": None if total == 0 else correct / total,
        "macro_f1": sum(f1_scores) / len(f1_scores),
        "parse_coverage": None if total == 0 else accounting["parsed_predictions"] / total,
    }


def _validate_final_accounting(value: Any, completed_count: int) -> None:
    required = {
        "confusion", "label_support", "parsed_predictions",
        "abstentions_or_parse_failures", "infrastructure_failures",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FinalIdentityError("FINAL aggregate accounting is invalid")
    confusion = value["confusion"]
    support = value["label_support"]
    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != {"yes", "no"}
        or not isinstance(support, Mapping)
        or set(support) != {"yes", "no"}
    ):
        raise FinalIdentityError("FINAL aggregate accounting is invalid")
    counts = [value["parsed_predictions"], value["abstentions_or_parse_failures"], value["infrastructure_failures"]]
    for label in ("yes", "no"):
        row = confusion[label]
        if not isinstance(row, Mapping) or set(row) != {"yes", "no"}:
            raise FinalIdentityError("FINAL aggregate accounting is invalid")
        counts.extend((support[label], row["yes"], row["no"]))
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise FinalIdentityError("FINAL aggregate accounting is invalid")
    if sum(support.values()) != completed_count:
        raise FinalIdentityError("FINAL aggregate accounting is incomplete")
    if sum(sum(confusion[label].values()) for label in ("yes", "no")) != value["parsed_predictions"]:
        raise FinalIdentityError("FINAL aggregate parsed count is inconsistent")
    if sum(counts[:3]) != completed_count:
        raise FinalIdentityError("FINAL aggregate outcome count is inconsistent")


def _validate_final_metrics(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"accuracy", "macro_f1", "parse_coverage"}:
        raise FinalIdentityError("FINAL aggregate metrics are invalid")
    for field in ("accuracy", "parse_coverage"):
        metric = value[field]
        if metric is not None and (isinstance(metric, bool) or not isinstance(metric, (int, float)) or not 0 <= metric <= 1):
            raise FinalIdentityError("FINAL aggregate metrics are invalid")
    macro_f1 = value["macro_f1"]
    if isinstance(macro_f1, bool) or not isinstance(macro_f1, (int, float)) or not 0 <= macro_f1 <= 1:
        raise FinalIdentityError("FINAL aggregate metrics are invalid")


def _safe_final_outcome_counts(accounting: Mapping[str, Any]) -> dict[str, Any]:
    _validate_final_accounting(accounting, sum(accounting["label_support"].values()))
    return {
        "confusion": _json_copy(accounting["confusion"]),
        "parsed_predictions": accounting["parsed_predictions"],
        "abstentions_or_parse_failures": accounting["abstentions_or_parse_failures"],
        "infrastructure_failures": accounting["infrastructure_failures"],
    }


def _validate_safe_final_outcome_counts(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "confusion", "parsed_predictions", "abstentions_or_parse_failures", "infrastructure_failures",
    }:
        raise FinalIdentityError("FINAL receipt outcome counts are invalid")
    counts = [
        value["parsed_predictions"],
        value["abstentions_or_parse_failures"],
        value["infrastructure_failures"],
    ]
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts):
        raise FinalIdentityError("FINAL receipt outcome counts are invalid")
    confusion = value["confusion"]
    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != {"yes", "no"}
        or any(
            not isinstance(confusion[label], Mapping)
            or set(confusion[label]) != {"yes", "no"}
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in confusion[label].values()
            )
            for label in ("yes", "no")
        )
        or sum(sum(confusion[label].values()) for label in ("yes", "no"))
        != value["parsed_predictions"]
        or sum(counts) != EXPERIMENT_FINAL_SIZE
    ):
        raise FinalIdentityError("FINAL receipt outcome counts are invalid")


def _completion_receipt(
    identity: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    receipt_variants = [
        {
            "prompt_variant": variant["prompt_variant"],
            "candidate_id": variant["candidate_id"],
            "prompt_sha256": variant["prompt_sha256"],
            "completed_request_count": len(variant["completed_request_sha256"]),
            "completed_request_hashes_sha256": _sha256_json(
                variant["completed_request_sha256"]
            ),
            "metrics": _metrics_from_final_accounting(variant["outcomes"]),
            "outcomes": _safe_final_outcome_counts(variant["outcomes"]),
        }
        for variant in variants
    ]
    payload = {
        "schema_version": EXPERIMENT_FINAL_STATE_SCHEMA_VERSION,
        "identity_sha256": _sha256_json(identity),
        "status": "complete",
        "variants": receipt_variants,
        "logical_calls": 2 * EXPERIMENT_FINAL_SIZE,
    }
    return {**payload, "receipt_sha256": _sha256_json(payload)}


def _validate_receipt_variants(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise FinalIdentityError("FINAL receipt variants are invalid")
    expected_names = ("cot", "meta_cot")
    for variant, expected_name in zip(value, expected_names):
        required = {
            "prompt_variant", "candidate_id", "prompt_sha256", "completed_request_count", "completed_request_hashes_sha256"
        }
        if not isinstance(variant, Mapping) or set(variant) not in (
            required,
            required | {"metrics", "outcomes"},
        ):
            raise FinalIdentityError("FINAL receipt variants are invalid")
        if variant["prompt_variant"] != expected_name or variant["completed_request_count"] != EXPERIMENT_FINAL_SIZE:
            raise FinalIdentityError("FINAL receipt variants are incomplete")
        _identifier(variant["candidate_id"])
        _sha256(variant["prompt_sha256"], "FINAL receipt prompt_sha256")
        _sha256(variant["completed_request_hashes_sha256"], "FINAL receipt request hashes")
        if "metrics" in variant:
            _validate_final_metrics(variant["metrics"])
            _validate_safe_final_outcome_counts(variant["outcomes"])


def _validate_receipt_identity(receipt: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if receipt["identity_sha256"] != _sha256_json(identity):
        raise FinalIdentityError("FINAL completion receipt identity is incompatible")


def _refresh_state_hash(state: dict[str, Any]) -> None:
    state["state_sha256"] = _sha256_json(
        {key: value for key, value in state.items() if key != "state_sha256"}
    )


def _final_state_path(repository_root: str | Path, run_id: str) -> Path:
    return (
        Path(repository_root)
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / _run_id(run_id)
        / "final"
        / _STATE_FILENAME
    )


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, FinalError) as exc:
        raise FinalError(f"{label} is unreadable") from exc
    if not isinstance(value, dict) or encoded != canonical_json(value).encode("utf-8"):
        raise FinalError(f"{label} is not canonical")
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(value).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".final.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            raise FinalIdentityError("a FINAL state was created concurrently")
        os.replace(temporary, path)
    except FinalError:
        raise
    except OSError as exc:
        raise FinalError("unable to atomically create FINAL state") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json(value).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".final.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise FinalError("unable to atomically update FINAL state") from exc
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _final_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FinalError("another process owns this FINAL run") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FinalError("FINAL identifier is invalid")
    return value


def _run_id(value: Any) -> str:
    try:
        return validate_run_identity(value)
    except ValueError as exc:
        raise FinalError(str(exc)) from exc


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FinalError(f"{field} must be a lowercase SHA-256")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> dict[str, Any]:
    return json.loads(canonical_json(value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalError("duplicate JSON key")
        result[key] = value
    return result


__all__ = [
    "EXPERIMENT_FINAL_PLAN_SCHEMA_VERSION",
    "EXPERIMENT_FINAL_STATE_SCHEMA_VERSION",
    "EXPERIMENT_FINAL_STATE_VERSION",
    "FinalAuthorizationError",
    "FinalError",
    "FinalIdentityError",
    "FinalSolverContract",
    "canonical_final_evaluation_solver_contract",
    "execute_final_evaluation",
    "final_evaluation_completion_receipt_path",
    "final_evaluation_state_path",
    "initialize_final_evaluation",
    "load_final_evaluation_completion_receipt",
    "load_final_evaluation_state",
    "preflight_final_evaluation",
]
