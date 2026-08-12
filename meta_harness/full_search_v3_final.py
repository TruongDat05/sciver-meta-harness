"""Isolated, offline planning and resumable state for full-SEARCH V3 FINAL.

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
    FULL_SEARCH_V3_FINAL_SIZE,
    FULL_SEARCH_V3_PROTOCOL_ID,
    canonical_full_search_v3_config,
)
from meta_harness.full_search_v3_evaluator import (
    FULL_SEARCH_V3_P0_CANDIDATE_ID,
    canonical_full_search_v3_p0_prompt_sha256,
)
from meta_harness.full_search_v3_freeze import (
    FullSearchV3FreezeError,
    full_search_v3_frozen_winner_path,
    load_full_search_v3_frozen_winner,
)
from meta_harness.full_search_v3_preparation import (
    FullSearchV3PreparationError,
    load_full_search_v3_search_safe_manifest,
    load_trusted_full_search_v3_private_manifest,
    verify_full_search_v3_search_safe_manifest,
)
from meta_harness.full_search_v3_solver import SolverGenerationSettings
from meta_harness.full_search_v3_solver import (
    SolverClient,
    build_solver_request,
    execute_solver_request,
    solver_request_payload_sha256,
)
from meta_harness.prompt_family import (
    TEMPLATE_KEYS,
    PromptFamily,
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)
from utils.answer_parser import PARSER_VERSION
from utils.constant import COT_PROMPT


FULL_SEARCH_V3_FINAL_PLAN_SCHEMA_VERSION = "sciver_full_search_v3_final_plan_v1"
FULL_SEARCH_V3_FINAL_STATE_SCHEMA_VERSION = "sciver_full_search_v3_final_state_v1"
FULL_SEARCH_V3_FINAL_STATE_VERSION = "sciver_full_search_v3_final_state_v2"
_STATE_FILENAME = "final_state.json"
_RECEIPT_FILENAME = "completion.json"
_LOCK_FILENAME = ".final.lock"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FullSearchV3FinalError(RuntimeError):
    """Raised when FINAL planning cannot safely cross the frozen boundary."""


class FullSearchV3FinalAuthorizationError(FullSearchV3FinalError):
    """Raised when durable FINAL work lacks its own explicit authorization."""


class FullSearchV3FinalIdentityError(FullSearchV3FinalError):
    """Raised when a FINAL state cannot resume under the supplied identity."""


@dataclass(frozen=True)
class FullSearchV3FinalSolverContract:
    """Non-secret solver/parser contract required before FINAL state creation."""

    solver_identity_sha256: str
    model: str
    generation: Mapping[str, Any]
    parser_version: str

    def __post_init__(self) -> None:
        _sha256(self.solver_identity_sha256, "solver_identity_sha256")
        expected = canonical_full_search_v3_config()
        if self.model != expected.solver_model:
            raise FullSearchV3FinalError("FINAL solver model is incompatible")
        if dict(self.generation) != SolverGenerationSettings.from_config(
            expected
        ).as_request_options():
            raise FullSearchV3FinalError("FINAL solver generation is incompatible")
        if self.parser_version != PARSER_VERSION:
            raise FullSearchV3FinalError("FINAL parser version is incompatible")

    def as_dict(self) -> dict[str, Any]:
        return {
            "solver_identity_sha256": self.solver_identity_sha256,
            "model": self.model,
            "generation": dict(self.generation),
            "parser_version": self.parser_version,
        }


def canonical_full_search_v3_final_solver_contract(
    *, solver_identity_sha256: str
) -> FullSearchV3FinalSolverContract:
    """Build the fixed non-secret contract without constructing a live client."""

    config = canonical_full_search_v3_config()
    return FullSearchV3FinalSolverContract(
        solver_identity_sha256=solver_identity_sha256,
        model=config.solver_model,
        generation=SolverGenerationSettings.from_config(config).as_request_options(),
        parser_version=PARSER_VERSION,
    )


def preflight_full_search_v3_final(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FullSearchV3FinalSolverContract,
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
        "schema_version": FULL_SEARCH_V3_FINAL_PLAN_SCHEMA_VERSION,
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "run_id": _identifier(run_id),
        "execution_identity_sha256": _sha256_json(identity),
        "p0_prompt_sha256": identity["p0"]["prompt_sha256"],
        "p_star_prompt_sha256": identity["p_star"]["prompt_sha256"],
    }


def initialize_full_search_v3_final(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FullSearchV3FinalSolverContract,
    authorize_final_execution: bool,
) -> dict[str, Any]:
    """Create or resume a FINAL-only planned state after explicit authorization.

    This authorization is intentionally unrelated to SEARCH and does not
    enable any solver work.  Dispatch is not implemented by this module.
    """

    if authorize_final_execution is not True:
        raise FullSearchV3FinalAuthorizationError(
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
            existing = load_full_search_v3_final_state(destination)
            if existing["identity"] != identity:
                raise FullSearchV3FinalIdentityError(
                    "existing FINAL state has an incompatible execution identity"
                )
            return existing
        _atomic_create_json(destination, state)
    return load_full_search_v3_final_state(destination)


def execute_full_search_v3_final(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FullSearchV3FinalSolverContract,
    solver: SolverClient,
    authorize_final_execution: bool,
) -> dict[str, Any]:
    """Run the frozen paired FINAL plan through an explicitly injected client.

    This function never constructs a client or reads credentials.  It records
    only deterministic request hashes, so interrupted work resumes without
    redispatching completed calls and without persisting FINAL records, IDs,
    labels, prompts, or model text.
    """

    if authorize_final_execution is not True:
        raise FullSearchV3FinalAuthorizationError(
            "FINAL execution requires explicit FINAL authorization"
        )
    complete = getattr(solver, "complete", None)
    if not callable(complete):
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
    initialize_full_search_v3_final(
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
        state = load_full_search_v3_final_state(destination)
        if state["identity"] != identity:
            raise FullSearchV3FinalIdentityError(
                "FINAL state has an incompatible execution identity"
            )
        if receipt_path.exists():
            receipt = load_full_search_v3_final_completion_receipt(receipt_path)
            _validate_receipt_identity(receipt, identity)
            if state["status"] != "complete":
                raise FullSearchV3FinalIdentityError(
                    "FINAL completion receipt conflicts with incomplete state"
                )
            return receipt
        frozen = load_full_search_v3_frozen_winner(
            frozen_winner_path
            if frozen_winner_path is not None
            else full_search_v3_frozen_winner_path(repository_root, run_id)
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
                raise FullSearchV3FinalIdentityError(
                    "FINAL checkpoint does not match immutable request order"
                )
            for index in range(len(completed_hashes), len(expected_hashes)):
                request = build_solver_request(final_records[index], prompts[prompt_variant])
                # The response is intentionally not persisted. Parser/scoring is
                # a later FINAL reporting concern and must never feed SEARCH.
                execute_solver_request(solver, request)
                completed_hashes.append(expected_hashes[index])
                state["status"] = "running"
                _refresh_state_hash(state)
                _atomic_replace_json(destination, state)
        state["status"] = "complete"
        _refresh_state_hash(state)
        _atomic_replace_json(destination, state)
        receipt = _completion_receipt(identity, state["variants"])
        _atomic_create_json(receipt_path, receipt)
    return load_full_search_v3_final_completion_receipt(receipt_path)


def full_search_v3_final_state_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the FINAL-only state location for a V3 run."""

    return _final_state_path(repository_root, run_id)


def load_full_search_v3_final_state(path: str | Path) -> dict[str, Any]:
    """Load a safe FINAL planning state without exposing private membership."""

    state = _load_canonical_object(Path(path), "FINAL state")
    required = {
        "schema_version", "state_version", "identity", "status", "variants", "state_sha256"
    }
    if set(state) != required:
        raise FullSearchV3FinalIdentityError("FINAL state fields are invalid")
    if state["schema_version"] != FULL_SEARCH_V3_FINAL_STATE_SCHEMA_VERSION or state[
        "state_version"
    ] != FULL_SEARCH_V3_FINAL_STATE_VERSION:
        raise FullSearchV3FinalIdentityError("FINAL state schema is incompatible")
    if state["status"] not in {"planned", "running", "complete"}:
        raise FullSearchV3FinalIdentityError("FINAL state status is invalid")
    _validate_execution_identity(state["identity"])
    _validate_variants(state["variants"], state["identity"])
    expected = _sha256_json({key: value for key, value in state.items() if key != "state_sha256"})
    if state["state_sha256"] != expected:
        raise FullSearchV3FinalIdentityError("FINAL state hash is inconsistent")
    return _json_copy(state)


def full_search_v3_final_completion_receipt_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the create-once paired FINAL completion receipt path."""

    return _final_state_path(repository_root, run_id).parent / _RECEIPT_FILENAME


def load_full_search_v3_final_completion_receipt(path: str | Path) -> dict[str, Any]:
    """Load a receipt containing only safe paired-execution identities."""

    receipt = _load_canonical_object(Path(path), "FINAL completion receipt")
    required = {
        "schema_version", "identity_sha256", "status", "variants", "logical_calls", "receipt_sha256"
    }
    if set(receipt) != required or receipt["schema_version"] != FULL_SEARCH_V3_FINAL_STATE_SCHEMA_VERSION:
        raise FullSearchV3FinalIdentityError("FINAL completion receipt fields are invalid")
    if receipt["status"] != "complete" or receipt["logical_calls"] != 2 * FULL_SEARCH_V3_FINAL_SIZE:
        raise FullSearchV3FinalIdentityError("FINAL completion receipt is incomplete")
    _sha256(receipt["identity_sha256"], "FINAL receipt identity_sha256")
    _validate_receipt_variants(receipt["variants"])
    expected = _sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    if receipt["receipt_sha256"] != expected:
        raise FullSearchV3FinalIdentityError("FINAL completion receipt hash is inconsistent")
    return _json_copy(receipt)


def _build_execution_identity(
    *,
    repository_root: str | Path,
    run_id: str,
    frozen_winner_path: str | Path | None,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    final_records: Sequence[Mapping[str, Any]],
    solver_contract: FullSearchV3FinalSolverContract,
) -> dict[str, Any]:
    safe_run_id = _identifier(run_id)
    if not isinstance(solver_contract, FullSearchV3FinalSolverContract):
        raise TypeError("solver_contract must be FullSearchV3FinalSolverContract")
    artifact_path = (
        Path(frozen_winner_path)
        if frozen_winner_path is not None
        else full_search_v3_frozen_winner_path(repository_root, safe_run_id)
    )
    try:
        frozen = load_full_search_v3_frozen_winner(artifact_path)
    except FullSearchV3FreezeError as exc:
        raise FullSearchV3FinalError("FINAL requires a valid immutable frozen winner") from exc
    if frozen["run_id"] != safe_run_id:
        raise FullSearchV3FinalError("frozen winner belongs to a different run")
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
        "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
        "run_id": safe_run_id,
        "config_sha256": private_manifest["config_sha256"],
        "split_sha256": private_manifest["split"]["split_sha256"],
        "final_membership_commitment": private_manifest["final_membership_commitment"],
        "final_record_order_sha256": ordered_membership_sha256,
        "final_size": FULL_SEARCH_V3_FINAL_SIZE,
        "frozen_winner_artifact_sha256": frozen["artifact_sha256"],
        "search_run_identity_sha256": frozen["hashes"]["run_identity_sha256"],
        "solver": solver_contract.as_dict(),
        "p0": {
            "candidate_id": FULL_SEARCH_V3_P0_CANDIDATE_ID,
            "prompt_sha256": canonical_full_search_v3_p0_prompt_sha256(),
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
        private_manifest = load_trusted_full_search_v3_private_manifest(
            private_manifest_path
        )
        safe_manifest = load_full_search_v3_search_safe_manifest(search_safe_manifest_path)
        verify_full_search_v3_search_safe_manifest(safe_manifest, private_manifest)
    except FullSearchV3PreparationError as exc:
        raise FullSearchV3FinalError(
            "FINAL private and SEARCH-safe manifests are incompatible"
        ) from exc
    if safe_manifest["final_membership_commitment"] != private_manifest[
        "final_membership_commitment"
    ]:
        raise FullSearchV3FinalError("FINAL membership commitment is incompatible")
    return private_manifest, safe_manifest


def _validate_final_records(
    records: Sequence[Mapping[str, Any]], private_manifest: Mapping[str, Any]
) -> str:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise FullSearchV3FinalError("FINAL records must be a sequence")
    expected_ids = tuple(private_manifest["split"]["FINAL"]["sample_ids"])
    if len(records) != FULL_SEARCH_V3_FINAL_SIZE:
        raise FullSearchV3FinalError("FINAL records must contain exactly 1,000 rows")
    observed_ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise FullSearchV3FinalError("FINAL record must be an object")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise FullSearchV3FinalError("FINAL record has an invalid sample identity")
        observed_ids.append(sample_id)
    if len(observed_ids) != len(set(observed_ids)):
        raise FullSearchV3FinalError("FINAL records contain duplicate sample identities")
    if tuple(observed_ids) != expected_ids:
        raise FullSearchV3FinalError("FINAL records are not in immutable manifest order")
    return _sha256_json(list(observed_ids))


def _load_search_identity(repository_root: str | Path, run_id: str) -> dict[str, Any]:
    path = (
        Path(repository_root)
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / _identifier(run_id)
        / "orchestration_state.json"
    )
    state = _load_canonical_object(path, "SEARCH state")
    identity = state.get("identity")
    if not isinstance(identity, Mapping):
        raise FullSearchV3FinalError("SEARCH run identity is missing")
    required = {
        "protocol_id", "config_sha256", "split_sha256", "search_membership_sha256",
        "sample_ids_sha256", "canonical_p0_prompt_sha256", "parser_version",
        "solver_identity_sha256", "proposer_identity",
    }
    if set(identity) != required or identity["protocol_id"] != FULL_SEARCH_V3_PROTOCOL_ID:
        raise FullSearchV3FinalError("SEARCH run identity is incompatible")
    for field in required - {"protocol_id", "parser_version", "proposer_identity"}:
        _sha256(identity[field], f"SEARCH identity {field}")
    if identity["parser_version"] != PARSER_VERSION:
        raise FullSearchV3FinalError("SEARCH parser identity is incompatible")
    return _json_copy(identity)


def _validate_cross_stage_identity(
    *,
    frozen: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    search_identity: Mapping[str, Any],
    solver_contract: FullSearchV3FinalSolverContract,
) -> None:
    hashes = frozen["hashes"]
    if hashes["config_sha256"] != private_manifest["config_sha256"] or hashes[
        "split_sha256"
    ] != private_manifest["split"]["split_sha256"]:
        raise FullSearchV3FinalError("frozen winner and private split are incompatible")
    if hashes["run_identity_sha256"] != _sha256_json(search_identity):
        raise FullSearchV3FinalError("frozen winner and SEARCH run identity are incompatible")
    if search_identity["config_sha256"] != private_manifest["config_sha256"] or search_identity[
        "split_sha256"
    ] != private_manifest["split"]["split_sha256"]:
        raise FullSearchV3FinalError("SEARCH run and private split are incompatible")
    if search_identity["solver_identity_sha256"] != solver_contract.solver_identity_sha256:
        raise FullSearchV3FinalError("FINAL solver identity is incompatible with SEARCH")
    if search_identity["canonical_p0_prompt_sha256"] != canonical_full_search_v3_p0_prompt_sha256():
        raise FullSearchV3FinalError("SEARCH P0 identity is incompatible")
    templates = {method: canonical_baseline_sources()[method] for method in TEMPLATE_KEYS}
    if template_source_sha256(templates) != canonical_full_search_v3_p0_prompt_sha256():
        raise FullSearchV3FinalError("canonical P0 prompt identity is inconsistent")
    if frozen["winner"]["candidate_id"] == FULL_SEARCH_V3_P0_CANDIDATE_ID:
        if frozen["hashes"]["prompt_sha256"] != canonical_full_search_v3_p0_prompt_sha256():
            raise FullSearchV3FinalError("frozen P0 winner identity is inconsistent")
    elif template_source_sha256(PromptFamily(frozen["templates"])) != frozen["hashes"]["prompt_sha256"]:
        raise FullSearchV3FinalError("frozen P* prompt identity is inconsistent")


def _planned_state(identity: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": FULL_SEARCH_V3_FINAL_STATE_SCHEMA_VERSION,
        "state_version": FULL_SEARCH_V3_FINAL_STATE_VERSION,
        "identity": _json_copy(identity),
        "status": "planned",
        "variants": [
            {
                "prompt_variant": "cot",
                "candidate_id": identity["p0"]["candidate_id"],
                "prompt_sha256": identity["p0"]["prompt_sha256"],
                "completed_request_sha256": [],
            },
            {
                "prompt_variant": "meta_cot",
                "candidate_id": identity["p_star"]["candidate_id"],
                "prompt_sha256": identity["p_star"]["prompt_sha256"],
                "completed_request_sha256": [],
            },
        ],
    }
    return {**payload, "state_sha256": _sha256_json(payload)}


def _validate_execution_identity(value: Any) -> None:
    required = {
        "protocol_id", "run_id", "config_sha256", "split_sha256", "final_membership_commitment",
        "final_record_order_sha256", "final_size", "frozen_winner_artifact_sha256",
        "search_run_identity_sha256", "solver", "p0", "p_star",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise FullSearchV3FinalIdentityError("FINAL execution identity fields are invalid")
    if value["protocol_id"] != FULL_SEARCH_V3_PROTOCOL_ID or value["final_size"] != FULL_SEARCH_V3_FINAL_SIZE:
        raise FullSearchV3FinalIdentityError("FINAL execution identity is incompatible")
    _identifier(value["run_id"])
    for field in required - {"protocol_id", "run_id", "final_size", "solver", "p0", "p_star"}:
        _sha256(value[field], f"FINAL identity {field}")
    contract = FullSearchV3FinalSolverContract(**value["solver"])
    _validate_prompt_identity(value["p0"], FULL_SEARCH_V3_P0_CANDIDATE_ID)
    _validate_prompt_identity(value["p_star"], None)
    if value["p0"]["prompt_sha256"] != canonical_full_search_v3_p0_prompt_sha256() or value[
        "p_star"
    ]["prompt_variant"] != "meta_cot":
        raise FullSearchV3FinalIdentityError("FINAL prompt identities are incompatible")
    if not isinstance(contract, FullSearchV3FinalSolverContract):
        raise AssertionError("validated contract unexpectedly has the wrong type")


def _validate_prompt_identity(value: Any, candidate_id: str | None) -> None:
    if not isinstance(value, Mapping):
        raise FullSearchV3FinalIdentityError("FINAL prompt identity is invalid")
    expected = {"candidate_id", "prompt_sha256"}
    if candidate_id is None:
        expected = {"candidate_id", "prompt_sha256", "prompt_variant"}
    if set(value) != expected or (candidate_id is not None and value["candidate_id"] != candidate_id):
        raise FullSearchV3FinalIdentityError("FINAL prompt identity is invalid")
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
        raise FullSearchV3FinalIdentityError("FINAL paired variants are incompatible")
    for actual, template in zip(value, expected):
        completed = actual.get("completed_request_sha256") if isinstance(actual, Mapping) else None
        if (
            not isinstance(actual, Mapping)
            or {key: item for key, item in actual.items() if key != "completed_request_sha256"}
            != {key: item for key, item in template.items() if key != "completed_request_sha256"}
            or not isinstance(completed, list)
            or len(completed) > FULL_SEARCH_V3_FINAL_SIZE
            or len(completed) != len(set(completed))
        ):
            raise FullSearchV3FinalIdentityError("FINAL paired variants are incompatible")
        for item in completed:
            _sha256(item, "FINAL completed request hash")


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
        }
        for variant in variants
    ]
    payload = {
        "schema_version": FULL_SEARCH_V3_FINAL_STATE_SCHEMA_VERSION,
        "identity_sha256": _sha256_json(identity),
        "status": "complete",
        "variants": receipt_variants,
        "logical_calls": 2 * FULL_SEARCH_V3_FINAL_SIZE,
    }
    return {**payload, "receipt_sha256": _sha256_json(payload)}


def _validate_receipt_variants(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise FullSearchV3FinalIdentityError("FINAL receipt variants are invalid")
    expected_names = ("cot", "meta_cot")
    for variant, expected_name in zip(value, expected_names):
        if not isinstance(variant, Mapping) or set(variant) != {
            "prompt_variant", "candidate_id", "prompt_sha256", "completed_request_count", "completed_request_hashes_sha256"
        }:
            raise FullSearchV3FinalIdentityError("FINAL receipt variants are invalid")
        if variant["prompt_variant"] != expected_name or variant["completed_request_count"] != FULL_SEARCH_V3_FINAL_SIZE:
            raise FullSearchV3FinalIdentityError("FINAL receipt variants are incomplete")
        _identifier(variant["candidate_id"])
        _sha256(variant["prompt_sha256"], "FINAL receipt prompt_sha256")
        _sha256(variant["completed_request_hashes_sha256"], "FINAL receipt request hashes")


def _validate_receipt_identity(receipt: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    if receipt["identity_sha256"] != _sha256_json(identity):
        raise FullSearchV3FinalIdentityError("FINAL completion receipt identity is incompatible")


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
        / _identifier(run_id)
        / "final"
        / _STATE_FILENAME
    )


def _load_canonical_object(path: Path, label: str) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, FullSearchV3FinalError) as exc:
        raise FullSearchV3FinalError(f"{label} is unreadable") from exc
    if not isinstance(value, dict) or encoded != canonical_json(value).encode("utf-8"):
        raise FullSearchV3FinalError(f"{label} is not canonical")
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
            raise FullSearchV3FinalIdentityError("a FINAL state was created concurrently")
        os.replace(temporary, path)
    except FullSearchV3FinalError:
        raise
    except OSError as exc:
        raise FullSearchV3FinalError("unable to atomically create FINAL state") from exc
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
        raise FullSearchV3FinalError("unable to atomically update FINAL state") from exc
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
            raise FullSearchV3FinalError("another process owns this FINAL run") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise FullSearchV3FinalError("FINAL identifier is invalid")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FullSearchV3FinalError(f"{field} must be a lowercase SHA-256")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> dict[str, Any]:
    return json.loads(canonical_json(value))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FullSearchV3FinalError("duplicate JSON key")
        result[key] = value
    return result


__all__ = [
    "FULL_SEARCH_V3_FINAL_PLAN_SCHEMA_VERSION",
    "FULL_SEARCH_V3_FINAL_STATE_SCHEMA_VERSION",
    "FULL_SEARCH_V3_FINAL_STATE_VERSION",
    "FullSearchV3FinalAuthorizationError",
    "FullSearchV3FinalError",
    "FullSearchV3FinalIdentityError",
    "FullSearchV3FinalSolverContract",
    "canonical_full_search_v3_final_solver_contract",
    "execute_full_search_v3_final",
    "full_search_v3_final_completion_receipt_path",
    "full_search_v3_final_state_path",
    "initialize_full_search_v3_final",
    "load_full_search_v3_final_completion_receipt",
    "load_full_search_v3_final_state",
    "preflight_full_search_v3_final",
]
