"""Thin, server-facing composition API for ``sciver_full_search_v3``.

The functions in this module deliberately delegate preparation, SEARCH,
freeze, and FINAL work to the M1--M5 engine.  They return compact,
notebook-friendly summaries and never persist runtime endpoint or credential
values.  SEARCH functions receive only SEARCH-safe artifacts; trusted FINAL
materialization is confined to the two FINAL functions below.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

from meta_harness.config import canonical_full_search_v3_config
from meta_harness.full_search_v3 import validate_sciver_full_search_v3_records
from meta_harness.full_search_v3_cache import FullSearchV3SearchCache
from meta_harness.full_search_v3_concurrency import FullSearchV3RequestExecutor
from meta_harness.full_search_v3_evaluator import (
    FullSearchV3SearchInput,
    canonical_full_search_v3_p0_prompt_sha256,
    load_full_search_v3_search_input,
)
from meta_harness.full_search_v3_final import (
    canonical_full_search_v3_final_solver_contract,
    execute_full_search_v3_final,
    full_search_v3_final_completion_receipt_path,
    full_search_v3_final_state_path,
    load_full_search_v3_final_completion_receipt,
    load_full_search_v3_final_state,
    preflight_full_search_v3_final,
)
from meta_harness.full_search_v3_freeze import (
    freeze_full_search_v3_winner,
    full_search_v3_frozen_winner_path,
)
from meta_harness.full_search_v3_orchestrator import (
    FullSearchV3Orchestrator,
    full_search_v3_orchestration_state_path,
    load_full_search_v3_orchestration_state,
)
from meta_harness.full_search_v3_preparation import (
    FullSearchV3PreparationArtifacts,
    load_trusted_full_search_v3_private_manifest,
    materialize_full_search_v3_final_records,
    prepare_full_search_v3,
)
from meta_harness.full_search_v3_proposer import (
    FULL_SEARCH_V3_PROPOSER_INSTRUCTION_VERSION,
    FullSearchV3Proposer,
    FullSearchV3ProposerConfig,
)
from meta_harness.full_search_v3_retry import SolverRetryPolicy
from meta_harness.full_search_v3_solver import create_live_solver_client
from meta_harness.full_search_v3_solver import (
    build_solver_request,
    execute_solver_request,
    solver_request_payload_sha256,
)
from meta_harness.prompt_family import canonical_json
from utils.answer_parser import PARSER_VERSION, parse_answer
from utils.constant import COT_PROMPT


_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SMOKE_RECEIPT_SCHEMA_VERSION = "sciver_full_search_v3_smoke_receipt_v1"


class FullSearchV3ServerError(RuntimeError):
    """Raised with a safe, actionable message by the server interface."""


class FullSearchV3ServerAuthorizationError(FullSearchV3ServerError):
    """Raised when a stage lacks its separate explicit live authorization."""


class _PreflightProposer:
    """Identity-only placeholder that guarantees dry-run cannot invoke a CLI."""

    def propose(self, **_kwargs: Any) -> None:
        raise AssertionError("SEARCH preflight must not invoke the proposer")


def prepare_full_search_v3_server_run(
    *,
    dataset_path: str | Path,
    repository_root: str | Path,
    run_id: str,
    preparation_directory: str | Path | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Delegate deterministic M1 preparation for one explicit run directory."""

    run_directory = _run_directory(repository_root, run_id)
    preparation_root = (
        Path(preparation_directory)
        if preparation_directory is not None
        else run_directory / "preparation"
    )
    artifacts = prepare_full_search_v3(
        source_path=dataset_path,
        private_directory=preparation_root / "private",
        search_directory=preparation_root / "search",
        config_path=config_path,
    )
    return _preparation_status(run_id, run_directory, artifacts)


def preflight_full_search_v3_server_run(
    *,
    repository_root: str | Path,
    run_id: str,
    search_safe_manifest_path: str | Path,
    search_records_path: str | Path,
    source_commit: str | None = None,
    solver_identity_sha256: str | None = None,
    proposer_config: FullSearchV3ProposerConfig | None = None,
) -> dict[str, Any]:
    """Plan SEARCH locally without reading credentials or constructing clients.

    A supplied solver identity is required only when checking an existing run.
    It is intentionally a hash, not an endpoint value.
    """

    search_input = load_full_search_v3_search_input(
        search_safe_manifest_path=search_safe_manifest_path,
        search_records_path=search_records_path,
    )
    safe_run_id = _run_id(run_id)
    commit = _source_commit(repository_root, source_commit)
    active_proposer_config = proposer_config or FullSearchV3ProposerConfig()
    proposer_identity = _proposer_identity(commit, active_proposer_config)
    _require_proposer_cli()
    if solver_identity_sha256 is not None:
        _require_sha256(solver_identity_sha256, "solver_identity_sha256")

    state_path = full_search_v3_orchestration_state_path(repository_root, safe_run_id)
    resume = state_path.exists()
    if resume:
        if solver_identity_sha256 is None:
            raise FullSearchV3ServerError(
                "existing SEARCH run requires solver_identity_sha256 for resume validation"
            )
        # M4 remains the authority for immutable identity compatibility.  This
        # call reads an existing state only; the path-exists guard prevents a
        # dry run from creating state.
        FullSearchV3Orchestrator(
            repository_root=repository_root,
            run_id=safe_run_id,
            search_input=search_input,
            solver_identity_sha256=solver_identity_sha256,
            cache=object(),
            executor=object(),
            proposer=_PreflightProposer(),
            proposer_identity=proposer_identity,
        ).state()

    return _search_preflight_status(
        repository_root=repository_root,
        run_id=safe_run_id,
        search_input=search_input,
        source_commit=commit,
        solver_identity_sha256=solver_identity_sha256,
        proposer_identity=proposer_identity,
        resume=resume,
    )


def start_or_resume_full_search_v3_server_run(
    *,
    repository_root: str | Path,
    run_id: str,
    search_safe_manifest_path: str | Path,
    search_records_path: str | Path,
    authorize_search_execution: bool,
    api_url: str | None = None,
    api_key: str | None = None,
    source_commit: str | None = None,
    proposer_config: FullSearchV3ProposerConfig | None = None,
    retry_policy: SolverRetryPolicy | None = None,
    maximum_in_flight_requests: int = 1,
) -> dict[str, Any]:
    """Run or resume M4 SEARCH only after explicit SEARCH authorization."""

    if authorize_search_execution is not True:
        raise FullSearchV3ServerAuthorizationError(
            "SEARCH dispatch requires explicit SEARCH authorization"
        )
    runtime_url, runtime_key = _runtime_credentials(api_url=api_url, api_key=api_key)
    solver_identity_sha256 = solver_identity_from_api_url(runtime_url)
    active_proposer_config = proposer_config or FullSearchV3ProposerConfig()
    preflight = preflight_full_search_v3_server_run(
        repository_root=repository_root,
        run_id=run_id,
        search_safe_manifest_path=search_safe_manifest_path,
        search_records_path=search_records_path,
        source_commit=source_commit,
        solver_identity_sha256=solver_identity_sha256,
        proposer_config=active_proposer_config,
    )
    search_input = load_full_search_v3_search_input(
        search_safe_manifest_path=search_safe_manifest_path,
        search_records_path=search_records_path,
    )
    _require_compatible_smoke_receipt(
        repository_root=repository_root,
        run_id=run_id,
        preflight=preflight,
        search_input=search_input,
    )
    client = _construct_live_solver(runtime_url, runtime_key)
    run_directory = _run_directory(repository_root, run_id)
    cache = FullSearchV3SearchCache(run_directory / "search_cache")
    executor = FullSearchV3RequestExecutor(
        cache=cache,
        client_factory=lambda: client,
        retry_policy=retry_policy or SolverRetryPolicy(),
        maximum_in_flight_requests=maximum_in_flight_requests,
        sleeper=time.sleep,
        clock=time.monotonic,
    )
    state = FullSearchV3Orchestrator(
        repository_root=repository_root,
        run_id=run_id,
        search_input=search_input,
        solver_identity_sha256=solver_identity_sha256,
        cache=cache,
        executor=executor,
        proposer=FullSearchV3Proposer(config=active_proposer_config),
        proposer_identity=_proposer_identity(
            _source_commit(repository_root, source_commit), active_proposer_config
        ),
    ).run()
    return {"preflight": preflight, "search": _search_status(state, run_id)}


def run_full_search_v3_server_smoke(
    *,
    repository_root: str | Path,
    run_id: str,
    search_safe_manifest_path: str | Path,
    search_records_path: str | Path,
    authorize_smoke_execution: bool,
    api_url: str | None = None,
    api_key: str | None = None,
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Dispatch one canonical P0 SEARCH request into isolated SMOKE state.

    The create-once receipt contains only hashes and stage metadata.  It never
    stores the record, model-visible messages, response text, label, images,
    endpoint, or credential.  A compatible existing receipt is idempotently
    returned without constructing a client or redispatching.
    """

    if authorize_smoke_execution is not True:
        raise FullSearchV3ServerAuthorizationError(
            "SMOKE dispatch requires separate explicit SMOKE authorization"
        )
    runtime_url, runtime_key = _runtime_credentials(api_url=api_url, api_key=api_key)
    solver_identity_sha256 = solver_identity_from_api_url(runtime_url)
    preflight = preflight_full_search_v3_server_run(
        repository_root=repository_root,
        run_id=run_id,
        search_safe_manifest_path=search_safe_manifest_path,
        search_records_path=search_records_path,
        source_commit=source_commit,
        solver_identity_sha256=solver_identity_sha256,
    )
    search_input = load_full_search_v3_search_input(
        search_safe_manifest_path=search_safe_manifest_path,
        search_records_path=search_records_path,
    )
    identity, request = _smoke_identity_and_request(preflight, search_input)
    receipt_path = full_search_v3_server_smoke_receipt_path(repository_root, run_id)
    with _smoke_lock(receipt_path.parent / ".smoke.lock"):
        if receipt_path.is_file():
            receipt = _load_compatible_smoke_receipt(receipt_path, identity)
            return _smoke_status(receipt, receipt_path, reused=True)

        result = execute_solver_request(
            _construct_live_solver(runtime_url, runtime_key), request
        )
        parsed = parse_answer(result.content)
        if parsed["parse_status"] != "parsed":
            raise FullSearchV3ServerError(
                "SMOKE response did not satisfy the canonical answer parser; no receipt was created"
            )
        receipt = {
            "schema_version": _SMOKE_RECEIPT_SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "identity_sha256": _sha256_json(identity),
            "logical_calls": 1,
            "parse_status": "parsed",
        }
        _create_once_json(receipt_path, receipt)
        persisted = _load_compatible_smoke_receipt(receipt_path, identity)
        return _smoke_status(persisted, receipt_path, reused=False)


def full_search_v3_server_smoke_receipt_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the isolated create-once SMOKE receipt path."""

    return _run_directory(repository_root, run_id) / "smoke" / "completion.json"


def inspect_full_search_v3_server_status(
    *, repository_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Return a compact SEARCH-only status without creating or advancing work."""

    path = full_search_v3_orchestration_state_path(repository_root, _run_id(run_id))
    if not path.is_file():
        raise FullSearchV3ServerError(
            "SEARCH state is missing; run preflight then start SEARCH with explicit authorization"
        )
    return _search_status(load_full_search_v3_orchestration_state(path), run_id)


def freeze_full_search_v3_server_winner(
    *, repository_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Delegate immutable M5 winner freezing after terminal SEARCH only."""

    artifact = freeze_full_search_v3_winner(repository_root=repository_root, run_id=run_id)
    return {
        "operation": "freeze",
        "protocol_id": artifact["schema_version"].split("_freeze_")[0],
        "run_id": artifact["run_id"],
        "prompt_variant": artifact["prompt_variant"],
        "winner_id": artifact["winner"]["candidate_id"],
        "artifact_sha256": artifact["artifact_sha256"],
        "prompt_sha256": artifact["hashes"]["prompt_sha256"],
        "frozen_winner_path": str(full_search_v3_frozen_winner_path(repository_root, run_id)),
    }


def preflight_full_search_v3_server_final(
    *,
    repository_root: str | Path,
    run_id: str,
    dataset_path: str | Path,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    solver_identity_sha256: str,
) -> dict[str, Any]:
    """Delegate M5 FINAL planning without a client, dispatch, or state mutation."""

    _require_sha256(solver_identity_sha256, "solver_identity_sha256")
    final_records = _trusted_final_records(dataset_path, private_manifest_path)
    return preflight_full_search_v3_final(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=None,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=canonical_full_search_v3_final_solver_contract(
            solver_identity_sha256=solver_identity_sha256
        ),
    )


def start_or_resume_full_search_v3_server_final(
    *,
    repository_root: str | Path,
    run_id: str,
    dataset_path: str | Path,
    private_manifest_path: str | Path,
    search_safe_manifest_path: str | Path,
    authorize_final_execution: bool,
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run only M5's paired FINAL stage after its separate authorization."""

    if authorize_final_execution is not True:
        raise FullSearchV3ServerAuthorizationError(
            "FINAL dispatch requires separate explicit FINAL authorization"
        )
    runtime_url, runtime_key = _runtime_credentials(api_url=api_url, api_key=api_key)
    solver_identity_sha256 = solver_identity_from_api_url(runtime_url)
    final_records = _trusted_final_records(dataset_path, private_manifest_path)
    contract = canonical_full_search_v3_final_solver_contract(
        solver_identity_sha256=solver_identity_sha256
    )
    preflight = preflight_full_search_v3_final(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=None,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=contract,
    )
    receipt = execute_full_search_v3_final(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=None,
        private_manifest_path=private_manifest_path,
        search_safe_manifest_path=search_safe_manifest_path,
        final_records=final_records,
        solver_contract=contract,
        solver=_construct_live_solver(runtime_url, runtime_key),
        authorize_final_execution=True,
    )
    return {"preflight": preflight, "final": _final_receipt_status(receipt, run_id)}


def inspect_full_search_v3_server_final_status(
    *, repository_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Inspect safe FINAL execution state or its create-once completion receipt."""

    receipt_path = full_search_v3_final_completion_receipt_path(repository_root, run_id)
    if receipt_path.is_file():
        return _final_receipt_status(
            load_full_search_v3_final_completion_receipt(receipt_path), run_id
        )
    state_path = full_search_v3_final_state_path(repository_root, run_id)
    if not state_path.is_file():
        raise FullSearchV3ServerError(
            "FINAL state is missing; freeze a terminal SEARCH winner and run FINAL preflight"
        )
    state = load_full_search_v3_final_state(state_path)
    return {
        "operation": "final_status",
        "run_id": _run_id(run_id),
        "status": state["status"],
        "execution_identity_sha256": _sha256_json(state["identity"]),
        "completed_logical_calls": sum(
            len(item["completed_request_sha256"]) for item in state["variants"]
        ),
        "expected_logical_calls": 2000,
    }


def inspect_full_search_v3_server_activity(
    *, repository_root: str | Path, run_id: str
) -> dict[str, Any]:
    """Report whether another process currently owns a run-stage lock.

    This is observational only: it neither creates a run directory nor changes
    a lock.  A lock may be released between this check and a later command, so
    the M4/M5 execution locks remain the authoritative race-safe gate.
    """

    run_directory = _run_directory(repository_root, _run_id(run_id))
    return {
        "operation": "activity",
        "run_id": _run_id(run_id),
        "smoke_lock": _lock_activity(run_directory / "smoke" / ".smoke.lock"),
        "search_lock": _lock_activity(run_directory / ".orchestration.lock"),
        "final_lock": _lock_activity(run_directory / "final" / ".final.lock"),
    }


def solver_identity_from_api_url(api_url: str) -> str:
    """Return a non-reversible solver deployment identity for runtime use only."""

    if not isinstance(api_url, str) or not api_url.strip():
        raise FullSearchV3ServerError("runtime API_URL is required for live execution")
    config = canonical_full_search_v3_config()
    return _sha256_json(
        {
            "protocol_id": config.protocol_id,
            "solver_model": config.solver_model,
            "api_url_sha256": hashlib.sha256(api_url.encode("utf-8")).hexdigest(),
        }
    )


def _preparation_status(
    run_id: str, run_directory: Path, artifacts: FullSearchV3PreparationArtifacts
) -> dict[str, Any]:
    return {
        "operation": "prepare",
        "run_id": _run_id(run_id),
        "run_directory": str(run_directory),
        **dict(artifacts.summary),
        "private_manifest_path": str(artifacts.private_manifest_path),
        "search_safe_manifest_path": str(artifacts.search_safe_manifest_path),
        "search_dataset_path": str(artifacts.search_dataset_path),
    }


def _search_preflight_status(
    *,
    repository_root: str | Path,
    run_id: str,
    search_input: FullSearchV3SearchInput,
    source_commit: str,
    solver_identity_sha256: str | None,
    proposer_identity: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    config = canonical_full_search_v3_config()
    manifest = search_input.manifest
    return {
        "operation": "search_preflight",
        "protocol_id": config.protocol_id,
        "run_id": run_id,
        "run_directory": str(_run_directory(repository_root, run_id)),
        "checkpoints": {
            "smoke_receipt_path": str(
                full_search_v3_server_smoke_receipt_path(repository_root, run_id)
            ),
            "search_state_path": str(
                full_search_v3_orchestration_state_path(repository_root, run_id)
            ),
            "search_cache_directory": str(
                _run_directory(repository_root, run_id) / "search_cache"
            ),
            "frozen_winner_path": str(
                full_search_v3_frozen_winner_path(repository_root, run_id)
            ),
            "final_state_path": str(
                full_search_v3_final_state_path(repository_root, run_id)
            ),
            "final_receipt_path": str(
                full_search_v3_final_completion_receipt_path(repository_root, run_id)
            ),
        },
        "resume": resume,
        "source_commit": source_commit,
        "config_sha256": config.sha256(),
        "split_sha256": manifest["split_sha256"],
        "search_membership_sha256": manifest["search_membership_sha256"],
        "search_sample_ids_sha256": _sha256_json(list(search_input.sample_ids)),
        "canonical_p0_prompt_sha256": canonical_full_search_v3_p0_prompt_sha256(),
        "solver": {
            "model": config.solver_model,
            "generation": {
                "temperature": config.solver_temperature,
                "top_p": config.solver_top_p,
                "seed": config.solver_seed,
                "n": config.solver_n,
                "stream": config.solver_stream,
                "max_tokens": config.solver_max_tokens,
            },
            "identity_sha256": solver_identity_sha256,
        },
        "parser_version": PARSER_VERSION,
        "proposer_identity_sha256": _sha256_json(proposer_identity),
        "workload": {
            "minimum_search_logical_calls": 16000,
            "maximum_search_logical_calls": 41000,
            "paired_final_logical_calls": 2000,
        },
    }


def _search_status(state: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    iterations = state["iterations"]
    completed = [item for item in iterations if item["status"] == "complete"]
    winner_report = _report_for_winner(state)
    return {
        "operation": "search_status",
        "run_id": _run_id(run_id),
        "status": state["status"],
        "stop_reason": state["stop_reason"],
        "completed_iterations": len(completed),
        "p0_status": state["p0"]["status"],
        "winner_id": state["winner_id"],
        "ranking": list(state["ranking"]),
        "patience": dict(state["patience"]),
        "winner_metrics": winner_report,
        "run_identity_sha256": _sha256_json(state["identity"]),
    }


def _smoke_identity_and_request(
    preflight: Mapping[str, Any], search_input: FullSearchV3SearchInput
) -> tuple[dict[str, Any], Any]:
    if not search_input.records:
        raise FullSearchV3ServerError("SMOKE requires at least one validated SEARCH record")
    request = build_solver_request(search_input.records[0], COT_PROMPT)
    identity = {
        "protocol_id": preflight["protocol_id"],
        "run_id": preflight["run_id"],
        "source_commit": preflight["source_commit"],
        "config_sha256": preflight["config_sha256"],
        "split_sha256": preflight["split_sha256"],
        "search_membership_sha256": preflight["search_membership_sha256"],
        "search_sample_ids_sha256": preflight["search_sample_ids_sha256"],
        "canonical_p0_prompt_sha256": preflight["canonical_p0_prompt_sha256"],
        "solver": preflight["solver"],
        "parser_version": preflight["parser_version"],
        "request_payload_sha256": solver_request_payload_sha256(request),
    }
    return identity, request


def _require_compatible_smoke_receipt(
    *,
    repository_root: str | Path,
    run_id: str,
    preflight: Mapping[str, Any],
    search_input: FullSearchV3SearchInput,
) -> None:
    identity, _request = _smoke_identity_and_request(preflight, search_input)
    path = full_search_v3_server_smoke_receipt_path(repository_root, run_id)
    if not path.is_file():
        raise FullSearchV3ServerError(
            "SEARCH requires a compatible completed SMOKE receipt; run the isolated SMOKE stage first"
        )
    _load_compatible_smoke_receipt(path, identity)


def _load_compatible_smoke_receipt(
    path: Path, expected_identity: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3ServerError(
            "SMOKE receipt is unreadable; use a fresh run ID after inspecting the artifact"
        ) from exc
    expected_hash = _sha256_json(expected_identity)
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema_version") != _SMOKE_RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "complete"
        or receipt.get("logical_calls") != 1
        or receipt.get("parse_status") != "parsed"
        or receipt.get("identity") != dict(expected_identity)
        or receipt.get("identity_sha256") != expected_hash
    ):
        raise FullSearchV3ServerError(
            "SMOKE receipt is incompatible with this run, split, prompt, parser, solver, or source revision"
        )
    return receipt


def _smoke_status(
    receipt: Mapping[str, Any], path: Path, *, reused: bool
) -> dict[str, Any]:
    return {
        "operation": "smoke",
        "run_id": receipt["identity"]["run_id"],
        "status": receipt["status"],
        "logical_calls": receipt["logical_calls"],
        "identity_sha256": receipt["identity_sha256"],
        "receipt_path": str(path),
        "reused": reused,
    }


def _create_once_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _report_for_winner(state: Mapping[str, Any]) -> dict[str, Any] | None:
    winner_id = state.get("winner_id")
    reports = [state["p0"].get("report")] + [
        entry.get("report") for entry in state["iterations"]
    ]
    for report in reports:
        if isinstance(report, Mapping) and report.get("candidate_id") == winner_id:
            metrics = report.get("metrics", {})
            return {
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
                "parse_coverage": metrics.get("parse_coverage"),
            }
    return None


def _final_receipt_status(receipt: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "operation": "final_status",
        "run_id": _run_id(run_id),
        "status": receipt["status"],
        "logical_calls": receipt["logical_calls"],
        "execution_identity_sha256": receipt["identity_sha256"],
        "variants": [
            {
                "prompt_variant": item["prompt_variant"],
                "candidate_id": item["candidate_id"],
                "prompt_sha256": item["prompt_sha256"],
                "completed_logical_calls": item["completed_request_count"],
            }
            for item in receipt["variants"]
        ],
    }


def _trusted_final_records(
    dataset_path: str | Path, private_manifest_path: str | Path
) -> list[dict[str, Any]]:
    """Materialize FINAL only within a FINAL operation and return nothing durable."""

    source = Path(dataset_path)
    try:
        raw_records = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullSearchV3ServerError("dataset must be readable valid JSON") from exc
    if not isinstance(raw_records, list) or any(
        not isinstance(record, Mapping) for record in raw_records
    ):
        raise FullSearchV3ServerError("dataset must be a JSON list of objects")
    try:
        return materialize_full_search_v3_final_records(
            validate_sciver_full_search_v3_records(raw_records, source_path=source),
            load_trusted_full_search_v3_private_manifest(private_manifest_path),
            source_path=source,
        )
    except Exception as exc:
        raise FullSearchV3ServerError(
            "trusted FINAL materialization failed; verify the dataset and private manifest"
        ) from exc


def _construct_live_solver(api_url: str, api_key: str) -> Any:
    with _runtime_environment(api_url=api_url, api_key=api_key):
        try:
            return create_live_solver_client(allow_live_requests=True)
        except Exception as exc:
            raise FullSearchV3ServerError(
                "unable to construct the live solver; verify runtime API_URL and API_KEY"
            ) from exc


def _runtime_credentials(*, api_url: str | None, api_key: str | None) -> tuple[str, str]:
    runtime_url = os.environ.get("API_URL") if api_url is None else api_url
    runtime_key = os.environ.get("API_KEY") if api_key is None else api_key
    if not isinstance(runtime_url, str) or not runtime_url.strip():
        raise FullSearchV3ServerError("live execution requires runtime API_URL")
    if not isinstance(runtime_key, str) or not runtime_key.strip():
        raise FullSearchV3ServerError("live execution requires runtime API_KEY")
    return runtime_url, runtime_key


def _lock_activity(path: Path) -> str:
    if not path.is_file():
        return "not_created"
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return "unavailable"
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "held"
        else:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            return "available"
    finally:
        os.close(descriptor)


@contextmanager
def _smoke_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FullSearchV3ServerError(
                "another process owns the isolated SMOKE stage for this run"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def _runtime_environment(*, api_url: str, api_key: str) -> Iterator[None]:
    """Temporarily supply runtime-only values to the existing M2 client factory."""

    original = {name: os.environ.get(name) for name in ("API_URL", "API_KEY")}
    os.environ["API_URL"] = api_url
    os.environ["API_KEY"] = api_key
    try:
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _proposer_identity(
    source_commit: str, config: FullSearchV3ProposerConfig
) -> dict[str, Any]:
    return {
        "kind": "local_cli",
        "instruction_version": FULL_SEARCH_V3_PROPOSER_INSTRUCTION_VERSION,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "timeout_seconds": config.timeout_seconds,
        "max_input_bytes": config.max_input_bytes,
        "max_output_bytes": config.max_output_bytes,
        "source_commit": source_commit,
    }


def _require_proposer_cli() -> None:
    if shutil.which("codex") is None:
        raise FullSearchV3ServerError(
            "local proposer CLI is unavailable; install it and ensure it is on PATH"
        )


def _source_commit(repository_root: str | Path, supplied: str | None) -> str:
    if supplied is not None:
        if not isinstance(supplied, str) or not _COMMIT.fullmatch(supplied):
            raise FullSearchV3ServerError("source_commit must be a lowercase Git commit hash")
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FullSearchV3ServerError(
            "source commit is unavailable; pass source_commit explicitly"
        ) from exc
    value = completed.stdout.strip()
    if not _COMMIT.fullmatch(value):
        raise FullSearchV3ServerError(
            "source commit is unavailable; pass source_commit explicitly"
        )
    if supplied is not None and supplied != value:
        raise FullSearchV3ServerError(
            "source_commit does not match the checked-out repository revision"
        )
    return value


def _run_directory(repository_root: str | Path, run_id: str) -> Path:
    return full_search_v3_orchestration_state_path(repository_root, _run_id(run_id)).parent


def _run_id(value: str) -> str:
    if not isinstance(value, str) or not value or "/" in value or ".." in value:
        raise FullSearchV3ServerError("run_id must be a safe local identifier")
    return value


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FullSearchV3ServerError(f"{field} must be a lowercase SHA-256")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "FullSearchV3ServerAuthorizationError",
    "FullSearchV3ServerError",
    "freeze_full_search_v3_server_winner",
    "full_search_v3_server_smoke_receipt_path",
    "inspect_full_search_v3_server_activity",
    "inspect_full_search_v3_server_final_status",
    "inspect_full_search_v3_server_status",
    "preflight_full_search_v3_server_final",
    "preflight_full_search_v3_server_run",
    "prepare_full_search_v3_server_run",
    "run_full_search_v3_server_smoke",
    "solver_identity_from_api_url",
    "start_or_resume_full_search_v3_server_final",
    "start_or_resume_full_search_v3_server_run",
]
