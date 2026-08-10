"""Immutable winner freezing and one-time final-test model transfer."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from meta_harness.candidate_store import CandidateStore
from meta_harness.config import (
    DEFAULT_MODEL,
    MetaHarnessConfig,
    SUPPORTED_SEARCH_MODELS,
)
from meta_harness.evaluator import (
    EVALUATOR_VERSION,
    PARSER_VERSION,
    evaluate_candidate,
)
from meta_harness.orchestrator import (
    EVALUATION_PROCEDURE,
    load_run_candidate,
    load_run_state,
    run_lock,
)
from meta_harness.prompt_family import PromptFamily, TEMPLATE_KEYS
from meta_harness.schemas import Candidate, canonical_json, template_source_sha256
from meta_harness.split_manager import verify_split_manifest


FINALIZATION_SCHEMA_VERSION = 1
FINALIZATION_VERSION = "meta_harness_finalization_v1"
META_COT_VARIANT = "meta_cot"
SEARCH_MODEL = DEFAULT_MODEL
TRANSFER_MODELS = (
    "Qwen2.5-VL-7B-Instruct",
    "gemma-4-31B-it",
    "gemma-3-27b-it",
)
FINAL_MODELS = (SEARCH_MODEL, *TRANSFER_MODELS)
STAGED_EVALUATION_PROCEDURE = "search_smoke_promote_validation_v1"
_TERMINAL_SEARCH_STATUSES = frozenset(
    {"completed", "early_stopped", "budget_exhausted", "failure_limit"}
)
_METRICS_VERSION = "binary_macro_f1_v1"
_FROZEN_FILE = "frozen_winner.json"
_FINALIZATION_DIRECTORY = "finalization"
_RECEIPT_FILE = "completion.json"
_MODEL_MANIFEST_FILE = "manifest.json"
_AGGREGATE_MANIFEST_FILE = "results_manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CODE_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


class FinalizationError(RuntimeError):
    """Raised when winner freezing or final-test execution is unsafe."""


class FrozenArtifactConflictError(FinalizationError):
    """Raised when immutable finalization content conflicts."""


class FinalTestConfirmationError(FinalizationError):
    """Raised when final-test execution lacks explicit confirmation."""


class CompletionReceiptError(FinalizationError):
    """Raised when one-time execution state is invalid or inconsistent."""


Evaluator = Callable[..., Mapping[str, Any]]


def select_global_best(run_state: Mapping[str, Any]) -> dict[str, Any]:
    """Select by Macro-F1, accuracy, token use, then candidate ID."""

    if not isinstance(run_state, Mapping):
        raise FinalizationError("run state must be a mapping")
    candidates = run_state.get("candidates")
    if not isinstance(candidates, Mapping):
        raise FinalizationError("run state candidates are missing")

    rankable: list[tuple[tuple[Any, ...], str, Mapping[str, Any]]] = []
    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate_id, str) or not isinstance(candidate, Mapping):
            raise FinalizationError("run state candidate entries are invalid")
        score = candidate.get("score")
        is_rankable = (
            score.get("rankable", score.get("eligible"))
            if isinstance(score, Mapping)
            else False
        )
        if (
            candidate.get("status") != "evaluated"
            or not isinstance(score, Mapping)
            or is_rankable is not True
        ):
            continue
        macro_f1 = _unit_metric(score.get("macro_f1"), "macro_f1")
        accuracy = _unit_metric(score.get("accuracy"), "accuracy")
        parse_coverage = _unit_metric(
            score.get("parse_coverage"),
            "parse_coverage",
        )
        unresolved_api_failures = _non_negative_int(
            score.get("unresolved_api_failures"),
            "unresolved_api_failures",
        )
        if parse_coverage != 1.0 or unresolved_api_failures != 0:
            continue
        solver_calls = _non_negative_int(
            score.get("solver_calls"),
            "solver_calls",
        )
        tokens = _non_negative_int(score.get("tokens"), "tokens")
        latency = _non_negative_number(
            score.get("latency_seconds"),
            "latency_seconds",
        )
        rankable.append(
            (
                (
                    -macro_f1,
                    -accuracy,
                    tokens,
                    candidate_id,
                ),
                candidate_id,
                score,
            )
        )

    if not rankable:
        raise FinalizationError(
            "run has no rankable validation candidate to freeze"
        )
    _, candidate_id, score = min(rankable, key=lambda item: item[0])
    return {
        "candidate_id": candidate_id,
        "macro_f1": float(score["macro_f1"]),
        "accuracy": float(score["accuracy"]),
        "solver_calls": score["solver_calls"],
        "tokens": score["tokens"],
        "latency_seconds": float(score["latency_seconds"]),
    }


select_winner = select_global_best


def frozen_winner_path(repository_root: str | Path, run_id: str) -> Path:
    safe_run_id = _safe_identifier(run_id, "run_id")
    return (
        Path(repository_root)
        / "workspace"
        / "meta_harness"
        / "runs"
        / safe_run_id
        / _FINALIZATION_DIRECTORY
        / _FROZEN_FILE
    )


def freeze_winner(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Select and atomically freeze the validation winner exactly once."""

    root = Path(repository_root)
    safe_run_id = _safe_identifier(run_id, "run_id")
    verify_split_manifest(split_manifest)
    state = load_run_state(root, safe_run_id)
    if state.get("status") not in _TERMINAL_SEARCH_STATUSES:
        raise FinalizationError(
            "search run must stop before its winner can be frozen"
        )
    configuration = state.get("configuration")
    if not isinstance(configuration, Mapping):
        raise FinalizationError("run configuration snapshot is missing")
    procedure = configuration.get("evaluation_procedure")
    if procedure not in {
        EVALUATION_PROCEDURE,
        STAGED_EVALUATION_PROCEDURE,
    }:
        raise FinalizationError(
            "winner scores must come from the fixed validation procedure"
        )
    if configuration.get("split_sha256") != split_manifest["split_sha256"]:
        raise FinalizationError("split manifest does not match the search run")
    if configuration.get("dataset_sha256") != split_manifest["dataset_sha256"]:
        raise FinalizationError("dataset hash does not match the search run")

    selected = select_global_best(state)
    candidate_id = selected["candidate_id"]
    candidate = load_run_candidate(root, safe_run_id, candidate_id)
    candidate_state = state["candidates"][candidate_id]
    if candidate.sha256() != candidate_state.get("candidate_sha256"):
        raise FinalizationError(
            "winner candidate hash does not match durable run state"
        )
    if candidate.source_sha256 != candidate_state.get("prompt_sha256"):
        raise FinalizationError(
            "winner prompt hash does not match durable run state"
        )

    revision = code_revision or _read_code_revision(root)
    if not isinstance(revision, str) or not _CODE_REVISION.fullmatch(revision):
        raise FinalizationError(
            "code revision must be a hexadecimal repository revision"
        )
    solver_configuration = configuration.get("config")
    if not isinstance(solver_configuration, Mapping):
        raise FinalizationError("solver configuration snapshot is missing")
    if solver_configuration.get("model") not in SUPPORTED_SEARCH_MODELS:
        raise FinalizationError(
            "search solver configuration has an unexpected model"
        )

    payload = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "finalization_version": FINALIZATION_VERSION,
        "run_id": safe_run_id,
        "prompt_variant": META_COT_VARIANT,
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate.sha256(),
        "prompt_sha256": candidate.source_sha256,
        "candidate": candidate.as_dict(),
        "templates": {
            method: candidate.templates[method] for method in TEMPLATE_KEYS
        },
        "validation": {
            "split": "validation",
            "macro_f1": selected["macro_f1"],
            "accuracy": selected["accuracy"],
            "rankable": True,
            "solver_calls": selected["solver_calls"],
            "tokens": selected["tokens"],
            "latency_seconds": selected["latency_seconds"],
            "selection_rule": (
                "macro_f1_desc_accuracy_desc_tokens_asc_"
                "candidate_id_asc"
            ),
        },
        "search_solver_configuration": _json_copy(solver_configuration),
        "search_solver_configuration_sha256": configuration[
            "config_sha256"
        ],
        "split_hashes": _split_hashes(split_manifest),
        "evaluator_version": EVALUATOR_VERSION,
        "parser_version": PARSER_VERSION,
        "metrics_version": _METRICS_VERSION,
        "code_revision": revision,
    }
    artifact = {
        **payload,
        "artifact_sha256": _sha256_json(payload),
    }

    destination = frozen_winner_path(root, safe_run_id)
    lock_path = destination.parent / ".freeze.lock"
    with run_lock(lock_path):
        if destination.exists():
            existing = load_frozen_winner(destination)
            if canonical_json(existing) != canonical_json(artifact):
                raise FrozenArtifactConflictError(
                    "a different immutable winner is already frozen"
                )
            return existing
        _atomic_create_json(destination, artifact, mode=0o444)
    return load_frozen_winner(destination)


finalize_run = freeze_winner


def load_frozen_winner(path: str | Path) -> dict[str, Any]:
    """Load and verify every identity hash in a frozen winner."""

    artifact = _load_json_object(Path(path), "frozen winner")
    required = {
        "schema_version",
        "finalization_version",
        "run_id",
        "prompt_variant",
        "candidate_id",
        "candidate_sha256",
        "prompt_sha256",
        "candidate",
        "templates",
        "validation",
        "search_solver_configuration",
        "search_solver_configuration_sha256",
        "split_hashes",
        "evaluator_version",
        "parser_version",
        "metrics_version",
        "code_revision",
        "artifact_sha256",
    }
    if set(artifact) != required:
        raise FinalizationError("frozen winner fields are invalid")
    if artifact["schema_version"] != FINALIZATION_SCHEMA_VERSION:
        raise FinalizationError("unsupported frozen winner schema_version")
    if artifact["finalization_version"] != FINALIZATION_VERSION:
        raise FinalizationError("unsupported finalization version")
    if artifact["prompt_variant"] != META_COT_VARIANT:
        raise FinalizationError("frozen prompt variant must be meta_cot")
    _require_sha256(artifact["artifact_sha256"], "artifact_sha256")
    without_hash = {
        key: value
        for key, value in artifact.items()
        if key != "artifact_sha256"
    }
    if _sha256_json(without_hash) != artifact["artifact_sha256"]:
        raise FinalizationError("frozen winner artifact hash mismatch")

    templates = artifact["templates"]
    family = PromptFamily(templates)
    normalized_templates = {
        method: family[method].template for method in TEMPLATE_KEYS
    }
    if normalized_templates != templates:
        raise FinalizationError("frozen templates are not canonical")
    if template_source_sha256(templates) != artifact["prompt_sha256"]:
        raise FinalizationError("frozen prompt hash mismatch")

    candidate = artifact["candidate"]
    if not isinstance(candidate, Mapping):
        raise FinalizationError("frozen candidate must be a mapping")
    if candidate.get("candidate_id") != artifact["candidate_id"]:
        raise FinalizationError("frozen candidate ID mismatch")
    if candidate.get("templates") != templates:
        raise FinalizationError("frozen candidate templates mismatch")
    if candidate.get("source_sha256") != artifact["prompt_sha256"]:
        raise FinalizationError("frozen candidate prompt hash mismatch")
    if (
        hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()
        != artifact["candidate_sha256"]
    ):
        raise FinalizationError("frozen candidate hash mismatch")
    _require_sha256(
        artifact["search_solver_configuration_sha256"],
        "search_solver_configuration_sha256",
    )
    try:
        search_config = MetaHarnessConfig.from_mapping(
            artifact["search_solver_configuration"]
        )
    except ValueError as exc:
        raise FinalizationError(
            "frozen search solver configuration is invalid"
        ) from exc
    if search_config.model not in SUPPORTED_SEARCH_MODELS:
        raise FinalizationError("frozen search model is invalid")
    if search_config.sha256() != artifact[
        "search_solver_configuration_sha256"
    ]:
        raise FinalizationError(
            "frozen search solver configuration hash mismatch"
        )
    if not _CODE_REVISION.fullmatch(artifact["code_revision"]):
        raise FinalizationError("frozen code revision is invalid")
    _validate_split_hashes(artifact["split_hashes"])
    validation = artifact["validation"]
    if (
        not isinstance(validation, Mapping)
        or validation.get("split") != "validation"
        or validation.get("rankable") is not True
    ):
        raise FinalizationError("frozen validation selection is invalid")
    _unit_metric(validation.get("macro_f1"), "validation.macro_f1")
    return artifact


def execute_final_test(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    final_samples: Sequence[Any],
    solver: Any,
    confirm_final_test: bool,
    evaluator: Evaluator = evaluate_candidate,
) -> dict[str, Any]:
    """Evaluate the frozen winner once with the original search model."""

    winner = load_frozen_winner(
        frozen_winner_path(repository_root, run_id)
    )
    return execute_frozen_model(
        repository_root=repository_root,
        run_id=run_id,
        split_manifest=split_manifest,
        final_samples=final_samples,
        solver=solver,
        model=_original_search_model(winner),
        confirm_final_test=confirm_final_test,
        evaluator=evaluator,
    )


def execute_final_test_pair(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    final_samples: Sequence[Any],
    solver: Any,
    confirm_final_test: bool,
    evaluator: Evaluator = evaluate_candidate,
) -> dict[str, dict[str, Any]]:
    """Evaluate unchanged ``cot`` and frozen ``meta_cot`` once on Qwen."""

    winner = load_frozen_winner(
        frozen_winner_path(repository_root, run_id)
    )
    search_model = _original_search_model(winner)
    baseline = execute_baseline_model(
        repository_root=repository_root,
        run_id=run_id,
        split_manifest=split_manifest,
        final_samples=final_samples,
        solver=solver,
        model=search_model,
        confirm_final_test=confirm_final_test,
        evaluator=evaluator,
    )
    frozen = execute_final_test(
        repository_root=repository_root,
        run_id=run_id,
        split_manifest=split_manifest,
        final_samples=final_samples,
        solver=solver,
        confirm_final_test=confirm_final_test,
        evaluator=evaluator,
    )
    return {"cot": baseline, META_COT_VARIANT: frozen}


def execute_baseline_model(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    final_samples: Sequence[Any],
    solver: Any,
    model: str,
    confirm_final_test: bool,
    evaluator: Evaluator = evaluate_candidate,
) -> dict[str, Any]:
    """Evaluate the unchanged canonical CoT family once for one final model."""

    if model not in FINAL_MODELS:
        raise FinalizationError("model is not part of the frozen transfer set")
    root = Path(repository_root)
    winner = load_frozen_winner(frozen_winner_path(root, run_id))
    verify_split_manifest(split_manifest)
    _verify_frozen_split(winner, split_manifest)
    final_ids = tuple(split_manifest["splits"]["final_test"]["sample_ids"])
    _validate_final_samples(final_samples, final_ids)
    baseline = load_run_candidate(root, run_id, "baseline_cot")

    directory = _baseline_execution_directory(root, run_id, model)
    receipt_path = directory / _RECEIPT_FILE
    lock_path = directory / ".execution.lock"
    with run_lock(lock_path):
        if receipt_path.is_file():
            receipt = load_completion_receipt(receipt_path)
            _verify_baseline_completion_identity(
                receipt,
                winner=winner,
                baseline=baseline,
                model=model,
                split_manifest=split_manifest,
                final_ids=final_ids,
            )
            return receipt
        if confirm_final_test is not True:
            raise FinalTestConfirmationError(
                "final-test execution requires explicit confirmation"
            )
        original_model = _original_search_model(winner)
        if model != original_model:
            original_receipt = _baseline_execution_directory(
                root,
                run_id,
                original_model,
            ) / _RECEIPT_FILE
            if not original_receipt.is_file():
                raise CompletionReceiptError(
                    "the original-model cot final test must complete before "
                    "cot transfer"
                )
            original = load_completion_receipt(original_receipt)
            _verify_baseline_completion_identity(
                original,
                winner=winner,
                baseline=baseline,
                model=original_model,
                split_manifest=split_manifest,
                final_ids=final_ids,
            )
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")

        model_configuration = _model_configuration(winner, model)
        model_config_sha256 = _sha256_json(model_configuration)
        execution_identity = {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "run_id": winner["run_id"],
            "candidate_id": baseline.candidate_id,
            "candidate_sha256": baseline.sha256(),
            "prompt_variant": "cot",
            "prompt_sha256": baseline.source_sha256,
            "model_configuration": model_configuration,
            "model_configuration_sha256": model_config_sha256,
            "split_sha256": split_manifest["split_sha256"],
            "final_test_sample_ids_sha256": _sha256_json(final_ids),
            "final_test_sample_count": len(final_ids),
        }
        manifest_path = directory / _MODEL_MANIFEST_FILE
        _atomic_create_or_verify(manifest_path, execution_identity)

        results_path = directory / "results.jsonl"
        metrics_path = directory / "metrics.json"
        if metrics_path.is_file():
            report = _load_json_object(metrics_path, "model metrics")
        else:
            store = _StaticCandidateStore(root, run_id, baseline)
            base_config = MetaHarnessConfig.from_mapping(
                winner["search_solver_configuration"]
            )
            report = evaluator(
                run_id=(
                    f"{winner['run_id']}:cot:"
                    f"{_safe_model_slug(model)}:final_test"
                ),
                candidate_id=baseline.candidate_id,
                candidate_store=store,
                split_manifest=split_manifest,
                split_name="final_test",
                sample_ids=final_ids,
                samples=tuple(final_samples),
                solver=solver,
                output_path=results_path,
                metrics_path=metrics_path,
                config=base_config,
                model=model,
                prompt_variant="cot",
                model_config_sha256=model_config_sha256,
            )
            if not isinstance(report, Mapping):
                raise CompletionReceiptError(
                    "evaluator must return a metrics mapping"
                )
            _atomic_write_json(metrics_path, report)

        _verify_baseline_model_report(
            report,
            baseline=baseline,
            model=model,
            split_manifest=split_manifest,
            model_config_sha256=model_config_sha256,
        )
        if not results_path.is_file():
            raise CompletionReceiptError(
                "evaluator did not create durable JSONL results"
            )
        csv_path = directory / "results.csv"
        _export_results_csv(results_path, csv_path)
        receipt_payload = {
            **execution_identity,
            "status": "complete",
            "artifacts": {
                "results_jsonl": {
                    "path": results_path.name,
                    "sha256": _sha256_file(results_path),
                },
                "metrics_json": {
                    "path": metrics_path.name,
                    "sha256": _sha256_file(metrics_path),
                },
                "results_csv": {
                    "path": csv_path.name,
                    "sha256": _sha256_file(csv_path),
                },
                "model_manifest_json": {
                    "path": manifest_path.name,
                    "sha256": _sha256_file(manifest_path),
                },
            },
        }
        if "search_protocol" in winner["search_solver_configuration"]:
            receipt_payload["stage_budget"] = _final_stage_budget(report)
        receipt = {
            **receipt_payload,
            "completion_sha256": _sha256_json(receipt_payload),
        }
        _atomic_create_json(receipt_path, receipt, mode=0o444)
        completed = load_completion_receipt(receipt_path)
        _verify_baseline_completion_identity(
            completed,
            winner=winner,
            baseline=baseline,
            model=model,
            split_manifest=split_manifest,
            final_ids=final_ids,
        )
        return completed


def execute_frozen_model(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    final_samples: Sequence[Any],
    solver: Any,
    model: str,
    confirm_final_test: bool,
    evaluator: Evaluator = evaluate_candidate,
) -> dict[str, Any]:
    """Evaluate one allowed model with the exact frozen prompt bytes."""

    if model not in FINAL_MODELS:
        raise FinalizationError("model is not part of the frozen transfer set")
    root = Path(repository_root)
    winner_path = frozen_winner_path(root, run_id)
    winner = load_frozen_winner(winner_path)
    verify_split_manifest(split_manifest)
    _verify_frozen_split(winner, split_manifest)
    final_ids = tuple(split_manifest["splits"]["final_test"]["sample_ids"])
    _validate_final_samples(final_samples, final_ids)

    directory = _execution_directory(root, run_id, model)
    receipt_path = directory / _RECEIPT_FILE
    lock_path = directory / ".execution.lock"
    with run_lock(lock_path):
        if receipt_path.is_file():
            receipt = load_completion_receipt(receipt_path)
            _verify_completion_identity(
                receipt,
                winner=winner,
                model=model,
                split_manifest=split_manifest,
                final_ids=final_ids,
            )
            return receipt
        if confirm_final_test is not True:
            raise FinalTestConfirmationError(
                "final-test execution requires explicit confirmation"
            )
        original_model = _original_search_model(winner)
        if model != original_model:
            original_receipt = _execution_directory(
                root,
                run_id,
                original_model,
            ) / _RECEIPT_FILE
            if not original_receipt.is_file():
                raise CompletionReceiptError(
                    "the original-model final test must complete before transfer"
                )
            original = load_completion_receipt(original_receipt)
            _verify_completion_identity(
                original,
                winner=winner,
                model=original_model,
                split_manifest=split_manifest,
                final_ids=final_ids,
            )
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")

        model_configuration = _model_configuration(winner, model)
        model_config_sha256 = _sha256_json(model_configuration)
        execution_identity = {
            "schema_version": FINALIZATION_SCHEMA_VERSION,
            "run_id": winner["run_id"],
            "candidate_id": winner["candidate_id"],
            "candidate_sha256": winner["candidate_sha256"],
            "prompt_variant": META_COT_VARIANT,
            "prompt_sha256": winner["prompt_sha256"],
            "model_configuration": model_configuration,
            "model_configuration_sha256": model_config_sha256,
            "split_sha256": split_manifest["split_sha256"],
            "final_test_sample_ids_sha256": _sha256_json(final_ids),
            "final_test_sample_count": len(final_ids),
        }
        manifest_path = directory / _MODEL_MANIFEST_FILE
        _atomic_create_or_verify(manifest_path, execution_identity)

        results_path = directory / "results.jsonl"
        metrics_path = directory / "metrics.json"
        if metrics_path.is_file():
            report = _load_json_object(metrics_path, "model metrics")
        else:
            store = _FrozenCandidateStore(root, run_id, winner)
            base_config = MetaHarnessConfig.from_mapping(
                winner["search_solver_configuration"]
            )
            report = evaluator(
                run_id=(
                    f"{winner['run_id']}:{META_COT_VARIANT}:"
                    f"{_safe_model_slug(model)}:final_test"
                ),
                candidate_id=winner["candidate_id"],
                candidate_store=store,
                split_manifest=split_manifest,
                split_name="final_test",
                sample_ids=final_ids,
                samples=tuple(final_samples),
                solver=solver,
                output_path=results_path,
                metrics_path=metrics_path,
                config=base_config,
                model=model,
                prompt_variant=META_COT_VARIANT,
                model_config_sha256=model_config_sha256,
            )
            if not isinstance(report, Mapping):
                raise CompletionReceiptError(
                    "evaluator must return a metrics mapping"
                )
            _atomic_write_json(metrics_path, report)

        _verify_model_report(
            report,
            winner=winner,
            model=model,
            split_manifest=split_manifest,
            model_config_sha256=model_config_sha256,
        )
        if not results_path.is_file():
            raise CompletionReceiptError(
                "evaluator did not create durable JSONL results"
            )
        csv_path = directory / "results.csv"
        _export_results_csv(results_path, csv_path)
        receipt_payload = {
            **execution_identity,
            "status": "complete",
            "artifacts": {
                "results_jsonl": {
                    "path": results_path.name,
                    "sha256": _sha256_file(results_path),
                },
                "metrics_json": {
                    "path": metrics_path.name,
                    "sha256": _sha256_file(metrics_path),
                },
                "results_csv": {
                    "path": csv_path.name,
                    "sha256": _sha256_file(csv_path),
                },
                "model_manifest_json": {
                    "path": manifest_path.name,
                    "sha256": _sha256_file(manifest_path),
                },
            },
        }
        if "search_protocol" in winner["search_solver_configuration"]:
            receipt_payload["stage_budget"] = _final_stage_budget(report)
        receipt = {
            **receipt_payload,
            "completion_sha256": _sha256_json(receipt_payload),
        }
        _atomic_create_json(receipt_path, receipt, mode=0o444)
        completed = load_completion_receipt(receipt_path)
        _verify_completion_identity(
            completed,
            winner=winner,
            model=model,
            split_manifest=split_manifest,
            final_ids=final_ids,
        )
        return completed


def execute_transfers(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    final_samples: Sequence[Any],
    solvers: Mapping[str, Any],
    confirm_final_test: bool,
    evaluator: Evaluator = evaluate_candidate,
) -> dict[str, dict[str, Any]]:
    """Run each transfer model once, with no proposal or prompt mutation."""

    if not isinstance(solvers, Mapping) or set(solvers) != set(TRANSFER_MODELS):
        raise FinalizationError(
            "solvers must contain exactly the three transfer models"
        )
    receipts = {
        model: execute_frozen_model(
            repository_root=repository_root,
            run_id=run_id,
            split_manifest=split_manifest,
            final_samples=final_samples,
            solver=solvers[model],
            model=model,
            confirm_final_test=confirm_final_test,
            evaluator=evaluator,
        )
        for model in TRANSFER_MODELS
    }
    _write_aggregate_manifest(
        Path(repository_root),
        run_id,
    )
    return receipts


def execute_transfer_matrix(
    *,
    repository_root: str | Path,
    run_id: str,
    split_manifest: Mapping[str, Any],
    final_samples: Sequence[Any],
    solvers: Mapping[str, Any],
    confirm_final_test: bool,
    evaluator: Evaluator = evaluate_candidate,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Evaluate unchanged ``cot`` and exact frozen ``meta_cot`` on Gemma."""

    if not isinstance(solvers, Mapping) or set(solvers) != set(TRANSFER_MODELS):
        raise FinalizationError(
            "solvers must contain exactly the three transfer models"
        )
    receipts: dict[str, dict[str, dict[str, Any]]] = {}
    for model in TRANSFER_MODELS:
        baseline = execute_baseline_model(
            repository_root=repository_root,
            run_id=run_id,
            split_manifest=split_manifest,
            final_samples=final_samples,
            solver=solvers[model],
            model=model,
            confirm_final_test=confirm_final_test,
            evaluator=evaluator,
        )
        frozen = execute_frozen_model(
            repository_root=repository_root,
            run_id=run_id,
            split_manifest=split_manifest,
            final_samples=final_samples,
            solver=solvers[model],
            model=model,
            confirm_final_test=confirm_final_test,
            evaluator=evaluator,
        )
        receipts[model] = {"cot": baseline, META_COT_VARIANT: frozen}
    _write_aggregate_manifest(Path(repository_root), run_id)
    return receipts


def load_completion_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path)
    receipt = _load_json_object(receipt_path, "completion receipt")
    completion_sha256 = receipt.get("completion_sha256")
    _require_sha256(completion_sha256, "completion_sha256")
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "completion_sha256"
    }
    if _sha256_json(payload) != completion_sha256:
        raise CompletionReceiptError("completion receipt hash mismatch")
    if receipt.get("status") != "complete":
        raise CompletionReceiptError("execution is not complete")
    if receipt.get("prompt_variant") not in {"cot", META_COT_VARIANT}:
        raise CompletionReceiptError("completion prompt variant is invalid")
    _require_sha256(receipt.get("prompt_sha256"), "prompt_sha256")
    model_configuration = receipt.get("model_configuration")
    if not isinstance(model_configuration, Mapping):
        raise CompletionReceiptError("model configuration is missing")
    if _sha256_json(model_configuration) != receipt.get(
        "model_configuration_sha256"
    ):
        raise CompletionReceiptError("model configuration hash mismatch")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise CompletionReceiptError("completion artifacts are missing")
    for artifact in artifacts.values():
        if not isinstance(artifact, Mapping):
            raise CompletionReceiptError("completion artifact entry is invalid")
        relative_path = artifact.get("path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).name != relative_path
        ):
            raise CompletionReceiptError(
                "completion artifact path is invalid"
            )
        artifact_path = receipt_path.parent / relative_path
        if not artifact_path.is_file():
            raise CompletionReceiptError("completion artifact is missing")
        if _sha256_file(artifact_path) != artifact.get("sha256"):
            raise CompletionReceiptError("completion artifact hash mismatch")
    return receipt


def _write_aggregate_manifest(repository_root: Path, run_id: str) -> Path:
    winner = load_frozen_winner(
        frozen_winner_path(repository_root, run_id)
    )
    receipts = {}
    for model in FINAL_MODELS:
        path = _execution_directory(
            repository_root,
            run_id,
            model,
        ) / _RECEIPT_FILE
        receipts[model] = load_completion_receipt(path)
    baseline_receipts = {}
    for model in FINAL_MODELS:
        path = _baseline_execution_directory(
            repository_root,
            run_id,
            model,
        ) / _RECEIPT_FILE
        if path.is_file():
            baseline_receipts[model] = load_completion_receipt(path)
    payload = {
        "schema_version": FINALIZATION_SCHEMA_VERSION,
        "run_id": winner["run_id"],
        "candidate_id": winner["candidate_id"],
        "candidate_sha256": winner["candidate_sha256"],
        "prompt_variant": META_COT_VARIANT,
        "prompt_sha256": winner["prompt_sha256"],
        "split_hashes": winner["split_hashes"],
        "models": {
            model: {
                "model_configuration": receipts[model][
                    "model_configuration"
                ],
                "model_configuration_sha256": receipts[model][
                    "model_configuration_sha256"
                ],
                "completion_sha256": receipts[model]["completion_sha256"],
                **(
                    {
                        "prompt_variants": {
                            "cot": {
                                "prompt_sha256": baseline_receipts[model][
                                    "prompt_sha256"
                                ],
                                "completion_sha256": baseline_receipts[model][
                                    "completion_sha256"
                                ],
                            },
                            META_COT_VARIANT: {
                                "prompt_sha256": receipts[model][
                                    "prompt_sha256"
                                ],
                                "completion_sha256": receipts[model][
                                    "completion_sha256"
                                ],
                            },
                        }
                    }
                    if model in baseline_receipts
                    else {}
                ),
            }
            for model in FINAL_MODELS
        },
    }
    manifest = {
        **payload,
        "manifest_sha256": _sha256_json(payload),
    }
    destination = (
        frozen_winner_path(repository_root, run_id).parent
        / _AGGREGATE_MANIFEST_FILE
    )
    _atomic_create_or_verify(destination, manifest)
    return destination


class _FrozenCandidateStore(CandidateStore):
    def __init__(
        self,
        repository_root: Path,
        run_id: str,
        winner: Mapping[str, Any],
    ) -> None:
        super().__init__(repository_root, f"{run_id}.frozen")
        self._candidate = _candidate_from_winner(winner)

    def load(self, candidate_id: str) -> Candidate:
        if candidate_id != self._candidate.candidate_id:
            raise FinalizationError("frozen candidate ID is immutable")
        return self._candidate


class _StaticCandidateStore(CandidateStore):
    """Read-only evaluator view over one already verified candidate."""

    def __init__(
        self,
        repository_root: Path,
        run_id: str,
        candidate: Candidate,
    ) -> None:
        super().__init__(repository_root, f"{run_id}.final")
        self._candidate = candidate

    def load(self, candidate_id: str) -> Candidate:
        if candidate_id != self._candidate.candidate_id:
            raise FinalizationError("final candidate ID is immutable")
        return self._candidate


def _candidate_from_winner(winner: Mapping[str, Any]) -> Candidate:
    value = winner["candidate"]
    try:
        return Candidate.from_mapping(value)
    except ValueError:
        if value.get("candidate_id") != "baseline_cot":
            raise
        candidate = object.__new__(Candidate)
        for field in (
            "candidate_id",
            "parent_id",
            "search_axis",
            "hypothesis",
            "expected_tradeoff",
            "source_sha256",
        ):
            object.__setattr__(candidate, field, value[field])
        object.__setattr__(candidate, "templates", value["templates"])
        return candidate


def _verify_model_report(
    report: Mapping[str, Any],
    *,
    winner: Mapping[str, Any],
    model: str,
    split_manifest: Mapping[str, Any],
    model_config_sha256: str,
) -> None:
    required = {
        "candidate_id": winner["candidate_id"],
        "prompt_variant": META_COT_VARIANT,
        "prompt_sha256": winner["prompt_sha256"],
        "model": model,
        "config_sha256": model_config_sha256,
        "split": "final_test",
        "split_sha256": split_manifest["split_sha256"],
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise CompletionReceiptError(
                f"model report has inconsistent {field}"
            )


def _verify_baseline_model_report(
    report: Mapping[str, Any],
    *,
    baseline: Candidate,
    model: str,
    split_manifest: Mapping[str, Any],
    model_config_sha256: str,
) -> None:
    required = {
        "candidate_id": baseline.candidate_id,
        "prompt_variant": "cot",
        "prompt_sha256": baseline.source_sha256,
        "model": model,
        "config_sha256": model_config_sha256,
        "split": "final_test",
        "split_sha256": split_manifest["split_sha256"],
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise CompletionReceiptError(
                f"baseline model report has inconsistent {field}"
            )


def _verify_completion_identity(
    receipt: Mapping[str, Any],
    *,
    winner: Mapping[str, Any],
    model: str,
    split_manifest: Mapping[str, Any],
    final_ids: Sequence[Any],
) -> None:
    expected_configuration = _model_configuration(winner, model)
    expected = {
        "run_id": winner["run_id"],
        "candidate_id": winner["candidate_id"],
        "candidate_sha256": winner["candidate_sha256"],
        "prompt_variant": META_COT_VARIANT,
        "prompt_sha256": winner["prompt_sha256"],
        "model_configuration": expected_configuration,
        "model_configuration_sha256": _sha256_json(expected_configuration),
        "split_sha256": split_manifest["split_sha256"],
        "final_test_sample_ids_sha256": _sha256_json(final_ids),
        "final_test_sample_count": len(final_ids),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise CompletionReceiptError(
                f"completion receipt has inconsistent {field}"
            )


def _verify_baseline_completion_identity(
    receipt: Mapping[str, Any],
    *,
    winner: Mapping[str, Any],
    baseline: Candidate,
    model: str,
    split_manifest: Mapping[str, Any],
    final_ids: Sequence[Any],
) -> None:
    expected_configuration = _model_configuration(winner, model)
    expected = {
        "run_id": winner["run_id"],
        "candidate_id": baseline.candidate_id,
        "candidate_sha256": baseline.sha256(),
        "prompt_variant": "cot",
        "prompt_sha256": baseline.source_sha256,
        "model_configuration": expected_configuration,
        "model_configuration_sha256": _sha256_json(expected_configuration),
        "split_sha256": split_manifest["split_sha256"],
        "final_test_sample_ids_sha256": _sha256_json(final_ids),
        "final_test_sample_count": len(final_ids),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise CompletionReceiptError(
                f"baseline completion receipt has inconsistent {field}"
            )


def _model_configuration(
    winner: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    generation = winner["search_solver_configuration"].get("generation")
    if not isinstance(generation, Mapping):
        raise FinalizationError("frozen generation configuration is missing")
    return {
        "model": model,
        "generation": _json_copy(generation),
        "n": 1,
        "stream": False,
        "evaluator_version": winner["evaluator_version"],
        "parser_version": winner["parser_version"],
        "metrics_version": winner["metrics_version"],
    }


def _final_stage_budget(report: Mapping[str, Any]) -> dict[str, Any]:
    usage = report.get("resource_usage", {})
    if not isinstance(usage, Mapping):
        usage = {}
    solver_calls = usage.get("solver_calls", report.get("sample_count", 0))
    tokens = usage.get("tokens", usage.get("total_tokens", 0))
    latency = usage.get(
        "latency_seconds",
        usage.get("latency", 0.0),
    )
    _non_negative_int(solver_calls, "final stage solver_calls")
    _non_negative_int(tokens, "final stage tokens")
    _non_negative_number(latency, "final stage latency_seconds")
    return {
        "stage": "final_test",
        "solver_calls": solver_calls,
        "tokens": tokens,
        "latency_seconds": float(latency),
    }


def _original_search_model(winner: Mapping[str, Any]) -> str:
    configuration = winner.get("search_solver_configuration")
    if not isinstance(configuration, Mapping):
        raise FinalizationError("frozen search solver configuration is missing")
    model = configuration.get("model")
    if model not in SUPPORTED_SEARCH_MODELS:
        raise FinalizationError("frozen search solver model is unsupported")
    return model


def _split_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    return {
        "split_sha256": manifest["split_sha256"],
        "dataset_sha256": manifest["dataset_sha256"],
        "search_sample_ids_sha256": _sha256_json(
            manifest["splits"]["search"]["sample_ids"]
        ),
        "validation_sample_ids_sha256": _sha256_json(
            manifest["splits"]["validation"]["sample_ids"]
        ),
        "final_test_sample_ids_sha256": _sha256_json(
            manifest["splits"]["final_test"]["sample_ids"]
        ),
    }


def _validate_split_hashes(value: Any) -> None:
    expected = {
        "split_sha256",
        "dataset_sha256",
        "search_sample_ids_sha256",
        "validation_sample_ids_sha256",
        "final_test_sample_ids_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FinalizationError("frozen split hashes are invalid")
    for field, digest in value.items():
        _require_sha256(digest, field)


def _verify_frozen_split(
    winner: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    if winner["split_hashes"] != _split_hashes(manifest):
        raise FinalizationError(
            "final-test split does not match the frozen winner"
        )


def _validate_final_samples(
    samples: Sequence[Any],
    final_ids: Sequence[Any],
) -> None:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise FinalizationError("final_samples must be a sequence")
    loaded = []
    for index, sample in enumerate(samples):
        if hasattr(sample, "sample_id"):
            sample_id = sample.sample_id
        elif isinstance(sample, Mapping):
            sample_id = sample.get("sample_id")
        else:
            raise FinalizationError(
                f"final sample at index {index} has no sample identity"
            )
        loaded.append(_json_identity(sample_id))
    expected = [_json_identity(sample_id) for sample_id in final_ids]
    if len(loaded) != len(set(loaded)) or set(loaded) != set(expected):
        raise FinalizationError(
            "final_samples must contain exactly the frozen final-test IDs"
        )


def _execution_directory(
    repository_root: Path,
    run_id: str,
    model: str,
) -> Path:
    return (
        frozen_winner_path(repository_root, run_id).parent
        / "executions"
        / _safe_model_slug(model)
    )


def _baseline_execution_directory(
    repository_root: Path,
    run_id: str,
    model: str,
) -> Path:
    return _execution_directory(repository_root, run_id, model) / "cot"


def _safe_model_slug(model: str) -> str:
    if model not in FINAL_MODELS:
        raise FinalizationError("model is not in the frozen model set")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def _export_results_csv(results_path: Path, destination: Path) -> None:
    records = []
    try:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CompletionReceiptError(
            "results JSONL must contain valid JSON objects"
        ) from exc
    fields = (
        "run_id",
        "sample_id",
        "dataset",
        "model",
        "prompt_variant",
        "candidate_id",
        "method",
        "reasoning_method",
        "gold_label",
        "prediction",
        "parse_status",
        "request_status",
        "latency",
        "usage",
        "attempt_count",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in records:
                row = {}
                for field in fields:
                    value = record.get(field)
                    row[field] = (
                        json.dumps(
                            value,
                            sort_keys=True,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        if isinstance(value, (dict, list))
                        else value
                    )
                writer.writerow(row)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_create_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = _load_json_object(path, path.name)
        if canonical_json(existing) != canonical_json(value):
            raise FrozenArtifactConflictError(
                f"{path.name} conflicts with immutable execution identity"
            )
        return
    _atomic_create_json(path, value)


def _atomic_create_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    mode: int = 0o600,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.chmod(temporary, mode)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = _load_json_object(path, path.name)
            if canonical_json(existing) != canonical_json(value):
                raise FrozenArtifactConflictError(
                    f"{path.name} was concurrently created with other content"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"{label} must be readable valid JSON") from exc
    if not isinstance(value, dict):
        raise FinalizationError(f"{label} must contain a JSON object")
    return value


def _read_code_revision(repository_root: Path) -> str:
    git_path = repository_root / ".git"
    if git_path.is_file():
        content = git_path.read_text(encoding="utf-8").strip()
        if not content.startswith("gitdir: "):
            raise FinalizationError("repository metadata is invalid")
        git_path = (repository_root / content[8:]).resolve()
    try:
        head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise FinalizationError("repository revision is unavailable") from exc
    if _CODE_REVISION.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        raise FinalizationError("repository HEAD is invalid")
    ref_name = head[5:]
    ref_path = git_path / ref_name
    if ref_path.is_file():
        revision = ref_path.read_text(encoding="utf-8").strip()
        if _CODE_REVISION.fullmatch(revision):
            return revision
    packed_refs = git_path / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.startswith(("#", "^")):
                continue
            fields = line.split(" ", 1)
            if len(fields) == 2 and fields[1] == ref_name:
                if _CODE_REVISION.fullmatch(fields[0]):
                    return fields[0]
    raise FinalizationError("repository revision cannot be resolved")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise CompletionReceiptError("result artifact is unreadable") from exc
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise FinalizationError(f"{field} must be a hexadecimal SHA-256")


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(canonical_json(value))
    except Exception as exc:
        raise FinalizationError("frozen data must be canonical JSON") from exc


def _json_identity(value: Any) -> str:
    try:
        return canonical_json(value)
    except Exception as exc:
        raise FinalizationError("sample IDs must be canonical JSON") from exc


def _safe_identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)
    ):
        raise FinalizationError(
            f"{field} must contain only letters, numbers, dot, underscore, "
            "or hyphen"
        )
    return value


def _unit_metric(value: Any, field: str) -> float:
    number = _non_negative_number(value, field)
    if number > 1:
        raise FinalizationError(f"{field} must be at most 1.0")
    return number


def _non_negative_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise FinalizationError(
            f"{field} must be a non-negative finite number"
        )
    return float(value)


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalizationError(f"{field} must be a non-negative integer")
    return value


__all__ = [
    "CompletionReceiptError",
    "FINALIZATION_SCHEMA_VERSION",
    "FINALIZATION_VERSION",
    "FINAL_MODELS",
    "FinalTestConfirmationError",
    "FinalizationError",
    "FrozenArtifactConflictError",
    "META_COT_VARIANT",
    "SEARCH_MODEL",
    "TRANSFER_MODELS",
    "execute_baseline_model",
    "execute_final_test",
    "execute_final_test_pair",
    "execute_frozen_model",
    "execute_transfer_matrix",
    "execute_transfers",
    "finalize_run",
    "freeze_winner",
    "frozen_winner_path",
    "load_completion_receipt",
    "load_frozen_winner",
    "select_global_best",
    "select_winner",
]
