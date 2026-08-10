#!/usr/bin/env python3
"""Provider-neutral command line entry point for Meta-Harness search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.config import load_meta_harness_config
from meta_harness.orchestrator import (
    BudgetLimits,
    EarlyStoppingConfig,
    MetaHarnessOrchestrator,
    OrchestratorError,
    load_run_state,
)
from meta_harness.proposer.codex_cli import (
    CodexCLIProposer,
    CodexCLIProposerConfig,
)
from meta_harness.proposer.feedback import load_reparsed_search_feedback
from meta_harness.split_manager import load_split_manifest
from meta_harness.hard_search import (
    load_hard_search_manifest,
)
from meta_harness.staged_orchestrator import StagedMetaHarnessOrchestrator
from utils.cli_environment import load_cli_environment
from utils.dataset_adapters import get_dataset_adapter


class _DryRunBoundary:
    def propose(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("dry-run must not call the proposer")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run staged Meta-Harness search: 10-example smoke, small search, "
            "top-K protected validation, and no final-test access."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--search-dataset-path",
        type=Path,
        help="materialized hard-search JSON (enables the staged protocol)",
    )
    parser.add_argument(
        "--hard-search-manifest",
        type=Path,
        help="immutable hard-search manifest (required with --search-dataset-path)",
    )
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument(
        "--proposer-history-reparse",
        type=Path,
        help=(
            "read-only parser-v2 validation history used to avoid strategies "
            "already tried in an earlier search"
        ),
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--run-id")
    identity.add_argument("--resume", metavar="RUN_ID")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--live-api", action="store_true")

    parser.add_argument("--max-iterations", type=_positive_int)
    parser.add_argument("--max-candidates", type=_positive_int)
    parser.add_argument("--max-solver-calls", type=_positive_int)
    parser.add_argument("--max-tokens", type=_positive_int)
    parser.add_argument("--max-wall-time", type=_positive_float)
    parser.add_argument("--max-consecutive-failures", type=_positive_int)
    parser.add_argument("--patience", type=_positive_int)
    parser.add_argument("--min-delta", type=_non_negative_float)
    parser.add_argument("--min-iterations", type=_non_negative_int)
    parser.add_argument(
        "--safe-resume-change",
        action="append",
        default=[],
        choices=(
            "max_iterations",
            "max_candidates",
            "max_solver_calls",
            "max_tokens",
            "max_wall_time_seconds",
            "max_consecutive_failures",
        ),
        help=(
            "Explicitly approve one non-decreasing resource-limit change "
            "when resuming."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    run_id = arguments.resume or arguments.run_id
    resume = arguments.resume is not None

    try:
        if arguments.live_api and not arguments.dry_run:
            load_cli_environment()
            from model_inference.remote_config import (
                validate_config_for_live_request,
            )

            validate_config_for_live_request()
        config = load_meta_harness_config(arguments.config)
        split_manifest = load_split_manifest(arguments.split_manifest)
        staged = arguments.search_dataset_path is not None
        if staged != (arguments.hard_search_manifest is not None):
            raise OrchestratorError(
                "--search-dataset-path and --hard-search-manifest are required "
                "together"
            )
        if config.search_protocol_explicit and not staged:
            raise OrchestratorError(
                "configs with search_protocol require --search-dataset-path "
                "and --hard-search-manifest; full-validation-per-candidate "
                "execution is disabled"
            )
        samples = _load_validation_samples(
            arguments.dataset_path,
            arguments.evidence_dir,
            split_manifest["splits"]["validation"]["sample_ids"],
        )
        if staged and arguments.proposer_history_reparse is not None:
            raise OrchestratorError(
                "staged runs cannot import legacy validation history into the "
                "proposer"
            )
        prior_search_feedback = (
            load_reparsed_search_feedback(
                REPOSITORY_ROOT,
                arguments.proposer_history_reparse,
            )
            if arguments.proposer_history_reparse is not None
            else None
        )
        limits, early_stopping = _settings(
            arguments,
            run_id,
            resume,
            config=config,
            staged=staged,
        )

        proposer: Any = _DryRunBoundary()
        solver: Any = object()
        if not arguments.dry_run:
            if not arguments.live_api:
                raise OrchestratorError(
                    "search execution is disabled without --live-api; "
                    "use --dry-run for offline validation"
                )
            from model_inference.remote_client import (
                RemoteChatCompletionsClient,
            )

            proposer = CodexCLIProposer(
                config=CodexCLIProposerConfig(
                    model=config.proposer_model,
                    reasoning_effort=config.proposer_reasoning_effort,
                )
            )
            solver = RemoteChatCompletionsClient()

        if staged:
            hard_search_manifest = load_hard_search_manifest(
                arguments.hard_search_manifest
            )
            search_samples = _load_exact_samples(
                arguments.search_dataset_path,
                arguments.evidence_dir,
                [
                    item["sample_id"]
                    for item in hard_search_manifest["items"]
                ],
                "hard search",
            )
            orchestrator = StagedMetaHarnessOrchestrator(
                repository_root=REPOSITORY_ROOT,
                run_id=run_id,
                config=config,
                split_manifest=split_manifest,
                hard_search_manifest=hard_search_manifest,
                search_samples=search_samples,
                validation_samples=samples,
                solver=solver,
                proposer=proposer,
                limits=limits,
                early_stopping=early_stopping,
                requested_settings=_requested_settings(arguments),
            )
        else:
            orchestrator = MetaHarnessOrchestrator(
                repository_root=REPOSITORY_ROOT,
                run_id=run_id,
                config=config,
                split_manifest=split_manifest,
                validation_samples=samples,
                solver=solver,
                proposer=proposer,
                limits=limits,
                early_stopping=early_stopping,
                prior_search_feedback=prior_search_feedback,
            )
        if arguments.dry_run:
            print(
                json.dumps(
                    orchestrator.planned_workload(),
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0

        state = orchestrator.run(
            resume=resume,
            safe_resume_changes=arguments.safe_resume_change,
        )
        print(
            json.dumps(
                {
                    "run_id": state["run_id"],
                    "status": state["status"],
                    "stop_reason": state["stop_reason"],
                    "best_candidate_id": state["best_candidate_id"],
                    "budgets": state["budgets"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, OrchestratorError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _settings(
    arguments: argparse.Namespace,
    run_id: str,
    resume: bool,
    *,
    config: Any,
    staged: bool,
) -> tuple[BudgetLimits, EarlyStoppingConfig]:
    previous_limits: dict[str, Any] = {}
    previous_early: dict[str, Any] = {}
    if resume:
        state = load_run_state(REPOSITORY_ROOT, run_id)
        previous_limits = state["configuration"]["limits"]
        previous_early = state["configuration"]["early_stopping"]

    if staged:
        protocol = config.search_protocol
        limit_defaults = BudgetLimits(
            max_iterations=protocol.target_iterations,
            max_candidates=(
                protocol.target_iterations
                * protocol.candidates_per_iteration
            ),
        ).as_dict()
        early_defaults = EarlyStoppingConfig(
            patience=protocol.early_stopping_patience,
            min_delta=0.0,
            min_iterations=protocol.min_iterations,
        ).as_dict()
    else:
        limit_defaults = BudgetLimits().as_dict()
        early_defaults = EarlyStoppingConfig().as_dict()
    limits = BudgetLimits(
        max_iterations=_selected(
            arguments.max_iterations,
            previous_limits,
            limit_defaults,
            "max_iterations",
        ),
        max_candidates=_selected(
            arguments.max_candidates,
            previous_limits,
            limit_defaults,
            "max_candidates",
        ),
        max_solver_calls=_selected(
            arguments.max_solver_calls,
            previous_limits,
            limit_defaults,
            "max_solver_calls",
        ),
        max_tokens=_selected(
            arguments.max_tokens,
            previous_limits,
            limit_defaults,
            "max_tokens",
        ),
        max_wall_time_seconds=_selected(
            arguments.max_wall_time,
            previous_limits,
            limit_defaults,
            "max_wall_time_seconds",
        ),
        max_consecutive_failures=_selected(
            arguments.max_consecutive_failures,
            previous_limits,
            limit_defaults,
            "max_consecutive_failures",
        ),
    )
    early = EarlyStoppingConfig(
        patience=_selected(
            arguments.patience,
            previous_early,
            early_defaults,
            "patience",
        ),
        min_delta=_selected(
            arguments.min_delta,
            previous_early,
            early_defaults,
            "min_delta",
        ),
        min_iterations=_selected(
            arguments.min_iterations,
            previous_early,
            early_defaults,
            "min_iterations",
        ),
    )
    return limits, early


def _requested_settings(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "limits": {
            "max_iterations": arguments.max_iterations,
            "max_candidates": arguments.max_candidates,
            "max_solver_calls": arguments.max_solver_calls,
            "max_tokens": arguments.max_tokens,
            "max_wall_time_seconds": arguments.max_wall_time,
            "max_consecutive_failures": arguments.max_consecutive_failures,
        },
        "early_stopping": {
            "patience": arguments.patience,
            "min_delta": arguments.min_delta,
            "min_iterations": arguments.min_iterations,
        },
    }


def _selected(
    explicit: Any,
    previous: dict[str, Any],
    defaults: dict[str, Any],
    field: str,
) -> Any:
    if explicit is not None:
        return explicit
    if field in previous:
        return previous[field]
    return defaults[field]


def _load_validation_samples(
    dataset_path: Path,
    evidence_dir: Path | None,
    validation_ids: Sequence[Any],
) -> tuple[Any, ...]:
    adapter = get_dataset_adapter("SciVer")
    loaded = adapter.load(dataset_path, evidence_dir=evidence_dir)
    wanted = {_identity(sample_id) for sample_id in validation_ids}
    loaded_ids = [_identity(sample.sample_id) for sample in loaded]
    if len(loaded_ids) != len(set(loaded_ids)) or set(loaded_ids) != wanted:
        raise ValueError(
            "--dataset-path must contain exactly the fixed validation split; "
            "search and final-test samples are forbidden"
        )
    return tuple(loaded)


def _load_exact_samples(
    dataset_path: Path,
    evidence_dir: Path | None,
    sample_ids: Sequence[Any],
    label: str,
) -> tuple[Any, ...]:
    adapter = get_dataset_adapter("SciVer")
    loaded = adapter.load(dataset_path, evidence_dir=evidence_dir)
    wanted = {_identity(sample_id) for sample_id in sample_ids}
    actual = [_identity(sample.sample_id) for sample in loaded]
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError(
            f"--search-dataset-path must contain exactly the immutable {label} "
            "samples"
        )
    return tuple(loaded)


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
        raise ValueError("sample IDs must be JSON serializable") from exc


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(cli())
