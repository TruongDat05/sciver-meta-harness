#!/usr/bin/env python3
"""Freeze a search winner and optionally run its one-time final test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.finalize import (
    FinalizationError,
    execute_final_test,
    execute_final_test_pair,
    freeze_winner,
    frozen_winner_path,
)
from meta_harness.split_manager import load_split_manifest
from utils.cli_environment import load_cli_environment
from utils.prompt_registry import register_frozen_prompt_family


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the validation winner before any final-test use.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--live-api", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        if arguments.live_api and not arguments.confirm_final_test:
            raise FinalizationError(
                "--live-api requires --confirm-final-test for final-test use"
            )
        if arguments.confirm_final_test and not arguments.live_api:
            raise FinalizationError(
                "final-test execution requires both "
                "--confirm-final-test and --live-api"
            )
        if arguments.live_api:
            load_cli_environment()
            from model_inference.remote_config import (
                validate_config_for_live_request,
            )

            validate_config_for_live_request()
        split_manifest = load_split_manifest(arguments.split_manifest)
        winner = freeze_winner(
            repository_root=REPOSITORY_ROOT,
            run_id=arguments.run_id,
            split_manifest=split_manifest,
        )
        register_frozen_prompt_family(
            frozen_winner_path(REPOSITORY_ROOT, arguments.run_id)
        )
        output: dict[str, Any] = {
            "run_id": winner["run_id"],
            "candidate_id": winner["candidate_id"],
            "prompt_variant": winner["prompt_variant"],
            "prompt_sha256": winner["prompt_sha256"],
            "validation_macro_f1": winner["validation"]["macro_f1"],
            "frozen_winner": str(
                frozen_winner_path(REPOSITORY_ROOT, arguments.run_id)
            ),
        }

        if arguments.confirm_final_test:
            if arguments.dataset_path is None:
                raise FinalizationError(
                    "--dataset-path is required for final-test execution"
                )
            final_samples = _load_final_samples(
                arguments.dataset_path,
                arguments.evidence_dir,
                split_manifest["splits"]["final_test"]["sample_ids"],
            )
            from model_inference.remote_client import (
                RemoteChatCompletionsClient,
            )

            execution_arguments = {
                "repository_root": REPOSITORY_ROOT,
                "run_id": arguments.run_id,
                "split_manifest": split_manifest,
                "final_samples": final_samples,
                "solver": RemoteChatCompletionsClient(),
                "confirm_final_test": True,
            }
            if "search_protocol" in winner["search_solver_configuration"]:
                receipt = execute_final_test(**execution_arguments)
                receipts = {winner["prompt_variant"]: receipt}
            else:
                receipts = execute_final_test_pair(**execution_arguments)
            output["final_test_completion_sha256"] = {
                prompt_variant: receipt["completion_sha256"]
                for prompt_variant, receipt in receipts.items()
            }
            output["final_test_stage_budget"] = {
                prompt_variant: receipt.get("stage_budget")
                for prompt_variant, receipt in receipts.items()
                if receipt.get("stage_budget") is not None
            }

        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, FinalizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _load_final_samples(
    dataset_path: Path,
    evidence_dir: Path | None,
    final_ids: Sequence[Any],
) -> tuple[Any, ...]:
    from utils.dataset_adapters import get_dataset_adapter

    loaded = get_dataset_adapter("SciVer").load(
        dataset_path,
        evidence_dir=evidence_dir,
    )
    wanted = {_identity(sample_id) for sample_id in final_ids}
    selected = tuple(
        sample for sample in loaded if _identity(sample.sample_id) in wanted
    )
    if {_identity(sample.sample_id) for sample in selected} != wanted:
        raise FinalizationError(
            "dataset does not contain every frozen final-test sample"
        )
    return selected


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
        raise FinalizationError(
            "sample IDs must be JSON serializable"
        ) from exc


if __name__ == "__main__":
    raise SystemExit(cli())
