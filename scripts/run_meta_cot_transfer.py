#!/usr/bin/env python3
"""Run unchanged cot and exact frozen meta_cot on three transfer models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.finalize import (
    FinalizationError,
    TRANSFER_MODELS,
    execute_transfer_matrix,
    frozen_winner_path,
    load_frozen_winner,
)
from meta_harness.split_manager import load_split_manifest
from scripts.finalize_meta_harness import _load_final_samples
from utils.cli_environment import load_cli_environment
from utils.prompt_registry import register_frozen_prompt_family


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate unchanged cot and frozen meta_cot bytes without "
            "re-optimization."
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--live-api", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        if (
            arguments.confirm_final_test is not True
            or arguments.live_api is not True
        ):
            raise FinalizationError(
                "transfer requires both --confirm-final-test and --live-api"
            )
        load_cli_environment()
        from model_inference.remote_config import (
            validate_config_for_live_request,
        )

        validate_config_for_live_request()
        winner_path = frozen_winner_path(
            REPOSITORY_ROOT,
            arguments.run_id,
        )
        winner = load_frozen_winner(winner_path)
        register_frozen_prompt_family(winner_path)
        split_manifest = load_split_manifest(arguments.split_manifest)
        final_samples = _load_final_samples(
            arguments.dataset_path,
            arguments.evidence_dir,
            split_manifest["splits"]["final_test"]["sample_ids"],
        )

        from model_inference.remote_client import (
            RemoteChatCompletionsClient,
        )

        client = RemoteChatCompletionsClient()
        receipts = execute_transfer_matrix(
            repository_root=REPOSITORY_ROOT,
            run_id=arguments.run_id,
            split_manifest=split_manifest,
            final_samples=final_samples,
            solvers={model: client for model in TRANSFER_MODELS},
            confirm_final_test=True,
        )
        print(
            json.dumps(
                {
                    "run_id": winner["run_id"],
                    "candidate_id": winner["candidate_id"],
                    "prompt_variant": winner["prompt_variant"],
                    "prompt_sha256": winner["prompt_sha256"],
                    "models": {
                        model: {
                            prompt_variant: receipt["completion_sha256"]
                            for prompt_variant, receipt in model_receipts.items()
                        }
                        for model, model_receipts in receipts.items()
                    },
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, FinalizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
