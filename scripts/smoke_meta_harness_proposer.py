#!/usr/bin/env python3
"""Run the production Codex proposer boundary without invoking the solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.candidate_store import CandidateStore
from meta_harness.config import load_meta_harness_config
from meta_harness.proposer.codex_cli import (
    CodexCLIProposer,
    CodexCLIProposerConfig,
    CodexCLIProposerError,
)
from meta_harness.proposer.feedback import load_reparsed_search_feedback


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Invoke the exact production proposer subprocess once without "
            "loading a dataset or constructing a solver client."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--proposer-history-reparse",
        type=Path,
        help=(
            "read-only parser-v2 validation history supplied to the proposer"
        ),
    )
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        config = load_meta_harness_config(arguments.config)
        proposer = CodexCLIProposer(
            config=CodexCLIProposerConfig(
                model=config.proposer_model,
                reasoning_effort=config.proposer_reasoning_effort,
            )
        )
        search_feedback = (
            load_reparsed_search_feedback(
                REPOSITORY_ROOT,
                arguments.proposer_history_reparse,
            )
            if arguments.proposer_history_reparse is not None
            else None
        )
        result = proposer.propose(
            CandidateStore(REPOSITORY_ROOT, arguments.run_id),
            iteration=1,
            parent_candidate_ids=["baseline_cot"],
            validation_scores={},
            aggregate_metrics={
                "completed_candidates": 0,
                "eligible_rate": 0.0,
                "solver_calls": 0,
                "tokens": 0,
            },
            failure_summaries={},
            search_feedback=search_feedback,
        )
        print(
            json.dumps(
                {
                    "status": "success",
                    "run_id": arguments.run_id,
                    "candidate_ids": [
                        candidate.candidate_id
                        for candidate in result.batch.candidates
                    ],
                    "cli_version": result.metadata["cli_version"],
                    "return_code": result.metadata["return_code"],
                    "audit_path": str(result.audit_path),
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, CodexCLIProposerError) as exc:
        payload = {
            "status": "error",
            "run_id": arguments.run_id,
            "error": str(exc),
        }
        audit_path = getattr(exc, "audit_path", None)
        if audit_path is not None:
            payload["audit_path"] = str(audit_path)
        print(json.dumps(payload, sort_keys=True, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())
