#!/usr/bin/env python3
"""Retry unresolved Meta-Harness API failures without rerunning validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.retry import RetryError, retry_api_failures


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append targeted attempts for unresolved validation API failures."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--max-attempts", type=_positive_int, required=True)
    parser.add_argument("--live-api", action="store_true")
    return parser


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    try:
        summary = retry_api_failures(
            repository_root=REPOSITORY_ROOT,
            run_id=arguments.run_id,
            candidate_id=arguments.candidate_id,
            max_attempts=arguments.max_attempts,
            live_api=arguments.live_api,
        )
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, RetryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


if __name__ == "__main__":
    raise SystemExit(cli())
