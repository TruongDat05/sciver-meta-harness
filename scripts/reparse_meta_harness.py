#!/usr/bin/env python3
"""Re-parse stored Meta-Harness raw responses without network access."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.metrics import EvaluationError
from meta_harness.reparse import ReparseError, reparse_run


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-parse stored raw_response fields into a new directory. "
            "No solver or network client is used."
        )
    )
    parser.add_argument("source_run", help="immutable source run directory")
    parser.add_argument("output", help="new output directory (must not exist)")
    parser.add_argument(
        "--expected-sample-count",
        type=int,
        help="require this many records in every evaluation",
    )
    parser.add_argument(
        "--expected-evaluation-count",
        type=int,
        help="require this many baseline/candidate evaluations",
    )
    return parser.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        summary = reparse_run(
            arguments.source_run,
            arguments.output,
            expected_sample_count=arguments.expected_sample_count,
            expected_evaluation_count=arguments.expected_evaluation_count,
        )
    except (EvaluationError, ReparseError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"re-parsed {summary['evaluation_count']} evaluations with "
        f"{summary['parser_version']} into {arguments.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
