"""Zero-configuration root launcher for the canonical SMOKE stage.

Resolves its own repository root, a shared stable run identity, the dataset,
and the trusted preparation layout internally, then orders trusted server
entry points: deterministic preparation followed by the isolated live SMOKE
request.  No experiment logic, argparse/dispatch/sanitization, or credential
handling is reimplemented here; those remain authoritative in the shared
operator helpers and the M6 server interface.  Live SMOKE runs only when
``--live-smoke`` is passed or ``RUN_LIVE_SMOKE`` is exactly ``"1"``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import scripts.run_meta_harness as cli_module  # noqa: E402
from utils.cli_environment import load_cli_environment  # noqa: E402

LIVE_FLAG = "--live-smoke"
AUTHORIZATION_ENV = "RUN_LIVE_SMOKE"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or reuse the isolated SMOKE stage.")
    parser.add_argument("--run-id", help="optional run override (default official_v3)")
    parser.add_argument("--dataset-path", help="dataset location when not discoverable")
    parser.add_argument(
        LIVE_FLAG,
        action="store_true",
        help="required separate authorization before the isolated SMOKE dispatch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    load_cli_environment()
    authorized = arguments.live_smoke or os.environ.get(AUTHORIZATION_ENV) == "1"

    def dispatch() -> object:
        run_id = cli_module.resolve_operator_run_id(override=arguments.run_id)
        dataset = cli_module.resolve_operator_dataset(
            REPOSITORY_ROOT, override=arguments.dataset_path
        )
        paths = cli_module.operator_preparation_paths(REPOSITORY_ROOT, run_id)
        cli_module.prepare_run(
            repository_root=paths["repository_root"],
            run_id=run_id,
            dataset_path=dataset,
            preparation_directory=paths["preparation_root"],
        )
        return cli_module.run_meta_harness_smoke(
            repository_root=paths["repository_root"],
            run_id=run_id,
            search_safe_manifest_path=paths["search_safe_manifest"],
            search_records_path=paths["search_records"],
            source_commit=None,
            authorize_smoke_execution=authorized,
        )

    return cli_module.operator_run(dispatch)


if __name__ == "__main__":
    sys.exit(main())
