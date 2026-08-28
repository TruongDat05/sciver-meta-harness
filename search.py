"""Zero-configuration root launcher for the canonical SEARCH stage.

Resolves its own repository root, a shared stable run identity, and the trusted
preparation layout internally, then delegates live SEARCH through the trusted
M6 server interface.  It validates that the SEARCH-safe artifacts exist (the
engine requires a compatible completed SMOKE receipt) but performs no
preparation itself.  No experiment logic is reimplemented here.  Live SEARCH
runs only when ``--live-search`` is passed or ``RUN_FULL_SEARCH`` is exactly
``"1"``.
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

LIVE_FLAG = "--live-search"
AUTHORIZATION_ENV = "RUN_FULL_SEARCH"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start or resume the SEARCH stage.")
    parser.add_argument("--run-id", help="optional run override (default official_v3)")
    parser.add_argument(
        LIVE_FLAG,
        action="store_true",
        help="required explicit authorization before SEARCH dispatch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    load_cli_environment()
    authorized = arguments.live_search or os.environ.get(AUTHORIZATION_ENV) == "1"

    def dispatch() -> object:
        run_id = cli_module.resolve_operator_run_id(override=arguments.run_id)
        paths = cli_module.operator_preparation_paths(REPOSITORY_ROOT, run_id)
        for artifact in (paths["search_safe_manifest"], paths["search_records"]):
            if not Path(artifact).is_file():
                raise cli_module.ServerError(
                    "SEARCH-safe artifacts missing; run `python smoke.py --live-smoke` for this run first"
                )
        return cli_module.start_or_resume_search_run(
            repository_root=paths["repository_root"],
            run_id=run_id,
            search_safe_manifest_path=paths["search_safe_manifest"],
            search_records_path=paths["search_records"],
            source_commit=None,
            authorize_search_execution=authorized,
        )

    return cli_module.operator_run(dispatch)


if __name__ == "__main__":
    sys.exit(main())
