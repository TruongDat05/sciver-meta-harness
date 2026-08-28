"""Zero-configuration root launcher for the canonical paired FINAL stage.

Resolves its own repository root, a shared stable run identity, the dataset,
and the trusted preparation layout internally.  It verifies the FINAL-safe
artifacts exist, then, only if SEARCH is terminal, delegates the create-once,
conflict-fail-closed winner freeze and finally the trusted paired FINAL server
interface (which runs FINAL preflight and resumes rather than redispatching
completed work).  No experiment logic is reimplemented here.  Live FINAL runs
only when ``--live-final`` is passed or ``RUN_FINAL_ONCE`` is exactly ``"1"``.
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

LIVE_FLAG = "--live-final"
AUTHORIZATION_ENV = "RUN_FINAL_ONCE"


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start or resume the paired FINAL stage.")
    parser.add_argument("--run-id", help="optional run override (default official_v3)")
    parser.add_argument("--dataset-path", help="dataset location when not discoverable")
    parser.add_argument(
        LIVE_FLAG,
        action="store_true",
        help="required separate explicit authorization before FINAL dispatch",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    load_cli_environment()
    authorized = arguments.live_final or os.environ.get(AUTHORIZATION_ENV) == "1"

    def dispatch() -> object:
        run_id = cli_module.resolve_operator_run_id(override=arguments.run_id)
        dataset = cli_module.resolve_operator_dataset(
            REPOSITORY_ROOT, override=arguments.dataset_path
        )
        paths = cli_module.operator_preparation_paths(REPOSITORY_ROOT, run_id)
        for artifact in (paths["search_safe_manifest"], paths["private_manifest"]):
            if not Path(artifact).is_file():
                raise cli_module.ServerError(
                    "FINAL artifacts missing; run smoke then search for this run first"
                )
        cli_module.freeze_server_run_winner(
            repository_root=paths["repository_root"], run_id=run_id
        )
        return cli_module.start_or_resume_server_run_final(
            repository_root=paths["repository_root"],
            run_id=run_id,
            dataset_path=dataset,
            private_manifest_path=paths["private_manifest"],
            search_safe_manifest_path=paths["search_safe_manifest"],
            authorize_final_execution=authorized,
        )

    return cli_module.operator_run(dispatch)


if __name__ == "__main__":
    sys.exit(main())
