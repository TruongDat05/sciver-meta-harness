"""Thin, opt-in command wrapper for the M6 full-SEARCH server interface.

This wrapper owns no experiment logic.  It prints only sanitized structured
status and delegates every operation to :mod:`meta_harness.server_run`.
Credentials are intentionally absent from its argument parser; live commands
read them only through the M6 interface from the process environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.server_run import (
    ServerError,
    freeze_server_run_winner,
    inspect_server_run_activity,
    inspect_server_run_final_status,
    inspect_server_run_status,
    preflight_server_run_final,
    preflight_search_run,
    prepare_run,
    run_meta_harness_smoke,
    start_or_resume_server_run_final,
    start_or_resume_search_run,
)

from meta_harness.final_evaluation import FinalError
from meta_harness.preparation import PreparationError
from meta_harness.run_identity import EXPERIMENT_RUN_ID_PATTERN
from meta_harness.winner_freeze import FreezeError


_SENSITIVE_KEY = re.compile(
    r"(?:api[_ -]?(?:key|url)|authorization|bearer|credential|secret|"
    r"token|endpoint|payload|base64|image|sample|record|label|trace|"
    r"response|completion|request)",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_BASE64 = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}(?![A-Za-z0-9+/=])")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate sciver_full_search_v3 through the M6 server interface."
    )
    subcommands = parser.add_subparsers(dest="operation", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repository-root", required=True)
        command.add_argument("--run-id", required=True)

    prepare = subcommands.add_parser("prepare", help="create or verify M1 preparation")
    common(prepare)
    prepare.add_argument("--dataset-path", required=True)
    prepare.add_argument("--preparation-directory")
    prepare.add_argument("--config-path")

    search_preflight = subcommands.add_parser(
        "search-preflight", help="offline SEARCH plan; never dispatches"
    )
    common(search_preflight)
    _search_artifact_arguments(search_preflight)
    search_preflight.add_argument("--source-commit", required=True)
    search_preflight.add_argument("--solver-identity-sha256")

    smoke = subcommands.add_parser(
        "smoke", help="explicitly run or reuse one isolated P0 SMOKE request"
    )
    common(smoke)
    _search_artifact_arguments(smoke)
    smoke.add_argument("--source-commit", required=True)
    smoke.add_argument(
        "--live-smoke",
        action="store_true",
        help="required separate authorization before the isolated SMOKE dispatch",
    )

    search = subcommands.add_parser("search", help="explicitly start or resume SEARCH")
    common(search)
    _search_artifact_arguments(search)
    search.add_argument("--source-commit", required=True)
    search.add_argument(
        "--live-search",
        action="store_true",
        help="required explicit authorization before SEARCH dispatch",
    )

    search_status = subcommands.add_parser("search-status", help="inspect SEARCH state")
    common(search_status)
    activity = subcommands.add_parser("activity", help="inspect run-stage locks")
    common(activity)
    freeze = subcommands.add_parser("freeze", help="freeze only terminal SEARCH winner")
    common(freeze)

    final_preflight = subcommands.add_parser(
        "final-preflight", help="offline paired FINAL plan after freeze"
    )
    common(final_preflight)
    _final_artifact_arguments(final_preflight)
    final_preflight.add_argument("--solver-identity-sha256", required=True)

    final = subcommands.add_parser("final", help="explicitly start or resume paired FINAL")
    common(final)
    _final_artifact_arguments(final)
    final.add_argument(
        "--live-final",
        action="store_true",
        help="required separate explicit authorization before FINAL dispatch",
    )
    final_status = subcommands.add_parser("final-status", help="inspect FINAL state")
    common(final_status)
    return parser


def _search_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--search-safe-manifest", required=True)
    parser.add_argument("--search-records", required=True)


def _final_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--private-manifest", required=True)
    parser.add_argument("--search-safe-manifest", required=True)


def cli(
    argv: Sequence[str] | None = None,
    *,
    output: Callable[[str], None] = print,
) -> int:
    """Parse a command and print a sanitized M6 status object."""

    arguments = build_argument_parser().parse_args(argv)
    try:
        result = _dispatch(arguments)
    except ServerError as exc:
        output(f"error: {_sanitize_text(str(exc))}")
        return 2
    except Exception:
        output("error: server operation failed; inspect the local sanitized status and retry")
        return 1
    output(operator_render(result))
    return 0


def _dispatch(arguments: argparse.Namespace) -> Mapping[str, Any]:
    common = {"repository_root": arguments.repository_root, "run_id": arguments.run_id}
    if arguments.operation == "prepare":
        return prepare_run(
            **common,
            dataset_path=arguments.dataset_path,
            preparation_directory=arguments.preparation_directory,
            config_path=arguments.config_path,
        )
    if arguments.operation == "search-preflight":
        return preflight_search_run(
            **common,
            search_safe_manifest_path=arguments.search_safe_manifest,
            search_records_path=arguments.search_records,
            source_commit=arguments.source_commit,
            solver_identity_sha256=arguments.solver_identity_sha256,
        )
    if arguments.operation == "smoke":
        if arguments.live_smoke is not True:
            raise ServerError(
                "SMOKE is offline by default; pass --live-smoke only after explicit authorization"
            )
        return run_meta_harness_smoke(
            **common,
            search_safe_manifest_path=arguments.search_safe_manifest,
            search_records_path=arguments.search_records,
            source_commit=arguments.source_commit,
            authorize_smoke_execution=True,
        )
    if arguments.operation == "search":
        if arguments.live_search is not True:
            raise ServerError(
                "SEARCH is offline by default; pass --live-search only after explicit authorization"
            )
        return start_or_resume_search_run(
            **common,
            search_safe_manifest_path=arguments.search_safe_manifest,
            search_records_path=arguments.search_records,
            source_commit=arguments.source_commit,
            authorize_search_execution=True,
        )
    if arguments.operation == "search-status":
        return inspect_server_run_status(**common)
    if arguments.operation == "activity":
        return inspect_server_run_activity(**common)
    if arguments.operation == "freeze":
        return freeze_server_run_winner(**common)
    if arguments.operation == "final-preflight":
        return preflight_server_run_final(
            **common,
            dataset_path=arguments.dataset_path,
            private_manifest_path=arguments.private_manifest,
            search_safe_manifest_path=arguments.search_safe_manifest,
            solver_identity_sha256=arguments.solver_identity_sha256,
        )
    if arguments.operation == "final":
        if arguments.live_final is not True:
            raise ServerError(
                "FINAL is offline by default; pass --live-final only after separate authorization"
            )
        return start_or_resume_server_run_final(
            **common,
            dataset_path=arguments.dataset_path,
            private_manifest_path=arguments.private_manifest,
            search_safe_manifest_path=arguments.search_safe_manifest,
            authorize_final_execution=True,
        )
    if arguments.operation == "final-status":
        return inspect_server_run_final_status(**common)
    raise AssertionError("argument parser produced an unknown operation")


def _sanitize_value(value: Any, *, inherited_sensitive: bool = False) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_value(
                item, inherited_sensitive=inherited_sensitive or bool(_SENSITIVE_KEY.search(str(key)))
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_value(item, inherited_sensitive=inherited_sensitive) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_value(item, inherited_sensitive=inherited_sensitive) for item in value]
    if isinstance(value, str):
        return "<redacted>" if inherited_sensitive else _sanitize_text(value)
    return value


def _sanitize_text(value: str) -> str:
    return _BASE64.sub("<redacted>", _URL.sub("<redacted>", value))


OFFICIAL_RUN_ID = "official_v3"
DEFAULT_DATASET_RELATIVE = Path("data/sciver/testset.json")


def operator_render(result: Any) -> str:
    """Return a sanitized structured status for an operator result."""
    return json.dumps(_sanitize_value(result), sort_keys=True, separators=(",", ":"))


def operator_run(
    dispatch: Callable[[], Any],
    *,
    output: Callable[[str], None] = print,
) -> int:
    """Execute a trusted operator dispatch and render a sanitized result.

    This is the single exit-code/error-render seam that the root launchers
    share with the canonical CLI.  ``output`` receives exactly one sanitized
    line.
    """
    try:
        result = dispatch()
    except (ServerError, PreparationError, FinalError, FreezeError) as exc:
        output(f"error: {_sanitize_text(str(exc))}")
        return 2
    except Exception:
        output("error: server operation failed; inspect the local sanitized status and retry")
        return 1
    output(operator_render(result))
    return 0


def resolve_operator_run_id(
    override: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the shared operator run identity (override > env > default)."""
    source = os.environ if env is None else env
    value = override or source.get("SCIVER_RUN_ID") or OFFICIAL_RUN_ID
    if not isinstance(value, str) or not EXPERIMENT_RUN_ID_PATTERN.fullmatch(value):
        raise ServerError("SCIVER_RUN_ID must be a safe operator run identifier")
    return value


def resolve_operator_dataset(
    repository_root: str | Path,
    override: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the dataset path (override > env > repository default) and validate it."""
    source = os.environ if env is None else env
    value = override or source.get("SCIVER_DATASET_PATH")
    candidate = (
        Path(value)
        if value
        else Path(repository_root) / DEFAULT_DATASET_RELATIVE
    )
    path = Path(candidate).expanduser()
    if not path.is_file():
        raise ServerError(
            f"dataset not found at {path}; pass --dataset-path or set SCIVER_DATASET_PATH"
        )
    return path.resolve()


def operator_preparation_paths(
    repository_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Derive the trusted repository preparation layout for one run."""
    run_root = (
        Path(repository_root) / "workspace" / "meta_harness" / "full_search_v3" / run_id
    )
    preparation = run_root / "preparation"
    return {
        "repository_root": Path(repository_root).resolve(),
        "run_id": run_id,
        "run_root": run_root,
        "preparation_root": preparation,
        "search_safe_manifest": preparation / "search" / "search_safe_manifest.json",
        "search_records": preparation / "search" / "search_records.json",
        "private_manifest": preparation / "private" / "private_split_manifest.json",
    }


if __name__ == "__main__":
    sys.exit(cli())
