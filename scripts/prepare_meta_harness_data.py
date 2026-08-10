#!/usr/bin/env python3
"""Build an immutable paper split and validation-only search input offline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.config import load_meta_harness_config
from meta_harness.full_search_v3_preparation import (
    FullSearchV3PreparationError,
    prepare_full_search_v3,
)
from meta_harness.schemas import canonical_json
from meta_harness.hard_search import (
    build_hard_search_manifest,
    materialize_hard_search_records,
    save_hard_search_manifest,
)
from meta_harness.split_manager import (
    build_split_manifest,
    save_split_manifest,
)
from utils.dataset_adapters import get_dataset_adapter


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare offline Meta-Harness data under an explicit protocol.",
    )
    parser.add_argument(
        "--protocol",
        choices=("legacy", "sciver_full_search_v3"),
        default="legacy",
        help="legacy preserves the staged preparation interface; V3 is isolated.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument(
        "--reserved-search-output",
        type=Path,
        help=(
            "optional immutable materialization of the complete reserved "
            "search split for baseline evaluation"
        ),
    )
    parser.add_argument(
        "--baseline-search-results",
        type=Path,
        help="offline JSONL baseline results for the complete reserved search split",
    )
    parser.add_argument("--hard-search-manifest", type=Path)
    parser.add_argument("--search-output", type=Path)
    parser.add_argument(
        "--v3-config",
        type=Path,
        help="optional complete locked sciver_full_search_v3 configuration JSON",
    )
    parser.add_argument(
        "--v3-private-dir",
        type=Path,
        help="trusted-only directory for the authoritative V3 private manifest",
    )
    parser.add_argument(
        "--v3-search-dir",
        type=Path,
        help="SEARCH-facing directory for the safe manifest and ordered records",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.protocol == "sciver_full_search_v3":
        return _v3_cli(arguments)
    try:
        _require_legacy_arguments(arguments)
        config = load_meta_harness_config(arguments.config)
        samples = get_dataset_adapter("SciVer").load(
            arguments.dataset_path,
            evidence_dir=arguments.evidence_dir,
        )
        records = [
            {
                **dict(sample.record),
                "sample_id": sample.sample_id,
            }
            for sample in samples
        ]
        manifest = build_split_manifest(records, config)
        save_split_manifest(arguments.split_manifest, manifest)

        validation_ids = {
            _identity(sample_id)
            for sample_id in manifest["splits"]["validation"]["sample_ids"]
        }
        validation_records = [
            record
            for record in records
            if _identity(record["sample_id"]) in validation_ids
        ]
        if {
            _identity(record["sample_id"])
            for record in validation_records
        } != validation_ids:
            raise ValueError(
                "failed to materialize every validation sample"
            )
        _atomic_create_or_verify_json(
            arguments.validation_output,
            validation_records,
        )
        search_ids = {
            _identity(sample_id)
            for sample_id in manifest["splits"]["search"]["sample_ids"]
        }
        reserved_search_records = [
            record
            for record in records
            if _identity(record["sample_id"]) in search_ids
        ]
        if arguments.reserved_search_output is not None:
            _atomic_create_or_verify_json(
                arguments.reserved_search_output,
                reserved_search_records,
            )
        hard_search_summary = None
        hard_arguments = (
            arguments.baseline_search_results,
            arguments.hard_search_manifest,
            arguments.search_output,
        )
        if any(value is not None for value in hard_arguments):
            if any(value is None for value in hard_arguments):
                raise ValueError(
                    "--baseline-search-results, --hard-search-manifest, and "
                    "--search-output are required together"
                )
            baseline_results = _load_jsonl(
                arguments.baseline_search_results
            )
            hard_manifest = build_hard_search_manifest(
                reserved_search_records,
                baseline_results,
                manifest,
                target_size=config.search_protocol.search_examples,
                guard_fraction=config.search_protocol.guard_fraction,
            )
            save_hard_search_manifest(
                arguments.hard_search_manifest,
                hard_manifest,
            )
            hard_records = materialize_hard_search_records(
                reserved_search_records,
                hard_manifest,
            )
            _atomic_create_or_verify_json(
                arguments.search_output,
                hard_records,
            )
            hard_search_summary = {
                "search_output": str(arguments.search_output),
                "hard_search_manifest": str(arguments.hard_search_manifest),
                "hard_search_sha256": hard_manifest["hard_search_sha256"],
                "search_samples": len(hard_records),
                "baseline_errors": hard_manifest["baseline_error_count"],
                "guard_examples": hard_manifest["guard_count"],
            }
        print(
            json.dumps(
                {
                    "split_manifest": str(arguments.split_manifest),
                    "validation_output": str(arguments.validation_output),
                    "split_sha256": manifest["split_sha256"],
                    "dataset_sha256": manifest["dataset_sha256"],
                    "total_samples": manifest["total_samples"],
                    "validation_samples": len(validation_records),
                    "reserved_search_output": (
                        None
                        if arguments.reserved_search_output is None
                        else str(arguments.reserved_search_output)
                    ),
                    "paper_counts": {
                        name: split["paper_count"]
                        for name, split in manifest["splits"].items()
                    },
                    "hard_search": hard_search_summary,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _v3_cli(arguments: argparse.Namespace) -> int:
    if arguments.v3_private_dir is None or arguments.v3_search_dir is None:
        print(
            "error: --v3-private-dir and --v3-search-dir are required for sciver_full_search_v3",
            file=sys.stderr,
        )
        return 1
    if any(
        value is not None
        for value in (
            arguments.config,
            arguments.split_manifest,
            arguments.validation_output,
            arguments.reserved_search_output,
            arguments.baseline_search_results,
            arguments.hard_search_manifest,
            arguments.search_output,
            arguments.evidence_dir,
        )
    ):
        print(
            "error: legacy preparation options cannot be combined with sciver_full_search_v3",
            file=sys.stderr,
        )
        return 1
    try:
        artifacts = prepare_full_search_v3(
            source_path=arguments.dataset_path,
            private_directory=arguments.v3_private_dir,
            search_directory=arguments.v3_search_dir,
            config_path=arguments.v3_config,
        )
        print(json.dumps(artifacts.summary, sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, FullSearchV3PreparationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _require_legacy_arguments(arguments: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--config", arguments.config),
            ("--split-manifest", arguments.split_manifest),
            ("--validation-output", arguments.validation_output),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "legacy preparation requires " + ", ".join(missing)
        )
    if any(
        value is not None
        for value in (
            arguments.v3_config,
            arguments.v3_private_dir,
            arguments.v3_search_dir,
        )
    ):
        raise ValueError("V3 preparation options require --protocol sciver_full_search_v3")


def _atomic_create_or_verify_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                "validation output must be readable valid JSON"
            ) from exc
        if canonical_json(existing) != canonical_json(value):
            raise ValueError(
                "refusing to replace different validation-only data"
            )
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o600)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _atomic_create_or_verify_json(path, value)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(value: Any) -> str:
    return canonical_json(value)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "baseline search results must be readable valid JSONL"
        ) from exc
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("baseline search result rows must be objects")
    return values


if __name__ == "__main__":
    raise SystemExit(cli())
