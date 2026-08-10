#!/usr/bin/env python3
"""Export one verified frozen winner as provider-neutral prompt artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from meta_harness.finalize import FinalizationError, load_frozen_winner
from meta_harness.prompt_family import TEMPLATE_KEYS
from meta_harness.schemas import canonical_json


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export exact frozen meta_cot templates and audit metadata "
            "without final-test access."
        ),
    )
    parser.add_argument("--frozen-winner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        winner = load_frozen_winner(arguments.frozen_winner)
        prompts = {
            method: winner["templates"][method] for method in TEMPLATE_KEYS
        }
        metadata = {
            "schema_version": 1,
            "method": "meta_cot",
            "candidate_id": winner["candidate_id"],
            "source_run_id": winner["run_id"],
            "optimized_for_model": winner[
                "search_solver_configuration"
            ]["model"],
            "baseline_method": "cot",
            "validation_macro_f1": winner["validation"]["macro_f1"],
            "prompt_sha256": winner["prompt_sha256"],
            "candidate_sha256": winner["candidate_sha256"],
            "split_sha256": winner["split_hashes"]["split_sha256"],
            "config_sha256": winner[
                "search_solver_configuration_sha256"
            ],
            "code_revision": winner["code_revision"],
            "frozen_artifact_sha256": winner["artifact_sha256"],
        }
        prompt_path = arguments.output_dir / "meta_cot.prompts.json"
        metadata_path = arguments.output_dir / "meta_cot.metadata.json"
        _atomic_create_or_verify(prompt_path, prompts)
        _atomic_create_or_verify(metadata_path, metadata)
        print(
            json.dumps(
                {
                    "prompt_path": str(prompt_path),
                    "metadata_path": str(metadata_path),
                    "prompt_sha256": winner["prompt_sha256"],
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, FinalizationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _atomic_create_or_verify(
    path: Path,
    value: Mapping[str, Any],
) -> None:
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
                f"existing export must be readable valid JSON: {path}"
            ) from exc
        if canonical_json(existing) != canonical_json(value):
            raise ValueError(f"refusing to replace different export: {path}")
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
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError:
            _atomic_create_or_verify(path, value)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(cli())
