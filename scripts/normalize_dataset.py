#!/usr/bin/env python3
"""Normalize one released benchmark into the unified SciVer evaluation schema."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.dataset_adapters import (
    DatasetAdapterError,
    SUPPORTED_DATASETS,
    get_dataset_adapter,
)


_NORMALIZED_FIELDS = (
    "sample_id",
    "claim_type",
    "claim",
    "context",
    "caption",
    "caption1",
    "caption2",
    "image_path",
    "item1_path",
    "item2_path",
    "paper_path",
    "section",
    "type",
    "item",
    "item1_type",
    "item1",
    "item2_type",
    "item2",
    "gold_label",
    "split",
)


def normalize_dataset(
    dataset_name: str,
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    evidence_dir: str | Path | None = None,
) -> int:
    """Write normalized JSON and return its sample count."""

    destination = Path(output_path)
    rendered_evidence = (
        Path(evidence_dir)
        if evidence_dir is not None
        else destination.parent / f"{destination.stem}_evidence"
    )
    samples = get_dataset_adapter(dataset_name).load(
        dataset_path,
        evidence_dir=rendered_evidence,
    )
    records = [
        {
            field: sample.record[field]
            for field in _NORMALIZED_FIELDS
            if field in sample.record
        }
        for sample in samples
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_name = output_file.name
            json.dump(records, output_file, indent=2, ensure_ascii=False, allow_nan=False)
            output_file.write("\n")
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return len(records)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize a released claim-verification dataset using verified fields."
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--evidence-dir",
        help="directory for rendered local evidence (required only to override the default)",
    )
    return parser.parse_args(argv)


def cli(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        count = normalize_dataset(
            arguments.dataset,
            arguments.dataset_path,
            arguments.output,
            evidence_dir=arguments.evidence_dir,
        )
    except (DatasetAdapterError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"dataset": arguments.dataset, "normalized_samples": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
