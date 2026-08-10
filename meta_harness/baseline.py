"""Canonical baseline snapshot helpers backed only by ``COT_PROMPT``."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Mapping

from meta_harness.prompt_family import (
    TEMPLATE_KEYS,
    load_prompt_family,
    serialize_prompt_family,
)
from utils.constant import COT_PROMPT


BASELINE_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "meta_harness"
    / "baseline_cot.json"
)


class BaselinePromptDriftError(ValueError):
    """Raised when a stored baseline differs from canonical SciVer prompts."""


def canonical_baseline_sources() -> Mapping[str, str]:
    """Return immutable raw template strings from the canonical mapping."""

    return MappingProxyType(
        {
            method: COT_PROMPT[method].template
            for method in TEMPLATE_KEYS
        }
    )


def canonical_baseline_json() -> str:
    """Serialize the canonical prompts deterministically for export."""

    return serialize_prompt_family(COT_PROMPT)


def validate_baseline_snapshot(
    path: str | Path = BASELINE_SNAPSHOT_PATH,
) -> Path:
    """Require exact source strings and canonical serialized bytes."""

    snapshot_path = Path(path)
    try:
        serialized = snapshot_path.read_text(encoding="utf-8")
        snapshot = load_prompt_family(snapshot_path)
    except (OSError, UnicodeError, ValueError) as exc:
        raise BaselinePromptDriftError(
            "baseline snapshot must be readable valid prompt-family JSON"
        ) from exc

    expected = canonical_baseline_sources()
    for method in TEMPLATE_KEYS:
        if snapshot[method].template != expected[method]:
            raise BaselinePromptDriftError(
                f"baseline snapshot template {method!r} differs from COT_PROMPT"
            )
    if serialized != canonical_baseline_json():
        raise BaselinePromptDriftError(
            "baseline snapshot bytes are not the canonical COT_PROMPT export"
        )
    return snapshot_path


def export_baseline_snapshot(
    path: str | Path = BASELINE_SNAPSHOT_PATH,
) -> Path:
    """Atomically export the exact canonical prompt mapping."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_baseline_json().encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    validate_baseline_snapshot(destination)
    return destination


__all__ = [
    "BASELINE_SNAPSHOT_PATH",
    "BaselinePromptDriftError",
    "canonical_baseline_json",
    "canonical_baseline_sources",
    "export_baseline_snapshot",
    "validate_baseline_snapshot",
]
