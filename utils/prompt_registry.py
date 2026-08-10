"""Provider-neutral registry for named SciVer prompt families."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from string import Template
from types import MappingProxyType

from meta_harness.prompt_family import validate_prompt_family
from utils.constant import COT_PROMPT


# Validation checks the fixed contract while retaining the original mapping and
# Template object identities as the source of truth for the baseline.
validate_prompt_family(COT_PROMPT)

_PROMPT_FAMILIES: dict[str, Mapping[str, Template]] = {
    "cot": COT_PROMPT,
}
_META_COT_ARTIFACT_SHA256: str | None = None


def get_prompt_family(name: str) -> Mapping[str, Template]:
    """Return a registered prompt family without rebuilding its templates."""

    try:
        return _PROMPT_FAMILIES[name]
    except (KeyError, TypeError) as exc:
        available = ", ".join(_PROMPT_FAMILIES)
        raise ValueError(
            f"Unknown prompt family {name!r}. Available prompt families: {available}"
        ) from exc


def register_frozen_prompt_family(
    frozen_winner_path: str | Path,
) -> Mapping[str, Template]:
    """Register one verified immutable winner under ``meta_cot``."""

    from meta_harness.finalize import (
        META_COT_VARIANT,
        load_frozen_winner,
    )

    global _META_COT_ARTIFACT_SHA256
    artifact = load_frozen_winner(frozen_winner_path)
    artifact_sha256 = artifact["artifact_sha256"]
    existing = _PROMPT_FAMILIES.get(META_COT_VARIANT)
    if existing is not None:
        if _META_COT_ARTIFACT_SHA256 != artifact_sha256:
            raise ValueError(
                "meta_cot is already registered from a different frozen winner"
            )
        return existing

    family = validate_prompt_family(artifact["templates"])
    registered = MappingProxyType(
        {method: family[method] for method in family}
    )
    _PROMPT_FAMILIES[META_COT_VARIANT] = registered
    _META_COT_ARTIFACT_SHA256 = artifact_sha256
    return registered


__all__ = ["get_prompt_family", "register_frozen_prompt_family"]
