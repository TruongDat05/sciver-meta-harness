"""One strict, persisted-compatible run identifier contract for boundaries."""

from __future__ import annotations

import re
from typing import Any


EXPERIMENT_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class RunIdError(ValueError):
    """Raised when a run identifier is unsafe or incompatible with paths."""


def validate_run_identity(value: Any) -> str:
    """Return a run ID accepted by every durable entry point."""

    if not isinstance(value, str) or not EXPERIMENT_RUN_ID_PATTERN.fullmatch(value):
        raise RunIdError("run_id must be a safe identifier")
    return value


__all__ = [
    "EXPERIMENT_RUN_ID_PATTERN",
    "RunIdError",
    "validate_run_identity",
]
