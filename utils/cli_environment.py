"""Explicit environment-file loading for executable command-line paths."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_cli_environment(
    path: str | Path = DEFAULT_ENV_PATH,
) -> bool:
    """Load local CLI settings without replacing an existing environment."""

    return bool(load_dotenv(dotenv_path=Path(path), override=False))


__all__ = ["DEFAULT_ENV_PATH", "load_cli_environment"]
