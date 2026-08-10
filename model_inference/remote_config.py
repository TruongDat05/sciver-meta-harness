"""Runtime configuration and supported-model registry for remote inference."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from typing import Mapping


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3


class RemoteConfigurationError(ValueError):
    """Raised when remote API configuration is missing or invalid."""


class UnsupportedModelError(ValueError):
    """Raised when a model identifier is not in the remote model registry."""


@dataclass(frozen=True)
class ModelSpecification:
    """Provider-neutral metadata for a supported remote model."""

    identifier: str


_MODEL_IDENTIFIERS = (
    "Qwen2.5-VL-7B-Instruct",
    "gemma-4-31B-it",
    "gemma-4-26B-A4B-it",
    "gemma-3-27b-it",
)

_MODEL_REGISTRY = {
    identifier: ModelSpecification(identifier=identifier)
    for identifier in _MODEL_IDENTIFIERS
}


def list_supported_models() -> tuple[str, ...]:
    """Return all supported remote model identifiers in registry order."""

    return tuple(_MODEL_REGISTRY)


def validate_model_identifier(model_identifier: str) -> str:
    """Validate and return a supported remote model identifier."""

    if model_identifier not in _MODEL_REGISTRY:
        supported = ", ".join(list_supported_models())
        raise UnsupportedModelError(
            f"Unsupported remote model {model_identifier!r}. "
            f"Supported models: {supported}."
        )
    return model_identifier


def get_model_specification(model_identifier: str) -> ModelSpecification:
    """Return the specification for a supported remote model."""

    return _MODEL_REGISTRY[validate_model_identifier(model_identifier)]


@dataclass(frozen=True)
class RemoteAPIConfig:
    """Remote API settings loaded explicitly at runtime."""

    api_key: str | None = field(repr=False)
    api_url: str | None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES

    @classmethod
    def from_environment(
        cls, environment: Mapping[str, str] | None = None
    ) -> "RemoteAPIConfig":
        """Load settings without requiring credentials for offline use."""

        source = os.environ if environment is None else environment
        return cls(
            api_key=source.get("API_KEY"),
            api_url=source.get("API_URL"),
            timeout_seconds=_read_positive_float(
                source, "API_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS
            ),
            max_retries=_read_non_negative_int(
                source, "API_MAX_RETRIES", DEFAULT_MAX_RETRIES
            ),
        )

    def validate_for_live_request(self) -> "RemoteAPIConfig":
        """Ensure required settings are present before any live request."""

        missing = []
        if not self.api_url or not self.api_url.strip():
            missing.append("API_URL")
        if not self.api_key or not self.api_key.strip():
            missing.append("API_KEY")

        if missing:
            names = " and ".join(missing)
            raise RemoteConfigurationError(
                f"Missing required remote API configuration: {names}."
            )
        return self


def load_remote_config(
    environment: Mapping[str, str] | None = None,
) -> RemoteAPIConfig:
    """Load remote configuration at call time."""

    return RemoteAPIConfig.from_environment(environment)


def validate_config_for_live_request(
    config: RemoteAPIConfig | None = None,
) -> RemoteAPIConfig:
    """Load, if needed, and validate configuration for a live request."""

    candidate = load_remote_config() if config is None else config
    return candidate.validate_for_live_request()


def _read_positive_float(
    environment: Mapping[str, str], name: str, default: float
) -> float:
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise RemoteConfigurationError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise RemoteConfigurationError(
            f"{name} must be a finite number greater than zero."
        )
    return value


def _read_non_negative_int(
    environment: Mapping[str, str], name: str, default: int
) -> int:
    raw_value = environment.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RemoteConfigurationError(f"{name} must be an integer.") from exc
    if value < 0:
        raise RemoteConfigurationError(f"{name} must be zero or greater.")
    return value
