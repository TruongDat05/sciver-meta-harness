"""Injected, offline-safe solver boundary for ``sciver_full_search_v3``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from string import Template
from typing import Any, Protocol

from meta_harness.config import (
    Config,
    canonical_experiment_config,
)
from model_inference.remote_client import (
    DEFAULT_CHAT_COMPLETIONS_PATH,
    DEFAULT_MODELS_PATH,
)
from model_inference.remote_api import prepare_remote_requests
from utils.answer_parser import PARSER_VERSION


SOLVER_REQUEST_IDENTITY_SCHEMA_VERSION = (
    "sciver_full_search_v3_solver_request_identity_v1"
)
EXPERIMENT_TRANSPORT_CONTRACT_VERSION = (
    "sciver_full_search_v3_compatible_transport_v1"
)
EXPERIMENT_MODEL_LIST_PREFLIGHT_VERSION = (
    "sciver_full_search_v3_model_list_preflight_v1"
)


class SolverError(ValueError):
    """Raised when the injected solver contract is invalid."""


class LiveSolverDisabledError(RuntimeError):
    """Raised when live client construction was not explicitly authorized."""


@dataclass(frozen=True)
class SolverGenerationSettings:
    """Frozen generation fields propagated to every solver request."""

    temperature: int
    top_p: int
    seed: int
    n: int
    stream: bool
    max_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise SolverError(
                "solver temperature must be a non-negative number"
            )
        if (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or self.top_p <= 0
            or self.top_p > 1
        ):
            raise SolverError(
                "solver top_p must be greater than zero and at most one"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise SolverError("solver seed must be an integer")
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n != 1:
            raise SolverError("solver n must equal one")
        if self.stream is not False:
            raise SolverError("solver stream must be false")
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise SolverError(
                "solver max_tokens must be a positive integer"
            )

    @classmethod
    def from_config(cls, config: Config) -> "SolverGenerationSettings":
        if not isinstance(config, Config):
            raise SolverError(
                "Meta-Harness solver requests require Config"
            )
        return cls(
            temperature=config.solver_temperature,
            top_p=config.solver_top_p,
            seed=config.solver_seed,
            n=config.solver_n,
            stream=config.solver_stream,
            max_tokens=config.solver_max_tokens,
        )

    def as_request_options(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "n": self.n,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
        }


@dataclass(frozen=True)
class SolverRequest:
    """One already-prepared, parser-independent solver request."""

    model: str
    messages: tuple[Mapping[str, Any], ...]
    generation: SolverGenerationSettings


@dataclass(frozen=True)
class SolverRequestIdentity:
    """Safe immutable identity for one exact SEARCH solver request."""

    protocol_id: str
    config_sha256: str
    split_sha256: str
    search_membership_sha256: str
    prompt_sha256: str
    candidate_id: str
    solver_identity_sha256: str
    solver_model: str
    generation: SolverGenerationSettings
    parser_version: str
    sample_id: str
    request_payload_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "config_sha256",
            "split_sha256",
            "search_membership_sha256",
            "prompt_sha256",
            "solver_identity_sha256",
            "request_payload_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        for field in (
            "protocol_id",
            "candidate_id",
            "solver_model",
            "parser_version",
            "sample_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise SolverError(f"{field} must be non-empty text")
        if not isinstance(self.generation, SolverGenerationSettings):
            raise SolverError(
                "generation must be SolverGenerationSettings"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SOLVER_REQUEST_IDENTITY_SCHEMA_VERSION,
            "protocol_id": self.protocol_id,
            "config_sha256": self.config_sha256,
            "split_sha256": self.split_sha256,
            "search_membership_sha256": self.search_membership_sha256,
            "prompt_sha256": self.prompt_sha256,
            "candidate_id": self.candidate_id,
            "solver_identity_sha256": self.solver_identity_sha256,
            "solver_model": self.solver_model,
            "generation": self.generation.as_request_options(),
            "parser_version": self.parser_version,
            "sample_id": self.sample_id,
            "request_payload_sha256": self.request_payload_sha256,
        }

    def sha256(self) -> str:
        return _canonical_sha256(self.as_dict())


@dataclass(frozen=True)
class SolverResult:
    """Completion text and optional provider-neutral token usage."""

    content: str
    usage: Mapping[str, int] | None = None


class SolverClient(Protocol):
    """Injectable boundary implemented by deterministic fakes or the live adapter."""

    def complete(self, request: SolverRequest) -> SolverResult: ...


def build_solver_request(
    record: Mapping[str, Any],
    prompt: Mapping[str, Template],
    *,
    config: Config | None = None,
) -> SolverRequest:
    """Build one request through the existing SciVer message constructor."""

    active_config = config or canonical_experiment_config()
    generation = SolverGenerationSettings.from_config(active_config)
    messages = prepare_remote_requests([record], prompt)[0]
    return SolverRequest(
        model=active_config.solver_model,
        messages=tuple(messages),
        generation=generation,
    )


def build_solver_request_identity(
    request: SolverRequest,
    *,
    sample_id: str,
    candidate_id: str,
    prompt_sha256: str,
    split_sha256: str,
    search_membership_sha256: str,
    solver_identity_sha256: str,
    config: Config | None = None,
    parser_version: str = PARSER_VERSION,
) -> SolverRequestIdentity:
    """Bind every cache-defining identity without retaining messages or images."""

    if not isinstance(request, SolverRequest):
        raise SolverError("request must be a SolverRequest")
    active_config = config or canonical_experiment_config()
    expected_generation = SolverGenerationSettings.from_config(active_config)
    if request.model != active_config.solver_model:
        raise SolverError(
            "request model does not match the immutable configuration"
        )
    if request.generation != expected_generation:
        raise SolverError(
            "request generation does not match the immutable configuration"
        )
    request_payload_sha256 = solver_request_payload_sha256(request)
    return SolverRequestIdentity(
        protocol_id=active_config.protocol_id,
        config_sha256=active_config.sha256(),
        split_sha256=split_sha256,
        search_membership_sha256=search_membership_sha256,
        prompt_sha256=prompt_sha256,
        candidate_id=candidate_id,
        solver_identity_sha256=solver_identity_sha256,
        solver_model=request.model,
        generation=request.generation,
        parser_version=parser_version,
        sample_id=sample_id,
        request_payload_sha256=request_payload_sha256,
    )


def solver_request_payload_sha256(request: SolverRequest) -> str:
    """Hash the exact model-visible request without retaining its payload."""

    if not isinstance(request, SolverRequest):
        raise SolverError("request must be a SolverRequest")
    return _canonical_sha256(
        {
            "model": request.model,
            "messages": list(request.messages),
            "generation": request.generation.as_request_options(),
        }
    )


def execute_solver_request(
    client: SolverClient,
    request: SolverRequest,
) -> SolverResult:
    """Execute one request only through an explicitly injected client."""

    if not isinstance(request, SolverRequest):
        raise SolverError("request must be a SolverRequest")
    complete = getattr(client, "complete", None)
    if not callable(complete):
        raise SolverError("client must provide a complete method")
    result = complete(request)
    if not isinstance(result, SolverResult):
        raise SolverError("client must return a SolverResult")
    return result


class _CompatibleHTTPSolverClient:
    """Private adapter around the existing compatible HTTP client."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(self, request: SolverRequest) -> SolverResult:
        content = self._client.create_chat_completion(
            request.model,
            request.messages,
            generation_options=request.generation.as_request_options(),
        )
        usage = _safe_usage(getattr(self._client, "last_usage", None))
        return SolverResult(content=content, usage=usage)

    def list_model_ids(self) -> tuple[str, ...]:
        """Delegate model discovery without exposing the response object."""

        return self._client.list_model_ids()


def create_live_solver_client(
    *,
    allow_live_requests: bool = False,
    api_url_is_base: bool = False,
) -> SolverClient:
    """Construct the compatible HTTP client only after explicit live opt-in."""

    if allow_live_requests is not True:
        raise LiveSolverDisabledError(
            "live solver construction is disabled without explicit authorization"
        )

    # Keep credential reads and HTTP-client construction behind the live gate.
    from model_inference.remote_client import (
        RemoteChatCompletionsClient,
        RetrySettings,
    )
    from model_inference.remote_config import validate_config_for_live_request

    live_config = validate_config_for_live_request()
    client_arguments = {
        "api_key": live_config.api_key,
        "api_url": live_config.api_url,
        "timeout": live_config.timeout_seconds,
        # The solver boundary owns POST retry accounting and classification.
        "retry_settings": RetrySettings(max_retries=0),
    }
    if api_url_is_base:
        client = RemoteChatCompletionsClient.from_base_url(
            **client_arguments,
            models_path=DEFAULT_MODELS_PATH,
            chat_completions_path=DEFAULT_CHAT_COMPLETIONS_PATH,
        )
    else:
        # Existing generic callers continue to supply a complete Chat
        # Completions URL unless they explicitly opt into base-URL mode.
        client = RemoteChatCompletionsClient(**client_arguments)
    return _CompatibleHTTPSolverClient(client)


def preflight_live_solver_model_list(
    client: SolverClient,
    *,
    required_model: str | None = None,
) -> dict[str, Any]:
    """Verify the locked model and return only safe model-list identities."""

    active_config = canonical_experiment_config()
    locked_model = (
        active_config.solver_model if required_model is None else required_model
    )
    if locked_model != active_config.solver_model:
        raise SolverError(
            "model-list preflight must require the immutable solver model"
        )
    list_model_ids = getattr(client, "list_model_ids", None)
    if not callable(list_model_ids):
        raise SolverError(
            "live solver does not provide model-list preflight"
        )
    model_ids = list_model_ids()
    if not isinstance(model_ids, tuple) or any(
        not isinstance(item, str) or not item for item in model_ids
    ):
        raise SolverError(
            "model-list preflight returned an incompatible sanitized result"
        )
    if locked_model not in model_ids:
        raise SolverError(
            "model-list preflight did not expose the immutable solver model"
        )
    return {
        "schema_version": EXPERIMENT_MODEL_LIST_PREFLIGHT_VERSION,
        "status": "passed",
        "model_count": len(model_ids),
        "model_ids_sha256": _canonical_sha256(sorted(model_ids)),
        "locked_model_sha256": hashlib.sha256(
            locked_model.encode("utf-8")
        ).hexdigest(),
    }


def _safe_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    allowed = {"input_tokens", "output_tokens", "total_tokens"}
    result = {
        key: item
        for key, item in value.items()
        if key in allowed
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
    }
    return result or None


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SolverError(
            "solver request identity inputs must contain JSON values"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SolverError(
            f"{field} must be a lowercase hexadecimal SHA-256"
        )


__all__ = [
    "EXPERIMENT_MODEL_LIST_PREFLIGHT_VERSION",
    "EXPERIMENT_TRANSPORT_CONTRACT_VERSION",
    "SolverError",
    "LiveSolverDisabledError",
    "SOLVER_REQUEST_IDENTITY_SCHEMA_VERSION",
    "SolverClient",
    "SolverGenerationSettings",
    "SolverRequest",
    "SolverRequestIdentity",
    "SolverResult",
    "build_solver_request",
    "build_solver_request_identity",
    "create_live_solver_client",
    "execute_solver_request",
    "preflight_live_solver_model_list",
    "solver_request_payload_sha256",
]
