"""Bounded transport retry policy for the injected solver boundary."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from dataclasses import dataclass
from enum import Enum
import math
import re
import time
from typing import Any, Callable

import requests

from meta_harness.solver import (
    SolverClient,
    SolverRequest,
    SolverResult,
    execute_solver_request,
)
from model_inference.remote_client import (
    InvalidConfigurationError,
    InvalidRequestError,
    MalformedJSONError,
    NetworkError,
    RateLimitError,
    RemoteClientError,
    RequestTimeoutError,
    UnexpectedHTTPStatusError,
    UnexpectedResponseSchemaError,
)


DEFAULT_PERMANENT_5XX = frozenset({501, 505})
DEFAULT_RETRYABLE_5XX = frozenset(range(500, 600)) - DEFAULT_PERMANENT_5XX
MAXIMUM_TRANSPORT_ATTEMPTS = 10


class FailureCategory(str, Enum):
    """Stable provider-neutral classification for solver failures."""

    TIMEOUT = "timeout"
    CONNECTION = "connection"
    HTTP_408 = "http_408"
    HTTP_429 = "http_429"
    HTTP_5XX_RETRYABLE = "http_5xx_retryable"
    HTTP_5XX_PERMANENT = "http_5xx_permanent"
    HTTP_4XX_PERMANENT = "http_4xx_permanent"
    PROTOCOL_OR_SCHEMA = "protocol_or_schema"
    PARSER = "parser"
    CANCELLATION = "cancellation"
    UNEXPECTED = "unexpected"


class SolverParserFailure(Exception):
    """Marker used by later parser-owning code; never a transport retry."""


@dataclass(frozen=True)
class SolverRetryPolicy:
    """Explicit bounded attempt and backoff limits for one request."""

    maximum_attempts: int = 4
    initial_backoff_seconds: float = 1.0
    maximum_backoff_seconds: float = 30.0
    retryable_5xx: frozenset[int] = DEFAULT_RETRYABLE_5XX

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_attempts, bool)
            or not isinstance(self.maximum_attempts, int)
            or self.maximum_attempts < 1
            or self.maximum_attempts > MAXIMUM_TRANSPORT_ATTEMPTS
        ):
            raise ValueError(
                "maximum_attempts must be between 1 and "
                f"{MAXIMUM_TRANSPORT_ATTEMPTS}"
            )
        for field in ("initial_backoff_seconds", "maximum_backoff_seconds"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{field} must be a finite non-negative number")
        try:
            statuses = frozenset(self.retryable_5xx)
        except TypeError as exc:
            raise ValueError("retryable_5xx must contain HTTP status integers") from exc
        if any(
            isinstance(status, bool)
            or not isinstance(status, int)
            or status < 500
            or status > 599
            for status in statuses
        ):
            raise ValueError("retryable_5xx must contain only 5xx status integers")
        object.__setattr__(self, "retryable_5xx", statuses)


@dataclass(frozen=True)
class FailureClassification:
    category: FailureCategory
    retryable: bool
    http_status_code: int | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True)
class SolverFailureMetadata:
    """Sanitized structured metadata safe for persistence as a failure."""

    category: FailureCategory
    retryable: bool
    exhausted: bool
    attempt_count: int
    maximum_attempts: int
    http_status_code: int | None
    retry_delays_seconds: tuple[float, ...]
    elapsed_seconds: float
    cause_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "retryable": self.retryable,
            "exhausted": self.exhausted,
            "attempt_count": self.attempt_count,
            "maximum_attempts": self.maximum_attempts,
            "http_status_code": self.http_status_code,
            "retry_delays_seconds": list(self.retry_delays_seconds),
            "elapsed_seconds": self.elapsed_seconds,
            "cause_type": self.cause_type,
        }


class SolverExecutionFailure(RuntimeError):
    """Raised with sanitized metadata for terminal solver failures."""

    def __init__(self, metadata: SolverFailureMetadata) -> None:
        self.metadata = metadata
        reason = (
            "bounded transport attempts were exhausted"
            if metadata.exhausted
            else "the failure is not retryable"
        )
        super().__init__(f"solver request failed because {reason}")


def classify_solver_failure(
    error: BaseException,
    policy: SolverRetryPolicy | None = None,
) -> FailureClassification:
    """Classify one failure without retaining its message or response payload."""

    active_policy = policy or SolverRetryPolicy()
    if not isinstance(active_policy, SolverRetryPolicy):
        raise TypeError("policy must be SolverRetryPolicy")

    if isinstance(error, (asyncio.CancelledError, FutureCancelledError)):
        return FailureClassification(FailureCategory.CANCELLATION, False)
    if isinstance(error, SolverParserFailure):
        return FailureClassification(FailureCategory.PARSER, False)

    status = _http_status(error)
    retry_after = _retry_after(error)
    if status == 408:
        return FailureClassification(
            FailureCategory.HTTP_408,
            True,
            status,
            retry_after,
        )
    if status == 429 or isinstance(error, RateLimitError):
        return FailureClassification(
            FailureCategory.HTTP_429,
            True,
            429 if status is None else status,
            retry_after,
        )
    if status is not None and 500 <= status <= 599:
        retryable = status in active_policy.retryable_5xx
        return FailureClassification(
            FailureCategory.HTTP_5XX_RETRYABLE
            if retryable
            else FailureCategory.HTTP_5XX_PERMANENT,
            retryable,
            status,
            retry_after,
        )
    if status is not None and 400 <= status <= 499:
        return FailureClassification(
            FailureCategory.HTTP_4XX_PERMANENT,
            False,
            status,
            retry_after,
        )
    if isinstance(error, (RequestTimeoutError, requests.exceptions.Timeout, TimeoutError)):
        return FailureClassification(FailureCategory.TIMEOUT, True)
    if isinstance(
        error,
        (
            NetworkError,
            requests.exceptions.ConnectionError,
            ConnectionError,
        ),
    ):
        return FailureClassification(FailureCategory.CONNECTION, True)
    if isinstance(
        error,
        (
            InvalidConfigurationError,
            InvalidRequestError,
            MalformedJSONError,
            UnexpectedHTTPStatusError,
            UnexpectedResponseSchemaError,
            requests.exceptions.InvalidJSONError,
        ),
    ):
        return FailureClassification(FailureCategory.PROTOCOL_OR_SCHEMA, False)
    if isinstance(error, RemoteClientError):
        return FailureClassification(FailureCategory.PROTOCOL_OR_SCHEMA, False)
    return FailureClassification(FailureCategory.UNEXPECTED, False)


def execute_solver_request_with_retry(
    client: SolverClient,
    request: SolverRequest,
    *,
    policy: SolverRetryPolicy,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> SolverResult:
    """Execute one logical request with bounded transport-only retries."""

    if not isinstance(policy, SolverRetryPolicy):
        raise TypeError("policy must be SolverRetryPolicy")
    if not callable(sleeper):
        raise TypeError("sleeper must be callable")
    if not callable(clock):
        raise TypeError("clock must be callable")
    started = _clock_value(clock)
    retry_delays: list[float] = []

    for attempt in range(1, policy.maximum_attempts + 1):
        try:
            return execute_solver_request(client, request)
        except Exception as error:
            classification = classify_solver_failure(error, policy)
            terminal = not classification.retryable
            exhausted = classification.retryable and attempt >= policy.maximum_attempts
            if terminal or exhausted:
                metadata = SolverFailureMetadata(
                    category=classification.category,
                    retryable=classification.retryable,
                    exhausted=exhausted,
                    attempt_count=attempt,
                    maximum_attempts=policy.maximum_attempts,
                    http_status_code=classification.http_status_code,
                    retry_delays_seconds=tuple(retry_delays),
                    elapsed_seconds=max(0.0, _clock_value(clock) - started),
                    cause_type=_safe_type_name(error),
                )
                raise SolverExecutionFailure(metadata) from None

            delay = _retry_delay(classification, attempt, policy)
            retry_delays.append(delay)
            sleeper(delay)

    raise AssertionError("unreachable")


def _retry_delay(
    classification: FailureClassification,
    attempt: int,
    policy: SolverRetryPolicy,
) -> float:
    retry_after = classification.retry_after_seconds
    if retry_after is not None:
        return min(retry_after, float(policy.maximum_backoff_seconds))
    initial = float(policy.initial_backoff_seconds)
    maximum = float(policy.maximum_backoff_seconds)
    if initial == 0 or maximum == 0:
        return 0.0
    if initial >= maximum:
        return maximum
    exponent = attempt - 1
    limit = math.ceil(math.log2(maximum / initial))
    if exponent >= limit:
        return maximum
    return min(math.ldexp(initial, exponent), maximum)


def _http_status(error: BaseException) -> int | None:
    value = getattr(error, "http_status_code", None)
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
        return value
    return None


def _retry_after(error: BaseException) -> float | None:
    value = getattr(error, "retry_after_seconds", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def _clock_value(clock: Callable[[], float]) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("clock must return a finite number")
    return float(value)


def _safe_type_name(error: BaseException) -> str:
    name = type(error).__name__
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:128]
    return sanitized or "Exception"


__all__ = [
    "DEFAULT_PERMANENT_5XX",
    "DEFAULT_RETRYABLE_5XX",
    "FailureCategory",
    "FailureClassification",
    "MAXIMUM_TRANSPORT_ATTEMPTS",
    "SolverExecutionFailure",
    "SolverFailureMetadata",
    "SolverParserFailure",
    "SolverRetryPolicy",
    "classify_solver_failure",
    "execute_solver_request_with_retry",
]
