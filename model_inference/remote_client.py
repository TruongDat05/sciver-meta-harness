"""Provider-neutral HTTP client for remote Chat Completions requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import os
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import requests


DEFAULT_TIMEOUT_SECONDS = 60.0


class RemoteClientError(Exception):
    """Base class for remote client failures."""

    def __init__(
        self,
        message: str,
        *,
        http_status_code: int | None = None,
        response_error_code: str | None = None,
        response_error_message: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status_code = http_status_code
        self.response_error_code = response_error_code
        self.response_error_message = response_error_message
        self.retry_after_seconds = retry_after_seconds


class InvalidConfigurationError(RemoteClientError, ValueError):
    """Raised when client configuration is missing or invalid."""


class AuthenticationError(RemoteClientError):
    """Raised when the remote API rejects authentication."""


class PermissionDeniedError(RemoteClientError):
    """Raised when the remote API denies access to the request."""


class RateLimitError(RemoteClientError):
    """Raised when rate limiting persists after configured retries."""


class InvalidRequestError(RemoteClientError):
    """Raised when the remote API rejects a request as invalid."""


class ServerError(RemoteClientError):
    """Raised when the remote API returns a server error."""


class NetworkError(RemoteClientError):
    """Raised when a request fails because of a network error."""


class RequestTimeoutError(NetworkError):
    """Raised when a request times out."""


class MalformedJSONError(RemoteClientError):
    """Raised when a successful response is not valid JSON."""


class UnexpectedResponseSchemaError(RemoteClientError):
    """Raised when a response does not match the expected schema."""


class UnexpectedHTTPStatusError(RemoteClientError):
    """Raised when a response has an unrecognized HTTP status."""


@dataclass(frozen=True)
class RetrySettings:
    """Retry count and bounded exponential-backoff settings."""

    max_retries: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.max_retries, bool) or not isinstance(
            self.max_retries, int
        ):
            raise InvalidConfigurationError(
                "max_retries must be a non-negative integer."
            )
        if self.max_retries < 0:
            raise InvalidConfigurationError(
                "max_retries must be a non-negative integer."
            )
        _validate_non_negative_number(
            self.initial_backoff_seconds, "initial_backoff_seconds"
        )
        _validate_non_negative_number(
            self.max_backoff_seconds, "max_backoff_seconds"
        )


class RemoteChatCompletionsClient:
    """Send non-streaming Chat Completions requests to a remote API."""

    _RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_settings: RetrySettings | None = None,
        session: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_api_key = (
            os.environ.get("API_KEY") if api_key is None else api_key
        )
        resolved_api_url = (
            os.environ.get("API_URL") if api_url is None else api_url
        )

        self._api_key = _validate_api_key(resolved_api_key)
        self._api_url = _validate_api_url(resolved_api_url)
        self._timeout = _validate_positive_number(timeout, "timeout")
        self._retry_settings = retry_settings or RetrySettings()
        if not isinstance(self._retry_settings, RetrySettings):
            raise InvalidConfigurationError(
                "retry_settings must be a RetrySettings instance."
            )
        if not callable(sleep):
            raise InvalidConfigurationError("sleep must be callable.")

        self._session = requests.Session() if session is None else session
        if not callable(getattr(self._session, "post", None)):
            raise InvalidConfigurationError("session must provide a post method.")
        self._sleep = sleep
        self.last_usage: dict[str, int] | None = None

    def create_chat_completion(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        """Return ``choices[0].message.content`` from a remote response."""

        if not isinstance(model, str) or not model.strip():
            raise InvalidConfigurationError("model must be a non-empty string.")
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise InvalidConfigurationError("messages must be a sequence.")

        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
        }
        payload.update(_validate_generation_options(generation_options))
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        self.last_usage = None
        response = self._post_with_retries(headers, payload)
        body = _decode_response_body(response)
        content = _extract_content(body)
        self.last_usage = _extract_usage(body)
        return content

    def _post_with_retries(
        self, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> Any:
        for attempt in range(self._retry_settings.max_retries + 1):
            try:
                response = self._session.post(
                    self._api_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout,
                )
            except requests.exceptions.InvalidJSONError:
                raise InvalidRequestError(
                    "Remote API request payload is not JSON serializable."
                ) from None
            except requests.exceptions.Timeout:
                if self._can_retry(attempt):
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise RequestTimeoutError(
                    "Remote API request timed out after configured retries."
                ) from None
            except requests.exceptions.RequestException:
                if self._can_retry(attempt):
                    self._sleep(self._backoff_delay(attempt))
                    continue
                raise NetworkError(
                    "Remote API request failed after configured retries."
                ) from None

            status_code = getattr(response, "status_code", None)
            if isinstance(status_code, bool) or not isinstance(status_code, int):
                raise UnexpectedResponseSchemaError(
                    "Remote API response did not include a valid HTTP status."
                )

            if status_code in self._RETRYABLE_STATUS_CODES and self._can_retry(
                attempt
            ):
                self._sleep(self._retry_delay(response, attempt))
                continue

            if 200 <= status_code < 300:
                return response
            self._raise_for_status(response, status_code)

        raise AssertionError("unreachable")

    def _can_retry(self, attempt: int) -> bool:
        return attempt < self._retry_settings.max_retries

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = _parse_retry_after(getattr(response, "headers", None))
        if retry_after is None:
            return self._backoff_delay(attempt)
        return min(retry_after, self._retry_settings.max_backoff_seconds)

    def _backoff_delay(self, attempt: int) -> float:
        initial = self._retry_settings.initial_backoff_seconds
        maximum = self._retry_settings.max_backoff_seconds
        if initial == 0 or maximum == 0:
            return 0.0
        if initial >= maximum:
            return maximum

        doublings_to_limit = math.ceil(math.log2(maximum / initial))
        if attempt >= doublings_to_limit:
            return maximum
        return min(math.ldexp(initial, attempt), maximum)

    @staticmethod
    def _raise_for_status(response: Any, status_code: int) -> None:
        error_code, error_message = _response_error_details(response)
        details = {
            "http_status_code": status_code,
            "response_error_code": error_code,
            "response_error_message": error_message,
            "retry_after_seconds": _parse_retry_after(
                getattr(response, "headers", None)
            ),
        }
        if status_code == 400:
            raise InvalidRequestError(
                "Remote API rejected the request (HTTP 400).",
                **details,
            )
        if status_code == 401:
            raise AuthenticationError(
                "Remote API authentication failed (HTTP 401).",
                **details,
            )
        if status_code == 403:
            raise PermissionDeniedError(
                "Remote API denied permission (HTTP 403).",
                **details,
            )
        if status_code == 429:
            raise RateLimitError(
                "Remote API rate limit persisted after configured retries (HTTP 429).",
                **details,
            )
        if 400 <= status_code < 500:
            raise InvalidRequestError(
                f"Remote API rejected the request (HTTP {status_code}).",
                **details,
            )
        if 500 <= status_code < 600:
            raise ServerError(
                f"Remote API server error (HTTP {status_code}).",
                **details,
            )
        raise UnexpectedHTTPStatusError(
            f"Remote API returned an unexpected HTTP status ({status_code}).",
            **details,
        )


def _validate_api_key(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidConfigurationError(
            "API_KEY is required to create a remote client."
        )
    if "\r" in value or "\n" in value:
        raise InvalidConfigurationError("API_KEY contains invalid characters.")
    return value


def _validate_generation_options(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the additive non-streaming generation request surface."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise InvalidConfigurationError(
            "generation_options must be a mapping when provided."
        )
    allowed = {"temperature", "top_p", "seed", "n", "stream", "max_tokens"}
    if any(not isinstance(key, str) for key in value):
        raise InvalidConfigurationError(
            "generation_options field names must be strings."
        )
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise InvalidConfigurationError(
            "generation_options contains unsupported fields: "
            + ", ".join(unexpected)
        )

    result = dict(value)
    if "temperature" in result:
        _validate_non_negative_number(result["temperature"], "temperature")
    if "top_p" in result:
        top_p = result["top_p"]
        if (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not math.isfinite(top_p)
            or top_p <= 0
            or top_p > 1
        ):
            raise InvalidConfigurationError(
                "top_p must be a finite number greater than zero and at most one."
            )
    if "seed" in result and (
        isinstance(result["seed"], bool) or not isinstance(result["seed"], int)
    ):
        raise InvalidConfigurationError("seed must be an integer.")
    if "n" in result and (
        isinstance(result["n"], bool)
        or not isinstance(result["n"], int)
        or result["n"] != 1
    ):
        raise InvalidConfigurationError("n must equal one.")
    if "stream" in result and result["stream"] is not False:
        raise InvalidConfigurationError("stream must be false.")
    if "max_tokens" in result and (
        isinstance(result["max_tokens"], bool)
        or not isinstance(result["max_tokens"], int)
        or result["max_tokens"] <= 0
    ):
        raise InvalidConfigurationError("max_tokens must be a positive integer.")
    return result


def _validate_api_url(value: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidConfigurationError(
            "API_URL is required to create a remote client."
        )
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise InvalidConfigurationError(
            "API_URL must be an HTTP(S) URL without embedded credentials."
        )
    return value


def _validate_positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise InvalidConfigurationError(
            f"{name} must be a finite number greater than zero."
        )
    return float(value)


def _validate_non_negative_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise InvalidConfigurationError(
            f"{name} must be a finite non-negative number."
        )
    return float(value)


def _parse_retry_after(headers: Any) -> float | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    value = headers.get("Retry-After")
    if value is None:
        return None

    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            retry_time = parsedate_to_datetime(str(value))
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_time.tzinfo is None:
            retry_time = retry_time.replace(tzinfo=timezone.utc)
        seconds = (retry_time - datetime.now(timezone.utc)).total_seconds()

    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _decode_response_body(response: Any) -> dict[str, Any]:
    try:
        body = response.json()
    except (TypeError, ValueError):
        raise MalformedJSONError(
            "Remote API returned malformed JSON."
        ) from None

    if not isinstance(body, dict):
        raise UnexpectedResponseSchemaError(
            "Remote API JSON response must be an object."
        )
    return body


def _response_error_details(response: Any) -> tuple[str | None, str | None]:
    """Extract bounded diagnostic fields without retaining the response body."""

    try:
        body = response.json()
    except (AttributeError, TypeError, ValueError):
        body = None

    code: Any = None
    message: Any = None
    if isinstance(body, Mapping):
        error = body.get("error")
        if isinstance(error, Mapping):
            code = error.get("code", error.get("type"))
            message = error.get("message")
        else:
            code = body.get("code", body.get("type"))
            message = body.get("message", body.get("detail"))

    return (
        _bounded_error_text(code),
        _bounded_error_text(message),
    )


def _bounded_error_text(value: Any) -> str | None:
    if value is None or isinstance(value, (Mapping, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:2000]


def _extract_content(body: Mapping[str, Any]) -> str:

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise UnexpectedResponseSchemaError(
            "Remote API response is missing choices[0]."
        )

    first_choice = choices[0]
    if not isinstance(first_choice, dict) or not isinstance(
        first_choice.get("message"), dict
    ):
        raise UnexpectedResponseSchemaError(
            "Remote API response is missing choices[0].message."
        )

    message = first_choice["message"]
    if "content" not in message or not isinstance(message["content"], str):
        raise UnexpectedResponseSchemaError(
            "Remote API response is missing choices[0].message.content."
        )
    return message["content"]


def _extract_usage(body: Mapping[str, Any]) -> dict[str, int] | None:
    usage = body.get("usage")
    if not isinstance(usage, Mapping):
        return None
    aliases = {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }
    normalized: dict[str, int] = {}
    for target, names in aliases.items():
        value = next((usage[name] for name in names if name in usage), None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[target] = value
    return normalized or None


__all__ = [
    "AuthenticationError",
    "DEFAULT_TIMEOUT_SECONDS",
    "InvalidConfigurationError",
    "InvalidRequestError",
    "MalformedJSONError",
    "NetworkError",
    "PermissionDeniedError",
    "RateLimitError",
    "RemoteChatCompletionsClient",
    "RemoteClientError",
    "RequestTimeoutError",
    "RetrySettings",
    "ServerError",
    "UnexpectedHTTPStatusError",
    "UnexpectedResponseSchemaError",
]
