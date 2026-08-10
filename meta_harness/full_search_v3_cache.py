"""Create-once SEARCH completion cache for ``sciver_full_search_v3``."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from meta_harness.full_search_v3_solver import (
    SolverRequestIdentity,
    SolverResult,
)


SEARCH_CACHE_SCHEMA_VERSION = "sciver_full_search_v3_search_completion_v1"
_ARTIFACT_TYPE = "completed_search_request"
_SENSITIVE_PATTERNS = (
    re.compile(r"(?i)\bapi_(?:key|url)\b\s*[:=]"),
    re.compile(r"(?i)\bauthorization(?:_header)?\b\s*[:=]"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)data:image/[a-z0-9.+-]+;base64,"),
    re.compile(
        r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{128,}={0,2}"
        r"(?![A-Za-z0-9+/=])"
    ),
)


class SearchCacheError(ValueError):
    """Base class for safe SEARCH cache failures."""


class SearchCacheIntegrityError(SearchCacheError):
    """Raised when an existing entry is incomplete, corrupt, or mismatched."""


class SearchCacheConflictError(SearchCacheError):
    """Raised rather than overwriting an immutable completed request."""


class SearchCacheSafetyError(SearchCacheError):
    """Raised before sensitive request or completion material can be persisted."""


class FullSearchV3SearchCache:
    """Store only complete, identity-matched SEARCH solver completions."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)

    def entry_path(self, identity: SolverRequestIdentity) -> Path:
        _require_identity(identity)
        digest = identity.sha256()
        return self._directory / SEARCH_CACHE_SCHEMA_VERSION / digest[:2] / (
            f"{digest}.json"
        )

    def coordination_directory(self) -> Path:
        """Return the payload-free directory reserved for cache coordination."""

        return self._directory / SEARCH_CACHE_SCHEMA_VERSION / ".coordination"

    def get(self, identity: SolverRequestIdentity) -> SolverResult | None:
        """Return a complete matching entry, a miss, or a fail-closed error."""

        _require_identity(identity)
        path = self.entry_path(identity)
        if not path.exists():
            return None
        if not path.is_file():
            raise SearchCacheIntegrityError(
                "SEARCH cache entry is not a regular file; move it aside and retry"
            )
        payload, encoded = _load_entry(path)
        expected_fields = {
            "schema_version",
            "artifact_type",
            "request_identity_sha256",
            "request_identity",
            "completion",
            "completion_sha256",
        }
        if set(payload) != expected_fields:
            raise _corrupt_entry_error()
        if (
            payload["schema_version"] != SEARCH_CACHE_SCHEMA_VERSION
            or payload["artifact_type"] != _ARTIFACT_TYPE
        ):
            raise SearchCacheIntegrityError(
                "SEARCH cache entry uses an incompatible schema; move it aside and retry"
            )
        expected_identity = identity.as_dict()
        expected_identity_sha256 = identity.sha256()
        if (
            payload["request_identity"] != expected_identity
            or payload["request_identity_sha256"] != expected_identity_sha256
        ):
            raise SearchCacheIntegrityError(
                "SEARCH cache entry identity does not match its request; "
                "move it aside and retry"
            )
        result = _result_from_payload(payload["completion"])
        if payload["completion_sha256"] != _completion_sha256(result):
            raise _corrupt_entry_error()
        if encoded != _json_bytes(payload):
            raise SearchCacheIntegrityError(
                "SEARCH cache entry is not canonical; move it aside and retry"
            )
        _ensure_safe(identity, result)
        return result

    def put(
        self,
        identity: SolverRequestIdentity,
        result: SolverResult,
    ) -> SolverResult:
        """Atomically create one completed entry without replacing existing work."""

        _require_identity(identity)
        normalized = _validate_result(result)
        _ensure_safe(identity, normalized)
        path = self.entry_path(identity)
        existing = self.get(identity)
        if existing is not None:
            return _require_same_completion(existing, normalized)

        payload = {
            "schema_version": SEARCH_CACHE_SCHEMA_VERSION,
            "artifact_type": _ARTIFACT_TYPE,
            "request_identity_sha256": identity.sha256(),
            "request_identity": identity.as_dict(),
            "completion": _completion_payload(normalized),
            "completion_sha256": _completion_sha256(normalized),
        }
        encoded = _json_bytes(payload)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
            )
        except OSError:
            raise SearchCacheError(
                "SEARCH cache storage is unavailable; verify permissions and retry"
            ) from None
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(encoded)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary, 0o400)
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.get(identity)
                if existing is None:
                    raise SearchCacheIntegrityError(
                        "SEARCH cache entry appeared without a complete artifact; "
                        "move it aside and retry"
                    )
                return _require_same_completion(existing, normalized)
            except OSError:
                raise SearchCacheError(
                    "SEARCH cache completion could not be created atomically; retry"
                ) from None
            try:
                _fsync_directory(path.parent)
            except OSError:
                raise SearchCacheError(
                    "SEARCH cache completion was created but directory sync failed; "
                    "retry to verify it"
                ) from None
            return normalized
        except SearchCacheError:
            raise
        except OSError:
            raise SearchCacheError(
                "SEARCH cache write was interrupted; retry"
            ) from None
        finally:
            temporary.unlink(missing_ok=True)


def _require_identity(identity: Any) -> None:
    if not isinstance(identity, SolverRequestIdentity):
        raise SearchCacheError("SEARCH cache identity must be SolverRequestIdentity")


def _validate_result(result: Any) -> SolverResult:
    if not isinstance(result, SolverResult):
        raise SearchCacheError(
            "SEARCH cache accepts only a complete SolverResult"
        )
    if not isinstance(result.content, str):
        raise SearchCacheError("SEARCH completion content must be text")
    usage = result.usage
    if usage is None:
        return SolverResult(content=result.content, usage=None)
    if not isinstance(usage, Mapping):
        raise SearchCacheError("SEARCH completion usage must be an object or null")
    allowed = {"input_tokens", "output_tokens", "total_tokens"}
    if set(usage) - allowed or any(
        not isinstance(key, str)
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for key, value in usage.items()
    ):
        raise SearchCacheError(
            "SEARCH completion usage contains invalid fields or values"
        )
    return SolverResult(content=result.content, usage=dict(usage))


def _completion_payload(result: SolverResult) -> dict[str, Any]:
    return {
        "content": result.content,
        "usage": None if result.usage is None else dict(result.usage),
    }


def _completion_sha256(result: SolverResult) -> str:
    return _sha256_json(_completion_payload(result))


def _result_from_payload(value: Any) -> SolverResult:
    if not isinstance(value, Mapping) or set(value) != {"content", "usage"}:
        raise _corrupt_entry_error()
    try:
        return _validate_result(
            SolverResult(content=value["content"], usage=value["usage"])
        )
    except SearchCacheError:
        raise _corrupt_entry_error() from None


def _require_same_completion(
    existing: SolverResult,
    proposed: SolverResult,
) -> SolverResult:
    if _completion_payload(existing) != _completion_payload(proposed):
        raise SearchCacheConflictError(
            "a different valid completion already exists for this request identity; "
            "the immutable entry was not overwritten"
        )
    return existing


def _ensure_safe(
    identity: SolverRequestIdentity,
    result: SolverResult,
) -> None:
    serialized_identity = json.dumps(
        identity.as_dict(),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    for value in (serialized_identity, result.content):
        if any(pattern.search(value) for pattern in _SENSITIVE_PATTERNS):
            raise SearchCacheSafetyError(
                "SEARCH cache input contains sensitive material and was not written"
            )


def _load_entry(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, SearchCacheIntegrityError):
        raise _corrupt_entry_error() from None
    if not isinstance(payload, dict):
        raise _corrupt_entry_error()
    return payload, encoded


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _corrupt_entry_error()
        result[key] = value
    return result


def _corrupt_entry_error() -> SearchCacheIntegrityError:
    return SearchCacheIntegrityError(
        "SEARCH cache entry is corrupt or truncated; move it aside and retry"
    )


def _json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SearchCacheError(
            "SEARCH cache completion must contain JSON values"
        ) from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FullSearchV3SearchCache",
    "SEARCH_CACHE_SCHEMA_VERSION",
    "SearchCacheConflictError",
    "SearchCacheError",
    "SearchCacheIntegrityError",
    "SearchCacheSafetyError",
]
