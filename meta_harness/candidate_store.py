"""Atomic, immutable local storage for validated prompt candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from meta_harness.schemas import (
    Candidate,
    CandidateValidationError,
    DEFAULT_MAX_PROMPT_CHARS,
    DEFAULT_ROOT_CANDIDATE_IDS,
    canonical_json,
    validate_candidate,
)


CANDIDATE_STATUSES = frozenset(
    {
        "proposed",
        "validated",
        "rejected",
        "smoke_passed",
        "screened",
        "evaluated",
    }
)
REGISTRY_SCHEMA_VERSION = 1


class CandidateStoreError(ValueError):
    """Raised when immutable candidate storage or integrity checks fail."""


class CandidateConflictError(CandidateStoreError):
    """Raised when an existing candidate ID has different content."""


class CandidateIntegrityError(CandidateStoreError):
    """Raised when stored canonical bytes do not match the registry hash."""


class CandidateStore:
    """Create-once candidate artifacts scoped to one local run."""

    def __init__(
        self,
        repository_root: str | Path,
        run_id: str,
        *,
        root_candidate_ids: Iterable[str] = DEFAULT_ROOT_CANDIDATE_IDS,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> None:
        self._repository_root = Path(repository_root)
        self._run_id = _safe_identifier(run_id, "run_id")
        self._root_candidate_ids = _safe_identifier_set(
            root_candidate_ids,
            "root_candidate_ids",
        )
        if (
            isinstance(max_prompt_chars, bool)
            or not isinstance(max_prompt_chars, int)
            or max_prompt_chars <= 0
        ):
            raise CandidateStoreError(
                "max_prompt_chars must be a positive integer"
            )
        self._max_prompt_chars = max_prompt_chars
        self._run_directory = (
            self._repository_root
            / "workspace"
            / "meta_harness"
            / "runs"
            / self._run_id
        )
        self._candidates_directory = self._run_directory / "candidates"
        self._registry_path = self._run_directory / "candidate_registry.json"

    @property
    def run_directory(self) -> Path:
        return self._run_directory

    @property
    def candidates_directory(self) -> Path:
        return self._candidates_directory

    @property
    def registry_path(self) -> Path:
        return self._registry_path

    def candidate_path(self, candidate_id: str) -> Path:
        safe_candidate_id = _safe_identifier(candidate_id, "candidate_id")
        return (
            self._candidates_directory
            / safe_candidate_id
            / "candidate.json"
        )

    def create(
        self,
        value: Candidate | Mapping[str, Any],
        *,
        status: str = "proposed",
    ) -> Candidate:
        """Atomically create one candidate or accept an identical retry."""

        _validate_status(status)
        known_parents = self._known_parent_ids()
        try:
            candidate = validate_candidate(
                value,
                existing_parent_ids=known_parents,
                max_prompt_chars=self._max_prompt_chars,
            )
        except CandidateValidationError as exc:
            raise CandidateStoreError(str(exc)) from exc
        if candidate.parent_id not in self._root_candidate_ids:
            self.load(candidate.parent_id)

        destination = self.candidate_path(candidate.candidate_id)
        encoded = candidate.canonical_json().encode("utf-8")
        candidate_sha256 = hashlib.sha256(encoded).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)

        lock_path = destination.parent / ".create.lock"
        with _exclusive_lock(lock_path):
            registry = self._read_registry(allow_missing=True)
            registered = registry["candidates"].get(candidate.candidate_id)
            if registered is not None and not destination.exists():
                raise CandidateIntegrityError(
                    "candidate registry references a missing immutable artifact"
                )
            if (
                registered is not None
                and registered["sha256"] != candidate_sha256
            ):
                raise CandidateConflictError(
                    "candidate ID already has a different registered hash; "
                    "repair requires a new candidate ID"
                )
            if destination.exists():
                existing = self._load_candidate_file(
                    destination,
                    expected_candidate_id=candidate.candidate_id,
                )
                if existing.canonical_json().encode("utf-8") != encoded:
                    raise CandidateConflictError(
                        "candidate ID already exists with different immutable "
                        "content; repair requires a new candidate ID"
                    )
                self._ensure_registry_entry(
                    candidate.candidate_id,
                    candidate_sha256,
                    status,
                )
                return existing

            _atomic_replace(destination, encoded, mode=0o444)
            try:
                self._ensure_registry_entry(
                    candidate.candidate_id,
                    candidate_sha256,
                    status,
                )
            except Exception:
                # The immutable candidate may already be durable. A retry with
                # identical content safely repairs the separate registry.
                raise
        return candidate

    def save_candidate(
        self,
        value: Candidate | Mapping[str, Any],
        *,
        status: str = "proposed",
    ) -> Candidate:
        return self.create(value, status=status)

    def load(self, candidate_id: str) -> Candidate:
        """Load a canonical candidate and verify its whole-file hash."""

        destination = self.candidate_path(candidate_id)
        if not destination.is_file():
            raise CandidateStoreError("candidate does not exist")
        encoded = _read_bytes(destination, "candidate artifact")
        registry = self._read_registry()
        entry = registry["candidates"].get(candidate_id)
        if entry is None:
            raise CandidateIntegrityError(
                "candidate registry entry is missing"
            )
        actual_sha256 = hashlib.sha256(encoded).hexdigest()
        if actual_sha256 != entry["sha256"]:
            raise CandidateIntegrityError(
                "candidate artifact hash does not match the registry"
            )
        candidate = self._decode_candidate(
            encoded,
            expected_candidate_id=candidate_id,
        )
        canonical = candidate.canonical_json().encode("utf-8")
        if encoded != canonical:
            raise CandidateIntegrityError(
                "candidate artifact is not canonical JSON"
            )
        return candidate

    def load_candidate(self, candidate_id: str) -> Candidate:
        return self.load(candidate_id)

    def update_status(self, candidate_id: str, status: str) -> str:
        """Atomically update only mutable registry status."""

        _validate_status(status)
        self.load(candidate_id)
        self._run_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self._run_directory / ".registry.lock"
        with _exclusive_lock(lock_path):
            registry = self._read_registry()
            entry = registry["candidates"].get(candidate_id)
            if entry is None:
                raise CandidateIntegrityError(
                    "candidate registry entry is missing"
                )
            if entry["status"] == status:
                return status
            updated = _copy_registry(registry)
            updated["candidates"][candidate_id]["status"] = status
            _atomic_replace(
                self._registry_path,
                canonical_json(updated).encode("utf-8"),
            )
        return status

    def get_status(self, candidate_id: str) -> str:
        self.load(candidate_id)
        registry = self._read_registry()
        return registry["candidates"][candidate_id]["status"]

    def read_registry(self) -> dict[str, Any]:
        """Return a detached copy of the status registry."""

        return _copy_registry(self._read_registry())

    def _known_parent_ids(self) -> frozenset[str]:
        registry = self._read_registry(allow_missing=True)
        return self._root_candidate_ids | frozenset(registry["candidates"])

    def _ensure_registry_entry(
        self,
        candidate_id: str,
        candidate_sha256: str,
        initial_status: str,
    ) -> None:
        self._run_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self._run_directory / ".registry.lock"
        with _exclusive_lock(lock_path):
            registry = self._read_registry(allow_missing=True)
            existing = registry["candidates"].get(candidate_id)
            if existing is not None:
                if existing["sha256"] != candidate_sha256:
                    raise CandidateConflictError(
                        "candidate registry hash conflicts with immutable content"
                    )
                return
            updated = _copy_registry(registry)
            updated["candidates"][candidate_id] = {
                "sha256": candidate_sha256,
                "status": initial_status,
            }
            _atomic_replace(
                self._registry_path,
                canonical_json(updated).encode("utf-8"),
            )

    def _read_registry(
        self,
        *,
        allow_missing: bool = False,
    ) -> dict[str, Any]:
        if not self._registry_path.exists():
            if allow_missing:
                return _empty_registry()
            raise CandidateIntegrityError("candidate registry is missing")
        encoded = _read_bytes(self._registry_path, "candidate registry")
        try:
            registry = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeError, json.JSONDecodeError, CandidateStoreError) as exc:
            raise CandidateIntegrityError(
                "candidate registry must be valid UTF-8 JSON"
            ) from exc
        _validate_registry(registry)
        if encoded != canonical_json(registry).encode("utf-8"):
            raise CandidateIntegrityError(
                "candidate registry is not canonical JSON"
            )
        return registry

    def _load_candidate_file(
        self,
        path: Path,
        *,
        expected_candidate_id: str,
    ) -> Candidate:
        encoded = _read_bytes(path, "candidate artifact")
        candidate = self._decode_candidate(
            encoded,
            expected_candidate_id=expected_candidate_id,
        )
        if encoded != candidate.canonical_json().encode("utf-8"):
            raise CandidateIntegrityError(
                "candidate artifact is not canonical JSON"
            )
        return candidate

    def _decode_candidate(
        self,
        encoded: bytes,
        *,
        expected_candidate_id: str,
    ) -> Candidate:
        try:
            payload = json.loads(
                encoded.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
            candidate = Candidate.from_mapping(
                payload,
                max_prompt_chars=self._max_prompt_chars,
            )
        except (
            UnicodeError,
            json.JSONDecodeError,
            CandidateValidationError,
            CandidateStoreError,
        ) as exc:
            raise CandidateIntegrityError(
                "candidate artifact violates the immutable schema"
            ) from exc
        if candidate.candidate_id != expected_candidate_id:
            raise CandidateIntegrityError(
                "candidate artifact ID does not match its storage path"
            )
        return candidate


def _empty_registry() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "candidates": {},
    }


def _copy_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": registry["schema_version"],
        "candidates": {
            candidate_id: {
                "sha256": entry["sha256"],
                "status": entry["status"],
            }
            for candidate_id, entry in registry["candidates"].items()
        },
    }


def _validate_registry(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise CandidateIntegrityError("candidate registry must be an object")
    if set(value) != {"schema_version", "candidates"}:
        raise CandidateIntegrityError(
            "candidate registry fields are invalid"
        )
    if value["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise CandidateIntegrityError(
            "candidate registry schema version is unsupported"
        )
    candidates = value["candidates"]
    if not isinstance(candidates, Mapping):
        raise CandidateIntegrityError(
            "candidate registry candidates must be an object"
        )
    for candidate_id, entry in candidates.items():
        _safe_identifier(candidate_id, "registry candidate ID")
        if not isinstance(entry, Mapping) or set(entry) != {"sha256", "status"}:
            raise CandidateIntegrityError(
                "candidate registry entry fields are invalid"
            )
        sha256 = entry["sha256"]
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise CandidateIntegrityError(
                "candidate registry hash is invalid"
            )
        _validate_status(entry["status"])


def _validate_status(status: Any) -> None:
    if not isinstance(status, str) or status not in CANDIDATE_STATUSES:
        raise CandidateStoreError(
            "candidate status must be one of: "
            + ", ".join(sorted(CANDIDATE_STATUSES))
        )


def _safe_identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or value[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in value
        )
    ):
        raise CandidateStoreError(f"{field} must be a safe local identifier")
    return value


def _safe_identifier_set(values: Iterable[str], field: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise CandidateStoreError(f"{field} must contain identifiers")
    try:
        normalized = frozenset(values)
    except TypeError as exc:
        raise CandidateStoreError(f"{field} must contain identifiers") from exc
    for value in normalized:
        _safe_identifier(value, field)
    return normalized


def _read_bytes(path: Path, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CandidateStoreError(f"{label} must be readable") from exc


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CandidateStoreError(
                "candidate store is busy with another atomic update"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_replace(
    destination: Path,
    encoded: bytes,
    *,
    mode: int = 0o600,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        os.chmod(temporary, mode)
        os.replace(temporary, destination)
    except Exception as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        if isinstance(exc, CandidateStoreError):
            raise
        raise CandidateStoreError("atomic candidate-store write failed") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateStoreError("stored JSON contains duplicate keys")
        result[key] = value
    return result


__all__ = [
    "CANDIDATE_STATUSES",
    "CandidateConflictError",
    "CandidateIntegrityError",
    "CandidateStore",
    "CandidateStoreError",
    "REGISTRY_SCHEMA_VERSION",
]
