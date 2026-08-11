"""Bounded, process-safe request execution for ``sciver_full_search_v3``."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable, Iterator

from meta_harness.full_search_v3_cache import FullSearchV3SearchCache
from meta_harness.full_search_v3_retry import (
    SolverRetryPolicy,
    execute_solver_request_with_retry,
)
from meta_harness.full_search_v3_solver import (
    SolverClient,
    SolverRequest,
    SolverRequestIdentity,
    SolverResult,
    solver_request_payload_sha256,
)


CONCURRENCY_SCHEMA_VERSION = "sciver_full_search_v3_concurrency_v1"
MAXIMUM_IN_FLIGHT_REQUESTS = 256


class SolverConcurrencyError(RuntimeError):
    """Raised when safe request coordination cannot be established."""


class FullSearchV3RequestExecutor:
    """Execute cached solver requests with bounded, identity-scoped locking."""

    def __init__(
        self,
        *,
        cache: FullSearchV3SearchCache,
        client_factory: Callable[[], SolverClient],
        retry_policy: SolverRetryPolicy,
        maximum_in_flight_requests: int,
        sleeper: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        if not isinstance(cache, FullSearchV3SearchCache):
            raise TypeError("cache must be FullSearchV3SearchCache")
        if not callable(client_factory):
            raise TypeError("client_factory must be callable")
        if not isinstance(retry_policy, SolverRetryPolicy):
            raise TypeError("retry_policy must be SolverRetryPolicy")
        if (
            isinstance(maximum_in_flight_requests, bool)
            or not isinstance(maximum_in_flight_requests, int)
            or maximum_in_flight_requests < 1
            or maximum_in_flight_requests > MAXIMUM_IN_FLIGHT_REQUESTS
        ):
            raise ValueError(
                "maximum_in_flight_requests must be between 1 and "
                f"{MAXIMUM_IN_FLIGHT_REQUESTS}"
            )
        if not callable(sleeper):
            raise TypeError("sleeper must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._cache = cache
        self._client_factory = client_factory
        self._retry_policy = retry_policy
        self._maximum_in_flight_requests = maximum_in_flight_requests
        self._sleeper = sleeper
        self._clock = clock
        self._coordination_directory = cache.coordination_directory()
        self._local_slots = threading.BoundedSemaphore(maximum_in_flight_requests)
        self._worker_state = threading.local()
        self._configuration_guard = threading.Lock()
        self._configuration_ready = False
        self._process_id = os.getpid()

    @property
    def maximum_in_flight_requests(self) -> int:
        return self._maximum_in_flight_requests

    @property
    def cache(self) -> FullSearchV3SearchCache:
        """Return the immutable completion cache used for request execution."""

        return self._cache

    def complete(
        self,
        identity: SolverRequestIdentity,
        request: SolverRequest,
    ) -> SolverResult:
        """Return a cached completion or perform one identity-locked request."""

        if not isinstance(identity, SolverRequestIdentity):
            raise TypeError("identity must be SolverRequestIdentity")
        if not isinstance(request, SolverRequest):
            raise TypeError("request must be SolverRequest")
        self._refresh_after_fork()
        if identity.request_payload_sha256 != solver_request_payload_sha256(request):
            raise SolverConcurrencyError(
                "request payload does not match its immutable identity"
            )

        cached = self._cache.get(identity)
        if cached is not None:
            return cached

        self._ensure_coordination_configuration()
        request_lock = (
            self._coordination_directory
            / "requests"
            / identity.sha256()[:2]
            / f"{identity.sha256()}.lock"
        )
        with _exclusive_file_lock(request_lock):
            # A concurrent holder may have completed while this caller waited.
            cached = self._cache.get(identity)
            if cached is not None:
                return cached
            with self._transport_slot():
                result = execute_solver_request_with_retry(
                    self._worker_client(),
                    request,
                    policy=self._retry_policy,
                    sleeper=self._sleeper,
                    clock=self._clock,
                )
            return self._cache.put(identity, result)

    def _refresh_after_fork(self) -> None:
        process_id = os.getpid()
        if process_id == self._process_id:
            return
        # Thread synchronization state cannot safely be inherited from a
        # potentially multithreaded parent. Durable file locks remain shared.
        self._local_slots = threading.BoundedSemaphore(
            self._maximum_in_flight_requests
        )
        self._worker_state = threading.local()
        self._configuration_guard = threading.Lock()
        self._configuration_ready = False
        self._process_id = process_id

    def _worker_client(self) -> SolverClient:
        process_id = os.getpid()
        if getattr(self._worker_state, "process_id", None) != process_id:
            client = self._client_factory()
            complete = getattr(client, "complete", None)
            if not callable(complete):
                raise SolverConcurrencyError(
                    "client_factory must return a solver client"
                )
            self._worker_state.process_id = process_id
            self._worker_state.client = client
        return self._worker_state.client

    def _ensure_coordination_configuration(self) -> None:
        if self._configuration_ready:
            return
        with self._configuration_guard:
            if self._configuration_ready:
                return
            directory = self._coordination_directory
            with _exclusive_file_lock(directory / "configuration.lock"):
                path = directory / "configuration.json"
                expected = {
                    "schema_version": CONCURRENCY_SCHEMA_VERSION,
                    "maximum_in_flight_requests": (
                        self._maximum_in_flight_requests
                    ),
                }
                if path.exists():
                    actual = _read_configuration(path)
                    if actual != expected:
                        raise SolverConcurrencyError(
                            "SEARCH cache concurrency configuration is incompatible; "
                            "use the configured maximum for this cache"
                        )
                else:
                    _write_configuration(path, expected)
            self._configuration_ready = True

    @contextmanager
    def _transport_slot(self) -> Iterator[None]:
        self._local_slots.acquire()
        try:
            slots = self._coordination_directory / "slots"
            with _one_of_file_slots(slots, self._maximum_in_flight_requests):
                yield
        finally:
            self._local_slots.release()


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError:
        raise SolverConcurrencyError(
            "request lock storage is unavailable; verify permissions and retry"
        ) from None
    locked = False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError:
            raise SolverConcurrencyError(
                "request locking failed; verify local lock storage and retry"
            ) from None
        locked = True
        yield
    finally:
        _close_locked_descriptor(descriptor, locked)


@contextmanager
def _one_of_file_slots(directory: Path, count: int) -> Iterator[None]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise SolverConcurrencyError(
            "concurrency slot storage is unavailable; verify permissions and retry"
        ) from None

    descriptor: int | None = None
    locked = False
    try:
        try:
            for index in range(count):
                candidate = os.open(
                    directory / f"slot-{index:03d}.lock",
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                try:
                    fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(candidate)
                    continue
                except OSError:
                    os.close(candidate)
                    raise
                descriptor = candidate
                locked = True
                break

            if descriptor is None:
                descriptor = os.open(
                    directory / "slot-000.lock",
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                locked = True
        except OSError:
            raise SolverConcurrencyError(
                "concurrency slot locking failed; "
                "verify local lock storage and retry"
            ) from None
        yield
    finally:
        if descriptor is not None:
            _close_locked_descriptor(descriptor, locked)


def _close_locked_descriptor(descriptor: int, locked: bool) -> None:
    if locked:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            # Closing the descriptor still releases a Linux flock and must not
            # mask an active solver exception or interruption.
            pass
    try:
        os.close(descriptor)
    except OSError:
        pass


def _read_configuration(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SolverConcurrencyError(
            "SEARCH cache concurrency configuration is corrupt; "
            "move the coordination directory aside and retry"
        ) from None
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "maximum_in_flight_requests",
    }:
        raise SolverConcurrencyError(
            "SEARCH cache concurrency configuration is corrupt; "
            "move the coordination directory aside and retry"
        )
    return value


def _write_configuration(path: Path, value: dict[str, object]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
    except OSError:
        raise SolverConcurrencyError(
            "SEARCH cache concurrency configuration could not be created"
        ) from None
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    except OSError:
        raise SolverConcurrencyError(
            "SEARCH cache concurrency configuration could not be persisted"
        ) from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CONCURRENCY_SCHEMA_VERSION",
    "FullSearchV3RequestExecutor",
    "MAXIMUM_IN_FLIGHT_REQUESTS",
    "SolverConcurrencyError",
]
