from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import threading

import pytest

from meta_harness.search_cache import SearchCache
from meta_harness.request_executor import (
    RequestExecutor,
    SolverConcurrencyError,
)
from meta_harness.retry import (
    SolverExecutionFailure,
    SolverParserFailure,
    SolverRetryPolicy,
)
from meta_harness.solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
    build_solver_request_identity,
)


def _request() -> SolverRequest:
    return SolverRequest(
        model="Qwen3.6-35B-A3B",
        messages=({"role": "user", "content": "safe request"},),
        generation=SolverGenerationSettings(
            temperature=0,
            top_p=1,
            seed=42,
            n=1,
            stream=False,
            max_tokens=8192,
        ),
    )


def _identity(index: int):
    return build_solver_request_identity(
        _request(),
        sample_id=f"sciver_sample_content_v1:{index:064x}",
        candidate_id="p0",
        prompt_sha256="2" * 64,
        split_sha256="3" * 64,
        search_membership_sha256="4" * 64,
        solver_identity_sha256="5" * 64,
    )


def _executor(tmp_path, factory, *, maximum=2, cache=None):
    return RequestExecutor(
        cache=cache or SearchCache(tmp_path / "cache"),
        client_factory=factory,
        retry_policy=SolverRetryPolicy(maximum_attempts=1),
        maximum_in_flight_requests=maximum,
        sleeper=lambda _delay: pytest.fail("test retry sleeper was called"),
        clock=lambda: 0.0,
    )


class _NotifyingProcessCache(SearchCache):
    def __init__(self, directory, checked):
        super().__init__(directory)
        self._checked = checked

    def get(self, identity):
        result = super().get(identity)
        self._checked.set()
        return result


class _ProcessClient:
    def __init__(self, calls, entered, release):
        self._calls = calls
        self._entered = entered
        self._release = release

    def complete(self, _request):
        with self._calls.get_lock():
            self._calls.value += 1
        self._entered.set()
        if not self._release.wait(5):
            raise AssertionError("process transport release timed out")
        return SolverResult("Answer: yes")


class _ImmediateProcessClient:
    def complete(self, _request):
        return SolverResult("Answer: yes")


class _CountingProcessFactory:
    def __init__(self, calls):
        self._calls = calls

    def __call__(self):
        with self._calls.get_lock():
            self._calls.value += 1
        return _ImmediateProcessClient()


def _process_complete(
    cache_directory,
    identity,
    checked,
    calls,
    entered,
    release,
    results,
):
    cache = _NotifyingProcessCache(cache_directory, checked)
    executor = RequestExecutor(
        cache=cache,
        client_factory=lambda: _ProcessClient(calls, entered, release),
        retry_policy=SolverRetryPolicy(maximum_attempts=1),
        maximum_in_flight_requests=2,
        sleeper=lambda _delay: None,
        clock=lambda: 0.0,
    )
    try:
        result = executor.complete(identity, _request())
    except BaseException as error:
        results.put(("error", type(error).__name__))
    else:
        results.put(("ok", result.content))


def _forked_executor_complete(executor, identity, results):
    try:
        result = executor.complete(identity, _request())
    except BaseException as error:
        results.put(("error", type(error).__name__))
    else:
        results.put(("ok", result.content))


def test_configured_concurrency_ceiling_is_never_exceeded(tmp_path):
    state_lock = threading.Lock()
    two_entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum_active = 0
    entered = 0

    class Client:
        def complete(self, _request):
            nonlocal active, maximum_active, entered
            with state_lock:
                active += 1
                entered += 1
                maximum_active = max(maximum_active, active)
                if entered == 2:
                    two_entered.set()
            assert release.wait(5)
            with state_lock:
                active -= 1
            return SolverResult("Answer: yes")

    executor = _executor(tmp_path, Client, maximum=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(executor.complete, _identity(index), _request())
            for index in range(1, 5)
        ]
        assert two_entered.wait(5)
        with state_lock:
            assert active == 2
            assert maximum_active == 2
        release.set()
        assert [future.result(timeout=5) for future in futures] == [
            SolverResult("Answer: yes")
        ] * 4

    assert maximum_active == 2


class _CountingCache(SearchCache):
    def __init__(self, directory):
        super().__init__(directory)
        self.get_count = 0
        self._count_lock = threading.Lock()
        self.waiter_checked_cache = threading.Event()

    def get(self, identity):
        with self._count_lock:
            self.get_count += 1
            if self.get_count >= 3:
                self.waiter_checked_cache.set()
        return super().get(identity)


def test_same_identity_calls_once_and_waiter_rechecks_cache(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    call_lock = threading.Lock()
    calls = 0

    class Client:
        def complete(self, _request):
            nonlocal calls
            with call_lock:
                calls += 1
            entered.set()
            assert release.wait(5)
            return SolverResult("Answer: no")

    cache = _CountingCache(tmp_path / "cache")
    executor = _executor(tmp_path, Client, cache=cache)
    identity = _identity(10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.complete, identity, _request())
        assert entered.wait(5)
        second = pool.submit(executor.complete, identity, _request())
        assert cache.waiter_checked_cache.wait(5)
        release.set()
        assert first.result(timeout=5) == SolverResult("Answer: no")
        assert second.result(timeout=5) == SolverResult("Answer: no")

    assert calls == 1
    assert cache.get_count >= 4


def test_same_identity_lock_deduplicates_across_linux_processes(tmp_path):
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    first_checked = context.Event()
    second_checked = context.Event()
    entered = context.Event()
    release = context.Event()
    results = context.Queue()
    cache_directory = tmp_path / "cache"
    identity = _identity(11)

    first = context.Process(
        target=_process_complete,
        args=(
            cache_directory,
            identity,
            first_checked,
            calls,
            entered,
            release,
            results,
        ),
    )
    second = context.Process(
        target=_process_complete,
        args=(
            cache_directory,
            identity,
            second_checked,
            calls,
            entered,
            release,
            results,
        ),
    )
    first.start()
    assert first_checked.wait(5)
    assert entered.wait(5)
    second.start()
    assert second_checked.wait(5)
    release.set()
    first.join(5)
    second.join(5)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert sorted((results.get(timeout=2), results.get(timeout=2))) == [
        ("ok", "Answer: yes"),
        ("ok", "Answer: yes"),
    ]
    assert calls.value == 1
    results.close()


def test_different_identities_execute_in_parallel(tmp_path):
    rendezvous = threading.Barrier(2)

    class Client:
        def complete(self, _request):
            rendezvous.wait(timeout=5)
            return SolverResult("Answer: yes")

    executor = _executor(tmp_path, Client, maximum=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.complete, _identity(20), _request())
        second = pool.submit(executor.complete, _identity(21), _request())
        assert first.result(timeout=5) == SolverResult("Answer: yes")
        assert second.result(timeout=5) == SolverResult("Answer: yes")


def test_complete_ordered_refills_bounded_workers_and_returns_manifest_order(tmp_path):
    first_started = threading.Event()
    third_started = threading.Event()
    release_first = threading.Event()
    active_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def request(content: str) -> SolverRequest:
        return SolverRequest(
            model="Qwen3.6-35B-A3B",
            messages=({"role": "user", "content": content},),
            generation=_request().generation,
        )

    def identity(index: int, value: SolverRequest):
        return build_solver_request_identity(
            value,
            sample_id=f"sciver_sample_content_v1:{index:064x}",
            candidate_id="p0",
            prompt_sha256="2" * 64,
            split_sha256="3" * 64,
            search_membership_sha256="4" * 64,
            solver_identity_sha256="5" * 64,
        )

    class Client:
        def complete(self, solver_request):
            nonlocal active, maximum_active
            content = solver_request.messages[0]["content"]
            with active_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                if content == "first":
                    first_started.set()
                    assert release_first.wait(5)
                elif content == "third":
                    third_started.set()
                return SolverResult(f"Answer: {content}")
            finally:
                with active_lock:
                    active -= 1

    requests = tuple(request(value) for value in ("first", "second", "third"))
    executor = _executor(tmp_path, Client, maximum=2)

    def collect():
        return list(
            executor.complete_ordered(
                tuple(
                    (identity(index, value), value)
                    for index, value in enumerate(requests)
                )
            )
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        results = pool.submit(collect)
        assert first_started.wait(5)
        assert third_started.wait(5)
        with active_lock:
            assert maximum_active == 2
        release_first.set()
        assert results.result(timeout=5) == [
            SolverResult("Answer: first"),
            SolverResult("Answer: second"),
            SolverResult("Answer: third"),
        ]


def test_complete_ordered_yields_retry_failures_without_abandoning_later_requests(
    tmp_path,
):
    outcomes = [
        SolverParserFailure("offline parser failure"),
        SolverResult("Answer: no"),
    ]

    class Client:
        def complete(self, _request):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    executor = _executor(tmp_path, Client, maximum=1)
    results = list(
        executor.complete_ordered(
            ((_identity(60), _request()), (_identity(61), _request()))
        )
    )

    assert isinstance(results[0], SolverExecutionFailure)
    assert results[1] == SolverResult("Answer: no")


def test_request_must_match_immutable_identity(tmp_path):
    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return object()

    executor = _executor(tmp_path, factory)
    changed_request = SolverRequest(
        model=_request().model,
        messages=({"role": "user", "content": "different request"},),
        generation=_request().generation,
    )

    with pytest.raises(SolverConcurrencyError, match="immutable identity"):
        executor.complete(_identity(22), changed_request)

    assert factory_calls == 0


def test_clients_are_created_once_per_worker_and_reused_worker_locally(tmp_path):
    factory_lock = threading.Lock()
    created_on_threads = []
    first_call_rendezvous = threading.Barrier(2)

    class Client:
        def __init__(self):
            self.owner = threading.get_ident()
            self.calls = 0

        def complete(self, _request):
            assert threading.get_ident() == self.owner
            self.calls += 1
            if self.calls == 1:
                first_call_rendezvous.wait(timeout=5)
            return SolverResult("Answer: yes")

    def factory():
        with factory_lock:
            created_on_threads.append(threading.get_ident())
        return Client()

    executor = _executor(tmp_path, factory, maximum=2)

    def worker(start):
        return [
            executor.complete(_identity(start + offset), _request())
            for offset in range(2)
        ]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(worker, 30)
        second = pool.submit(worker, 40)
        assert first.result(timeout=5) == [SolverResult("Answer: yes")] * 2
        assert second.result(timeout=5) == [SolverResult("Answer: yes")] * 2

    assert len(created_on_threads) == 2
    assert len(set(created_on_threads)) == 2


def test_forked_worker_does_not_reuse_parent_client_or_thread_state(tmp_path):
    context = multiprocessing.get_context("fork")
    factory_calls = context.Value("i", 0)
    results = context.Queue()
    executor = _executor(
        tmp_path,
        _CountingProcessFactory(factory_calls),
        maximum=1,
    )

    assert executor.complete(_identity(45), _request()) == SolverResult(
        "Answer: yes"
    )
    child = context.Process(
        target=_forked_executor_complete,
        args=(executor, _identity(46), results),
    )
    child.start()
    child.join(5)

    assert child.exitcode == 0
    assert results.get(timeout=2) == ("ok", "Answer: yes")
    assert factory_calls.value == 2
    results.close()


@pytest.mark.parametrize(
    ("failure", "expected_exception"),
    [
        (SolverParserFailure("unsafe detail"), SolverExecutionFailure),
        (KeyboardInterrupt(), KeyboardInterrupt),
    ],
)
def test_identity_lock_is_released_after_failure_or_interruption(
    tmp_path,
    failure,
    expected_exception,
):
    outcomes = [failure, SolverResult("Answer: yes")]

    class Client:
        def complete(self, _request):
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    executor = _executor(tmp_path, Client, maximum=1)
    identity = _identity(50)

    with pytest.raises(expected_exception):
        executor.complete(identity, _request())

    assert executor.complete(identity, _request()) == SolverResult("Answer: yes")
    assert outcomes == []
