from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import threading
from unittest.mock import Mock

from PIL import Image
import pytest
import requests

from meta_harness.full_search_v3_cache import FullSearchV3SearchCache
from meta_harness.full_search_v3_concurrency import FullSearchV3RequestExecutor
from meta_harness.full_search_v3_retry import (
    SolverExecutionFailure,
    SolverRetryPolicy,
)
from meta_harness.full_search_v3_solver import (
    SolverRequest,
    SolverResult,
    build_solver_request,
    build_solver_request_identity,
    solver_request_payload_sha256,
)
from model_inference.remote_client import InvalidRequestError, RequestTimeoutError
from utils.constant import COT_PROMPT


FAKE_SECRET = "UNMISTAKABLY_FAKE_MILESTONE2_SECRET"
FAKE_ENDPOINT = "https://invalid.example.test/private/chat/completions"
IMAGE_PREFIX = "data:image/png;base64,"


@pytest.fixture(autouse=True)
def deny_external_activity(monkeypatch):
    """Prove this integration layer cannot reach any external boundary."""

    http = Mock(side_effect=AssertionError("integration test attempted HTTP"))
    connect = Mock(side_effect=AssertionError("integration test opened a socket"))
    run = Mock(side_effect=AssertionError("integration test ran a subprocess"))
    popen = Mock(side_effect=AssertionError("integration test started a subprocess"))
    monkeypatch.setattr(requests.sessions.Session, "request", http)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setenv("API_KEY", FAKE_SECRET)
    monkeypatch.setenv("API_URL", FAKE_ENDPOINT)

    yield

    http.assert_not_called()
    connect.assert_not_called()
    run.assert_not_called()
    popen.assert_not_called()


class _FakeTransportState:
    def __init__(self, outcomes, *, block_keys=()):
        self.outcomes = {key: list(value) for key, value in outcomes.items()}
        self.block_keys = frozenset(block_keys)
        self.calls = Counter()
        self.completed_calls = Counter()
        self.delivered_images = {}
        self.factory_threads = []
        self.client_call_threads = []
        self.active = 0
        self.maximum_active = 0
        self._lock = threading.Lock()
        self.two_active = threading.Event()
        self.release = threading.Event()
        if not self.block_keys:
            self.release.set()

    def client_factory(self):
        owner = threading.get_ident()
        with self._lock:
            self.factory_threads.append(owner)
        return _FakeTransportClient(self, owner)

    def invoke(self, owner, request):
        current_thread = threading.get_ident()
        assert current_thread == owner
        key = solver_request_payload_sha256(request)
        with self._lock:
            self.calls[key] += 1
            self.client_call_threads.append((owner, current_thread))
            self.delivered_images.setdefault(key, []).append(_request_images(request))
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            if self.active == 2:
                self.two_active.set()
            try:
                outcome = self.outcomes[key].pop(0)
            except (KeyError, IndexError):
                raise AssertionError("unexpected fake transport attempt") from None

        try:
            if key in self.block_keys:
                assert self.release.wait(5)
            if isinstance(outcome, BaseException):
                raise outcome
            with self._lock:
                self.completed_calls[key] += 1
            return outcome
        finally:
            with self._lock:
                self.active -= 1


class _FakeTransportClient:
    def __init__(self, state, owner):
        self._state = state
        self._owner = owner

    def complete(self, request):
        return self._state.invoke(self._owner, request)


def _request_images(request: SolverRequest) -> tuple[bytes, ...]:
    return tuple(
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for message in request.messages
        for block in message["content"]
        if block["type"] == "image_url"
    )


def _make_case(tmp_path: Path, index: int):
    first = tmp_path / f"sample-{index}-first.png"
    second = tmp_path / f"sample-{index}-second.png"
    Image.new(
        "RGB",
        (2, 2),
        color=((index * 17) % 256, 10, 20),
    ).save(first, format="PNG")
    Image.new(
        "RGB",
        (2, 2),
        color=(30, (index * 29) % 256, 40),
    ).save(second, format="PNG")
    record = {
        "sample_id": f"offline-search-{index}",
        "claim_type": "parallel",
        "claim": f"Immutable offline claim {index}.",
        "context": f"Offline evidence context {index}.",
        "caption1": f"First image {index}.",
        "caption2": f"Second image {index}.",
        "item1_path": str(first),
        "item2_path": str(second),
        "gold_label": "GROUND_TRUTH_MUST_NOT_LEAK",
    }
    request = build_solver_request(record, COT_PROMPT)
    sample_digest = hashlib.sha256(
        f"offline-search-{index}".encode("ascii")
    ).hexdigest()
    identity = build_solver_request_identity(
        request,
        sample_id=f"sciver_sample_content_v1:{sample_digest}",
        candidate_id="p0",
        prompt_sha256="2" * 64,
        split_sha256="3" * 64,
        search_membership_sha256="4" * 64,
        solver_identity_sha256="5" * 64,
    )
    return request, identity, (first.read_bytes(), second.read_bytes())


def _executor(cache, state, sleeps, *, maximum=2):
    return FullSearchV3RequestExecutor(
        cache=cache,
        client_factory=state.client_factory,
        retry_policy=SolverRetryPolicy(
            maximum_attempts=3,
            initial_backoff_seconds=0.25,
            maximum_backoff_seconds=0.5,
        ),
        maximum_in_flight_requests=maximum,
        sleeper=sleeps.append,
        clock=lambda: 0.0,
    )


def _safe_result(answer: str) -> SolverResult:
    return SolverResult(
        content=f"Therefore, the final answer is: Answer: {answer}",
        usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def test_milestone2_offline_lifecycle_retry_interruption_and_resume(tmp_path):
    cases = [_make_case(tmp_path, index) for index in range(1, 8)]
    keys = [solver_request_payload_sha256(case[0]) for case in cases]
    assert len(set(keys)) == len(keys)
    assert all(
        "GROUND_TRUTH_MUST_NOT_LEAK"
        not in json.dumps(request.messages, sort_keys=True)
        for request, _identity, _images in cases
    )
    yes = _safe_result("yes")
    no = _safe_result("no")
    sensitive_timeout = RequestTimeoutError(
        f"Authorization: Bearer {FAKE_SECRET}; "
        f"data:image/png;base64,{'A' * 160}"
    )
    permanent = InvalidRequestError(
        f"HTTP 400 at {FAKE_ENDPOINT} with {FAKE_SECRET}",
        http_status_code=400,
        response_error_message=FAKE_SECRET,
    )
    outcomes = {
        keys[0]: [yes],
        keys[1]: [RequestTimeoutError("transient"), no],
        keys[2]: [permanent],
        keys[3]: [sensitive_timeout, sensitive_timeout, sensitive_timeout],
        keys[4]: [yes],
        keys[5]: [no],
        keys[6]: [KeyboardInterrupt(), yes],
    }
    state = _FakeTransportState(outcomes)
    sleeps = []
    cache_root = tmp_path / "search-cache"
    cache = FullSearchV3SearchCache(cache_root)
    executor = _executor(cache, state, sleeps, maximum=2)

    # Miss, durable completion, then hit without another logical or transport call.
    assert cache.get(cases[0][1]) is None
    assert executor.complete(cases[0][1], cases[0][0]) == yes
    assert executor.complete(cases[0][1], cases[0][0]) == yes

    # A transport retry recovers; permanent and exhausted failures never complete.
    assert executor.complete(cases[1][1], cases[1][0]) == no
    with pytest.raises(SolverExecutionFailure) as permanent_failure:
        executor.complete(cases[2][1], cases[2][0])
    with pytest.raises(SolverExecutionFailure) as exhausted_failure:
        executor.complete(cases[3][1], cases[3][0])
    assert permanent_failure.value.metadata.attempt_count == 1
    assert permanent_failure.value.metadata.exhausted is False
    assert exhausted_failure.value.metadata.attempt_count == 3
    assert exhausted_failure.value.metadata.exhausted is True

    # Two durable completions survive interruption of the following request.
    assert executor.complete(cases[4][1], cases[4][0]) == yes
    assert executor.complete(cases[5][1], cases[5][0]) == no
    with pytest.raises(KeyboardInterrupt):
        executor.complete(cases[6][1], cases[6][0])

    calls_before_resume = dict(state.calls)
    resumed = _executor(cache, state, sleeps, maximum=2)
    assert resumed.complete(cases[4][1], cases[4][0]) == yes
    assert resumed.complete(cases[5][1], cases[5][0]) == no
    assert dict(state.calls) == calls_before_resume
    assert resumed.complete(cases[6][1], cases[6][0]) == yes

    assert state.calls == Counter(
        {
            keys[0]: 1,
            keys[1]: 2,
            keys[2]: 1,
            keys[3]: 3,
            keys[4]: 1,
            keys[5]: 1,
            keys[6]: 2,
        }
    )
    assert sum(state.calls.values()) == 11
    assert state.completed_calls == Counter(
        {keys[0]: 1, keys[1]: 1, keys[4]: 1, keys[5]: 1, keys[6]: 1}
    )
    assert sum(state.completed_calls.values()) == 5
    assert sleeps == [0.25, 0.25, 0.5]

    completed = [cache.get(identity) for _request, identity, _images in cases]
    assert completed == [yes, no, None, None, yes, no, yes]
    assert sum(result is not None for result in completed) == 5
    assert not cache.entry_path(cases[2][1]).exists()
    assert not cache.entry_path(cases[3][1]).exists()

    # Rebuilding requests is byte/hash stable and every attempt kept image order.
    rebuilt = [_make_case(tmp_path, index) for index in range(1, 8)]
    assert [case[1].sha256() for case in rebuilt] == [
        case[1].sha256() for case in cases
    ]
    for key, (_request, _identity, expected_images) in zip(keys, cases):
        assert all(
            delivered == expected_images for delivered in state.delivered_images[key]
        )

    failure_json = json.dumps(
        [
            permanent_failure.value.metadata.as_dict(),
            exhausted_failure.value.metadata.as_dict(),
        ],
        sort_keys=True,
    )
    artifacts = b"".join(
        path.read_bytes() for path in cache_root.rglob("*") if path.is_file()
    )
    artifact_text = artifacts.decode("utf-8")
    request_image_urls = [
        IMAGE_PREFIX + base64.b64encode(image).decode("ascii")
        for _request, _identity, images in cases
        for image in images
    ]
    for sensitive in (
        FAKE_SECRET,
        FAKE_ENDPOINT,
        "Authorization",
        "Bearer",
        "data:image",
        "base64",
        "GROUND_TRUTH_MUST_NOT_LEAK",
        *request_image_urls,
    ):
        assert sensitive not in artifact_text
        assert sensitive not in failure_json


def test_milestone2_bounded_concurrency_duplicate_dedup_and_worker_clients(
    tmp_path,
):
    first = _make_case(tmp_path, 20)
    second = _make_case(tmp_path, 21)
    first_key = solver_request_payload_sha256(first[0])
    second_key = solver_request_payload_sha256(second[0])
    result = _safe_result("yes")
    state = _FakeTransportState(
        {first_key: [result], second_key: [result]},
        block_keys={first_key, second_key},
    )
    cache = FullSearchV3SearchCache(tmp_path / "concurrent-cache")
    executor = _executor(cache, state, [], maximum=2)

    with ThreadPoolExecutor(max_workers=3) as pool:
        unique_first = pool.submit(executor.complete, first[1], first[0])
        duplicate_first = pool.submit(executor.complete, first[1], first[0])
        unique_second = pool.submit(executor.complete, second[1], second[0])
        assert state.two_active.wait(5)
        with state._lock:
            assert state.active == 2
            assert state.maximum_active == 2
        state.release.set()
        assert unique_first.result(timeout=5) == result
        assert duplicate_first.result(timeout=5) == result
        assert unique_second.result(timeout=5) == result

    assert state.calls == Counter({first_key: 1, second_key: 1})
    assert state.completed_calls == Counter({first_key: 1, second_key: 1})
    assert state.maximum_active == 2
    assert len(state.factory_threads) == 2
    assert len(set(state.factory_threads)) == 2
    assert all(owner == caller for owner, caller in state.client_call_threads)
    assert cache.get(first[1]) == result
    assert cache.get(second[1]) == result
    assert state.delivered_images == {
        first_key: [first[2]],
        second_key: [second[2]],
    }

    # A new executor resumes only from durable entries and creates no client.
    resumed_state = _FakeTransportState({})
    resumed = _executor(cache, resumed_state, [], maximum=2)
    assert resumed.complete(first[1], first[0]) == result
    assert resumed.complete(second[1], second[0]) == result
    assert resumed_state.calls == Counter()
    assert resumed_state.factory_threads == []
