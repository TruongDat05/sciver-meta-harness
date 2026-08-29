from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import json
from unittest.mock import Mock, call

import pytest
import requests

from meta_harness.search_cache import SearchCache
from meta_harness.retry import (
    FailureCategory,
    SolverExecutionFailure,
    SolverParserFailure,
    SolverRetryPolicy,
    classify_solver_failure,
    execute_solver_request_with_retry,
)
from meta_harness.solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
    build_solver_request_identity,
)
from model_inference.remote_client import (
    AuthenticationError,
    InvalidRequestError,
    MalformedJSONError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    UnexpectedResponseSchemaError,
)


FAKE_SECRET = "UNMISTAKABLY_FAKE_RETRY_SECRET"


class SequenceSolver:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def complete(self, _request):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _request():
    return SolverRequest(
        model="gemma-4-26B-A4B-it",
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


def _identity():
    return build_solver_request_identity(
        _request(),
        sample_id="sciver_sample_content_v1:" + "1" * 64,
        candidate_id="p0",
        prompt_sha256="2" * 64,
        split_sha256="3" * 64,
        search_membership_sha256="4" * 64,
        solver_identity_sha256="5" * 64,
    )


def _http_error(error_type, status, retry_after=None):
    return error_type(
        "sensitive remote detail must not persist",
        http_status_code=status,
        response_error_message=FAKE_SECRET,
        retry_after_seconds=retry_after,
    )


@pytest.mark.parametrize(
    ("error", "category", "retryable"),
    [
        (RequestTimeoutError("timeout"), FailureCategory.TIMEOUT, True),
        (requests.exceptions.Timeout(), FailureCategory.TIMEOUT, True),
        (NetworkError("connection"), FailureCategory.CONNECTION, True),
        (requests.exceptions.ConnectionError(), FailureCategory.CONNECTION, True),
        (
            _http_error(InvalidRequestError, 408),
            FailureCategory.HTTP_408,
            True,
        ),
        (
            _http_error(RateLimitError, 429),
            FailureCategory.HTTP_429,
            True,
        ),
        (
            _http_error(ServerError, 503),
            FailureCategory.HTTP_5XX_RETRYABLE,
            True,
        ),
        (
            _http_error(ServerError, 501),
            FailureCategory.HTTP_5XX_PERMANENT,
            False,
        ),
        (
            _http_error(AuthenticationError, 401),
            FailureCategory.HTTP_4XX_PERMANENT,
            False,
        ),
        (MalformedJSONError("bad JSON"), FailureCategory.PROTOCOL_OR_SCHEMA, False),
        (
            UnexpectedResponseSchemaError("bad schema"),
            FailureCategory.PROTOCOL_OR_SCHEMA,
            False,
        ),
        (SolverParserFailure("parse failed"), FailureCategory.PARSER, False),
        (FutureCancelledError(), FailureCategory.CANCELLATION, False),
        (asyncio.CancelledError(), FailureCategory.CANCELLATION, False),
        (RuntimeError("unexpected"), FailureCategory.UNEXPECTED, False),
    ],
)
def test_every_failure_classification(error, category, retryable):
    classification = classify_solver_failure(error)

    assert classification.category is category
    assert classification.retryable is retryable


def test_only_configured_5xx_statuses_are_retryable():
    policy = SolverRetryPolicy(retryable_5xx=frozenset({503}))

    assert classify_solver_failure(
        _http_error(ServerError, 503), policy
    ).category is FailureCategory.HTTP_5XX_RETRYABLE
    assert classify_solver_failure(
        _http_error(ServerError, 500), policy
    ).category is FailureCategory.HTTP_5XX_PERMANENT


@pytest.mark.parametrize(
    "error",
    [
        RequestTimeoutError("timeout"),
        NetworkError("connection"),
        _http_error(InvalidRequestError, 408),
        _http_error(RateLimitError, 429),
        _http_error(ServerError, 503),
    ],
)
def test_retryable_failures_use_exact_maximum_attempts(error):
    solver = SequenceSolver([error, error, error])
    sleeper = Mock()
    ticks = iter((10.0, 12.5))

    with pytest.raises(SolverExecutionFailure) as raised:
        execute_solver_request_with_retry(
            solver,
            _request(),
            policy=SolverRetryPolicy(
                maximum_attempts=3,
                initial_backoff_seconds=1,
                maximum_backoff_seconds=4,
            ),
            sleeper=sleeper,
            clock=lambda: next(ticks),
        )

    assert solver.calls == 3
    assert sleeper.call_args_list == [call(1.0), call(2.0)]
    assert raised.value.metadata.attempt_count == 3
    assert raised.value.metadata.exhausted is True
    assert raised.value.metadata.elapsed_seconds == 2.5


@pytest.mark.parametrize(
    "error",
    [
        _http_error(InvalidRequestError, 400),
        MalformedJSONError("protocol"),
        SolverParserFailure("parser"),
        FutureCancelledError(),
    ],
)
def test_permanent_failures_are_attempted_once_without_sleep(error):
    solver = SequenceSolver([error])
    sleeper = Mock()
    ticks = iter((4.0, 4.5))

    with pytest.raises(SolverExecutionFailure) as raised:
        execute_solver_request_with_retry(
            solver,
            _request(),
            policy=SolverRetryPolicy(maximum_attempts=4),
            sleeper=sleeper,
            clock=lambda: next(ticks),
        )

    assert solver.calls == 1
    sleeper.assert_not_called()
    assert raised.value.metadata.attempt_count == 1
    assert raised.value.metadata.exhausted is False


def test_retry_after_and_exponential_backoff_are_bounded():
    transient = _http_error(RateLimitError, 429, retry_after=90)
    solver = SequenceSolver(
        [
            transient,
            RequestTimeoutError("timeout"),
            RequestTimeoutError("timeout"),
            SolverResult("Answer: yes"),
        ]
    )
    sleeper = Mock()

    result = execute_solver_request_with_retry(
        solver,
        _request(),
        policy=SolverRetryPolicy(
            maximum_attempts=4,
            initial_backoff_seconds=1.5,
            maximum_backoff_seconds=4,
        ),
        sleeper=sleeper,
        clock=lambda: 0.0,
    )

    assert result == SolverResult("Answer: yes")
    assert solver.calls == 4
    assert sleeper.call_args_list == [
        call(4.0),
        call(3.0),
        call(4.0),
    ]


def test_transient_failure_recovers_without_extra_attempts():
    solver = SequenceSolver(
        [RequestTimeoutError("first"), SolverResult("Answer: no")]
    )
    sleeper = Mock()

    result = execute_solver_request_with_retry(
        solver,
        _request(),
        policy=SolverRetryPolicy(maximum_attempts=4),
        sleeper=sleeper,
        clock=lambda: 0.0,
    )

    assert result == SolverResult("Answer: no")
    assert solver.calls == 2
    sleeper.assert_called_once_with(1.0)


def test_retry_exhaustion_metadata_is_sanitized_and_cache_remains_a_miss(tmp_path):
    sensitive_error = RequestTimeoutError(
        f"Authorization: Bearer {FAKE_SECRET}; "
        f"data:image/png;base64,{'A' * 160}"
    )
    solver = SequenceSolver([sensitive_error, sensitive_error])
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    ticks = iter((20.0, 21.0))

    with pytest.raises(SolverExecutionFailure) as raised:
        execute_solver_request_with_retry(
            solver,
            _request(),
            policy=SolverRetryPolicy(maximum_attempts=2),
            sleeper=lambda _delay: None,
            clock=lambda: next(ticks),
        )

    metadata = raised.value.metadata.as_dict()
    serialized = json.dumps(metadata, sort_keys=True)
    assert metadata == {
        "category": "timeout",
        "retryable": True,
        "exhausted": True,
        "attempt_count": 2,
        "maximum_attempts": 2,
        "http_status_code": None,
        "retry_delays_seconds": [1.0],
        "elapsed_seconds": 1.0,
        "cause_type": "RequestTimeoutError",
    }
    assert FAKE_SECRET not in str(raised.value)
    assert FAKE_SECRET not in serialized
    assert "Authorization" not in serialized
    assert "base64" not in serialized
    assert cache.get(identity) is None
    assert not cache.entry_path(identity).exists()


def test_keyboard_interrupt_and_async_cancellation_propagate_unchanged():
    sleeper = Mock()
    keyboard_solver = SequenceSolver([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        execute_solver_request_with_retry(
            keyboard_solver,
            _request(),
            policy=SolverRetryPolicy(maximum_attempts=4),
            sleeper=sleeper,
            clock=lambda: 0.0,
        )

    cancelled_solver = SequenceSolver([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        execute_solver_request_with_retry(
            cancelled_solver,
            _request(),
            policy=SolverRetryPolicy(maximum_attempts=4),
            sleeper=sleeper,
            clock=lambda: 0.0,
        )

    sleeper.assert_not_called()
