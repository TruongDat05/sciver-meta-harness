from unittest.mock import Mock, call

import pytest
import requests

from model_inference.remote_client import (
    AuthenticationError,
    InvalidConfigurationError,
    InvalidRequestError,
    MalformedJSONError,
    NetworkError,
    PermissionDeniedError,
    RateLimitError,
    RemoteChatCompletionsClient,
    RequestTimeoutError,
    RetrySettings,
    ServerError,
    UnexpectedResponseSchemaError,
    build_compatible_api_endpoints,
)


FAKE_API_KEY = "obviously-fake-key"
FAKE_API_URL = "https://invalid.example.test/chat/completions"
MODEL = "test-model"
MESSAGES = [{"role": "user", "content": "Test prompt"}]


def _response(status_code=200, body=None, headers=None):
    response = Mock()
    response.status_code = status_code
    response.headers = {} if headers is None else headers
    response.json.return_value = (
        {"choices": [{"message": {"content": "Test response"}}]}
        if body is None
        else body
    )
    return response


def _client(session, *, max_retries=0, sleep=None):
    return RemoteChatCompletionsClient(
        api_key=FAKE_API_KEY,
        api_url=FAKE_API_URL,
        timeout=12.5,
        retry_settings=RetrySettings(
            max_retries=max_retries,
            initial_backoff_seconds=1,
            max_backoff_seconds=4,
        ),
        session=session,
        sleep=Mock() if sleep is None else sleep,
    )


def test_successful_request_extracts_content_and_uses_expected_request():
    session = Mock()
    session.post.return_value = _response(
        body={
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
            },
        }
    )

    client = _client(session)
    content = client.create_chat_completion(MODEL, MESSAGES)

    assert content == "Test response"
    assert client.last_usage == {
        "input_tokens": 11,
        "output_tokens": 3,
        "total_tokens": 14,
    }
    session.post.assert_called_once_with(
        FAKE_API_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FAKE_API_KEY}",
        },
        json={"model": MODEL, "messages": MESSAGES, "stream": False},
        timeout=12.5,
    )
    response = session.post.return_value
    response.iter_lines.assert_not_called()


def test_generation_options_are_propagated_without_changing_legacy_defaults():
    session = Mock()
    session.post.return_value = _response()
    generation = {
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "n": 1,
        "stream": False,
        "max_tokens": 8192,
    }

    _client(session).create_chat_completion(
        MODEL,
        MESSAGES,
        generation_options=generation,
    )

    assert session.post.call_args.kwargs["json"] == {
        "model": MODEL,
        "messages": MESSAGES,
        **generation,
    }


@pytest.mark.parametrize(
    "generation_options",
    [
        {"unknown": 1},
        {"stream": True},
        {"n": 2},
        {"max_tokens": 0},
    ],
)
def test_invalid_generation_options_fail_before_http(
    generation_options,
):
    session = Mock()

    with pytest.raises(InvalidConfigurationError):
        _client(session).create_chat_completion(
            MODEL,
            MESSAGES,
            generation_options=generation_options,
        )

    session.post.assert_not_called()


def test_missing_or_invalid_usage_is_ignored_without_stale_accounting():
    session = Mock()
    session.post.side_effect = [
        _response(
            body={
                "choices": [{"message": {"content": "first"}}],
                "usage": {"total_tokens": 4},
            }
        ),
        _response(
            body={
                "choices": [{"message": {"content": "second"}}],
                "usage": {"total_tokens": "not-an-integer"},
            }
        ),
    ]
    client = _client(session)

    assert client.create_chat_completion(MODEL, MESSAGES) == "first"
    assert client.last_usage == {"total_tokens": 4}
    assert client.create_chat_completion(MODEL, MESSAGES) == "second"
    assert client.last_usage is None


def test_client_reads_credentials_and_endpoint_from_environment(monkeypatch):
    monkeypatch.setenv("API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("API_URL", FAKE_API_URL)
    session = Mock()
    session.post.return_value = _response()

    client = RemoteChatCompletionsClient(
        session=session,
        retry_settings=RetrySettings(max_retries=0),
    )
    client.create_chat_completion(MODEL, MESSAGES)

    assert session.post.call_args.args == (FAKE_API_URL,)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api_key": ""},
        {"api_url": ""},
        {"api_url": "not-a-url"},
        {"timeout": 0},
        {"session": object()},
        {"sleep": None},
    ],
)
def test_invalid_configuration_has_clear_exception(overrides):
    arguments = {
        "api_key": FAKE_API_KEY,
        "api_url": FAKE_API_URL,
        "session": Mock(),
    }
    arguments.update(overrides)

    with pytest.raises(InvalidConfigurationError):
        RemoteChatCompletionsClient(**arguments)


@pytest.mark.parametrize(
    ("status_code", "exception_type"),
    [
        (400, InvalidRequestError),
        (401, AuthenticationError),
        (403, PermissionDeniedError),
        (404, InvalidRequestError),
    ],
)
def test_permanent_client_errors_are_not_retried(status_code, exception_type):
    session = Mock()
    session.post.return_value = _response(status_code)

    with pytest.raises(exception_type):
        _client(session, max_retries=3).create_chat_completion(MODEL, MESSAGES)

    session.post.assert_called_once()


def test_http_error_does_not_expose_response_body_diagnostics():
    session = Mock()
    session.post.return_value = _response(
        400,
        body={
            "error": {
                "code": "invalid_image",
                "message": "The image dimensions are unsupported.",
            }
        },
    )

    with pytest.raises(InvalidRequestError) as raised:
        _client(session).create_chat_completion(MODEL, MESSAGES)

    assert raised.value.http_status_code == 400
    assert raised.value.response_error_code is None
    assert raised.value.response_error_message is None
    assert "unsupported" not in str(raised.value)


@pytest.mark.parametrize(
    ("base_url", "models_url", "chat_url"),
    [
        (
            " https://invalid.example.test/api/// ",
            "https://invalid.example.test/api/models",
            "https://invalid.example.test/api/chat/completions",
        ),
        (
            "https://invalid.example.test//nested///base/",
            "https://invalid.example.test/nested/base/models",
            "https://invalid.example.test/nested/base/chat/completions",
        ),
    ],
)
def test_base_url_joining_normalizes_without_inserting_v1(
    base_url, models_url, chat_url
):
    endpoints = build_compatible_api_endpoints(base_url)

    assert endpoints.models_url == models_url
    assert endpoints.chat_completions_url == chat_url
    assert "/v1/" not in endpoints.models_url
    assert "/v1/" not in endpoints.chat_completions_url


@pytest.mark.parametrize(
    "base_url",
    [
        "https://user:pass@invalid.example.test/base",
        "https://invalid.example.test/base?query=value",
        "https://invalid.example.test/base#fragment",
    ],
)
def test_base_url_rejects_unsafe_components(base_url):
    with pytest.raises(InvalidConfigurationError):
        build_compatible_api_endpoints(base_url)


def test_base_url_client_gets_models_and_posts_valid_json_to_exact_paths():
    session = Mock()
    session.get.return_value = _response(
        body={
            "object": "list",
            "data": [
                {"id": " locked-model "},
                {"id": "locked-model"},
                {"id": ""},
                {"missing": "id"},
                "invalid",
            ],
        }
    )
    session.post.return_value = _response()
    client = RemoteChatCompletionsClient.from_base_url(
        api_key=f"  {FAKE_API_KEY}  ",
        api_url=" https://invalid.example.test/base/// ",
        timeout=7.5,
        retry_settings=RetrySettings(max_retries=0),
        session=session,
    )

    assert client.list_model_ids() == ("locked-model",)
    client.create_chat_completion(
        MODEL,
        MESSAGES,
        generation_options={"temperature": 0, "top_p": 1},
    )

    session.get.assert_called_once_with(
        "https://invalid.example.test/base/models",
        headers={"Authorization": f"Bearer {FAKE_API_KEY}"},
        timeout=7.5,
    )
    session.post.assert_called_once_with(
        "https://invalid.example.test/base/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FAKE_API_KEY}",
        },
        json={
            "model": MODEL,
            "messages": MESSAGES,
            "stream": False,
            "temperature": 0,
            "top_p": 1,
        },
        timeout=7.5,
    )


@pytest.mark.parametrize(
    ("status_code", "exception_type"),
    [(401, AuthenticationError), (403, PermissionDeniedError)],
)
def test_model_list_authentication_and_permission_fail_closed_without_retry(
    status_code, exception_type
):
    session = Mock()
    session.get.return_value = _response(status_code)
    client = RemoteChatCompletionsClient.from_base_url(
        api_key=FAKE_API_KEY,
        api_url="https://invalid.example.test/base",
        retry_settings=RetrySettings(max_retries=3),
        session=session,
    )

    with pytest.raises(exception_type):
        client.list_model_ids()

    session.get.assert_called_once()


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": {}},
        {"data": [None, {}, {"id": ""}]},
        {"object": "unexpected", "data": [{"id": "model"}]},
    ],
)
def test_model_list_incompatible_schema_fails_closed(body):
    session = Mock()
    session.get.return_value = _response(body=body)
    client = RemoteChatCompletionsClient.from_base_url(
        api_key=FAKE_API_KEY,
        api_url="https://invalid.example.test/base",
        retry_settings=RetrySettings(max_retries=0),
        session=session,
    )

    with pytest.raises(UnexpectedResponseSchemaError):
        client.list_model_ids()


def test_model_list_invalid_json_and_timeout_fail_closed_without_payload_exposure():
    malformed_session = Mock()
    malformed = _response()
    malformed.json.side_effect = ValueError(FAKE_API_KEY)
    malformed_session.get.return_value = malformed
    malformed_client = RemoteChatCompletionsClient.from_base_url(
        api_key=FAKE_API_KEY,
        api_url="https://invalid.example.test/base",
        retry_settings=RetrySettings(max_retries=0),
        session=malformed_session,
    )

    with pytest.raises(MalformedJSONError) as malformed_error:
        malformed_client.list_model_ids()
    assert FAKE_API_KEY not in str(malformed_error.value)

    timeout_session = Mock()
    timeout_session.get.side_effect = requests.exceptions.Timeout(FAKE_API_KEY)
    timeout_client = RemoteChatCompletionsClient.from_base_url(
        api_key=FAKE_API_KEY,
        api_url="https://invalid.example.test/base",
        retry_settings=RetrySettings(max_retries=0),
        session=timeout_session,
    )

    with pytest.raises(RequestTimeoutError) as timeout_error:
        timeout_client.list_model_ids()
    assert FAKE_API_KEY not in str(timeout_error.value)
    timeout_session.get.assert_called_once()


def test_api_key_newline_is_rejected_and_surrounding_whitespace_is_normalized():
    with pytest.raises(InvalidConfigurationError, match="invalid characters"):
        RemoteChatCompletionsClient(
            api_key=FAKE_API_KEY + "\n",
            api_url=FAKE_API_URL,
            session=Mock(),
        )


def test_rate_limit_is_retried_and_respects_retry_after():
    session = Mock()
    session.post.side_effect = [
        _response(429, headers={"Retry-After": "3"}),
        _response(),
    ]
    sleep = Mock()

    content = _client(session, max_retries=1, sleep=sleep).create_chat_completion(
        MODEL, MESSAGES
    )

    assert content == "Test response"
    assert session.post.call_count == 2
    sleep.assert_called_once_with(3.0)


def test_terminal_http_error_exposes_numeric_retry_after_for_outer_boundary():
    session = Mock()
    session.post.return_value = _response(
        429,
        headers={"Retry-After": "90"},
    )

    with pytest.raises(RateLimitError) as raised:
        _client(session, max_retries=0).create_chat_completion(MODEL, MESSAGES)

    assert raised.value.retry_after_seconds == 90.0
    assert "90" not in str(raised.value)


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_retryable_server_errors_are_retried(status_code):
    session = Mock()
    session.post.side_effect = [_response(status_code), _response()]
    sleep = Mock()

    content = _client(session, max_retries=1, sleep=sleep).create_chat_completion(
        MODEL, MESSAGES
    )

    assert content == "Test response"
    assert session.post.call_count == 2
    sleep.assert_called_once_with(1.0)


def test_non_retryable_server_error_is_not_retried():
    session = Mock()
    session.post.return_value = _response(501)

    with pytest.raises(ServerError):
        _client(session, max_retries=3).create_chat_completion(MODEL, MESSAGES)

    session.post.assert_called_once()


def test_retryable_error_raises_typed_exception_after_retries():
    session = Mock()
    session.post.side_effect = [_response(429), _response(429)]

    with pytest.raises(RateLimitError):
        _client(session, max_retries=1).create_chat_completion(MODEL, MESSAGES)

    assert session.post.call_count == 2


def test_timeout_is_retried_with_bounded_exponential_backoff():
    session = Mock()
    session.post.side_effect = [
        requests.exceptions.Timeout(),
        requests.exceptions.Timeout(),
        requests.exceptions.Timeout(),
        _response(),
    ]
    sleep = Mock()

    content = _client(session, max_retries=3, sleep=sleep).create_chat_completion(
        MODEL, MESSAGES
    )

    assert content == "Test response"
    assert sleep.call_args_list == [call(1.0), call(2.0), call(4.0)]


def test_timeout_raises_typed_exception_after_retries():
    session = Mock()
    session.post.side_effect = requests.exceptions.Timeout()

    with pytest.raises(RequestTimeoutError):
        _client(session, max_retries=1).create_chat_completion(MODEL, MESSAGES)

    assert session.post.call_count == 2


def test_network_failure_is_retried_and_has_typed_exception():
    session = Mock()
    session.post.side_effect = requests.exceptions.ConnectionError(
        FAKE_API_KEY
    )

    with pytest.raises(NetworkError, match="configured retries") as raised:
        _client(session, max_retries=1).create_chat_completion(MODEL, MESSAGES)

    assert session.post.call_count == 2
    assert FAKE_API_KEY not in str(raised.value)


def test_malformed_json_has_clear_exception():
    session = Mock()
    response = _response()
    response.json.side_effect = ValueError(FAKE_API_KEY)
    session.post.return_value = response

    with pytest.raises(MalformedJSONError, match="malformed JSON") as raised:
        _client(session).create_chat_completion(MODEL, MESSAGES)

    session.post.assert_called_once()
    assert FAKE_API_KEY not in str(raised.value)


@pytest.mark.parametrize(
    ("body", "missing_path"),
    [
        ({}, "choices\\[0\\]"),
        ({"choices": []}, "choices\\[0\\]"),
        ({"choices": [{}]}, "choices\\[0\\]\\.message"),
        (
            {"choices": [{"message": {}}]},
            "choices\\[0\\]\\.message\\.content",
        ),
        (
            {"choices": [{"message": {"content": None}}]},
            "choices\\[0\\]\\.message\\.content",
        ),
    ],
)
def test_missing_response_schema_fields_raise_clear_exception(body, missing_path):
    session = Mock()
    session.post.return_value = _response(body=body)

    with pytest.raises(UnexpectedResponseSchemaError, match=missing_path):
        _client(session).create_chat_completion(MODEL, MESSAGES)


@pytest.mark.parametrize("status_code", [400, 401, 403, 429, 500])
def test_http_exceptions_do_not_contain_api_key(status_code):
    session = Mock()
    session.post.return_value = _response(status_code)

    with pytest.raises(Exception) as raised:
        _client(session).create_chat_completion(MODEL, MESSAGES)

    assert FAKE_API_KEY not in str(raised.value)
