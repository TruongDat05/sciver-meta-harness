import os
from pathlib import Path
import subprocess
import sys

import pytest

from model_inference.remote_config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    RemoteAPIConfig,
    RemoteConfigurationError,
    UnsupportedModelError,
    get_model_specification,
    list_supported_models,
    load_remote_config,
    validate_config_for_live_request,
    validate_model_identifier,
)


EXPECTED_MODELS = (
    "Qwen2.5-VL-7B-Instruct",
    "gemma-4-31B-it",
    "gemma-4-26B-A4B-it",
    "gemma-3-27b-it",
)


def test_registry_contains_exactly_requested_models():
    assert list_supported_models() == EXPECTED_MODELS
    for model_identifier in EXPECTED_MODELS:
        assert validate_model_identifier(model_identifier) == model_identifier
        specification = get_model_specification(model_identifier)
        assert specification.identifier == model_identifier


def test_unsupported_model_has_clear_error():
    with pytest.raises(UnsupportedModelError, match="Unsupported remote model"):
        get_model_specification("unsupported-model")


def test_api_url_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("API_URL", "https://invalid.example.test/api")

    config = load_remote_config()

    assert config.api_url == "https://invalid.example.test/api"


def test_missing_api_url_is_reported_before_live_request(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("API_KEY", "obviously-fake-key")

    with pytest.raises(RemoteConfigurationError, match="API_URL"):
        validate_config_for_live_request()


def test_missing_api_key_is_reported_before_live_request(monkeypatch):
    monkeypatch.setenv("API_URL", "https://invalid.example.test/api")
    monkeypatch.delenv("API_KEY", raising=False)

    with pytest.raises(RemoteConfigurationError, match="API_KEY"):
        validate_config_for_live_request()


def test_importing_module_requires_no_credentials():
    environment = os.environ.copy()
    environment.pop("API_URL", None)
    environment.pop("API_KEY", None)

    subprocess.run(
        [sys.executable, "-c", "import model_inference.remote_config"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=Path(__file__).resolve().parents[1],
    )


def test_missing_credentials_are_allowed_for_offline_configuration():
    config = RemoteAPIConfig.from_environment({})

    assert config.api_key is None
    assert config.api_url is None
    assert config.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert config.max_retries == DEFAULT_MAX_RETRIES


def test_timeout_and_retries_are_read_from_environment():
    config = RemoteAPIConfig.from_environment(
        {
            "API_TIMEOUT_SECONDS": "12.5",
            "API_MAX_RETRIES": "5",
        }
    )

    assert config.timeout_seconds == 12.5
    assert config.max_retries == 5


@pytest.mark.parametrize(
    ("environment", "expected_message"),
    [
        ({"API_TIMEOUT_SECONDS": "0"}, "API_TIMEOUT_SECONDS"),
        ({"API_TIMEOUT_SECONDS": "nan"}, "API_TIMEOUT_SECONDS"),
        ({"API_TIMEOUT_SECONDS": "not-a-number"}, "API_TIMEOUT_SECONDS"),
        ({"API_MAX_RETRIES": "-1"}, "API_MAX_RETRIES"),
        ({"API_MAX_RETRIES": "1.5"}, "API_MAX_RETRIES"),
    ],
)
def test_invalid_numeric_configuration_has_clear_error(
    environment, expected_message
):
    with pytest.raises(RemoteConfigurationError, match=expected_message):
        RemoteAPIConfig.from_environment(environment)


def test_api_key_is_not_in_configuration_representation():
    config = RemoteAPIConfig(
        api_key="obviously-fake-key",
        api_url="https://invalid.example.test/api",
    )

    assert "obviously-fake-key" not in repr(config)
