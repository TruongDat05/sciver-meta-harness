from __future__ import annotations

import base64
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from meta_harness.full_search_v3_solver import (
    FullSearchV3SolverError,
    LiveSolverDisabledError,
    SolverResult,
    build_solver_request,
    create_live_solver_client,
    execute_solver_request,
    preflight_live_solver_model_list,
)
from model_inference.remote_api import prepare_remote_requests
from model_inference.remote_config import RemoteAPIConfig
from utils.constant import COT_PROMPT


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FAKE_API_KEY = "UNMISTAKABLY_FAKE_SOLVER_KEY"
FAKE_API_URL = "https://invalid.example.test/chat/completions"


class DeterministicSolver:
    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return SolverResult(
            content="Therefore, the final answer is: Answer: yes",
            usage={"total_tokens": 7},
        )


def _paired_record(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), color="red").save(first, format="PNG")
    Image.new("RGB", (2, 2), color="green").save(second, format="PNG")
    return {
        "sample_id": "safe-search-sample",
        "claim_type": "parallel",
        "claim": "The claim remains unchanged.",
        "context": "Local evidence context.",
        "caption1": "First caption.",
        "caption2": "Second caption.",
        "item1_path": str(first),
        "item2_path": str(second),
        "gold_label": "GROUND_TRUTH_MUST_NOT_LEAK",
    }, first, second


def test_injected_fake_receives_existing_request_builder_output(tmp_path):
    record, first, second = _paired_record(tmp_path)
    expected_messages = prepare_remote_requests([record], COT_PROMPT)[0]
    request = build_solver_request(record, COT_PROMPT)
    fake = DeterministicSolver()

    result = execute_solver_request(fake, request)

    assert fake.requests == [request]
    assert list(request.messages) == expected_messages
    assert request.model == "Qwen3.6-35B-A3B"
    assert request.generation.as_request_options() == {
        "temperature": 0,
        "top_p": 1,
        "seed": 42,
        "n": 1,
        "stream": False,
        "max_tokens": 8192,
    }
    content = request.messages[0]["content"]
    image_bytes = [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in content
        if block["type"] == "image_url"
    ]
    assert image_bytes == [first.read_bytes(), second.read_bytes()]
    assert "GROUND_TRUTH_MUST_NOT_LEAK" not in json.dumps(request.messages)
    assert result == SolverResult(
        content="Therefore, the final answer is: Answer: yes",
        usage={"total_tokens": 7},
    )


def test_injected_boundary_rejects_invalid_client_result(tmp_path):
    record, _first, _second = _paired_record(tmp_path)
    client = Mock()
    client.complete.return_value = "not-a-typed-result"

    with pytest.raises(FullSearchV3SolverError, match="SolverResult"):
        execute_solver_request(client, build_solver_request(record, COT_PROMPT))


def test_live_factory_requires_authorization_before_config_or_client(monkeypatch):
    import model_inference.remote_client as remote_client
    import model_inference.remote_config as remote_config

    config_loader = Mock(side_effect=AssertionError("credentials were read"))
    client_factory = Mock(side_effect=AssertionError("client was constructed"))
    monkeypatch.setattr(
        remote_config,
        "validate_config_for_live_request",
        config_loader,
    )
    monkeypatch.setattr(
        remote_client,
        "RemoteChatCompletionsClient",
        client_factory,
    )

    with pytest.raises(LiveSolverDisabledError, match="authorization"):
        create_live_solver_client()

    config_loader.assert_not_called()
    client_factory.assert_not_called()


def test_live_factory_propagates_exact_model_and_generation(monkeypatch, tmp_path):
    import model_inference.remote_client as remote_client
    import model_inference.remote_config as remote_config

    record, _first, _second = _paired_record(tmp_path)
    low_level_client = Mock()
    low_level_client.create_chat_completion.return_value = "Answer: no"
    low_level_client.last_usage = {
        "input_tokens": 10,
        "output_tokens": 2,
        "ignored": 99,
    }
    config = RemoteAPIConfig(
        api_key=FAKE_API_KEY,
        api_url=FAKE_API_URL,
        timeout_seconds=12.5,
        max_retries=3,
    )
    config_loader = Mock(return_value=config)
    client_factory = Mock(return_value=low_level_client)
    monkeypatch.setattr(
        remote_config,
        "validate_config_for_live_request",
        config_loader,
    )
    monkeypatch.setattr(
        remote_client,
        "RemoteChatCompletionsClient",
        client_factory,
    )

    client = create_live_solver_client(allow_live_requests=True)
    request = build_solver_request(record, COT_PROMPT)
    result = execute_solver_request(client, request)

    config_loader.assert_called_once_with()
    client_factory.assert_called_once()
    assert client_factory.call_args.kwargs["retry_settings"].max_retries == 0
    low_level_client.create_chat_completion.assert_called_once_with(
        "Qwen3.6-35B-A3B",
        request.messages,
        generation_options={
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "n": 1,
            "stream": False,
            "max_tokens": 8192,
        },
    )
    assert result == SolverResult(
        content="Answer: no",
        usage={"input_tokens": 10, "output_tokens": 2},
    )


def test_live_factory_base_mode_and_sanitized_locked_model_preflight(monkeypatch):
    import model_inference.remote_client as remote_client
    import model_inference.remote_config as remote_config

    low_level_client = Mock()
    low_level_client.list_model_ids.return_value = (
        "Qwen3.6-35B-A3B",
        "another-model",
    )
    config = RemoteAPIConfig(
        api_key=FAKE_API_KEY,
        api_url="https://invalid.example.test/base",
        timeout_seconds=9.0,
        max_retries=3,
    )
    monkeypatch.setattr(
        remote_config,
        "validate_config_for_live_request",
        Mock(return_value=config),
    )
    base_factory = Mock(return_value=low_level_client)
    monkeypatch.setattr(
        remote_client.RemoteChatCompletionsClient,
        "from_base_url",
        base_factory,
    )

    client = create_live_solver_client(
        allow_live_requests=True,
        api_url_is_base=True,
    )
    summary = preflight_live_solver_model_list(client)

    assert summary["status"] == "passed"
    assert summary["model_count"] == 2
    assert set(summary) == {
        "schema_version",
        "status",
        "model_count",
        "model_ids_sha256",
        "locked_model_sha256",
    }
    assert "Qwen3.6-35B-A3B" not in str(summary)
    base_factory.assert_called_once()
    assert base_factory.call_args.kwargs["api_url"] == config.api_url
    assert base_factory.call_args.kwargs["models_path"] == "/models"
    assert (
        base_factory.call_args.kwargs["chat_completions_path"]
        == "/chat/completions"
    )


def test_locked_model_preflight_never_selects_an_arbitrary_model():
    client = Mock()
    client.list_model_ids.return_value = ("another-model",)

    with pytest.raises(FullSearchV3SolverError, match="immutable V3 solver"):
        preflight_live_solver_model_list(client)


def test_import_has_no_client_network_or_credential_side_effect():
    probe = """
import importlib
import socket
import sys
from unittest.mock import patch

import model_inference.remote_client as remote_client
import model_inference.remote_config as remote_config

sys.modules.pop("meta_harness.full_search_v3_solver", None)
with patch.object(
    remote_client,
    "RemoteChatCompletionsClient",
    side_effect=AssertionError("import constructed a client"),
), patch.object(
    remote_config,
    "validate_config_for_live_request",
    side_effect=AssertionError("import read credentials"),
), patch.object(
    socket.socket,
    "connect",
    side_effect=AssertionError("import opened a socket"),
):
    importlib.import_module("meta_harness.full_search_v3_solver")
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
