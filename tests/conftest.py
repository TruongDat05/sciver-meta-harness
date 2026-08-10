"""Suite-wide safety controls for offline benchmark tests."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import requests


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch):
    """Make credentials, accelerators, model downloads, and network opt-in only."""

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def deny_http(*args, **kwargs):
        raise AssertionError("tests must mock every HTTP request")

    def deny_socket(*args, **kwargs):
        raise AssertionError("tests must not open network connections")

    monkeypatch.setattr(requests.sessions.Session, "request", deny_http)
    monkeypatch.setattr(socket.socket, "connect", deny_socket)

