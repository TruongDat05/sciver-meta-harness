from __future__ import annotations

from dataclasses import replace
import json
import os

import pytest

import meta_harness.search_cache as cache_module
from meta_harness.search_cache import (
    SearchCache,
    SearchCacheConflictError,
    SearchCacheError,
    SearchCacheIntegrityError,
    SearchCacheSafetyError,
)
from meta_harness.solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
    build_solver_request_identity,
)


FAKE_SECRET = "UNMISTAKABLY_FAKE_CACHE_SECRET"
IMAGE_DATA = "A" * 160


def _request():
    return SolverRequest(
        model="gemma-4-26B-A4B-it",
        messages=(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{IMAGE_DATA}",
                        },
                    },
                    {
                        "type": "text",
                        "text": f"Private request marker: {FAKE_SECRET}",
                    },
                ],
            },
        ),
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


def _result(content="Answer: yes"):
    return SolverResult(
        content=content,
        usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    )


def test_cache_miss_then_complete_hit(tmp_path):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    result = _result()

    assert cache.get(identity) is None
    assert cache.put(identity, result) == result
    assert cache.get(identity) == result

    artifact = json.loads(cache.entry_path(identity).read_text(encoding="utf-8"))
    assert artifact["request_identity"] == identity.as_dict()
    assert artifact["request_identity_sha256"] == identity.sha256()
    assert artifact["completion"]["content"] == "Answer: yes"


@pytest.mark.parametrize(
    "changed",
    [
        lambda value: replace(value, protocol_id="incompatible_protocol"),
        lambda value: replace(value, config_sha256="a" * 64),
        lambda value: replace(value, split_sha256="b" * 64),
        lambda value: replace(value, search_membership_sha256="c" * 64),
        lambda value: replace(value, prompt_sha256="d" * 64),
        lambda value: replace(value, candidate_id="different_candidate"),
        lambda value: replace(value, solver_identity_sha256="e" * 64),
        lambda value: replace(value, solver_model="different-model"),
        lambda value: replace(value, parser_version="different_parser"),
        lambda value: replace(value, sample_id="different-sample"),
        lambda value: replace(value, request_payload_sha256="f" * 64),
        lambda value: replace(
            value,
            generation=replace(value.generation, max_tokens=4096),
        ),
    ],
)
def test_cache_isolated_across_every_request_identity_dimension(tmp_path, changed):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    cache.put(identity, _result())

    incompatible = changed(identity)

    assert incompatible.sha256() != identity.sha256()
    assert cache.get(incompatible) is None
    assert cache.get(identity) == _result()


def test_previous_model_cache_entry_remains_isolated_and_untouched(tmp_path):
    cache = SearchCache(tmp_path / "cache")
    current_identity = _identity()
    previous_identity = replace(
        current_identity,
        config_sha256="9cd31a36b1763de32ed3e3878176aee5d7521c645d8ca4bfe4e4f91dc5019517",
        solver_model="Qwen/Qwen3.5-35B-A3B",
        request_payload_sha256="6" * 64,
    )
    cache.put(previous_identity, _result())
    previous_bytes = cache.entry_path(previous_identity).read_bytes()

    assert current_identity.sha256() != previous_identity.sha256()
    assert cache.get(current_identity) is None
    assert cache.get(previous_identity) == _result()
    assert cache.entry_path(previous_identity).read_bytes() == previous_bytes


def test_atomic_interruption_leaves_a_miss_and_retry_recovers(
    tmp_path,
    monkeypatch,
):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    real_link = os.link

    def interrupted_link(_source, _destination):
        raise OSError(f"Authorization: Bearer {FAKE_SECRET}")

    monkeypatch.setattr(cache_module.os, "link", interrupted_link)
    with pytest.raises(SearchCacheError, match="atomically") as raised:
        cache.put(identity, _result())

    assert FAKE_SECRET not in str(raised.value)
    assert cache.get(identity) is None
    assert not list((tmp_path / "cache").rglob("*.tmp"))

    monkeypatch.setattr(cache_module.os, "link", real_link)
    assert cache.put(identity, _result()) == _result()
    assert cache.get(identity) == _result()


@pytest.mark.parametrize(
    "damaged",
    [
        b'{"schema_version":',
        b'{"schema_version":"wrong"}\n',
        b'{"schema_version":"x","schema_version":"y"}\n',
    ],
)
def test_corrupt_or_truncated_entries_fail_closed_with_sanitized_error(
    tmp_path,
    damaged,
):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    path = cache.entry_path(identity)
    path.parent.mkdir(parents=True)
    path.write_bytes(damaged + FAKE_SECRET.encode("ascii"))

    with pytest.raises(SearchCacheIntegrityError) as raised:
        cache.get(identity)

    message = str(raised.value)
    assert "move it aside and retry" in message
    assert FAKE_SECRET not in message


def test_entry_stored_under_wrong_identity_fails_closed(tmp_path):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    cache.put(identity, _result())
    incompatible = replace(identity, prompt_sha256="a" * 64)
    wrong_path = cache.entry_path(incompatible)
    wrong_path.parent.mkdir(parents=True, exist_ok=True)
    wrong_path.write_bytes(cache.entry_path(identity).read_bytes())

    with pytest.raises(SearchCacheIntegrityError, match="identity"):
        cache.get(incompatible)


def test_completed_entry_is_immutable_and_idempotent(tmp_path):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()
    original = _result("Answer: yes")
    cache.put(identity, original)
    path = cache.entry_path(identity)
    original_bytes = path.read_bytes()

    assert cache.put(identity, original) == original
    assert path.read_bytes() == original_bytes
    with pytest.raises(SearchCacheConflictError, match="not overwritten"):
        cache.put(identity, _result("Answer: no"))

    assert path.read_bytes() == original_bytes
    assert cache.get(identity) == original


def test_failed_or_sensitive_completion_never_creates_an_entry(tmp_path):
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()

    with pytest.raises(SearchCacheError, match="complete SolverResult"):
        cache.put(identity, RuntimeError("request failed"))
    assert cache.get(identity) is None

    sensitive = SolverResult(
        content=f"Authorization: Bearer {FAKE_SECRET}",
        usage=None,
    )
    with pytest.raises(SearchCacheSafetyError, match="not written") as raised:
        cache.put(identity, sensitive)
    assert FAKE_SECRET not in str(raised.value)
    assert cache.get(identity) is None

    encoded_image = SolverResult(
        content=f"data:image/png;base64,{IMAGE_DATA}",
        usage=None,
    )
    with pytest.raises(SearchCacheSafetyError, match="not written"):
        cache.put(identity, encoded_image)
    assert cache.get(identity) is None


def test_cache_artifact_excludes_request_secret_and_image_base64(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("API_KEY", FAKE_SECRET)
    monkeypatch.setenv("API_URL", "https://invalid.example.test/private")
    cache = SearchCache(tmp_path / "cache")
    identity = _identity()

    cache.put(identity, _result())

    serialized = cache.entry_path(identity).read_text(encoding="utf-8")
    assert FAKE_SECRET not in serialized
    assert IMAGE_DATA not in serialized
    assert "data:image" not in serialized
    assert "Authorization" not in serialized
    assert "invalid.example.test" not in serialized
    assert set(json.loads(serialized)["request_identity"]) == {
        "schema_version",
        "protocol_id",
        "config_sha256",
        "split_sha256",
        "search_membership_sha256",
        "prompt_sha256",
        "candidate_id",
        "solver_identity_sha256",
        "solver_model",
        "generation",
        "parser_version",
        "sample_id",
        "request_payload_sha256",
    }
