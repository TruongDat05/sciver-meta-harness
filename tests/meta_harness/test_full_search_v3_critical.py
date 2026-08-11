"""Compact offline regressions for full-SEARCH V3 server-readiness invariants."""

from __future__ import annotations

import base64
import copy
import inspect
import json
from pathlib import Path
import socket
from string import Template
from types import MappingProxyType

from PIL import Image
import pytest
import requests

import meta_harness.full_search_v3_evaluator as evaluator
from meta_harness.config import canonical_full_search_v3_config
from meta_harness.full_search_v3_cache import FullSearchV3SearchCache
from meta_harness.full_search_v3_concurrency import FullSearchV3RequestExecutor
from meta_harness.full_search_v3_evaluator import (
    FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256,
    FullSearchV3PredictionOutcome,
    FullSearchV3SearchInput,
    load_full_search_v3_search_input,
    validate_full_search_v3_search_input,
)
from meta_harness.full_search_v3_preparation import prepare_full_search_v3
from meta_harness.full_search_v3_retry import SolverRetryPolicy
from meta_harness.full_search_v3_solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
    build_solver_request_identity,
)
from utils.constant import COT_PROMPT


TESTSET_PATH = Path(__file__).resolve().parents[2] / "data" / "sciver" / "testset.json"


@pytest.fixture(scope="module")
def production_search_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("full-search-v3-critical")
    artifacts = prepare_full_search_v3(
        source_path=TESTSET_PATH,
        private_directory=root / "trusted",
        search_directory=root / "search",
    )
    return load_full_search_v3_search_input(
        search_safe_manifest_path=artifacts.search_safe_manifest_path,
        search_records_path=artifacts.search_dataset_path,
    )


def test_production_input_requires_exact_unique_manifest_order(
    production_search_input,
):
    assert len(production_search_input.records) == 1000
    assert production_search_input.sample_ids == tuple(
        production_search_input.manifest["SEARCH"]["sample_ids"]
    )

    manifest = _mutable(production_search_input.manifest)
    records = _mutable(production_search_input.records)
    with pytest.raises(evaluator.FullSearchV3EvaluatorInputError, match="1,000"):
        validate_full_search_v3_search_input(
            manifest=manifest,
            records=records[:-1],
        )

    duplicate = copy.deepcopy(records)
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(evaluator.FullSearchV3EvaluatorInputError, match="duplicate"):
        validate_full_search_v3_search_input(manifest=manifest, records=duplicate)

    with pytest.raises(
        evaluator.FullSearchV3EvaluatorInputError, match="immutable manifest order"
    ):
        validate_full_search_v3_search_input(
            manifest=manifest,
            records=list(reversed(records)),
        )


def test_p0_and_candidate_share_frozen_request_parser_and_image_path(tmp_path):
    search_input, image_paths = _four_record_input(tmp_path)
    candidate_prompt = {
        method: Template(template.template) for method, template in COT_PROMPT.items()
    }

    p0_requests = evaluator._candidate_requests(
        search_input=search_input,
        candidate_id="cot",
        prompt=COT_PROMPT,
        prompt_sha256=evaluator.canonical_full_search_v3_p0_prompt_sha256(),
        solver_identity_sha256="1" * 64,
    )
    candidate_requests = evaluator._candidate_requests(
        search_input=search_input,
        candidate_id="candidate_1",
        prompt=candidate_prompt,
        prompt_sha256=evaluator._prompt_sha256(evaluator.PromptFamily(candidate_prompt)),
        solver_identity_sha256="1" * 64,
    )

    assert evaluator.canonical_full_search_v3_p0_prompt_sha256() == evaluator.template_source_sha256(COT_PROMPT)
    assert tuple(item["sample_id"] for item in p0_requests) == search_input.sample_ids
    assert tuple(item["sample_id"] for item in candidate_requests) == search_input.sample_ids
    for p0, candidate, record, expected_paths in zip(
        p0_requests, candidate_requests, search_input.records, image_paths
    ):
        assert p0["request"].messages == candidate["request"].messages
        assert _image_bytes(p0["request"].messages) == [
            path.read_bytes() for path in expected_paths
        ]
        assert record["gold_label"] not in json.dumps(p0["request"].messages)

    outcomes = tuple(
        FullSearchV3PredictionOutcome(sample_id, "Answer: yes")
        for sample_id in search_input.sample_ids
    )
    assert [record["prediction"] for record in evaluator._metric_records(search_input, "cot", outcomes)] == [
        record["prediction"]
        for record in evaluator._metric_records(search_input, "candidate_1", outcomes)
    ]


def test_metrics_are_trusted_and_hand_computable():
    search_input = FullSearchV3SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256,
                "search_membership_sha256": "a" * 64,
            }
        ),
        _records=tuple(
            MappingProxyType(
                {
                    "sample_id": f"metric-{index}",
                    "claim_type": "direct",
                    "gold_label": label,
                }
            )
            for index, label in enumerate(("yes", "no", "yes", "no"))
        ),
    )
    outcomes = tuple(
        FullSearchV3PredictionOutcome(sample_id, f"Answer: {prediction}")
        for sample_id, prediction in zip(search_input.sample_ids, ("yes", "yes", "yes", "no"))
    )

    metrics = evaluator._compute_prediction_metrics(search_input, outcomes)

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(11 / 15)


def test_m2_cache_reuses_completed_work_and_resumes_missing_work(tmp_path, monkeypatch):
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        lambda *_args, **_kwargs: pytest.fail("offline test opened HTTP"),
    )
    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("offline test opened a socket"),
    )
    cache = FullSearchV3SearchCache(tmp_path / "cache")
    request = _cache_request()
    first = _cache_identity(request, "cache-first")
    second = _cache_identity(request, "cache-second")
    cache.put(first, SolverResult("Answer: yes"))
    first_client = _FakeSolver("Answer: no")
    executor = _executor(cache, first_client)

    assert executor.complete(first, request) == SolverResult("Answer: yes")
    assert first_client.calls == 0
    assert executor.complete(second, request) == SolverResult("Answer: no")
    assert first_client.calls == 1

    resumed_client = _FakeSolver("Answer: yes")
    resumed_executor = _executor(cache, resumed_client)
    assert resumed_executor.complete(second, request) == SolverResult("Answer: no")
    assert resumed_client.calls == 0


def test_evaluator_rejects_a_mismatched_m2_cache_before_dispatch(
    production_search_input, tmp_path
):
    evaluator_cache = FullSearchV3SearchCache(tmp_path / "evaluator-cache")
    executor_cache = FullSearchV3SearchCache(tmp_path / "executor-cache")
    client = _FakeSolver("Answer: yes")

    with pytest.raises(
        evaluator.FullSearchV3EvaluatorInputError, match="same SEARCH cache"
    ):
        evaluator.evaluate_full_search_v3_p0(
            search_input=production_search_input,
            solver_identity_sha256="4" * 64,
            cache=evaluator_cache,
            executor=_executor(executor_cache, client),
            checkpoint_path=tmp_path / "checkpoint.json",
            result_path=tmp_path / "result.json",
        )

    assert client.calls == 0
    assert not (tmp_path / "checkpoint.json").exists()
    assert not (tmp_path / "result.json").exists()


def test_search_interface_has_no_private_final_input(production_search_input):
    source = Path(evaluator.__file__).read_text(encoding="utf-8")
    assert "load_trusted_full_search_v3_private_manifest" not in source
    assert "load_full_search_v3_private" not in source
    assert "FINAL" not in production_search_input.manifest
    assert all(
        "final" not in parameter.lower() and "private" not in parameter.lower()
        for function in (
            evaluator.load_full_search_v3_search_input,
            evaluator.validate_full_search_v3_search_input,
            evaluator.evaluate_full_search_v3_candidate,
            evaluator.evaluate_full_search_v3_p0,
        )
        for parameter in inspect.signature(function).parameters
    )


class _FakeSolver:
    def __init__(self, response):
        self._response = response
        self.calls = 0

    def complete(self, _request):
        self.calls += 1
        return SolverResult(self._response)


def _executor(cache, client):
    return FullSearchV3RequestExecutor(
        cache=cache,
        client_factory=lambda: client,
        retry_policy=SolverRetryPolicy(maximum_attempts=1),
        maximum_in_flight_requests=1,
        sleeper=lambda _delay: pytest.fail("offline test retried a fake request"),
        clock=lambda: 0.0,
    )


def _cache_request():
    config = canonical_full_search_v3_config()
    return SolverRequest(
        model=config.solver_model,
        messages=({"role": "user", "content": "offline request"},),
        generation=SolverGenerationSettings.from_config(config),
    )


def _cache_identity(request, sample_id):
    return build_solver_request_identity(
        request,
        sample_id=sample_id,
        candidate_id="candidate_1",
        prompt_sha256="2" * 64,
        split_sha256=FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256,
        search_membership_sha256="a" * 64,
        solver_identity_sha256="3" * 64,
    )


def _four_record_input(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), color="red").save(first, format="PNG")
    Image.new("RGB", (2, 2), color="green").save(second, format="PNG")
    records = (
        {
            "sample_id": "direct",
            "claim_type": "direct",
            "claim": "Direct claim.",
            "context": "Direct context.",
            "caption": "Direct caption.",
            "image_path": str(first),
            "gold_label": "gold_direct_sentinel",
        },
        {
            "sample_id": "analytical",
            "claim_type": "analytical",
            "claim": "Analytical claim.",
            "context": "Analytical context.",
            "caption": "Analytical caption.",
            "image_path": str(first),
            "gold_label": "gold_analytical_sentinel",
        },
        {
            "sample_id": "parallel",
            "claim_type": "parallel",
            "claim": "Parallel claim.",
            "context": "Parallel context.",
            "caption1": "First caption.",
            "caption2": "Second caption.",
            "item1_path": str(first),
            "item2_path": str(second),
            "gold_label": "gold_parallel_sentinel",
        },
        {
            "sample_id": "sequential",
            "claim_type": "sequential",
            "claim": "Sequential claim.",
            "context": "Sequential context.",
            "caption1": "First caption.",
            "caption2": "Second caption.",
            "item1_path": str(first),
            "item2_path": str(second),
            "gold_label": "gold_sequential_sentinel",
        },
    )
    return (
        FullSearchV3SearchInput(
            _manifest=MappingProxyType(
                {
                    "split_sha256": FULL_SEARCH_V3_REQUIRED_SPLIT_SHA256,
                    "search_membership_sha256": "a" * 64,
                }
            ),
            _records=tuple(MappingProxyType(record) for record in records),
        ),
        ((first,), (first,), (first, second), (first, second)),
    )


def _image_bytes(messages):
    return [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in messages[0]["content"]
        if block["type"] == "image_url"
    ]


def _mutable(value):
    if hasattr(value, "items"):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return copy.deepcopy(value)
