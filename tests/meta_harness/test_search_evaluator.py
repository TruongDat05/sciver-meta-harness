from __future__ import annotations

import base64
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
from string import Template
from types import MappingProxyType

from PIL import Image
import pytest

import meta_harness
import meta_harness.search_evaluator as search_evaluator
from evaluation.metrics import evaluate_dataset_records
from meta_harness.search_evaluator import (
    EXPERIMENT_CHECKPOINT_SCHEMA_VERSION,
    EXPERIMENT_P0_CANDIDATE_ID,
    EXPERIMENT_REQUIRED_SPLIT_SHA256,
    EXPERIMENT_STAGE,
    EvaluationIncomplete,
    EvaluationResumeError,
    EvaluatorInputError,
    PredictionOutcome,
    SearchInput,
    account_experiment_predictions,
    canonical_experiment_p0_prompt_sha256,
    evaluate_experiment_candidate,
    evaluate_experiment_p0,
    load_experiment_search_input,
    validate_experiment_search_input,
)
from meta_harness.search_cache import SearchCache
from meta_harness.request_executor import RequestExecutor
from meta_harness.config import (
    EXPERIMENT_PROTOCOL_ID,
    canonical_experiment_config,
)
from meta_harness.preparation import prepare_experiment
from meta_harness.preparation import (
    EXPERIMENT_PREPARATION_IDENTITY_SCHEMA,
    EXPERIMENT_SEARCH_DATASET_SCHEMA,
    EXPERIMENT_SEARCH_SAFE_MANIFEST_SCHEMA,
)
from meta_harness.records import (
    EXPERIMENT_ALLOCATION_ALGORITHM,
    EXPERIMENT_PAPER_IDENTITY_VERSION,
    EXPERIMENT_SAMPLE_IDENTITY_VERSION,
    EXPERIMENT_SPLIT_SCHEMA_VERSION,
)
from meta_harness.retry import (
    FailureCategory,
    SolverExecutionFailure,
    SolverFailureMetadata,
    SolverRetryPolicy,
)
from meta_harness.solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
    build_solver_request_identity,
)
from meta_harness.prompt_family import PromptFamily, REQUIRED_PLACEHOLDERS
from meta_harness.prompt_family import template_source_sha256
from model_inference.remote_api import prepare_remote_requests
from model_inference.remote_client import RequestTimeoutError
from utils.answer_parser import parse_answer
from utils.constant import COT_PROMPT
from utils.result_writer import SUCCESS


TESTSET_PATH = Path(__file__).resolve().parents[2] / "data" / "sciver" / "testset.json"


class RecordingSolver:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self) -> None:
        self.calls += 1


@pytest.fixture(scope="module")
def prepared_search_input(tmp_path_factory):
    root = tmp_path_factory.mktemp("meta-harness-evaluator")
    artifacts = prepare_experiment(
        source_path=TESTSET_PATH,
        private_directory=root / "private",
        search_directory=root / "search",
    )
    return artifacts.search_safe_manifest_path, artifacts.search_dataset_path


@pytest.fixture
def loaded_search_input(prepared_search_input):
    return load_experiment_search_input(
        search_safe_manifest_path=prepared_search_input[0],
        search_records_path=prepared_search_input[1],
    )


def test_loads_only_complete_required_manifest_ordered_search_input(
    loaded_search_input,
):
    assert loaded_search_input.stage == EXPERIMENT_STAGE
    assert loaded_search_input.manifest["protocol_id"] == "sciver_full_search_v3"
    assert (
        loaded_search_input.manifest["split_sha256"]
        == EXPERIMENT_REQUIRED_SPLIT_SHA256
    )
    assert len(loaded_search_input.records) == 1000
    assert loaded_search_input.sample_ids == tuple(
        loaded_search_input.manifest["SEARCH"]["sample_ids"]
    )

    with pytest.raises(TypeError):
        loaded_search_input.records[0]["sample_id"] = "mutated"
    with pytest.raises(TypeError):
        loaded_search_input.manifest["SEARCH"]["sample_ids"] = ()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest, _records: manifest.__setitem__(
                "protocol_id", "different_protocol"
            ),
            "locked protocol",
        ),
        (
            lambda manifest, _records: manifest.__setitem__("split_sha256", "0" * 64),
            "required split SHA-256",
        ),
        (
            lambda manifest, _records: manifest.__setitem__("stage", "FINAL"),
            "only the SEARCH stage",
        ),
        (lambda _manifest, records: records.pop(), "exactly 1,000"),
        (
            lambda _manifest, records: records.__setitem__(-1, copy.deepcopy(records[0])),
            "duplicate sample IDs",
        ),
        (
            lambda manifest, _records: manifest["SEARCH"]["sample_ids"].__setitem__(
                -1, manifest["SEARCH"]["sample_ids"][0]
            ),
            "SEARCH manifest contains duplicate sample IDs",
        ),
        (lambda _manifest, records: records.reverse(), "immutable manifest order"),
        (
            lambda _manifest, records: records.__setitem__(
                0, {**records[0], "sample_id": "foreign-sample"}
            ),
            "complete manifest ID coverage",
        ),
    ],
)
def test_invalid_input_is_rejected_before_any_solver_dispatch(
    loaded_search_input,
    mutate,
    message,
):
    manifest = _mutable(loaded_search_input.manifest)
    records = _mutable(loaded_search_input.records)
    solver = RecordingSolver()

    mutate(manifest, records)

    with pytest.raises(EvaluatorInputError, match=message):
        validated = validate_experiment_search_input(
            manifest=manifest,
            records=records,
        )
        for _record in validated.records:
            solver.complete()

    assert solver.calls == 0


def test_loader_rejects_mismatched_artifact_before_any_solver_dispatch(
    prepared_search_input,
    tmp_path,
):
    manifest_path, records_path = prepared_search_input
    tampered = tmp_path / "tampered-search-safe-manifest.json"
    text = manifest_path.read_text(encoding="utf-8").replace(
        EXPERIMENT_REQUIRED_SPLIT_SHA256,
        "0" * 64,
    )
    tampered.write_text(text, encoding="utf-8")
    solver = RecordingSolver()

    with pytest.raises(EvaluatorInputError):
        loaded = load_experiment_search_input(
            search_safe_manifest_path=tampered,
            search_records_path=records_path,
        )
        for _record in loaded.records:
            solver.complete()

    assert solver.calls == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest, _records: manifest.__setitem__(
                "protocol_id", "different_protocol"
            ),
            "locked protocol",
        ),
        (
            lambda manifest, _records: manifest.__setitem__("split_sha256", "0" * 64),
            "required split SHA-256",
        ),
        (
            lambda manifest, _records: manifest.__setitem__("stage", "FINAL"),
            "only the SEARCH stage",
        ),
        (lambda _manifest, records: records.pop(), "exactly 1,000"),
        (
            lambda _manifest, records: records.__setitem__(-1, copy.deepcopy(records[0])),
            "duplicate sample IDs",
        ),
        (
            lambda manifest, _records: manifest["SEARCH"]["sample_ids"].__setitem__(
                -1, manifest["SEARCH"]["sample_ids"][0]
            ),
            "SEARCH manifest contains duplicate sample IDs",
        ),
        (lambda _manifest, records: records.reverse(), "immutable manifest order"),
        (
            lambda _manifest, records: records.__setitem__(
                0, {**records[0], "sample_id": "foreign-sample"}
            ),
            "complete manifest ID coverage",
        ),
    ],
)
def test_unvalidated_public_input_never_persists_or_dispatches(
    loaded_search_input,
    tmp_path,
    mutate,
    message,
):
    manifest = _mutable(loaded_search_input.manifest)
    records = _mutable(loaded_search_input.records)
    mutate(manifest, records)
    unvalidated = SearchInput(
        _manifest=manifest,
        _records=tuple(records),
    )
    metrics_path = tmp_path / "metrics.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    result_path = tmp_path / "result.json"
    cache_path = tmp_path / "cache"
    client = _CountingClient()
    cache = SearchCache(cache_path)

    with pytest.raises(EvaluatorInputError, match=message):
        account_experiment_predictions(
            search_input=unvalidated,
            candidate_id="candidate_safe",
            outcomes=(),
            metrics_path=metrics_path,
        )
    assert not metrics_path.exists()

    with pytest.raises(EvaluatorInputError, match=message):
        evaluate_experiment_candidate(
            search_input=unvalidated,
            candidate_id="candidate_safe",
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=checkpoint_path,
            result_path=result_path,
        )
    assert client.calls == 0
    assert not checkpoint_path.exists()
    assert not result_path.exists()
    assert not cache_path.exists()


def test_raw_metrics_writer_is_private_and_forged_mapping_has_no_public_path(tmp_path):
    """Only validated public boundaries may reach the aggregate result writer."""

    writer_name = "persist_experiment_prediction_metrics"
    forged_path = tmp_path / "forged-metrics.json"
    forged_report = {"metrics": {"rankable": True}}

    assert writer_name not in search_evaluator.__all__
    assert not hasattr(search_evaluator, writer_name)
    assert not hasattr(meta_harness, writer_name)
    with pytest.raises(AttributeError):
        getattr(search_evaluator, writer_name)(forged_path, forged_report)
    assert not forged_path.exists()

    public_result_paths = {
        name: set(inspect.signature(getattr(search_evaluator, name)).parameters)
        for name in search_evaluator.__all__
        if inspect.isfunction(getattr(search_evaluator, name))
        and {"metrics_path", "result_path"}
        & set(inspect.signature(getattr(search_evaluator, name)).parameters)
    }
    assert public_result_paths == {
        "account_experiment_predictions": {
            "search_input",
            "candidate_id",
            "outcomes",
            "metrics_path",
        },
        "evaluate_experiment_candidate": {
            "search_input",
            "candidate_id",
            "prompt",
            "solver_identity_sha256",
            "cache",
            "executor",
            "checkpoint_path",
            "result_path",
            "resume",
            "progress_hook",
        },
        "evaluate_experiment_p0": {
            "search_input",
            "solver_identity_sha256",
            "cache",
            "executor",
            "checkpoint_path",
            "result_path",
            "resume",
            "progress_hook",
        },
    }
    assert all("search_input" in parameters for parameters in public_result_paths.values())


def test_incomplete_outcomes_cannot_persist_rankable_metrics(
    loaded_search_input,
    tmp_path,
):
    metrics_path = tmp_path / "metrics.json"
    incomplete_outcomes = tuple(
        PredictionOutcome(sample_id, raw_response="Answer: yes")
        for sample_id in loaded_search_input.sample_ids[:-1]
    )

    with pytest.raises(EvaluatorInputError, match="cover every"):
        account_experiment_predictions(
            search_input=loaded_search_input,
            candidate_id="candidate_safe",
            outcomes=incomplete_outcomes,
            metrics_path=metrics_path,
        )

    assert not metrics_path.exists()


@pytest.mark.parametrize(
    "candidate_id",
    [
        "api_key_placeholder",
        "authorization-placeholder",
        "bearer-placeholder",
        "endpoint-placeholder",
        "https://placeholder.invalid",
        "QUJD" * 16,
        "candidate\nnext",
        "../candidate",
        "x" * 129,
    ],
)
def test_unsafe_candidate_id_never_persists_or_dispatches(
    loaded_search_input,
    tmp_path,
    candidate_id,
):
    metrics_path = tmp_path / "metrics.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    result_path = tmp_path / "result.json"
    cache_path = tmp_path / "cache"
    client = _CountingClient()
    cache = SearchCache(cache_path)

    with pytest.raises(EvaluatorInputError, match="safe opaque"):
        account_experiment_predictions(
            search_input=loaded_search_input,
            candidate_id=candidate_id,
            outcomes=(),
            metrics_path=metrics_path,
        )
    with pytest.raises(EvaluatorInputError, match="safe opaque"):
        evaluate_experiment_candidate(
            search_input=loaded_search_input,
            candidate_id=candidate_id,
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=checkpoint_path,
            result_path=result_path,
        )
    assert client.calls == 0
    assert not metrics_path.exists()
    assert not checkpoint_path.exists()
    assert not result_path.exists()
    assert not cache_path.exists()


def test_pure_prediction_metrics_perfect_and_all_wrong_are_hand_computable():
    search_input = _synthetic_search_input(["yes", "no", "yes", "no"])

    perfect = search_evaluator._compute_prediction_metrics(
        search_input,
        _outcomes(search_input, ["yes", "no", "yes", "no"]),
    )
    wrong = search_evaluator._compute_prediction_metrics(
        search_input,
        _outcomes(search_input, ["no", "yes", "no", "yes"]),
    )

    assert {
        key: perfect[key]
        for key in ("macro_f1", "accuracy", "parse_coverage")
    } == {
        "macro_f1": 1.0,
        "accuracy": 1.0,
        "parse_coverage": 1.0,
    }
    assert perfect["total_records"] == perfect["completed_solver_responses"] == 4
    assert perfect["parsed_predictions"] == 4
    assert perfect["abstentions_or_parse_failures"] == 0
    assert perfect["infrastructure_failures"] == 0
    assert wrong["macro_f1"] == 0.0
    assert wrong["accuracy"] == 0.0
    assert "rankable" not in perfect
    assert "protocol_id" not in perfect


def test_pure_prediction_metrics_class_imbalance_and_one_class_predictions():
    imbalanced = _synthetic_search_input(["yes", "no", "no", "no"])
    imbalanced_report = search_evaluator._compute_prediction_metrics(
        imbalanced,
        _outcomes(imbalanced, ["yes", "yes", "no", "no"]),
    )
    one_class = _synthetic_search_input(["yes", "no", "no"])
    one_class_report = search_evaluator._compute_prediction_metrics(
        one_class,
        _outcomes(one_class, ["yes", "yes", "yes"]),
    )

    assert imbalanced_report["accuracy"] == pytest.approx(0.75)
    assert imbalanced_report["macro_f1"] == pytest.approx(11 / 15)
    assert one_class_report["accuracy"] == pytest.approx(1 / 3)
    assert one_class_report["macro_f1"] == pytest.approx(0.25)


def test_abstention_and_parse_failure_remain_in_full_metric_denominator():
    search_input = _synthetic_search_input(["yes", "no"])
    abstention = search_evaluator._compute_prediction_metrics(
        search_input,
        _outcomes(search_input, ["yes", "I cannot determine."]),
    )
    parse_failure = search_evaluator._compute_prediction_metrics(
        search_input,
        _outcomes(
            search_input,
            ["Answer: yes or no", "Answer: no"],
        ),
    )

    for report in (abstention, parse_failure):
        assert report["total_records"] == 2
        assert report["completed_solver_responses"] == 2
        assert report["parsed_predictions"] == 1
        assert report["abstentions_or_parse_failures"] == 1
        assert report["infrastructure_failures"] == 0
        assert report["accuracy"] == pytest.approx(0.5)
        assert report["macro_f1"] == pytest.approx(0.5)
        assert "rankable" not in report


def test_pure_metrics_infrastructure_failure_is_not_a_prediction(tmp_path):
    search_input = _synthetic_search_input(["yes", "no"])
    metadata = SolverFailureMetadata(
        category=FailureCategory.TIMEOUT,
        retryable=True,
        exhausted=True,
        attempt_count=4,
        maximum_attempts=4,
        http_status_code=None,
        retry_delays_seconds=(1.0, 2.0, 4.0),
        elapsed_seconds=7.0,
        cause_type="RequestTimeoutError",
    )
    outcome = PredictionOutcome.from_infrastructure_failure(
        search_input.sample_ids[1], SolverExecutionFailure(metadata)
    )
    report = search_evaluator._compute_prediction_metrics(
        search_input,
        (
            PredictionOutcome(
                sample_id=search_input.sample_ids[0], raw_response="Answer: yes"
            ),
            outcome,
        ),
    )

    assert report["completed_solver_responses"] == 1
    assert report["parsed_predictions"] == 1
    assert report["abstentions_or_parse_failures"] == 0
    assert report["infrastructure_failures"] == 1
    assert report["infrastructure_failure_categories"] == {"timeout": 1}
    assert report["accuracy"] == pytest.approx(0.5)
    assert report["macro_f1"] == pytest.approx(0.5)
    assert "rankable" not in report
    assert not (tmp_path / "metrics.json").exists()


def test_full_search_checkpoint_resume_is_atomic_deterministic_and_noops_completed(
    loaded_search_input,
    tmp_path,
    monkeypatch,
):
    """A simulated interruption resumes the remaining immutable IDs only."""

    monkeypatch.setattr(
        "meta_harness.search_evaluator._candidate_requests",
        _safe_candidate_requests,
    )
    cache = SearchCache(tmp_path / "cache")
    checkpoint = tmp_path / "checkpoint.json"
    result = tmp_path / "result.json"
    first_client = _InterruptingClient(after_successes=2)

    with pytest.raises(KeyboardInterrupt):
        evaluate_experiment_p0(
            search_input=loaded_search_input,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, first_client),
            checkpoint_path=checkpoint,
            result_path=result,
        )

    interrupted = checkpoint.read_bytes()
    interrupted_payload = json.loads(interrupted)
    assert interrupted_payload["schema_version"] == EXPERIMENT_CHECKPOINT_SCHEMA_VERSION
    assert interrupted_payload["status"] == "running"
    assert len(interrupted_payload["completed_samples"]) == 2
    assert interrupted == _canonical_snapshot_bytes(interrupted_payload)
    assert not result.exists()

    second_client = _CountingClient()
    report = evaluate_experiment_p0(
        search_input=loaded_search_input,
        solver_identity_sha256="1" * 64,
        cache=cache,
        executor=_executor(cache, second_client),
        checkpoint_path=checkpoint,
        result_path=result,
        resume=True,
    )

    assert first_client.successes == 2
    assert second_client.calls == 998
    assert report["total_records"] == 1000
    assert report["completed_solver_responses"] == 1000
    assert report["infrastructure_failures"] == 0
    assert report["metrics"]["rankable"] is True
    assert report["prompt_sha256"] == canonical_experiment_p0_prompt_sha256()
    assert result.exists()
    final_checkpoint = checkpoint.read_bytes()
    assert final_checkpoint == _canonical_snapshot_bytes(json.loads(final_checkpoint))
    persisted = result.read_text(encoding="utf-8")
    checkpoint_text = final_checkpoint.decode("utf-8")
    for forbidden in ("API_KEY", "API_URL", "Authorization", "data:image", "Answer: yes"):
        assert forbidden not in persisted
        assert forbidden not in checkpoint_text

    no_call_client = _CountingClient()
    resumed_report = evaluate_experiment_p0(
        search_input=loaded_search_input,
        solver_identity_sha256="1" * 64,
        cache=cache,
        executor=_executor(cache, no_call_client),
        checkpoint_path=checkpoint,
        result_path=result,
        resume=True,
    )
    assert resumed_report == report
    assert no_call_client.calls == 0


def test_evaluator_dispatches_missing_search_requests_concurrently_in_manifest_order(
    loaded_search_input,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "meta_harness.search_evaluator._candidate_requests",
        _safe_candidate_requests,
    )
    two_started = threading.Event()
    release = threading.Event()
    state_lock = threading.Lock()
    active = 0
    maximum_active = 0

    class Client:
        def complete(self, _request):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
                if active == 2:
                    two_started.set()
            try:
                assert release.wait(5)
                return SolverResult("Therefore, the final answer is: Answer: yes")
            finally:
                with state_lock:
                    active -= 1

    cache = SearchCache(tmp_path / "cache")
    executor = RequestExecutor(
        cache=cache,
        client_factory=Client,
        retry_policy=SolverRetryPolicy(maximum_attempts=1),
        maximum_in_flight_requests=2,
        sleeper=lambda _delay: pytest.fail("offline test must not sleep"),
        clock=lambda: 0.0,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        evaluation = pool.submit(
            evaluate_experiment_candidate,
            search_input=loaded_search_input,
            candidate_id="p0",
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=executor,
            checkpoint_path=tmp_path / "checkpoint.json",
            result_path=tmp_path / "result.json",
        )
        try:
            assert two_started.wait(5)
            with state_lock:
                assert maximum_active == 2
        finally:
            release.set()
        report = evaluation.result(timeout=10)

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert report["completed_solver_responses"] == 1000
    assert [entry["sample_id"] for entry in checkpoint["completed_samples"]] == list(
        loaded_search_input.sample_ids
    )


def test_resume_rejects_incompatible_and_corrupt_checkpoint_before_solver(
    loaded_search_input,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "meta_harness.search_evaluator._candidate_requests",
        _safe_candidate_requests,
    )
    cache = SearchCache(tmp_path / "cache")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{not-json", encoding="utf-8")
    client = _CountingClient()

    with pytest.raises(EvaluationResumeError, match="corrupt"):
        evaluate_experiment_candidate(
            search_input=loaded_search_input,
            candidate_id="p0",
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=checkpoint,
            result_path=tmp_path / "result.json",
            resume=True,
        )
    assert client.calls == 0

    checkpoint.write_text(
        json.dumps(
            {
                "schema_version": EXPERIMENT_CHECKPOINT_SCHEMA_VERSION,
                "artifact_type": "full_search_candidate_checkpoint",
                "identity": {"candidate_id": "wrong"},
                "status": "running",
                "completed_samples": [],
                "last_infrastructure_failures": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationResumeError, match="incompatible"):
        evaluate_experiment_candidate(
            search_input=loaded_search_input,
            candidate_id="p0",
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=checkpoint,
            result_path=tmp_path / "result.json",
            resume=True,
        )
    assert client.calls == 0


def test_infrastructure_incomplete_checkpoint_never_persists_eligible_result(
    loaded_search_input,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "meta_harness.search_evaluator._candidate_requests",
        _safe_candidate_requests,
    )
    cache = SearchCache(tmp_path / "cache")
    checkpoint = tmp_path / "checkpoint.json"
    result = tmp_path / "result.json"
    client = _PermanentFailureClient()

    with pytest.raises(EvaluationIncomplete) as raised:
        evaluate_experiment_candidate(
            search_input=loaded_search_input,
            candidate_id="p0",
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=checkpoint,
            result_path=result,
        )

    assert raised.value.report["infrastructure_failures"] == 1
    assert raised.value.report["metrics"]["rankable"] is False
    assert not result.exists()
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"
    assert len(payload["completed_samples"]) == 999
    assert len(payload["last_infrastructure_failures"]) == 1


def test_public_evaluator_rejects_non_full_input_before_solver(
    loaded_search_input,
    tmp_path,
):
    cache = SearchCache(tmp_path / "cache")
    client = _CountingClient()
    shortened = SearchInput(
        _manifest=loaded_search_input.manifest,
        _records=loaded_search_input.records[:2],
    )

    with pytest.raises(EvaluatorInputError, match="exactly 1,000"):
        evaluate_experiment_candidate(
            search_input=shortened,
            candidate_id="p0",
            prompt=COT_PROMPT,
            solver_identity_sha256="1" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=tmp_path / "checkpoint.json",
            result_path=tmp_path / "result.json",
        )
    assert client.calls == 0


def test_p0_request_path_is_exactly_canonical_and_preserves_multimodal_order(
    tmp_path,
):
    search_input, expected_images = _p0_equivalence_input(tmp_path)
    config = canonical_experiment_config()
    p0_hash = canonical_experiment_p0_prompt_sha256()

    assert p0_hash == template_source_sha256(COT_PROMPT)
    prompt = PromptFamily(COT_PROMPT)
    assert {
        method: prompt[method].template for method in prompt
    } == {
        method: COT_PROMPT[method].template for method in COT_PROMPT
    }
    for method, placeholders in REQUIRED_PLACEHOLDERS.items():
        for placeholder in placeholders:
            assert f"${placeholder}" in prompt[method].template

    full_search_requests = search_evaluator._candidate_requests(
        search_input=search_input,
        candidate_id=EXPERIMENT_P0_CANDIDATE_ID,
        prompt=prompt,
        prompt_sha256=p0_hash,
        solver_identity_sha256="2" * 64,
    )
    assert len(full_search_requests) == 4
    for item, record, paths in zip(
        full_search_requests,
        search_input.records,
        expected_images,
    ):
        canonical_messages = prepare_remote_requests([record], COT_PROMPT)[0]
        request = item["request"]
        identity = item["identity"]

        assert list(request.messages) == canonical_messages
        assert request.model == config.solver_model
        assert request.generation.as_request_options() == {
            "temperature": config.solver_temperature,
            "top_p": config.solver_top_p,
            "seed": config.solver_seed,
            "n": config.solver_n,
            "stream": config.solver_stream,
            "max_tokens": config.solver_max_tokens,
        }
        assert identity.candidate_id == EXPERIMENT_P0_CANDIDATE_ID
        assert identity.prompt_sha256 == p0_hash
        assert _message_image_bytes(request.messages) == [path.read_bytes() for path in paths]
        assert record["gold_label"] not in json.dumps(request.messages)

        prompt_values = {
            "claim": record["claim"],
            "context": record["context"],
        }
        if record["claim_type"] in {"direct", "analytical"}:
            prompt_values["caption"] = record["caption"]
        else:
            prompt_values["caption1"] = record["caption1"]
            prompt_values["caption2"] = record["caption2"]
        assert request.messages[0]["content"][-1]["text"] == COT_PROMPT[
            record["claim_type"]
        ].substitute(**prompt_values)


def test_p0_parser_labels_and_metrics_match_existing_canonical_boundary(tmp_path):
    search_input, _expected_images = _p0_equivalence_input(tmp_path)
    raw_responses = (
        "Reasoning. Therefore, the final answer is: Answer: yes",
        "Reasoning. Therefore, the final answer is: Answer: no",
        "Reasoning. Therefore, the final answer is: Answer: no",
        "Reasoning. Therefore, the final answer is: Answer: yes",
    )
    outcomes = tuple(
        PredictionOutcome(sample_id, raw_response=response)
        for sample_id, response in zip(search_input.sample_ids, raw_responses)
    )
    full_search_records = search_evaluator._metric_records(
        search_input,
        EXPERIMENT_P0_CANDIDATE_ID,
        outcomes,
    )
    canonical_records = []
    for record, raw_response in zip(search_input.records, raw_responses):
        parsed = parse_answer(raw_response)
        canonical_records.append(
            {
                "run_id": "sciver_full_search_v3",
                "dataset": "SciVer",
                "model": canonical_experiment_config().solver_model,
                "prompt_variant": EXPERIMENT_P0_CANDIDATE_ID,
                "method": record["claim_type"],
                "sample_id": record["sample_id"],
                "gold_label": record["gold_label"],
                "attempt_count": 1,
                "prediction": parsed["prediction"],
                "parse_status": parsed["parse_status"],
                "request_status": SUCCESS,
            }
        )

    assert full_search_records == canonical_records
    assert [parse_answer(response)["prediction"] for response in raw_responses] == [
        record["prediction"] for record in full_search_records
    ]
    canonical_summary = evaluate_dataset_records(canonical_records)[0]
    report = search_evaluator._compute_prediction_metrics(
        search_input,
        outcomes,
    )
    assert canonical_summary["per_class"]["yes"]["support"] == 2
    assert canonical_summary["per_class"]["no"]["support"] == 2
    assert report["macro_f1"] == canonical_summary["macro_f1"]
    assert report["accuracy"] == canonical_summary["accuracy"]["value"]


def test_p0_entrypoint_uses_frozen_mapping_and_validates_its_hash(monkeypatch):
    captured = {}
    expected_hash = canonical_experiment_p0_prompt_sha256()

    def canonical_result(**kwargs):
        captured.update(kwargs)
        return {"prompt_sha256": expected_hash}

    monkeypatch.setattr(
        search_evaluator,
        "evaluate_experiment_candidate",
        canonical_result,
    )
    result = evaluate_experiment_p0(
        search_input=object(),
        solver_identity_sha256="3" * 64,
        cache=object(),
        executor=object(),
        checkpoint_path="offline-checkpoint.json",
        result_path="offline-result.json",
    )

    assert result["prompt_sha256"] == expected_hash
    assert captured["candidate_id"] == EXPERIMENT_P0_CANDIDATE_ID
    assert captured["prompt"] is COT_PROMPT
    assert captured["resume"] is False

    monkeypatch.setattr(
        search_evaluator,
        "evaluate_experiment_candidate",
        lambda **_kwargs: {"prompt_sha256": "0" * 64},
    )
    with pytest.raises(EvaluatorInputError, match="frozen COT prompt hash"):
        evaluate_experiment_p0(
            search_input=object(),
            solver_identity_sha256="3" * 64,
            cache=object(),
            executor=object(),
            checkpoint_path="offline-checkpoint.json",
            result_path="offline-result.json",
        )


def test_p0_dispatches_original_cot_mapping_to_request_builder(
    loaded_search_input,
    tmp_path,
    monkeypatch,
):
    captured = {}

    class StopBeforeSolver(RuntimeError):
        pass

    def capture_requests(**kwargs):
        captured.update(kwargs)
        raise StopBeforeSolver()

    monkeypatch.setattr(
        search_evaluator,
        "_candidate_requests",
        capture_requests,
    )
    cache = SearchCache(tmp_path / "cache")
    client = _CountingClient()
    with pytest.raises(StopBeforeSolver):
        evaluate_experiment_p0(
            search_input=loaded_search_input,
            solver_identity_sha256="3" * 64,
            cache=cache,
            executor=_executor(cache, client),
            checkpoint_path=tmp_path / "checkpoint.json",
            result_path=tmp_path / "result.json",
        )

    assert captured["prompt"] is COT_PROMPT
    assert captured["prompt_sha256"] == canonical_experiment_p0_prompt_sha256()
    assert client.calls == 0


def test_experiment_offline_1000_record_integration(
    tmp_path,
    monkeypatch,
):
    """Exercise P0/candidates, retry, durable resume, and safe artifacts."""

    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("offline integration opened a socket"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("offline integration started a subprocess"),
    )
    search_input, image_paths = _synthetic_1000_search_artifact(tmp_path)
    expected_ids = search_input.sample_ids
    cache = SearchCache(tmp_path / "cache")
    retry_delays = []

    interrupted_client = _IntegrationClient(
        interrupt_at=12,
        transient_timeout_at=0,
        image_paths=image_paths,
    )
    with pytest.raises(KeyboardInterrupt):
        evaluate_experiment_p0(
            search_input=search_input,
            solver_identity_sha256="4" * 64,
            cache=cache,
            executor=_integration_executor(cache, interrupted_client, retry_delays),
            checkpoint_path=tmp_path / "p0-checkpoint.json",
            result_path=tmp_path / "p0-result.json",
        )
    partial_checkpoint = tmp_path / "p0-checkpoint.json"
    assert len(json.loads(partial_checkpoint.read_text(encoding="utf-8"))["completed_samples"]) == 12

    resumed_input = load_experiment_search_input(
        search_safe_manifest_path=tmp_path / "search_safe_manifest.json",
        search_records_path=tmp_path / "search_records.json",
    )
    resumed_client = _IntegrationClient(image_paths=image_paths)
    p0_report = evaluate_experiment_p0(
        search_input=resumed_input,
        solver_identity_sha256="4" * 64,
        cache=cache,
        executor=_integration_executor(cache, resumed_client, retry_delays),
        checkpoint_path=partial_checkpoint,
        result_path=tmp_path / "p0-result.json",
        resume=True,
    )
    assert retry_delays == [0.0]
    assert interrupted_client.calls == 14  # timeout retry + 12 completions + interrupt
    assert resumed_client.calls == 988
    assert p0_report["total_records"] == 1000
    assert p0_report["metrics"]["macro_f1"] == 1.0
    assert p0_report["metrics"]["accuracy"] == 1.0
    p0_indices = _unique_in_order(interrupted_client.indices + resumed_client.indices)
    assert p0_indices == list(range(1000))
    assert tuple(f"synthetic-{index:04d}" for index in p0_indices) == expected_ids

    candidate_a = _candidate_prompt("CANDIDATE_A")
    candidate_a_client = _IntegrationClient(
        parse_failure_indices={41},
        abstention_indices={42},
        image_paths=image_paths,
    )
    candidate_a_report = evaluate_experiment_candidate(
        search_input=resumed_input,
        candidate_id="candidate_a",
        prompt=candidate_a,
        solver_identity_sha256="4" * 64,
        cache=cache,
        executor=_integration_executor(cache, candidate_a_client, []),
        checkpoint_path=tmp_path / "candidate-a-checkpoint.json",
        result_path=tmp_path / "candidate-a-result.json",
    )
    assert candidate_a_client.calls == 1000
    assert candidate_a_report["total_records"] == 1000
    assert candidate_a_report["completed_solver_responses"] == 1000
    assert candidate_a_report["parsed_predictions"] == 998
    assert candidate_a_report["abstentions_or_parse_failures"] == 2
    assert candidate_a_report["metrics"]["accuracy"] == pytest.approx(0.998)
    assert candidate_a_report["metrics"]["macro_f1"] == pytest.approx(998 / 999)
    assert candidate_a_report["metrics"]["rankable"] is False
    assert not (tmp_path / "candidate-a-result.json").exists()

    candidate_b = _candidate_prompt("CANDIDATE_B")
    candidate_b_client = _IntegrationClient(
        permanent_failure_at=73,
        image_paths=image_paths,
    )
    with pytest.raises(EvaluationIncomplete) as incomplete:
        evaluate_experiment_candidate(
            search_input=resumed_input,
            candidate_id="candidate_b",
            prompt=candidate_b,
            solver_identity_sha256="4" * 64,
            cache=cache,
            executor=_integration_executor(cache, candidate_b_client, []),
            checkpoint_path=tmp_path / "candidate-b-checkpoint.json",
            result_path=tmp_path / "candidate-b-result.json",
        )
    assert candidate_b_client.calls == 1000
    assert incomplete.value.report["infrastructure_failures"] == 1
    assert incomplete.value.report["metrics"]["rankable"] is False
    assert not (tmp_path / "candidate-b-result.json").exists()

    for client in (interrupted_client, resumed_client, candidate_a_client, candidate_b_client):
        assert _unique_in_order(client.indices) == sorted(set(client.indices))
        _assert_recorded_image_order(client.image_orders, image_paths)
    candidate_a_indices = _unique_in_order(candidate_a_client.indices)
    candidate_b_indices = _unique_in_order(candidate_b_client.indices)
    assert candidate_a_indices == list(range(1000))
    assert candidate_b_indices == list(range(1000))
    assert tuple(f"synthetic-{index:04d}" for index in candidate_a_indices) == expected_ids
    assert tuple(f"synthetic-{index:04d}" for index in candidate_b_indices) == expected_ids


    p0_result = (tmp_path / "p0-result.json").read_bytes()
    p0_checkpoint = partial_checkpoint.read_bytes()
    assert p0_result == _canonical_snapshot_bytes(json.loads(p0_result))
    assert p0_checkpoint == _canonical_snapshot_bytes(json.loads(p0_checkpoint))
    for snapshot in (p0_result, p0_checkpoint):
        text = snapshot.decode("utf-8")
        for forbidden in ("Answer: yes", "data:image", "Authorization", "API_KEY"):
            assert forbidden not in text

def test_candidate_reports_live_progress_hook_for_every_completed_case(
    tmp_path, monkeypatch
):
    """The evaluator drives a progress hook once per durably decided request."""

    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("offline progress test opened a socket"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("offline progress test started a subprocess"),
    )
    search_input, image_paths = _synthetic_1000_search_artifact(tmp_path)
    cache = SearchCache(tmp_path / "cache")
    client = _IntegrationClient(
        parse_failure_indices={41},
        abstention_indices={42},
        image_paths=image_paths,
    )
    calls = []
    report = evaluate_experiment_candidate(
        search_input=search_input,
        candidate_id="candidate_progress",
        prompt=_candidate_prompt("PROGRESS"),
        solver_identity_sha256="4" * 64,
        cache=cache,
        executor=_integration_executor(cache, client, []),
        checkpoint_path=tmp_path / "chk.json",
        result_path=tmp_path / "res.json",
        progress_hook=lambda completed, total, passed, failed: calls.append(
            (completed, total, passed, failed)
        ),
    )

    assert report["total_records"] == 1000
    assert len(calls) == 1000
    assert calls[-1][0] == 1000 and calls[-1][1] == 1000
    assert calls[-1][2] + calls[-1][3] == 1000
    assert calls[-1][2] == 998 and calls[-1][3] == 2
    for previous, current in zip(calls, calls[1:]):
        assert current[0] == previous[0] + 1


def test_resumed_candidate_progress_hook_starts_from_durable_completed_count(
    tmp_path, monkeypatch
):
    """A resumed evaluation reports live progress that includes completed cases."""

    monkeypatch.setattr(
        socket.socket,
        "connect",
        lambda *_args, **_kwargs: pytest.fail("offline resumed progress opened a socket"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("offline resumed progress started a subprocess"),
    )
    search_input, image_paths = _synthetic_1000_search_artifact(tmp_path)
    cache = SearchCache(tmp_path / "cache")
    interrupted = _IntegrationClient(
        interrupt_at=12, image_paths=image_paths
    )
    with pytest.raises(KeyboardInterrupt):
        evaluate_experiment_candidate(
            search_input=search_input,
            candidate_id="candidate_resume",
            prompt=_candidate_prompt("RESUME"),
            solver_identity_sha256="4" * 64,
            cache=cache,
            executor=_integration_executor(cache, interrupted, []),
            checkpoint_path=tmp_path / "chk.json",
            result_path=tmp_path / "res.json",
        )
    checkpoint = json.loads((tmp_path / "chk.json").read_text(encoding="utf-8"))
    resumed_count = len(checkpoint["completed_samples"])
    assert resumed_count == 12

    resumed = _IntegrationClient(image_paths=image_paths)
    calls = []
    evaluate_experiment_candidate(
        search_input=search_input,
        candidate_id="candidate_resume",
        prompt=_candidate_prompt("RESUME"),
        solver_identity_sha256="4" * 64,
        cache=cache,
        executor=_integration_executor(cache, resumed, []),
        checkpoint_path=tmp_path / "chk.json",
        result_path=tmp_path / "res.json",
        resume=True,
        progress_hook=lambda completed, total, passed, failed: calls.append(
            (completed, total, passed, failed)
        ),
    )
    assert calls and calls[0][0] == resumed_count + 1
    assert calls[-1][0] == 1000

def test_search_evaluator_has_no_legacy_stage_or_final_dependencies():
    """The dedicated evaluator must not import or call legacy execution paths."""

    source = Path(search_evaluator.__file__).read_text(encoding="utf-8")
    forbidden_dependencies = (
        "meta_harness.hard_search",
        "meta_harness.staged_orchestrator",
        "meta_harness.orchestrator",
        "meta_harness.finalize",
        "protected_validation",
        "promotion",
    )
    assert all(dependency not in source for dependency in forbidden_dependencies)
    for evaluator in (
        evaluate_experiment_candidate,
        evaluate_experiment_p0,
    ):
        parameter_names = set(inspect.signature(evaluator).parameters)
        assert not {
            "hard_set",
            "sample_limit",
            "smoke_set",
            "promotion",
            "protected_validation",
            "final_test",
        } & parameter_names


def _synthetic_search_input(labels):
    records = tuple(
        MappingProxyType(
            {
                "sample_id": f"sample-{index}",
                "claim_type": "direct",
                "gold_label": label,
            }
        )
        for index, label in enumerate(labels)
    )
    return SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": EXPERIMENT_REQUIRED_SPLIT_SHA256,
                "search_membership_sha256": "a" * 64,
            }
        ),
        _records=records,
    )


def _outcomes(search_input, answers):
    return tuple(
        PredictionOutcome(
            sample_id=sample_id,
            raw_response=f"Therefore, the final answer is: Answer: {answer}",
        )
        for sample_id, answer in zip(search_input.sample_ids, answers)
    )


class _CountingClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, _request):
        self.calls += 1
        return SolverResult("Therefore, the final answer is: Answer: yes")


class _InterruptingClient(_CountingClient):
    def __init__(self, *, after_successes: int) -> None:
        super().__init__()
        self.successes = 0
        self._after_successes = after_successes

    def complete(self, _request):
        self.calls += 1
        if self.successes >= self._after_successes:
            raise KeyboardInterrupt()
        self.successes += 1
        return SolverResult("Therefore, the final answer is: Answer: yes")


class _PermanentFailureClient(_CountingClient):
    def complete(self, _request):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("offline injected failure")
        return SolverResult("Therefore, the final answer is: Answer: yes")


def _executor(cache, client):
    return RequestExecutor(
        cache=cache,
        client_factory=lambda: client,
        retry_policy=SolverRetryPolicy(maximum_attempts=1),
        maximum_in_flight_requests=1,
        sleeper=lambda _delay: pytest.fail("offline test must not sleep"),
        clock=lambda: 0.0,
    )


def _safe_candidate_requests(
    *,
    search_input,
    candidate_id,
    prompt_sha256,
    solver_identity_sha256,
    **_unused,
):
    """Avoid image serialization while exercising persistence semantics only."""

    config = canonical_experiment_config()
    request = SolverRequest(
        model=config.solver_model,
        messages=({"role": "user", "content": "offline evaluator test"},),
        generation=SolverGenerationSettings.from_config(config),
    )
    return tuple(
        {
            "sample_id": sample_id,
            "request": request,
            "identity": build_solver_request_identity(
                request,
                sample_id=sample_id,
                candidate_id=candidate_id,
                prompt_sha256=prompt_sha256,
                split_sha256=search_input.manifest["split_sha256"],
                search_membership_sha256=search_input.manifest[
                    "search_membership_sha256"
                ],
                solver_identity_sha256=solver_identity_sha256,
                config=config,
            ),
        }
        for sample_id in search_input.sample_ids
    )


def _canonical_snapshot_bytes(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
        + "\n"
    ).encode("utf-8")


def _p0_equivalence_input(tmp_path):
    first = tmp_path / "p0-first.png"
    second = tmp_path / "p0-second.png"
    Image.new("RGB", (2, 2), color="red").save(first, format="PNG")
    Image.new("RGB", (2, 2), color="green").save(second, format="PNG")
    records = (
        MappingProxyType(
            {
                "sample_id": "p0-direct",
                "claim_type": "direct",
                "claim": "Direct claim.",
                "context": "Direct context.",
                "caption": "Direct caption.",
                "image_path": str(first),
                "gold_label": "true",
            }
        ),
        MappingProxyType(
            {
                "sample_id": "p0-analytical",
                "claim_type": "analytical",
                "claim": "Analytical claim.",
                "context": "Analytical context.",
                "caption": "Analytical caption.",
                "image_path": str(first),
                "gold_label": "false",
            }
        ),
        MappingProxyType(
            {
                "sample_id": "p0-parallel",
                "claim_type": "parallel",
                "claim": "Parallel claim.",
                "context": "Parallel context.",
                "caption1": "Parallel first caption.",
                "caption2": "Parallel second caption.",
                "item1_path": str(first),
                "item2_path": str(second),
                "gold_label": "true",
            }
        ),
        MappingProxyType(
            {
                "sample_id": "p0-sequential",
                "claim_type": "sequential",
                "claim": "Sequential claim.",
                "context": "Sequential context.",
                "caption1": "Sequential first caption.",
                "caption2": "Sequential second caption.",
                "item1_path": str(first),
                "item2_path": str(second),
                "gold_label": "false",
            }
        ),
    )
    return (
        SearchInput(
            _manifest=MappingProxyType(
                {
                    "split_sha256": EXPERIMENT_REQUIRED_SPLIT_SHA256,
                    "search_membership_sha256": "a" * 64,
                }
            ),
            _records=records,
        ),
        ((first,), (first,), (first, second), (first, second)),
    )


def _message_image_bytes(messages):
    return [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in messages[0]["content"]
        if block["type"] == "image_url"
    ]


def _synthetic_1000_search_artifact(tmp_path):
    """Write a valid SEARCH-safe artifact with no FINAL records or IDs."""

    first = tmp_path / "synthetic-first.png"
    second = tmp_path / "synthetic-second.png"
    Image.new("RGB", (2, 2), color="red").save(first, format="PNG")
    Image.new("RGB", (2, 2), color="green").save(second, format="PNG")
    sample_ids = [f"synthetic-{index:04d}" for index in range(1000)]
    search_pool = {
        "sample_ids": sample_ids,
        "paper_identities": [f"paper-{index:04d}" for index in range(1000)],
        "sample_count": 1000,
        "paper_count": 1000,
    }
    config = canonical_experiment_config()
    manifest = {
        "schema_version": EXPERIMENT_SEARCH_SAFE_MANIFEST_SCHEMA,
        "artifact_type": "search_safe_manifest",
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "split_seed": config.split_seed,
        "source_dataset_sha256": "0" * 64,
        "config_sha256": config.sha256(),
        "sample_identity_version": EXPERIMENT_SAMPLE_IDENTITY_VERSION,
        "paper_identity_version": EXPERIMENT_PAPER_IDENTITY_VERSION,
        "split_schema_version": EXPERIMENT_SPLIT_SCHEMA_VERSION,
        "split_algorithm": EXPERIMENT_ALLOCATION_ALGORITHM,
        "preparation_identity_schema_version": EXPERIMENT_PREPARATION_IDENTITY_SCHEMA,
        "preparation_identity_sha256": "1" * 64,
        "split_sha256": EXPERIMENT_REQUIRED_SPLIT_SHA256,
        "search_materialization_schema_version": EXPERIMENT_SEARCH_DATASET_SCHEMA,
        "search_membership_sha256": _canonical_sha256(search_pool),
        "SEARCH": search_pool,
        "final_membership_commitment": "2" * 64,
        "search_safe_manifest_sha256": "",
    }
    manifest["search_safe_manifest_sha256"] = _canonical_sha256(manifest)
    records = []
    for index, sample_id in enumerate(sample_ids):
        method = ("direct", "analytical", "parallel", "sequential")[index % 4]
        common = {
            "sample_id": sample_id,
            "claim_type": method,
            "claim": f"Synthetic claim {index:04d}.",
            "context": f"Synthetic context {index:04d}.",
            "gold_label": "yes" if index % 2 == 0 else "no",
        }
        if method in {"direct", "analytical"}:
            records.append(
                {
                    **common,
                    "caption": f"Synthetic caption {index:04d}.",
                    "image_path": str(first),
                }
            )
        else:
            records.append(
                {
                    **common,
                    "caption1": f"Synthetic first caption {index:04d}.",
                    "caption2": f"Synthetic second caption {index:04d}.",
                    "item1_path": str(first),
                    "item2_path": str(second),
                }
            )
    manifest_path = tmp_path / "search_safe_manifest.json"
    records_path = tmp_path / "search_records.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    records_path.write_text(json.dumps(records), encoding="utf-8")
    return (
        load_experiment_search_input(
            search_safe_manifest_path=manifest_path,
            search_records_path=records_path,
        ),
        (first, second),
    )


class _IntegrationClient:
    def __init__(
        self,
        *,
        interrupt_at=None,
        transient_timeout_at=None,
        permanent_failure_at=None,
        parse_failure_indices=frozenset(),
        abstention_indices=frozenset(),
        image_paths=(),
    ):
        self.calls = 0
        self.indices = []
        self.image_orders = {}
        self._interrupt_at = interrupt_at
        self._transient_timeout_at = transient_timeout_at
        self._permanent_failure_at = permanent_failure_at
        self._parse_failure_indices = set(parse_failure_indices)
        self._abstention_indices = set(abstention_indices)
        self._timeout_fired = False
        self._permanent_failure_fired = False
        self._completed = 0
        self._image_paths = image_paths

    def complete(self, request):
        content = request.messages[0]["content"]
        text = content[-1]["text"]
        index = int(re.search(r"Synthetic claim (\d{4})", text).group(1))
        self.calls += 1
        self.indices.append(index)
        self.image_orders[index] = tuple(
            base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
            for block in content
            if block["type"] == "image_url"
        )
        if (
            self._transient_timeout_at == index
            and not self._timeout_fired
        ):
            self._timeout_fired = True
            raise RequestTimeoutError("offline transient timeout")
        if self._interrupt_at is not None and self._completed >= self._interrupt_at:
            raise KeyboardInterrupt()
        if (
            self._permanent_failure_at == index
            and not self._permanent_failure_fired
        ):
            self._permanent_failure_fired = True
            raise RuntimeError("offline permanent failure")
        self._completed += 1
        if index in self._parse_failure_indices:
            return SolverResult("Answer: yes or no")
        if index in self._abstention_indices:
            return SolverResult("I cannot determine.")
        return SolverResult(
            "Therefore, the final answer is: Answer: "
            + ("yes" if index % 2 == 0 else "no")
        )


def _integration_executor(cache, client, retry_delays):
    return RequestExecutor(
        cache=cache,
        client_factory=lambda: client,
        retry_policy=SolverRetryPolicy(
            maximum_attempts=2,
            initial_backoff_seconds=0,
            maximum_backoff_seconds=0,
        ),
        maximum_in_flight_requests=1,
        sleeper=retry_delays.append,
        clock=lambda: 0.0,
    )


def _candidate_prompt(prefix):
    return {
        method: Template(prefix + "\n" + template.template)
        for method, template in COT_PROMPT.items()
    }


def _unique_in_order(values):
    seen = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _assert_recorded_image_order(image_orders, image_paths):
    first, second = image_paths
    for index, observed in image_orders.items():
        expected = (first.read_bytes(),) if index % 4 < 2 else (
            first.read_bytes(),
            second.read_bytes(),
        )
        assert observed == expected


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _mutable(value):
    if hasattr(value, "items"):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable(item) for item in value]
    return copy.deepcopy(value)
