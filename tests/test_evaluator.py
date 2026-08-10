import base64
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from meta_harness.candidate_store import CandidateStore, CandidateStoreError
from meta_harness.config import MetaHarnessConfig
from meta_harness.evaluator import (
    EvaluatorError,
    evaluate_candidate,
    metrics_path_for_results,
)
from meta_harness.schemas import template_source_sha256
from meta_harness.split_manager import build_split_manifest
from utils.dataset_adapters import AdaptedSample
from utils.result_writer import iter_result_records


METHODS = ("direct", "analytical", "parallel", "sequential")


class FakeSolver:
    def __init__(self, outcomes):
        self.outcomes = dict(outcomes)
        self.calls = []
        self.last_usage = None

    def create_chat_completion(self, model, messages):
        self.calls.append((model, messages))
        prompt = messages[0]["content"][-1]["text"]
        method = next(method for method in METHODS if f"claim-{method}" in prompt)
        outcome = self.outcomes[method]
        if isinstance(outcome, BaseException):
            raise outcome
        self.last_usage = {"input_tokens": 10, "output_tokens": 2}
        return outcome


def _templates(sentinel):
    return {
        "direct": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption: $caption\nConclude with exactly Answer: yes or Answer: no."
        ),
        "analytical": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption: $caption\nConclude with exactly Answer: yes or Answer: no."
        ),
        "parallel": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption 1: $caption1\nCaption 2: $caption2\n"
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "sequential": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption 1: $caption1\nCaption 2: $caption2\n"
            "Conclude with exactly Answer: yes or Answer: no."
        ),
    }


def _store_candidate(tmp_path, candidate_id="candidate_one", sentinel="ONE"):
    store = CandidateStore(tmp_path, "run_001")
    templates = _templates(sentinel)
    store.create(
        {
            "candidate_id": candidate_id,
            "parent_id": "baseline_cot",
            "search_axis": "exploration",
            "hypothesis": "The sentinel wording will preserve exact predictions.",
            "templates": templates,
            "expected_tradeoff": "The prompt will contain one extra sentinel line.",
            "source_sha256": template_source_sha256(templates),
        }
    )
    return store


def _images(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "green").save(second)
    return first, second


def _samples_and_manifest(tmp_path):
    first, second = _images(tmp_path)
    samples = []
    records = []
    for paper_index in range(5):
        for method_index, method in enumerate(METHODS):
            sample_id = f"paper-{paper_index}-{method}"
            record = {
                "sample_id": sample_id,
                "paper_id": f"paper-{paper_index}",
                "claim_type": method,
                "claim": f"claim-{method}",
                "context": f"context-{paper_index}",
                "gold_label": (
                    "yes"
                    if method in {"direct", "sequential"}
                    else "no"
                ),
                "gold_explanation": f"GOLD_ONLY_{method}",
            }
            if method in {"direct", "analytical"}:
                record.update(
                    {
                        "caption": f"caption-{method}",
                        "image_path": str(first),
                    }
                )
            else:
                record.update(
                    {
                        "caption1": f"caption-1-{method}",
                        "caption2": f"caption-2-{method}",
                        "item1_path": str(first),
                        "item2_path": str(second),
                    }
                )
            samples.append(AdaptedSample(sample_id, record))
            records.append(record)
    manifest = build_split_manifest(records)
    search_ids = manifest["splits"]["search"]["sample_ids"]
    selected_paper = search_ids[0].rsplit("-", 1)[0]
    selected_ids = [
        f"{selected_paper}-{method}"
        for method in METHODS
    ]
    return samples, manifest, selected_ids, (first, second)


def _image_hashes(messages):
    return [
        hashlib.sha256(
            base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        ).hexdigest()
        for block in messages[0]["content"]
        if block["type"] == "image_url"
    ]


def test_all_methods_exact_metrics_image_order_and_model_visible_boundary(tmp_path):
    store = _store_candidate(tmp_path)
    samples, manifest, selected_ids, images = _samples_and_manifest(tmp_path)
    solver = FakeSolver(
        {
            "direct": "Therefore, the final answer is: Answer: yes",
            "analytical": "Therefore, the final answer is: Answer: yes",
            "parallel": "Therefore, the final answer is: Answer: no",
            "sequential": "Therefore, the final answer is: Answer: no",
        }
    )
    ticks = iter(range(8))
    output = tmp_path / "results.jsonl"

    report = evaluate_candidate(
        run_id="evaluation_001",
        candidate_id="candidate_one",
        candidate_store=store,
        split_manifest=manifest,
        split_name="search",
        sample_ids=selected_ids,
        samples=samples,
        solver=solver,
        output_path=output,
        clock=lambda: next(ticks),
    )

    assert len(solver.calls) == len(selected_ids) == 4
    assert {model for model, _ in solver.calls} == {
        "gemma-4-26B-A4B-it"
    }
    assert {record["reasoning_method"] for record in iter_result_records(output)} == set(
        METHODS
    )
    assert {record["candidate_id"] for record in iter_result_records(output)} == {
        "candidate_one"
    }
    assert all(record["latency"] == 1.0 for record in iter_result_records(output))
    assert all(
        record["usage"] == {"input_tokens": 10, "output_tokens": 2}
        for record in iter_result_records(output)
    )
    assert report["metrics"] == {
        "accuracy": 0.5,
        "precision": {"yes": 0.5, "no": 0.5},
        "recall": {"yes": 0.5, "no": 0.5},
        "F1": {"yes": 0.5, "no": 0.5},
        "Macro-F1": 0.5,
        "coverage": 1.0,
        "parse_coverage": 1.0,
        "request_coverage": 1.0,
        "confusion_matrix": {
            "actual_yes": {"predicted_yes": 1, "predicted_no": 1},
            "actual_no": {"predicted_yes": 1, "predicted_no": 1},
        },
    }
    assert report["rankable"] is True
    assert json.loads(metrics_path_for_results(output).read_text()) == report

    for method, (_, messages) in zip(METHODS, solver.calls):
        prompt_text = messages[0]["content"][-1]["text"]
        assert "gold_label" not in prompt_text.casefold()
        assert "gold_explanation" not in prompt_text.casefold()
        assert "GOLD_ONLY_" not in prompt_text
        assert prompt_text.startswith("ONE\n")
        expected = [images[0]] if method in {"direct", "analytical"} else list(images)
        assert _image_hashes(messages) == [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in expected
        ]


def test_selection_is_isolated_to_one_declared_split_before_solver_calls(tmp_path):
    store = _store_candidate(tmp_path)
    samples, manifest, selected_ids, _ = _samples_and_manifest(tmp_path)
    solver = FakeSolver({})
    outside = manifest["splits"]["validation"]["sample_ids"][0]

    with pytest.raises(EvaluatorError, match="declared split"):
        evaluate_candidate(
            run_id="evaluation_001",
            candidate_id="candidate_one",
            candidate_store=store,
            split_manifest=manifest,
            split_name="search",
            sample_ids=[selected_ids[0], outside],
            samples=samples,
            solver=solver,
            output_path=tmp_path / "results.jsonl",
        )

    assert solver.calls == []


def test_request_budget_guard_stops_before_an_extra_solver_call(tmp_path):
    store = _store_candidate(tmp_path)
    samples, manifest, selected_ids, _ = _samples_and_manifest(tmp_path)
    solver = FakeSolver(
        {
            method: "Therefore, the final answer is: Answer: yes"
            for method in METHODS
        }
    )

    def before_solver_call():
        if len(solver.calls) >= 2:
            raise RuntimeError("solver-call ceiling")

    with pytest.raises(RuntimeError, match="solver-call ceiling"):
        evaluate_candidate(
            run_id="evaluation_001",
            candidate_id="candidate_one",
            candidate_store=store,
            split_manifest=manifest,
            split_name="search",
            sample_ids=selected_ids,
            samples=samples,
            solver=solver,
            output_path=tmp_path / "results.jsonl",
            before_solver_call=before_solver_call,
        )

    assert len(solver.calls) == 2
    assert len(list(iter_result_records(tmp_path / "results.jsonl"))) == 2


def test_api_parse_and_invalid_input_failures_are_distinct_and_not_rankable(tmp_path):
    store = _store_candidate(tmp_path)
    samples, manifest, selected_ids, _ = _samples_and_manifest(tmp_path)
    selected = {
        sample.sample_id: sample
        for sample in samples
        if sample.sample_id in selected_ids
    }
    invalid = dict(selected[selected_ids[3]].record)
    invalid.pop("item2_path")
    selected[selected_ids[3]] = AdaptedSample(selected_ids[3], invalid)
    solver = FakeSolver(
        {
            "direct": "Therefore, the final answer is: Answer: yes",
            "analytical": "response without an explicit conclusion",
            "parallel": RuntimeError("temporary solver failure"),
        }
    )

    report = evaluate_candidate(
        run_id="evaluation_001",
        candidate_id="candidate_one",
        candidate_store=store,
        split_manifest=manifest,
        split_name="search",
        sample_ids=selected_ids,
        samples=list(selected.values()),
        solver=solver,
        output_path=tmp_path / "results.jsonl",
    )

    records = {
        record["reasoning_method"]: record
        for record in iter_result_records(tmp_path / "results.jsonl")
    }
    assert len(solver.calls) == 3
    assert records["direct"]["request_status"] == "success"
    assert records["analytical"]["request_status"] == "parse_failure"
    assert records["parallel"]["request_status"] == "api_failure"
    assert records["sequential"]["request_status"] == "invalid_input"
    assert report["failure_counts"] == {
        "invalid_input": 1,
        "api_failure": 1,
        "parse_failure": 1,
    }
    assert report["metrics"]["coverage"] == 0.25
    assert report["metrics"]["parse_coverage"] == 0.25
    assert report["metrics"]["request_coverage"] == 0.5
    assert report["unresolved_api_failures"] == 1
    assert report["rankable"] is False

    calls_before_resume = len(solver.calls)
    resumed_report = evaluate_candidate(
        run_id="evaluation_001",
        candidate_id="candidate_one",
        candidate_store=store,
        split_manifest=manifest,
        split_name="search",
        sample_ids=selected_ids,
        samples=list(selected.values()),
        solver=solver,
        output_path=tmp_path / "results.jsonl",
    )
    resumed_records = list(iter_result_records(tmp_path / "results.jsonl"))
    result_keys = {
        (
            record["candidate_id"],
            record["sample_id"],
            record["split_sha256"],
        )
        for record in resumed_records
    }
    assert resumed_report == report
    assert len(solver.calls) == calls_before_resume + 1
    assert len(resumed_records) == len(selected_ids) + 1
    assert len(result_keys) == len(selected_ids)


def test_candidate_must_be_loaded_from_store_and_model_is_fixed(tmp_path):
    store = _store_candidate(tmp_path)
    samples, manifest, selected_ids, _ = _samples_and_manifest(tmp_path)

    with pytest.raises(EvaluatorError, match="CandidateStore"):
        evaluate_candidate(
            run_id="evaluation_001",
            candidate_id="candidate_one",
            candidate_store=None,
            split_manifest=manifest,
            split_name="search",
            sample_ids=selected_ids,
            samples=samples,
            solver=FakeSolver({}),
            output_path=tmp_path / "results.jsonl",
        )

    with pytest.raises(CandidateStoreError, match="candidate does not exist"):
        evaluate_candidate(
            run_id="evaluation_001",
            candidate_id="missing",
            candidate_store=store,
            split_manifest=manifest,
            split_name="search",
            sample_ids=selected_ids,
            samples=samples,
            solver=FakeSolver({}),
            output_path=tmp_path / "results.jsonl",
        )

    solver = FakeSolver({})
    with pytest.raises(EvaluatorError, match="generation request options"):
        evaluate_candidate(
            run_id="evaluation_001",
            candidate_id="candidate_one",
            candidate_store=store,
            split_manifest=manifest,
            split_name="search",
            sample_ids=selected_ids,
            samples=samples,
            solver=solver,
            output_path=tmp_path / "results.jsonl",
            config=MetaHarnessConfig(temperature=0.1),
        )
    assert solver.calls == []
