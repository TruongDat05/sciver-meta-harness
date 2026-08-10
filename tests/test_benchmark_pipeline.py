"""Offline integration coverage for the complete remote benchmark pipeline."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from PIL import Image
import pytest
import requests

import main as cli_module
from evaluation.metrics import evaluate_records, load_jsonl
from utils.dataset_adapters import get_dataset_adapter


FAKE_API_KEY = "obviously-fake-integration-key"
FAKE_API_URL = "https://invalid.example.test/chat/completions"

BENCHMARK_CASES = (
    ("SciVer", "Qwen2.5-VL-7B-Instruct", "direct", "yes"),
    ("SciAtomicBench", "gemma-4-31B-it", "analytical", "no"),
    ("MuSciClaims", "gemma-4-26B-A4B-it", "parallel", "yes"),
    ("SciClaimEval", "gemma-3-27b-it", "sequential", "no"),
)


@dataclass(frozen=True)
class BenchmarkFixture:
    dataset: str
    model: str
    method: str
    answer: str
    dataset_path: Path
    sample_id: str
    ground_truth_marker: str
    image_bytes: tuple[bytes, ...]


@pytest.fixture
def benchmark_fixture_factory(tmp_path):
    """Create tiny normalized records without guessing raw dataset mappings."""

    def create(case: tuple[str, str, str, str]) -> BenchmarkFixture:
        dataset, model, method, answer = case
        fixture_root = tmp_path / dataset
        fixture_root.mkdir()

        first_image = fixture_root / "first.png"
        second_image = fixture_root / "second.png"
        Image.new("RGB", (2, 2), color="red").save(first_image, format="PNG")
        Image.new("RGB", (2, 2), color="green").save(second_image, format="PNG")

        paper_path = fixture_root / "paper.json"
        paper_path.write_text(
            json.dumps(
                {
                    "sections": [
                        {
                            "section_id": "1.1",
                            "section_name": "Results",
                            "text": "Small local evidence paragraph.",
                        }
                    ],
                    "image_paths": [{"caption": "First local chart."}],
                    "tables": [{"capture": "Second local table."}],
                }
            ),
            encoding="utf-8",
        )

        marker = f"PRIVATE_GROUND_TRUTH_{dataset.upper()}"
        sample_id = f"{dataset}-sample"
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "claim_type": method,
            "claim": f"A synthetic claim for {dataset}.",
            "paper_path": str(paper_path),
            "section": ["1.1"],
            "gold_label": answer,
            "label": marker,
            "rationale": f"{marker}_RATIONALE",
        }
        if method in {"direct", "analytical"}:
            record.update(
                {
                    "type": "chart",
                    "item": 0,
                    "image_path": str(first_image),
                }
            )
            expected_images = (first_image.read_bytes(),)
        else:
            record.update(
                {
                    "item1_type": "chart",
                    "item1": 0,
                    "item1_path": str(first_image),
                    "item2_type": "table",
                    "item2": 0,
                    "item2_path": str(second_image),
                }
            )
            expected_images = (first_image.read_bytes(), second_image.read_bytes())

        dataset_path = fixture_root / "records.json"
        dataset_path.write_text(json.dumps([record]), encoding="utf-8")
        return BenchmarkFixture(
            dataset=dataset,
            model=model,
            method=method,
            answer=answer,
            dataset_path=dataset_path,
            sample_id=sample_id,
            ground_truth_marker=marker,
            image_bytes=expected_images,
        )

    return create


def _remote_args(fixture: BenchmarkFixture, output_dir: Path, *extra: str) -> list[str]:
    return [
        "--provider",
        "remote",
        "--model",
        fixture.model,
        "--dataset",
        fixture.dataset,
        "--dataset-path",
        str(fixture.dataset_path),
        "--method",
        "cot",
        "--output-dir",
        str(output_dir),
        "--max-num",
        "-1",
        *extra,
    ]


def _output_path(fixture: BenchmarkFixture, output_dir: Path) -> Path:
    return output_dir / f"{fixture.dataset}_cot" / f"{fixture.model}.jsonl"


def _mock_response(answer: str, status_code: int = 200) -> Mock:
    response = Mock()
    response.status_code = status_code
    response.headers = {}
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": (
                        "Local reasoning. Therefore, the final answer is: "
                        f"Answer: {answer}"
                    )
                }
            }
        ]
    }
    return response


def _image_bytes_from_payload(payload: dict[str, Any]) -> tuple[bytes, ...]:
    content = payload["messages"][0]["content"]
    return tuple(
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in content
        if block["type"] == "image_url"
    )


@pytest.mark.parametrize("case", BENCHMARK_CASES)
def test_complete_pipeline_is_offline_for_every_dataset_model_and_method(
    case,
    benchmark_fixture_factory,
    tmp_path,
    monkeypatch,
    capsys,
):
    fixture = benchmark_fixture_factory(case)
    output_dir = tmp_path / "outputs"
    calls: list[tuple[str, str, dict[str, Any]]] = []

    def fake_request(session, method, url, **kwargs):
        calls.append((method, url, kwargs))
        return _mock_response(fixture.answer)

    monkeypatch.setenv("API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("API_URL", FAKE_API_URL)
    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)

    adapter = get_dataset_adapter(fixture.dataset)
    loaded = adapter.load(fixture.dataset_path)
    assert adapter.dataset_name == fixture.dataset
    assert loaded[0].sample_id == fixture.sample_id
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""

    assert cli_module.cli(_remote_args(fixture, output_dir, "--live-api")) == 0
    assert len(calls) == 1
    method, url, request = calls[0]
    assert method == "POST"
    assert url == FAKE_API_URL

    payload = request["json"]
    serialized_payload = json.dumps(payload)
    assert payload["model"] == fixture.model
    assert payload["stream"] is False
    assert fixture.ground_truth_marker not in serialized_payload
    assert '"gold_label"' not in serialized_payload
    assert '"label"' not in serialized_payload
    assert '"rationale"' not in serialized_payload
    assert FAKE_API_KEY not in serialized_payload
    assert _image_bytes_from_payload(payload) == fixture.image_bytes

    output_path = _output_path(fixture, output_dir)
    records = load_jsonl(output_path)
    assert len(records) == 1
    assert records[0]["prediction"] == fixture.answer
    assert records[0]["gold_label"] == fixture.answer
    assert records[0]["request_status"] == "success"
    assert "base64" not in output_path.read_text(encoding="utf-8")

    summary = evaluate_records(records)[0]
    assert summary["dataset"] == fixture.dataset
    assert summary["model"] == fixture.model
    assert summary["method"] == fixture.method
    assert summary["parse_coverage"] == {
        "value": 1.0,
        "numerator": 1,
        "denominator": 1,
    }
    assert summary["accuracy_all_labeled"] == {
        "value": 1.0,
        "numerator": 1,
        "denominator": 1,
    }

    captured = capsys.readouterr()
    observable = captured.out + captured.err + output_path.read_text(encoding="utf-8")
    assert FAKE_API_KEY not in observable


@pytest.mark.parametrize("case", BENCHMARK_CASES)
def test_cli_dry_run_needs_no_credentials_or_http(
    case, benchmark_fixture_factory, tmp_path, capsys
):
    fixture = benchmark_fixture_factory(case)
    output_dir = tmp_path / "dry-run"

    assert cli_module.cli(_remote_args(fixture, output_dir, "--dry-run")) == 0

    captured = capsys.readouterr()
    metadata = json.loads(captured.out)
    assert metadata["dataset"] == fixture.dataset
    assert metadata["model"] == fixture.model
    assert metadata["reasoning_method"] == fixture.method
    assert metadata["image_count"] == len(fixture.image_bytes)
    assert fixture.ground_truth_marker not in captured.out + captured.err
    assert not _output_path(fixture, output_dir).exists()


def test_resume_skips_completed_request_without_credentials_or_http(
    benchmark_fixture_factory, tmp_path, monkeypatch
):
    fixture = benchmark_fixture_factory(BENCHMARK_CASES[0])
    output_dir = tmp_path / "resume"
    calls = 0

    def fake_request(session, method, url, **kwargs):
        nonlocal calls
        calls += 1
        return _mock_response(fixture.answer)

    monkeypatch.setenv("API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("API_URL", FAKE_API_URL)
    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)
    assert cli_module.cli(_remote_args(fixture, output_dir, "--live-api")) == 0
    assert calls == 1

    monkeypatch.delenv("API_KEY")
    monkeypatch.delenv("API_URL")

    def unexpected_request(*args, **kwargs):
        raise AssertionError("resume must not repeat a completed request")

    monkeypatch.setattr(requests.sessions.Session, "request", unexpected_request)
    assert cli_module.cli(
        _remote_args(fixture, output_dir, "--live-api", "--resume")
    ) == 0
    assert len(load_jsonl(_output_path(fixture, output_dir))) == 1


def test_mocked_http_failure_never_exposes_api_key(
    benchmark_fixture_factory,
    tmp_path,
    monkeypatch,
    capsys,
    caplog,
):
    fixture = benchmark_fixture_factory(BENCHMARK_CASES[0])
    output_dir = tmp_path / "failure"

    def fake_request(session, method, url, **kwargs):
        return _mock_response(fixture.answer, status_code=401)

    monkeypatch.setenv("API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("API_URL", FAKE_API_URL)
    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)

    assert cli_module.cli(_remote_args(fixture, output_dir, "--live-api")) == 1

    output_path = _output_path(fixture, output_dir)
    records = load_jsonl(output_path)
    assert records[0]["request_status"] == "api_failure"
    assert records[0]["error_type"] == "AuthenticationError"
    captured = capsys.readouterr()
    observable = (
        captured.out
        + captured.err
        + caplog.text
        + output_path.read_text(encoding="utf-8")
    )
    assert FAKE_API_KEY not in observable


def test_unparseable_mocked_response_is_persisted_and_counted(
    benchmark_fixture_factory,
    tmp_path,
    monkeypatch,
):
    fixture = benchmark_fixture_factory(BENCHMARK_CASES[2])
    output_dir = tmp_path / "parse-failure"
    response = _mock_response(fixture.answer)
    response.json.return_value = {
        "choices": [{"message": {"content": "Reasoning without a final marker."}}]
    }

    def fake_request(session, method, url, **kwargs):
        return response

    monkeypatch.setenv("API_KEY", FAKE_API_KEY)
    monkeypatch.setenv("API_URL", FAKE_API_URL)
    monkeypatch.setattr(requests.sessions.Session, "request", fake_request)

    assert cli_module.cli(_remote_args(fixture, output_dir, "--live-api")) == 1

    records = load_jsonl(_output_path(fixture, output_dir))
    assert records[0]["prediction"] == "invalid"
    assert records[0]["parse_status"] == "invalid"
    assert records[0]["request_status"] == "parse_failure"
    summary = evaluate_records(records)[0]
    assert summary["parse_failures"] == 1
    assert summary["parse_coverage"] == {
        "value": 0.0,
        "numerator": 0,
        "denominator": 1,
    }
