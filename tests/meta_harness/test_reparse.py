import json
from pathlib import Path

import pytest

from meta_harness.reparse import ReparseError, reparse_run


def _record(
    sample_id,
    gold_label,
    raw_response,
    *,
    prediction="invalid",
    parse_status="invalid",
    request_status="parse_failure",
):
    return {
        "run_id": "test-run",
        "sample_id": sample_id,
        "dataset": "SciVer",
        "model": "test-model",
        "method": "direct",
        "prompt_variant": "candidate_a",
        "candidate_id": "candidate_a",
        "reasoning_method": "direct",
        "gold_label": gold_label,
        "prediction": prediction,
        "parse_status": parse_status,
        "raw_response": raw_response,
        "request_status": request_status,
        "error_type": "AnswerParseError" if request_status == "parse_failure" else None,
        "error_message": "old parse failure" if request_status == "parse_failure" else None,
        "timestamp": "2026-01-01T00:00:00Z",
        "attempt_count": 1,
    }


def _source_run(tmp_path, records):
    source = tmp_path / "source-run"
    evaluation = source / "evaluations" / "candidate_a"
    evaluation.mkdir(parents=True)
    results = evaluation / "validation.results.jsonl"
    results.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    metrics = {
        "schema_version": 1,
        "parser_version": "answer_parser_v1",
        "run_id": "test-run",
        "candidate_id": "candidate_a",
        "prompt_variant": "candidate_a",
        "sample_count": len(records),
        "metrics": {
            "coverage": 0,
            "parse_coverage": 0,
            "accuracy": 0,
            "Macro-F1": 0,
        },
        "failure_counts": {
            "invalid_input": 0,
            "api_failure": 0,
            "parse_failure": len(records),
        },
    }
    (evaluation / "validation.metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    return source, results


def _load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_reparse_run_is_offline_and_recomputes_full_denominator_metrics(tmp_path):
    records = [
        _record("sample-1", "yes", "yes\n\nRationale: clearly supported."),
        _record("sample-2", "no", "The claim is **no**."),
        _record(
            "sample-3",
            "yes",
            "The evidence could support yes, but another reading suggests no.",
        ),
    ]
    source, source_results = _source_run(tmp_path, records)
    source_bytes = source_results.read_bytes()
    output = tmp_path / "parser_v2_reparse"

    summary = reparse_run(
        source,
        output,
        expected_sample_count=3,
        expected_evaluation_count=1,
    )

    assert source_results.read_bytes() == source_bytes
    assert summary["parser_version"] == "answer_parser_v2"
    assert summary["evaluation_count"] == 1
    reparsed = _load_jsonl(
        output / "candidate_a" / "validation.results.jsonl"
    )
    assert [record["prediction"] for record in reparsed] == [
        "yes",
        "no",
        "invalid",
    ]
    assert [record["request_status"] for record in reparsed] == [
        "success",
        "success",
        "parse_failure",
    ]

    metrics = json.loads(
        (output / "candidate_a" / "validation.metrics.json").read_text(
            encoding="utf-8"
        )
    )
    assert metrics["metrics"]["parse_coverage"] == pytest.approx(2 / 3)
    assert metrics["metrics"]["accuracy"] == pytest.approx(2 / 3)
    assert metrics["metrics"]["Macro-F1"] == pytest.approx(5 / 6)
    assert metrics["failure_counts"]["parse_failure"] == 1


def test_reparse_preserves_response_less_api_failure(tmp_path):
    records = [
        _record(
            "sample-1",
            "yes",
            None,
            prediction=None,
            parse_status="not_attempted",
            request_status="api_failure",
        )
    ]
    source, _ = _source_run(tmp_path, records)
    output = tmp_path / "parser_v2_reparse"

    reparse_run(source, output)

    [record] = _load_jsonl(
        output / "candidate_a" / "validation.results.jsonl"
    )
    assert record["request_status"] == "api_failure"
    assert record["parse_status"] == "not_attempted"
    assert record["prediction"] is None


def test_reparse_refuses_existing_or_nested_output(tmp_path):
    source, source_results = _source_run(
        tmp_path,
        [_record("sample-1", "yes", "yes")],
    )
    source_bytes = source_results.read_bytes()
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(ReparseError, match="already exists"):
        reparse_run(source, existing)
    with pytest.raises(ReparseError, match="outside"):
        reparse_run(source, source / "parser_v2_reparse")

    assert source_results.read_bytes() == source_bytes
