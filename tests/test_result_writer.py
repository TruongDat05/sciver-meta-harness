import json
from datetime import datetime

import pytest

from utils.result_writer import (
    API_FAILURE,
    INVALID_INPUT,
    PARSE_FAILURE,
    SUCCESS,
    ResultWriter,
    ResultWriterError,
    iter_result_records,
)


BASE_RESULT = {
    "run_id": "run-1",
    "sample_id": "sample-1",
    "dataset": "SciVer",
    "model": "test-model",
    "method": "zero-shot",
    "prediction": "yes",
    "parse_status": "parsed",
    "raw_response": "Answer: yes",
    "request_status": SUCCESS,
}


def _write(writer, **overrides):
    result = dict(BASE_RESULT)
    result.update(overrides)
    return writer.write_result(**result)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_writes_complete_record_immediately_and_syncs(tmp_path, monkeypatch):
    output = tmp_path / "nested" / "results.jsonl"
    fsync_calls = []
    monkeypatch.setattr("utils.result_writer.os.fsync", fsync_calls.append)

    assert _write(ResultWriter(output), gold_label="yes") is True

    records = _read(output)
    assert len(records) == 1
    assert records[0] == {
        **BASE_RESULT,
        "prompt_variant": "cot",
        "gold_label": "yes",
        "error_type": None,
        "error_message": None,
        "timestamp": records[0]["timestamp"],
        "attempt_count": 1,
    }
    assert datetime.fromisoformat(records[0]["timestamp"].replace("Z", "+00:00"))
    assert fsync_calls


def test_gold_label_is_omitted_when_unavailable(tmp_path):
    output = tmp_path / "results.jsonl"
    _write(ResultWriter(output))

    assert "gold_label" not in _read(output)[0]


@pytest.mark.parametrize(
    ("request_status", "parse_status", "prediction", "raw_response"),
    [
        (API_FAILURE, "not_attempted", None, None),
        (PARSE_FAILURE, "invalid", "invalid", "Unstructured response"),
        (INVALID_INPUT, "not_attempted", None, None),
    ],
)
def test_failure_types_are_distinct_and_retryable(
    tmp_path, request_status, parse_status, prediction, raw_response
):
    output = tmp_path / "results.jsonl"
    writer = ResultWriter(output)

    assert _write(
        writer,
        request_status=request_status,
        parse_status=parse_status,
        prediction=prediction,
        raw_response=raw_response,
        error_type="FakeError",
        error_message="safe diagnostic",
    )
    assert writer.should_skip(**{
        key: BASE_RESULT[key]
        for key in ("run_id", "sample_id", "dataset", "model", "method")
    }) is False
    assert writer.next_attempt_count(**{
        key: BASE_RESULT[key]
        for key in ("run_id", "sample_id", "dataset", "model", "method")
    }) == 2
    assert _read(output)[0]["request_status"] == request_status


def test_resume_skips_success_and_prevents_duplicate_success(tmp_path):
    output = tmp_path / "results.jsonl"
    first_writer = ResultWriter(output)
    _write(
        first_writer,
        request_status=API_FAILURE,
        parse_status="not_attempted",
        prediction=None,
        raw_response=None,
    )
    _write(first_writer)

    resumed = ResultWriter(output)
    identity = {
        key: BASE_RESULT[key]
        for key in ("run_id", "sample_id", "dataset", "model", "method")
    }
    assert resumed.is_successful(**identity)
    assert resumed.successful_sample_ids(
        run_id="run-1",
        dataset="SciVer",
        model="test-model",
        method="zero-shot",
    ) == {"sample-1"}
    assert resumed.next_attempt_count(**identity) == 3
    assert _write(resumed) is False
    assert len(_read(output)) == 2
    assert [record["attempt_count"] for record in _read(output)] == [1, 2]


def test_same_sample_in_different_configuration_is_not_skipped(tmp_path):
    output = tmp_path / "results.jsonl"
    writer = ResultWriter(output)
    _write(writer)

    assert _write(writer, method="few-shot")
    assert len(_read(output)) == 2


def test_prompt_variant_is_a_separate_resume_identity(tmp_path):
    output = tmp_path / "results.jsonl"
    writer = ResultWriter(output)
    _write(writer)

    assert _write(writer, prompt_variant="candidate-1")
    assert writer.is_successful(
        run_id="run-1",
        sample_id="sample-1",
        dataset="SciVer",
        model="test-model",
        method="zero-shot",
        prompt_variant="candidate-1",
    )
    assert len(_read(output)) == 2


def test_legacy_record_defaults_to_cot_without_rewriting_file(tmp_path):
    output = tmp_path / "results.jsonl"
    legacy_record = {
        **BASE_RESULT,
        "error_type": None,
        "error_message": None,
        "timestamp": "2026-01-01T00:00:00Z",
        "attempt_count": 1,
    }
    output.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
    original_bytes = output.read_bytes()

    writer = ResultWriter(output)

    assert writer.is_successful(
        run_id="run-1",
        sample_id="sample-1",
        dataset="SciVer",
        model="test-model",
        method="zero-shot",
    )
    assert not writer.is_successful(
        run_id="run-1",
        sample_id="sample-1",
        dataset="SciVer",
        model="test-model",
        method="zero-shot",
        prompt_variant="candidate-1",
    )
    assert list(iter_result_records(output))[0]["prompt_variant"] == "cot"
    assert output.read_bytes() == original_bytes


@pytest.mark.parametrize("initial", [None, b""])
def test_missing_and_empty_files_are_supported(tmp_path, initial):
    output = tmp_path / "results.jsonl"
    if initial is not None:
        output.write_bytes(initial)

    assert _write(ResultWriter(output))
    assert len(_read(output)) == 1


def test_partial_final_line_is_ignored_repaired_and_retryable(tmp_path):
    output = tmp_path / "results.jsonl"
    writer = ResultWriter(output)
    _write(
        writer,
        request_status=API_FAILURE,
        parse_status="not_attempted",
        prediction=None,
        raw_response=None,
    )
    with output.open("ab") as stream:
        stream.write(b'{"run_id":"interrupted"')

    resumed = ResultWriter(output)
    assert _write(resumed)

    records = _read(output)
    assert len(records) == 2
    assert [record["attempt_count"] for record in records] == [1, 2]


def test_valid_final_line_without_newline_is_preserved(tmp_path):
    output = tmp_path / "results.jsonl"
    writer = ResultWriter(output)
    _write(writer, sample_id="sample-0")
    output.write_bytes(output.read_bytes().rstrip(b"\n"))

    resumed = ResultWriter(output)
    assert _write(resumed, sample_id="sample-1")

    assert [record["sample_id"] for record in _read(output)] == [
        "sample-0",
        "sample-1",
    ]


def test_invalid_nonfinal_record_is_reported(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text('{"broken":\n{}\n')

    with pytest.raises(ResultWriterError, match="line 1"):
        ResultWriter(output)


def test_sensitive_values_are_redacted_before_write(tmp_path, monkeypatch):
    output = tmp_path / "results.jsonl"
    fake_key = "obviously-fake-placeholder"
    image_data = "A" * 160
    monkeypatch.setenv("API_KEY", fake_key)

    _write(
        ResultWriter(output),
        request_status=API_FAILURE,
        parse_status="not_attempted",
        prediction=None,
        raw_response=f"data:image/png;base64,{image_data}",
        error=RuntimeError(
            f"Authorization: Bearer {fake_key}; payload={image_data}"
        ),
    )

    written = output.read_text()
    record = _read(output)[0]
    assert fake_key not in written
    assert image_data not in written
    assert "Bearer obviously" not in written
    assert record["error_type"] == "RuntimeError"
    assert "[REDACTED]" in record["error_message"]
    assert record["raw_response"] == "[REDACTED]"


def test_status_and_success_parse_status_are_validated(tmp_path):
    writer = ResultWriter(tmp_path / "results.jsonl")

    with pytest.raises(ResultWriterError, match="request_status"):
        _write(writer, request_status="unknown")
    with pytest.raises(ResultWriterError, match="parse_status='parsed'"):
        _write(writer, parse_status="invalid")


def test_iter_records_ignores_only_partial_final_line(tmp_path):
    output = tmp_path / "results.jsonl"
    _write(ResultWriter(output))
    with output.open("ab") as stream:
        stream.write(b'{"partial":')

    assert list(iter_result_records(output))[0]["sample_id"] == "sample-1"
    assert list(iter_result_records(tmp_path / "missing.jsonl")) == []
