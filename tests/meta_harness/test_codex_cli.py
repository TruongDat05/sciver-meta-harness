from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pytest

from meta_harness.baseline import canonical_baseline_sources
from meta_harness.candidate_store import CandidateStore
from meta_harness.proposer.codex_cli import (
    CodexCLIProposer,
    CodexCLIProposerConfig,
    CodexCLIProposerError,
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_PROPOSER_REASONING_EFFORT,
)
from meta_harness.schemas import template_source_sha256


FIXED_TIME = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _templates(suffix: str = "") -> dict[str, str]:
    return {
        method: (
            source
            + f"\nBefore deciding, verify one decisive relation{suffix}. "
            "Choose yes or no and use exactly Answer: yes or Answer: no."
        )
        for method, source in canonical_baseline_sources().items()
    }


def _candidate(candidate_id: str, suffix: str, axis: str) -> dict:
    templates = _templates(suffix)
    return {
        "candidate_id": candidate_id,
        "parent_id": "baseline_cot",
        "search_axis": axis,
        "hypothesis": (
            "Explicit evidence checks will reduce unsupported positive decisions."
        ),
        "templates": templates,
        "expected_tradeoff": "Reasoning may require slightly more text.",
    }


def _batch() -> dict:
    return {
        "iteration": 1,
        "candidates": [
            _candidate(
                "iter_001_candidate_01",
                " before deciding",
                "exploitation",
            ),
            _candidate(
                "iter_001_candidate_02",
                " with an independent consistency pass",
                "exploration",
            ),
        ],
    }


def _search_feedback() -> dict:
    metrics = {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "parse_coverage": 1.0,
        "request_coverage": 1.0,
        "yes_precision": 1.0,
        "yes_recall": 1.0,
        "yes_f1": 1.0,
        "no_precision": 1.0,
        "no_recall": 1.0,
        "no_f1": 1.0,
    }
    confusion = {
        "actual_no": {"predicted_no": 1, "predicted_yes": 0},
        "actual_yes": {"predicted_no": 0, "predicted_yes": 0},
    }
    paired_counts = {
        "both_correct": 1,
        "corrected": 0,
        "regressed": 0,
        "still_incorrect": 0,
        "resolved_to_correct": 0,
        "resolved_to_incorrect": 0,
        "became_unresolved_from_correct": 0,
        "became_unresolved_from_incorrect": 0,
        "still_unresolved": 0,
    }
    paired_keys = {
        bucket: []
        for bucket in paired_counts
        if bucket != "both_correct"
    }
    return {
        "schema_version": 1,
        "histories": [
            {
                "history_key": "prior_search",
                "parser_version": "answer_parser_v2",
                "evaluator_version": "meta_harness_evaluator_v1",
                "dataset": "SciVer",
                "model": "gemma-4-26B-A4B-it",
                "split_sha256": "a" * 64,
                "case_count": 1,
                "case_catalog": [
                    {
                        "case_key": "case_0001",
                        "reasoning_method": "direct",
                    }
                ],
                "baseline": {
                    "metrics": metrics,
                    "confusion_matrix": confusion,
                    "failure_counts": {
                        "api_failure": 0,
                        "invalid_input": 0,
                        "parse_failure": 0,
                    },
                    "incorrect_case_keys": [],
                    "unresolved_case_keys": [],
                },
                "tried_candidates": [
                    {
                        "strategy": {
                            "candidate_id": "tried_strategy_v1",
                            "parent_id": "baseline_cot",
                            "search_axis": "exploitation",
                            "hypothesis": (
                                "A balance rule is expected to increase "
                                "validation Macro-F1 relative to baseline_cot."
                            ),
                            "expected_tradeoff": (
                                "The rule may shift class-specific recall."
                            ),
                            "source_sha256": template_source_sha256(
                                _templates(" from prior search")
                            ),
                        },
                        "metrics": metrics,
                        "confusion_matrix": confusion,
                        "failure_counts": {
                            "api_failure": 0,
                            "invalid_input": 0,
                            "parse_failure": 0,
                        },
                        "delta_vs_baseline": {
                            "metrics": {
                                field: 0.0 for field in metrics
                            },
                            "confusion_matrix": {
                                row: {column: 0 for column in columns}
                                for row, columns in confusion.items()
                            },
                        },
                        "paired_outcome_counts": paired_counts,
                        "paired_error_case_keys": paired_keys,
                    }
                ],
            }
        ],
    }


class FakeRunner:
    def __init__(
        self,
        output: str | bytes | None,
        *,
        returncode: int = 0,
        stderr: bytes = b"",
        timeout: bool = False,
        stdout: bytes = b"",
    ) -> None:
        self.output = output
        self.returncode = returncode
        self.stderr = stderr
        self.timeout = timeout
        self.stdout = stdout
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), dict(kwargs)))
        if args == ["codex", "--version"]:
            return subprocess.CompletedProcess(
                args,
                0,
                stdout=b"codex-cli 0.test.0\n",
                stderr=b"",
            )
        if self.timeout:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        if self.output is not None:
            output_index = args.index("--output-last-message") + 1
            Path(args[output_index]).write_bytes(
                self.output
                if isinstance(self.output, bytes)
                else self.output.encode("utf-8")
            )
        return subprocess.CompletedProcess(
            args,
            self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


def _proposer(runner, **config_overrides):
    return CodexCLIProposer(
        runner=runner,
        config=CodexCLIProposerConfig(**config_overrides),
        timestamp_factory=lambda: FIXED_TIME,
    )


def _propose(proposer, store):
    return proposer.propose(
        store,
        iteration=1,
        parent_candidate_ids=["baseline_cot"],
        validation_scores={
            "baseline_cot": {
                "macro_f1": 0.61,
                "accuracy": 0.64,
                "parse_coverage": 1.0,
                "unresolved_api_failures": 0,
            }
        },
        aggregate_metrics={"completed_candidates": 1, "eligible_rate": 1.0},
        failure_summaries={"parse_failures": 0, "api_failures": 0},
    )


def _audit(error: CodexCLIProposerError) -> dict:
    assert error.audit_path is not None
    return json.loads(error.audit_path.read_text(encoding="utf-8"))


def test_candidate_schema_uses_codex_0145_compatible_array_constraints():
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "meta_harness"
        / "proposer"
        / "candidate_batch.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    candidates = schema["properties"]["candidates"]

    assert candidates["minItems"] == candidates["maxItems"] == 2
    assert "uniqueItems" not in candidates


def test_valid_output_uses_fixed_noninteractive_command_and_stores_candidates(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("API_KEY", "FAKE_TEST_SECRET_DO_NOT_USE")
    monkeypatch.setenv("API_URL", "https://invalid.example.test/v1")
    runner = FakeRunner(json.dumps(_batch()))
    store = CandidateStore(tmp_path, "run_001")

    result = _propose(_proposer(runner), store)

    assert [candidate.candidate_id for candidate in result.batch.candidates] == [
        "iter_001_candidate_01",
        "iter_001_candidate_02",
    ]
    assert store.load("iter_001_candidate_01") == result.batch.candidates[0]
    assert result.batch.candidates[0].source_sha256 == template_source_sha256(
        payload_templates := _batch()["candidates"][0]["templates"]
    )
    assert payload_templates == result.batch.candidates[0].templates
    assert result.metadata["cli_version"] == "codex-cli 0.test.0"
    assert result.metadata["parent_candidate_ids"] == ["baseline_cot"]
    assert result.metadata["timestamp"] == "2026-07-27T12:00:00Z"

    assert len(runner.calls) == 2
    command, kwargs = runner.calls[1]
    assert command[:2] == ["codex", "exec"]
    assert "--ephemeral" in command
    assert "--ignore-user-config" in command
    assert "--ignore-rules" in command
    assert (
        command[command.index("--model") + 1]
        == DEFAULT_PROPOSER_MODEL
        == "gpt-5.6-terra"
    )
    assert command[command.index("--config") + 1] == (
        'model_reasoning_effort="medium"'
    )
    assert DEFAULT_PROPOSER_REASONING_EFFORT == "medium"
    assert "--json" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" in command
    assert "--output-last-message" in command
    assert command[-1] == "-"
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == command[command.index("--cd") + 1]
    assert kwargs["input"].startswith(b"You are a read-only prompt-family proposer")
    assert (
        b"<localized change> is expected to increase/decrease <named validation "
        b"metric or paired error category> relative to <parent candidate>"
        in kwargs["input"]
    )
    assert b"confusion matrices" in kwargs["input"]
    assert b"delta_vs_baseline" in kwargs["input"]
    assert b"Do not repeat, rename, combine, or intensify" in kwargs["input"]
    assert b"one localized rule" in kwargs["input"]
    assert b"final_test" not in kwargs["input"]
    assert b"gold_label" not in kwargs["input"]
    assert b"raw_response" not in kwargs["input"]
    assert b"FAKE_TEST_SECRET_DO_NOT_USE" not in kwargs["input"]
    assert "API_KEY" not in kwargs["env"]
    assert "API_URL" not in kwargs["env"]
    assert Path(kwargs["cwd"]).is_dir()
    assert Path(kwargs["cwd"]).name == "experience"

    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["status"] == "success"
    assert audit["proposer_configuration"]["model"] == "gpt-5.6-terra"
    assert audit["proposer_configuration"]["reasoning_effort"] == "medium"
    assert audit["return_code"] == 0
    assert audit["stdout_excerpt"] == ""
    assert audit["stderr_excerpt"] == ""
    assert audit["codex_error_events"] == []
    assert audit["command"][audit["command"].index("--cd") + 1] == (
        "<run_experience_directory>"
    )
    assert "FAKE_TEST_SECRET_DO_NOT_USE" not in result.audit_path.read_text()


def test_invalid_json_is_rejected_without_saving_candidates(tmp_path):
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="valid JSON") as raised:
        _propose(_proposer(FakeRunner("not-json")), store)

    assert raised.value.category == "invalid_output"
    assert _audit(raised.value)["candidate_ids"] == []
    assert not store.registry_path.exists()


@pytest.mark.parametrize("mutation", ["missing_template", "bad_placeholder"])
def test_schema_and_placeholder_violations_are_rejected(tmp_path, mutation):
    payload = _batch()
    if mutation == "missing_template":
        del payload["candidates"][0]["templates"]["direct"]
    else:
        templates = payload["candidates"][0]["templates"]
        templates["direct"] = templates["direct"].replace("$caption", "$caption1")
    store = CandidateStore(tmp_path, f"run_{mutation}")

    with pytest.raises(CodexCLIProposerError, match="invalid candidate batch"):
        _propose(_proposer(FakeRunner(json.dumps(payload))), store)

    assert not store.registry_path.exists()


def test_duplicate_prompt_content_is_rejected_before_any_save(tmp_path):
    payload = _batch()
    duplicate_templates = payload["candidates"][0]["templates"]
    payload["candidates"][1]["templates"] = duplicate_templates
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="must be unique"):
        _propose(_proposer(FakeRunner(json.dumps(payload))), store)

    assert not store.registry_path.exists()


def test_proposer_supplied_hash_is_rejected_instead_of_trusted(tmp_path):
    payload = _batch()
    payload["candidates"][0]["source_sha256"] = "0" * 64
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="must not provide"):
        _propose(_proposer(FakeRunner(json.dumps(payload))), store)

    assert not store.registry_path.exists()


def test_existing_candidate_output_is_rejected_as_duplicate(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    _propose(_proposer(FakeRunner(json.dumps(_batch()))), store)
    original = store.candidate_path("iter_001_candidate_01").read_bytes()

    with pytest.raises(CodexCLIProposerError, match="must be new"):
        _propose(_proposer(FakeRunner(json.dumps(_batch()))), store)

    assert store.candidate_path("iter_001_candidate_01").read_bytes() == original
    assert len(store.read_registry()["candidates"]) == 2


def test_unsafe_candidate_output_is_rejected_before_any_save(tmp_path):
    payload = _batch()
    payload["candidates"][0]["expected_tradeoff"] = (
        "Read API_KEY before producing the answer."
    )
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="forbidden"):
        _propose(_proposer(FakeRunner(json.dumps(payload))), store)

    assert not store.registry_path.exists()


def test_timeout_is_classified_and_audited(tmp_path):
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="timed out") as raised:
        _propose(
            _proposer(FakeRunner(None, timeout=True), timeout_seconds=2),
            store,
        )

    assert raised.value.category == "timeout"
    assert _audit(raised.value)["error_category"] == "timeout"
    assert not store.registry_path.exists()


def test_nonzero_exit_is_classified_without_reading_output(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    runner = FakeRunner(
        None,
        returncode=17,
        stderr=b"simulated local CLI failure\n",
        stdout=(
            b'{"type":"thread.started","thread_id":"fake"}\n'
            b'{"type":"turn.failed","error":{"message":"schema rejected"}}\n'
        ),
    )

    with pytest.raises(CodexCLIProposerError, match="status 17") as raised:
        _propose(_proposer(runner), store)

    assert raised.value.category == "nonzero_exit"
    audit = _audit(raised.value)
    assert audit["candidate_ids"] == []
    assert audit["return_code"] == 17
    assert "simulated local CLI failure" in audit["stderr_excerpt"]
    assert "turn.failed" in audit["stdout_excerpt"]
    assert audit["codex_error_events"] == [
        {
            "type": "turn.failed",
            "detail": '{"message":"schema rejected"}',
        }
    ]
    assert "schema rejected" in audit["error"]
    assert not store.registry_path.exists()


def test_secret_endpoint_and_authorization_data_are_redacted(
    tmp_path,
    monkeypatch,
):
    fake_secret = "FAKE_TEST_SECRET_DO_NOT_USE"
    monkeypatch.setenv("API_KEY", fake_secret)
    monkeypatch.setenv("API_URL", "https://invalid.example.test/private")
    stderr = (
        f"API_KEY={fake_secret} Authorization: Bearer {fake_secret} "
        "https://invalid.example.test/private"
    ).encode()
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError) as raised:
        _propose(
            _proposer(FakeRunner(None, returncode=1, stderr=stderr)),
            store,
        )

    audit_text = raised.value.audit_path.read_text(encoding="utf-8")
    assert fake_secret not in str(raised.value)
    assert fake_secret not in audit_text
    assert "invalid.example.test" not in audit_text
    assert "Bearer" not in audit_text
    assert "[REDACTED]" in audit_text


def test_minimal_environment_preserves_auth_paths_tls_and_proxy_only(
    tmp_path,
    monkeypatch,
):
    values = {
        "HOME": str(tmp_path / "home"),
        "CODEX_HOME": str(tmp_path / "codex-home"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "PATH": "/bin",
        "SSL_CERT_FILE": str(tmp_path / "fake-ca.pem"),
        "REQUESTS_CA_BUNDLE": str(tmp_path / "fake-bundle.pem"),
        "HTTPS_PROXY": "http://proxy.invalid.test",
        "NO_PROXY": "localhost",
        "API_KEY": "FAKE_TEST_SECRET_DO_NOT_USE",
        "API_URL": "https://solver.invalid.test",
        "UNRELATED_VARIABLE": "must-not-pass",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    runner = FakeRunner(json.dumps(_batch()))
    store = CandidateStore(tmp_path, "run_001")

    result = _propose(_proposer(runner), store)

    environment = runner.calls[1][1]["env"]
    preserved = {
        "HOME",
        "CODEX_HOME",
        "XDG_CONFIG_HOME",
        "PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "HTTPS_PROXY",
        "NO_PROXY",
    }
    assert {name: environment[name] for name in preserved} == {
        name: values[name] for name in preserved
    }
    assert "API_KEY" not in environment
    assert "API_URL" not in environment
    assert "UNRELATED_VARIABLE" not in environment
    assert result.metadata["environment_variable_names"] == sorted(environment)


def test_diagnostic_excerpts_are_bounded_and_secret_redacted(
    tmp_path,
    monkeypatch,
):
    fake_secret = "FAKE_TEST_SECRET_DO_NOT_USE"
    monkeypatch.setenv("API_KEY", fake_secret)
    runner = FakeRunner(
        None,
        returncode=1,
        stdout=(
            b'{"type":"error","message":"API_KEY='
            + fake_secret.encode()
            + b'"}\n'
            + b"x" * 10_000
        ),
        stderr=b"y" * 10_000,
    )

    with pytest.raises(CodexCLIProposerError) as raised:
        _propose(_proposer(runner), CandidateStore(tmp_path, "run_001"))

    audit = _audit(raised.value)
    assert len(audit["stdout_excerpt"]) <= 2_000
    assert len(audit["stderr_excerpt"]) <= 2_000
    assert fake_secret not in json.dumps(audit)
    assert audit["codex_error_events"][0]["type"] == "error"


def test_deterministic_fake_runner_produces_identical_prompts_and_candidates(
    tmp_path,
):
    first_runner = FakeRunner(json.dumps(_batch()))
    second_runner = FakeRunner(json.dumps(_batch()))
    first_store = CandidateStore(tmp_path / "first", "run_001")
    second_store = CandidateStore(tmp_path / "second", "run_001")

    first = _propose(_proposer(first_runner), first_store)
    second = _propose(_proposer(second_runner), second_store)

    assert first.batch.canonical_json() == second.batch.canonical_json()
    assert first_runner.calls[1][1]["input"] == second_runner.calls[1][1]["input"]
    assert first.metadata == second.metadata
    assert (
        first_store.candidate_path("iter_001_candidate_01").read_bytes()
        == second_store.candidate_path("iter_001_candidate_01").read_bytes()
    )


def test_search_feedback_reaches_prompt_without_case_labels_or_identities(
    tmp_path,
):
    runner = FakeRunner(json.dumps(_batch()))
    proposer = _proposer(runner)

    proposer.propose(
        CandidateStore(tmp_path, "run_001"),
        iteration=1,
        parent_candidate_ids=["baseline_cot"],
        search_feedback=_search_feedback(),
    )

    prompt = runner.calls[1][1]["input"]
    assert b'"candidate_id":"tried_strategy_v1"' in prompt
    assert b'"case_key":"case_0001"' in prompt
    assert b'"confusion_matrix"' in prompt
    assert b'"delta_vs_baseline"' in prompt
    assert b"gold_label" not in prompt
    assert b"sample_id" not in prompt
    assert b'"prediction"' not in prompt


def test_tried_strategy_id_is_rejected_before_storage(tmp_path):
    payload = _batch()
    payload["candidates"][0]["candidate_id"] = "tried_strategy_v1"
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="repeats a tried"):
        _proposer(FakeRunner(json.dumps(payload))).propose(
            store,
            iteration=1,
            parent_candidate_ids=["baseline_cot"],
            search_feedback=_search_feedback(),
        )

    assert not store.registry_path.exists()


def test_broad_parent_rewrite_is_rejected(tmp_path):
    payload = _batch()
    payload["candidates"][0]["templates"] = {
        "direct": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            "Check a relation. Answer: yes or Answer: no."
        ),
        "analytical": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            "Check a relation. Answer: yes or Answer: no."
        ),
        "parallel": (
            "Claim: $claim\nContext: $context\nCaption 1: $caption1\n"
            "Caption 2: $caption2\nCheck a relation. "
            "Answer: yes or Answer: no."
        ),
        "sequential": (
            "Claim: $claim\nContext: $context\nCaption 1: $caption1\n"
            "Caption 2: $caption2\nCheck a relation. "
            "Answer: yes or Answer: no."
        ),
    }

    with pytest.raises(CodexCLIProposerError, match="small edits"):
        _propose(
            _proposer(FakeRunner(json.dumps(payload))),
            CandidateStore(tmp_path, "run_001"),
        )


def test_unsafe_feedback_field_is_rejected_before_cli_invocation(tmp_path):
    runner = FakeRunner(json.dumps(_batch()))
    proposer = _proposer(runner)

    with pytest.raises(ValueError, match="unsafe field"):
        proposer.propose(
            CandidateStore(tmp_path, "run_001"),
            iteration=1,
            parent_candidate_ids=["baseline_cot"],
            aggregate_metrics={"final_test_score": 1.0},
        )

    assert runner.calls == []


def test_excessive_output_is_rejected(tmp_path):
    runner = FakeRunner(json.dumps(_batch()), stdout=b"x" * 200)
    store = CandidateStore(tmp_path, "run_001")

    with pytest.raises(CodexCLIProposerError, match="max_output_bytes") as raised:
        _propose(
            _proposer(runner, max_output_bytes=100),
            store,
        )

    assert raised.value.category == "output_too_large"
    assert not store.registry_path.exists()


def test_sol_reasoning_effort_and_jsonl_activity_are_audited(tmp_path):
    stdout = (
        b'{"type":"item.completed","item":{"type":"command_execution",'
        b'"command":"rg error traces","path":"traces/baseline_cot.jsonl"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":123,'
        b'"output_tokens":45,"total_tokens":168}}\n'
    )
    runner = FakeRunner(json.dumps(_batch()), stdout=stdout)
    store = CandidateStore(tmp_path, "run_001")

    result = _propose(
        _proposer(
            runner,
            model="gpt-5.6-sol",
            reasoning_effort="high",
        ),
        store,
    )

    command = runner.calls[1][0]
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert command[command.index("--config") + 1] == (
        'model_reasoning_effort="high"'
    )
    audited_command = result.metadata["command"]
    assert audited_command[audited_command.index("--model") + 1] == (
        "gpt-5.6-sol"
    )
    assert audited_command[audited_command.index("--config") + 1] == (
        'model_reasoning_effort="high"'
    )
    assert result.metadata["token_usage"]["total_tokens"] == 168
    assert result.metadata["tool_calls"] == ["rg error traces"]
    assert result.metadata["files_read"] == ["traces/baseline_cot.jsonl"]
    event_log = store.run_directory / result.metadata["codex_jsonl_log"]
    assert event_log.is_file()
    assert len(event_log.read_text(encoding="utf-8").splitlines()) == 2
