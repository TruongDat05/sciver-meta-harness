"""Offline contract tests for the one-candidate full SEARCH proposer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from meta_harness import prompt_proposer as pp
from meta_harness.prompt_proposer import (
    CandidateValidationError,
    ProposerInfrastructureError,
    ProposalExhausted,
    Proposer,
    ProposerConfig,
    build_prompt_proposer_input,
)
from meta_harness.config import (
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_PROPOSER_REASONING_EFFORT,
    EXPERIMENT_PROPOSAL_ATTEMPTS,
)
from meta_harness.prompt_family import canonical_baseline_sources, template_source_sha256


def _templates(suffix: str = "") -> dict[str, str]:
    ending = "\nApply one independent support check{suffix}. Respond with 'yes' or 'no'.".format(
        suffix=suffix
    )
    return {
        method: source + ending
        for method, source in canonical_baseline_sources().items()
    }


def _payload(*, candidate_id: str = "v3_candidate_001", suffix: str = "") -> dict:
    return {
        "iteration": 1,
        "candidate": {
            "candidate_id": candidate_id,
            "parent_id": "baseline_cot",
            "hypothesis": "Independent support checks may reduce ambiguous decisions.",
            "expected_tradeoff": "The added check may make responses longer.",
            "templates": _templates(suffix),
        },
    }


class FakeRunner:
    def __init__(self, outputs: list[dict | str]) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, command, **kwargs):
        self.calls.append((list(command), dict(kwargs)))
        output = self.outputs.pop(0)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(
            output if isinstance(output, str) else json.dumps(output),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


class ScriptedRunner:
    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.inputs: list[bytes] = []

    def __call__(self, command, **kwargs):
        self.inputs.append(kwargs["input"])
        output = self.payloads.pop(0)
        output_path = Path(command[command.index("--output-last-message") + 1])
        output_path.write_text(json.dumps(output), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


class SerializedRunner:
    def __init__(self, outputs: list) -> None:
        self.outputs = list(outputs)
        self.inputs: list[bytes] = []

    def __call__(self, command, **kwargs):
        self.inputs.append(kwargs["input"])
        output = self.outputs.pop(0)
        output_path = Path(command[command.index("--output-last-message") + 1])
        if isinstance(output, str):
            output_path.write_text(output, encoding="utf-8")
        else:
            output_path.write_text(json.dumps(output), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")


def _propose(tmp_path, runner, **kwargs):
    kwargs.setdefault("lineage", [])
    return Proposer(runner=runner).propose(
        proposal_directory=tmp_path,
        run_id="offline_v3",
        iteration=1,
        aggregate_search_metrics={"macro_f1": 0.6, "accuracy": 0.65},
        representative_search_failures=[
            {
                "pattern": "ambiguous_support",
                "summary": "Ambiguous support relationships recur across methods.",
                "count": 4,
                "methods": ["direct", "parallel"],
            }
        ],
        **kwargs,
    )


def test_valid_one_candidate_output_is_accepted_and_receipted(tmp_path):
    runner = FakeRunner([_payload()])

    result = _propose(tmp_path, runner)

    assert result.attempt == 1
    assert result.candidate.candidate_id == "v3_candidate_001"
    assert set(result.candidate.templates) == {
        "direct",
        "analytical",
        "parallel",
        "sequential",
    }
    assert result.candidate.source_sha256 == template_source_sha256(_templates())
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "accepted"
    assert receipt["candidate_id"] == "v3_candidate_001"
    command, kwargs = runner.calls[0]
    assert command[:2] == ["codex", "exec"]
    assert command[command.index("--sandbox") + 1] == "danger-full-access"
    assert kwargs["shell"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_missing_or_extra_template_keys_are_rejected_before_any_solver_call(
    tmp_path, mutation
):
    payload = _payload()
    if mutation == "missing":
        del payload["candidate"]["templates"]["direct"]
    else:
        payload["candidate"]["templates"]["extra"] = "not allowed"
    runner = FakeRunner([payload, payload, payload])
    solver_calls = 0

    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner)

    assert solver_calls == 0
    receipts = sorted(
        (tmp_path / "workspace" / "meta_harness" / "full_search_v3" / "offline_v3" / "proposals" / "iteration_0001").glob("attempt_*.json")
    )
    assert len(receipts) == 3
    assert all(json.loads(path.read_text())["status"] == "rejected" for path in receipts)


def test_placeholder_mutation_is_rejected_before_any_solver_call(tmp_path):
    payload = _payload()
    payload["candidate"]["templates"]["direct"] = payload["candidate"][
        "templates"
    ]["direct"].replace("$caption", "$caption1")
    runner = FakeRunner([payload, payload, payload])

    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner)

    assert len(runner.calls) == 3


def test_duplicate_proposal_is_rejected_before_any_solver_call(tmp_path):
    payload = _payload()
    runner = FakeRunner([payload, payload, payload])
    duplicate_hash = template_source_sha256(payload["candidate"]["templates"])
    solver_calls = 0

    with pytest.raises(ProposalExhausted):
        _propose(
            tmp_path,
            runner,
            existing_candidate_ids=["earlier_candidate"],
            existing_source_sha256=[duplicate_hash],
        )

    assert solver_calls == 0
    receipt_path = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_v3"
        / "proposals"
        / "iteration_0001"
        / "attempt_00003.json"
    )
    assert json.loads(receipt_path.read_text())["category"] == "duplicate_prompt_content"


def test_three_attempt_exhaustion_records_each_rejection(tmp_path):
    invalid = _payload()
    invalid["candidate"]["templates"] = {"direct": "invalid"}
    runner = FakeRunner([invalid, invalid, invalid])

    with pytest.raises(ProposalExhausted, match="three invalid"):
        _propose(tmp_path, runner)

    assert len(runner.calls) == 3


def test_rejected_attempt_receipts_resume_without_repeating_an_attempt(tmp_path):
    invalid = _payload()
    invalid["candidate"]["templates"] = {"direct": "invalid"}

    class InterruptingRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, command, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated proposer interruption")
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text(json.dumps(invalid), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    interrupted = InterruptingRunner()
    with pytest.raises(RuntimeError, match="simulated proposer interruption"):
        _propose(tmp_path, interrupted)

    resumed = FakeRunner([invalid, invalid])
    with pytest.raises(ProposalExhausted, match="three invalid"):
        _propose(tmp_path, resumed)

    receipts = sorted(
        (
            tmp_path
            / "workspace"
            / "meta_harness"
            / "full_search_v3"
            / "offline_v3"
            / "proposals"
            / "iteration_0001"
        ).glob("attempt_*.json")
    )
    assert [path.name for path in receipts] == [
        "attempt_00001.json",
        "attempt_00002.json",
        "attempt_00003.json",
    ]
    assert len(resumed.calls) == 2


def test_infrastructure_failure_receipt_retries_without_consuming_invalid_budget(tmp_path):
    class UnavailableRunner:
        def __call__(self, command, **kwargs):
            raise OSError("simulated unavailable proposer")

    with pytest.raises(ProposerInfrastructureError):
        _propose(tmp_path, UnavailableRunner())

    resumed = FakeRunner([_payload()])
    result = _propose(tmp_path, resumed)

    assert result.attempt == 2
    receipt_directory = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_v3"
        / "proposals"
        / "iteration_0001"
    )
    receipt = json.loads((receipt_directory / "attempt_00002.json").read_text())
    assert receipt["status"] == "accepted"
    assert len(resumed.calls) == 1


@pytest.mark.parametrize("run_id", [".", "..", "-leading", "space run"])
def test_proposer_rejects_run_ids_outside_the_shared_contract(tmp_path, run_id):
    with pytest.raises(CandidateValidationError, match="run_id"):
        Proposer(runner=FakeRunner([_payload()])).propose(
            proposal_directory=tmp_path,
            run_id=run_id,
            iteration=1,
        )


def test_proposer_input_is_sanitized_and_rejects_prohibited_failure_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("API_KEY", "FAKE_TEST_SECRET_DO_NOT_USE")
    runner = FakeRunner([_payload()])

    _propose(tmp_path, runner)

    serialized = runner.calls[0][1]["input"].decode("utf-8")
    assert "FAKE_TEST_SECRET_DO_NOT_USE" not in serialized
    assert "gold_label" not in serialized
    assert "PRIVATE_FINAL_RECORD" not in serialized
    assert "sample_id" not in serialized
    assert "raw_trace" not in serialized
    assert "aggregate_search_metrics" in serialized
    assert "representative_search_failures" in serialized

    with pytest.raises(ValueError, match="prohibited"):
        build_prompt_proposer_input(
            iteration=1,
            parent_id="baseline_cot",
            parent_templates=canonical_baseline_sources(),
            aggregate_search_metrics={},
            lineage=[],
            representative_search_failures=[
                {
                    "pattern": "private_final_record",
                    "summary": "This must never be visible.",
                    "count": 1,
                    "methods": ["direct"],
                }
            ],
        )
    assert len(runner.calls) == 1


def test_proposer_defaults_to_sol_model_with_high_reasoning_effort(tmp_path):
    assert DEFAULT_PROPOSER_MODEL == "gpt-5.6-sol"
    assert DEFAULT_PROPOSER_REASONING_EFFORT == "high"
    assert ProposerConfig().model == "gpt-5.6-sol"
    assert ProposerConfig().reasoning_effort == "high"
    assert EXPERIMENT_PROPOSAL_ATTEMPTS == 3

    runner = FakeRunner([_payload()])
    _propose(tmp_path, runner)
    command, _ = runner.calls[0]
    assert command[command.index("--model") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="high"' in command


def test_prompt_explicitly_requires_json_schema_and_frozen_contract(tmp_path):
    runner = FakeRunner([_payload()])
    _propose(tmp_path, runner)
    serialized = runner.calls[0][1]["input"].decode("utf-8")

    assert "code fences" in serialized
    assert "Answer: $$ANSWER" in serialized
    assert "$claim" in serialized and "$caption1" in serialized
    assert "globally unique" in serialized
    assert "All four templates" in serialized
    assert "materially distinct" in serialized
    assert "falsifiable" in serialized
    assert "no Markdown" in serialized


def test_lineage_metrics_and_deltas_appear_in_prompt_history(tmp_path):
    lineage = [
        {
            "candidate_id": "earlier_ablation",
            "parent_id": "cot",
            "source_sha256": "a" * 64,
            "macro_f1": 0.79,
            "accuracy": 0.8,
            "delta_macro_f1": -0.033,
            "delta_accuracy": -0.024,
        }
    ]
    runner = FakeRunner([_payload()])
    _propose(tmp_path, runner, lineage=lineage)
    serialized = runner.calls[0][1]["input"].decode("utf-8")

    assert "earlier_ablation" in serialized
    assert "macro_f1" in serialized
    assert "delta_macro_f1" in serialized
    assert "0.79" in serialized
    assert "-0.033" in serialized


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_id", "duplicate_content", "placeholder"],
)
def test_rejection_code_and_retry_feedback_for_each_category(tmp_path, mutation):
    payload = _payload()
    kwargs: dict = {}
    if mutation == "duplicate_id":
        kwargs = dict(existing_candidate_ids=["v3_candidate_001"])
        code = "duplicate_candidate_id"
    elif mutation == "duplicate_content":
        kwargs = dict(existing_source_sha256=[template_source_sha256(_templates())])
        code = "duplicate_prompt_content"
    else:
        payload["candidate"]["templates"]["direct"] = payload["candidate"][
            "templates"
        ]["direct"].replace("$caption", "$caption1")
        code = "placeholder_contract"

    runner = ScriptedRunner([dict(payload), dict(payload), dict(payload)])
    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner, **kwargs)

    assert len(runner.inputs) == 3
    assert b"REJECTION_FEEDBACK" not in runner.inputs[0]
    for inp in runner.inputs[1:]:
        assert b"REJECTION_FEEDBACK" in inp
        assert code.encode("utf-8") in inp
    receipt = json.loads(
        (
            tmp_path
            / "workspace"
            / "meta_harness"
            / "full_search_v3"
            / "offline_v3"
            / "proposals"
            / "iteration_0001"
            / "attempt_00003.json"
        ).read_text()
    )
    assert receipt["category"] == code


def test_retry_feedback_is_only_injected_after_a_rejection(tmp_path):
    class TwoCallRunner:
        def __init__(self):
            self.inputs: list[bytes] = []

        def __call__(self, command, **kwargs):
            self.inputs.append(kwargs["input"])
            output_path = Path(command[command.index("--output-last-message") + 1])
            if len(self.inputs) == 1:
                output_path.write_text("this is not valid JSON", encoding="utf-8")
            else:
                output_path.write_text(json.dumps(_payload()), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    runner = TwoCallRunner()
    result = _propose(tmp_path, runner)

    assert result.attempt == 2
    assert len(runner.inputs) == 2
    assert b"REJECTION_FEEDBACK" not in runner.inputs[0]
    assert b"REJECTION_FEEDBACK" in runner.inputs[1]
    assert b"invalid_output" in runner.inputs[1]


def test_resume_replays_durable_rejection_feedback(tmp_path):
    invalid = "this is not valid JSON".encode("utf-8")

    class InterruptingRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, command, **kwargs):
            self.calls += 1
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_bytes(invalid)
            if self.calls == 2:
                raise RuntimeError("simulated interrupt after first rejection")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    with pytest.raises(RuntimeError, match="interrupt"):
        _propose(tmp_path, InterruptingRunner())

    resumed_inputs: list[bytes] = []

    class ResumingRunner:
        def __init__(self):
            self.calls = 0

        def __call__(self, command, **kwargs):
            self.calls += 1
            resumed_inputs.append(kwargs["input"])
            output_path = Path(command[command.index("--output-last-message") + 1])
            if self.calls == 1:
                output_path.write_bytes(invalid)
            else:
                output_path.write_text(json.dumps(_payload()), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    result = _propose(tmp_path, ResumingRunner())

    assert result.attempt == 3
    assert b"REJECTION_FEEDBACK" in resumed_inputs[0]
    assert b"invalid_output" in resumed_inputs[0]


def test_proposer_receipt_schema_and_instruction_versions_are_current():
    assert pp.EXPERIMENT_PROPOSER_SCHEMA_VERSION == 2
    assert (
        pp.EXPERIMENT_PROPOSER_INSTRUCTION_VERSION
        == "sciver_full_search_v3_proposer_v2"
    )


def test_attempt_prompt_hash_is_persisted_and_matches_sent_prompt(tmp_path):
    class TwoCallRunner:
        def __init__(self):
            self.inputs: list[bytes] = []

        def __call__(self, command, **kwargs):
            self.inputs.append(kwargs["input"])
            output_path = Path(command[command.index("--output-last-message") + 1])
            if len(self.inputs) == 1:
                output_path.write_text("not json", encoding="utf-8")
            else:
                output_path.write_text(json.dumps(_payload()), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    runner = TwoCallRunner()
    result = _propose(tmp_path, runner)

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["attempt_prompt_sha256"] == hashlib.sha256(
        runner.inputs[1]
    ).hexdigest()
    assert receipt["input_sha256"] == hashlib.sha256(
        runner.inputs[0]
    ).hexdigest()


def test_attempt_prompt_max_input_bytes_is_enforced(tmp_path):
    envelope = build_prompt_proposer_input(
        iteration=1,
        parent_id="baseline_cot",
        parent_templates=canonical_baseline_sources(),
        aggregate_search_metrics={"macro_f1": 0.6, "accuracy": 0.65},
        lineage=[],
        representative_search_failures=[],
    )
    base_bytes = len(pp._build_prompt(envelope).encode("utf-8"))
    config = ProposerConfig(max_input_bytes=base_bytes + 10)

    class RejectingRunner:
        def __call__(self, command, **kwargs):
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("not json", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    with pytest.raises(ValueError, match="attempt prompt exceeds max_input_bytes"):
        Proposer(runner=RejectingRunner(), config=config).propose(
            proposal_directory=tmp_path,
            run_id="offline_v3",
            iteration=1,
        )


def test_resume_fails_closed_on_tampered_attempt_prompt_hash(tmp_path):
    class RejectingRunner:
        def __call__(self, command, **kwargs):
            output_path = Path(command[command.index("--output-last-message") + 1])
            output_path.write_text("not json", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    runner = RejectingRunner()
    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner)

    directory = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_v3"
        / "proposals"
        / "iteration_0001"
    )
    first = sorted(directory.glob("attempt_*.json"))[0]
    record = json.loads(first.read_text(encoding="utf-8"))
    record["attempt_prompt_sha256"] = "f" * 64
    first.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ProposerInfrastructureError, match="prompt identity"):
        _propose(tmp_path, RejectingRunner())


def test_resume_fails_closed_on_legacy_receipt_version(tmp_path):
    directory = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / "offline_v3"
        / "proposals"
        / "iteration_0001"
    )
    directory.mkdir(parents=True)
    (directory / "attempt_00001.json").write_text(
        json.dumps({"schema_version": 1}), encoding="utf-8"
    )

    with pytest.raises(ProposerInfrastructureError, match="incompatible"):
        _propose(tmp_path, FakeRunner([_payload()]))


def test_prompt_states_exact_placeholder_sets_per_method(tmp_path):
    runner = FakeRunner([_payload()])
    _propose(tmp_path, runner)
    serialized = runner.calls[0][1]["input"].decode("utf-8")

    assert (
        "direct and analytical templates must contain exactly "
        "$claim, $context, and $caption" in serialized
    )
    assert (
        "parallel and sequential templates must contain exactly "
        "$claim, $context, $caption1, and $caption2" in serialized
    )


_REJECTION_CODE_BY_KIND = {
    "invalid_output": "invalid_output",
    "top_level_structure": "top_level_structure",
    "iteration": "iteration",
    "parent": "parent",
    "metadata": "metadata",
    "template_keys": "template_keys",
    "placeholder": "placeholder_contract",
    "unchanged": "unchanged_template",
    "answer": "answer_contract",
    "prohibited": "prohibited_content",
    "duplicate_id": "duplicate_candidate_id",
    "duplicate_content": "duplicate_prompt_content",
}


def _output_for_kind(kind: str, payload: dict):
    if kind == "invalid_output":
        return "not json"
    if kind == "top_level_structure":
        return {"iteration": 1, "wrong": True}
    if kind == "iteration":
        adjusted = dict(payload)
        adjusted["iteration"] = 99
        return adjusted
    adjusted = json.loads(json.dumps(payload))
    candidate = adjusted["candidate"]
    if kind == "parent":
        candidate["parent_id"] = "wrong_parent"
    elif kind == "metadata":
        candidate["candidate_id"] = "!!!not-valid-id"
    elif kind == "template_keys":
        del candidate["templates"]["direct"]
    elif kind == "placeholder":
        candidate["templates"]["direct"] = candidate["templates"]["direct"].replace(
            "$caption", "$caption1"
        )
    elif kind == "unchanged":
        candidate["templates"] = dict(canonical_baseline_sources())
    elif kind == "answer":
        candidate["templates"]["direct"] = candidate["templates"]["direct"].replace(
            "$$ANSWER", "$$answer"
        )
    elif kind == "prohibited":
        candidate["templates"]["direct"] = (
            candidate["templates"]["direct"] + "\ngold_label"
        )
    return adjusted


@pytest.mark.parametrize("kind", sorted(_REJECTION_CODE_BY_KIND))
def test_typed_rejection_categories_and_retry_feedback(tmp_path, kind):
    code = _REJECTION_CODE_BY_KIND[kind]
    payload = _payload()
    kwargs: dict = {}
    if kind == "duplicate_id":
        kwargs = dict(existing_candidate_ids=["v3_candidate_001"])
    elif kind == "duplicate_content":
        kwargs = dict(existing_source_sha256=[template_source_sha256(_templates())])

    outputs = [_output_for_kind(kind, payload) for _ in range(3)]
    runner = SerializedRunner(outputs)
    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner, **kwargs)

    assert len(runner.inputs) == 3
    for attempt_input in runner.inputs[1:]:
        assert b"REJECTION_FEEDBACK" in attempt_input
        assert code.encode("utf-8") in attempt_input
    receipt = json.loads(
        (
            tmp_path
            / "workspace"
            / "meta_harness"
            / "full_search_v3"
            / "offline_v3"
            / "proposals"
            / "iteration_0001"
            / "attempt_00003.json"
        ).read_text()
    )
    assert receipt["category"] == code


@pytest.mark.parametrize("label", ["yes", "no"])
def test_missing_either_answer_label_is_rejected_as_answer_contract(tmp_path, label):
    payload = _payload()
    payload["candidate"]["templates"]["direct"] = payload["candidate"][
        "templates"
    ]["direct"].replace(label, "zz")
    runner = SerializedRunner([payload, payload, payload])

    with pytest.raises(ProposalExhausted):
        _propose(tmp_path, runner)

    assert b"REJECTION_FEEDBACK" in runner.inputs[1]
    assert b"answer_contract" in runner.inputs[1]
    receipt = json.loads(
        (
            tmp_path
            / "workspace"
            / "meta_harness"
            / "full_search_v3"
            / "offline_v3"
            / "proposals"
            / "iteration_0001"
            / "attempt_00003.json"
        ).read_text()
    )
    assert receipt["category"] == "answer_contract"


def _recovery_envelope():
    return build_prompt_proposer_input(
        iteration=1,
        parent_id="baseline_cot",
        parent_templates=canonical_baseline_sources(),
        aggregate_search_metrics={"macro_f1": 0.6, "accuracy": 0.65},
        lineage=[],
        representative_search_failures=[],
    )


def _recovery_candidate(parent="baseline_cot", **overrides):
    templates = {
        method: source + "\nRecovered check. Respond with 'yes' or 'no'."
        for method, source in canonical_baseline_sources().items()
    }
    candidate = {
        "candidate_id": "recovered_candidate",
        "parent_id": parent,
        "hypothesis": "Recovered hypothesis.",
        "expected_tradeoff": "Recovered tradeoff.",
        "templates": templates,
        "source_sha256": template_source_sha256(templates),
    }
    candidate.update(overrides)
    return candidate


def _receipt_chain(envelope, *, prior=(), candidate_factory=None, parent="baseline_cot"):
    base = pp._build_prompt(envelope)
    base_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()
    spec = list(prior) + ["accepted"]
    categories: list[str] = []
    records = []
    for index, status in enumerate(spec, start=1):
        feedback = pp._ordered_feedback(categories)
        prompt = pp._build_prompt(envelope, tuple(feedback)) if feedback else base
        attempt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        category = None
        candidate = None
        if status == "rejected":
            category = "invalid_output"
        elif status == "infrastructure_failure":
            category = "subprocess"
        elif status == "accepted":
            candidate = (
                candidate_factory()
                if candidate_factory
                else _recovery_candidate(parent=parent)
            )
        records.append(
            {
                "schema_version": pp.EXPERIMENT_PROPOSER_SCHEMA_VERSION,
                "protocol_id": pp.EXPERIMENT_PROTOCOL_ID,
                "instruction_version": pp.EXPERIMENT_PROPOSER_INSTRUCTION_VERSION,
                "timestamp": "2026-01-01T00:00:00Z",
                "status": status,
                "iteration": envelope["iteration"],
                "attempt": index,
                "input_sha256": base_hash,
                "attempt_prompt_sha256": attempt_hash,
                "category": category,
                "candidate_id": None if candidate is None else candidate["candidate_id"],
                "candidate_source_sha256": (
                    None if candidate is None else candidate["source_sha256"]
                ),
                "candidate": candidate,
            }
        )
        if status == "rejected":
            categories.append(category)
    return records


def _write_receipts(directory, records, *, mutate=None):
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        if mutate is not None:
            mutate(record, index)
        path = directory / f"attempt_{index:05d}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
    return directory


def _recover(tmp_path, envelope, records, **kwargs):
    directory = tmp_path / "proposals" / "iter"
    mutate = kwargs.pop("mutate", None)
    _write_receipts(directory, records, mutate=mutate)
    return pp.recover_accepted_experiment_proposal(
        directory,
        expected_iteration=1,
        expected_parent_id="baseline_cot",
        parent_templates=canonical_baseline_sources(),
        envelope=envelope,
        **kwargs,
    )


def test_recover_accepted_chain_happy_path(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope)

    result = _recover(tmp_path, envelope, records)

    assert result.attempt == 1
    assert result.candidate.candidate_id == "recovered_candidate"


def test_recover_accepted_chain_after_rejection_feedback(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected"])

    result = _recover(tmp_path, envelope, records)

    assert result.attempt == 2
    assert result.candidate.candidate_id == "recovered_candidate"


def test_recover_returns_none_when_no_attempt_accepted(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected", "rejected"])[:-1]

    result = _recover(tmp_path, envelope, records)

    assert result is None


def test_recover_fails_on_missing_attempt_prompt_hash(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope)

    def drop(record, _index):
        del record["attempt_prompt_sha256"]

    with pytest.raises(ProposerInfrastructureError, match="incompatible"):
        _recover(tmp_path, envelope, records, mutate=drop)


def test_recover_fails_on_tampered_attempt_prompt_hash(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope)

    def tamper(record, _index):
        record["attempt_prompt_sha256"] = "f" * 64

    with pytest.raises(ProposerInfrastructureError, match="prompt identity"):
        _recover(tmp_path, envelope, records, mutate=tamper)


def test_recover_fails_on_altered_rejection_chain(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected"])

    def alter(record, index):
        if index == 1:
            record["category"] = "placeholder_contract"

    with pytest.raises(ProposerInfrastructureError):
        _recover(tmp_path, envelope, records, mutate=alter)


def test_recover_fails_on_wrong_parent(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(
        envelope, candidate_factory=lambda: _recovery_candidate(parent="wrong_parent")
    )

    with pytest.raises(ProposerInfrastructureError, match="candidate is invalid"):
        _recover(tmp_path, envelope, records)


def test_recover_fails_on_invalid_templates(tmp_path):
    envelope = _recovery_envelope()

    def invalid_templates():
        bad = _recovery_candidate()
        bad["templates"] = {
            "direct": "\nClaim: $claim\nContext: $context\nCaption: $caption\nAssess the claim.\n",
            "analytical": "\nClaim: $claim\nContext: $context\nCaption: $caption\nAssess the claim.\n",
            "parallel": (
                "\nClaim: $claim\nContext: $context\nitem1 Caption: $caption1\n"
                "item2 Caption: $caption2\nAssess the claim.\n"
            ),
            "sequential": (
                "\nClaim: $claim\nContext: $context\nitem1 Caption: $caption1\n"
                "item2 Caption: $caption2\nAssess the claim.\n"
            ),
        }
        bad["source_sha256"] = template_source_sha256(bad["templates"])
        return bad

    records = _receipt_chain(envelope, candidate_factory=invalid_templates)

    with pytest.raises(ProposerInfrastructureError, match="candidate is invalid"):
        _recover(tmp_path, envelope, records)


def test_recover_fails_on_non_contiguous_chain(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected", "rejected"])

    directory = tmp_path / "proposals" / "iter"
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        if index == 2:
            continue
        (directory / f"attempt_{index:05d}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )

    with pytest.raises(ProposerInfrastructureError, match="contiguous"):
        pp.recover_accepted_experiment_proposal(
            directory,
            expected_iteration=1,
            expected_parent_id="baseline_cot",
            parent_templates=canonical_baseline_sources(),
            envelope=envelope,
        )


def test_load_accepted_proposal_rejects_mirror_mismatch(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope)
    directory = tmp_path / "proposals" / "iter"
    _write_receipts(
        directory,
        records,
        mutate=lambda rec, _i: rec.update({"candidate_id": "other_id"}),
    )

    with pytest.raises(ProposerInfrastructureError, match="mirrors"):
        pp.recover_accepted_experiment_proposal(
            directory,
            expected_iteration=1,
            expected_parent_id="baseline_cot",
            parent_templates=canonical_baseline_sources(),
            envelope=envelope,
        )


@pytest.mark.parametrize("field", ["candidate_id", "candidate_source_sha256"])
def test_recover_rejects_tampered_candidate_mirror_in_rejected_receipt(
    tmp_path, field
):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected"])

    def tamper(record, index):
        if index == 1:
            record[field] = "z" * 64

    with pytest.raises(ProposerInfrastructureError, match="incompatible"):
        _recover(tmp_path, envelope, records, mutate=tamper)


@pytest.mark.parametrize("field", ["candidate_id", "candidate_source_sha256"])
def test_recover_rejects_tampered_candidate_mirror_in_infrastructure_receipt(
    tmp_path, field
):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["infrastructure_failure"])

    def tamper(record, index):
        if index == 1:
            record[field] = "z" * 64

    with pytest.raises(ProposerInfrastructureError, match="incompatible"):
        _recover(tmp_path, envelope, records, mutate=tamper)


def test_recover_fails_closed_when_rejected_exceeds_three(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected"] * 4)

    with pytest.raises(ProposerInfrastructureError, match="exceed the limit"):
        _recover(tmp_path, envelope, records)


def test_recover_infrastructure_only_does_not_consume_rejection_budget(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["infrastructure_failure"] * 5)

    result = _recover(tmp_path, envelope, records)

    assert result is not None
    assert result.attempt == len(records)
    assert result.candidate.candidate_id == "recovered_candidate"


def test_recover_two_rejections_with_infrastructure_then_acceptance(tmp_path):
    envelope = _recovery_envelope()
    prior = [
        "rejected",
        "infrastructure_failure",
        "rejected",
        "infrastructure_failure",
    ]
    records = _receipt_chain(envelope, prior=prior)

    result = _recover(tmp_path, envelope, records)

    assert result is not None
    assert result.attempt == len(records)
    assert result.candidate.candidate_id == "recovered_candidate"
    assert result.candidate.parent_id == "baseline_cot"


def test_recover_fails_closed_on_three_rejections_then_acceptance(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected"] * 3)

    with pytest.raises(ProposerInfrastructureError, match="incompatible"):
        _recover(tmp_path, envelope, records)


def test_recover_exactly_three_rejections_without_accepted_returns_none(tmp_path):
    envelope = _recovery_envelope()
    records = _receipt_chain(envelope, prior=["rejected"] * 3)[:-1]

    result = _recover(tmp_path, envelope, records)

    assert result is None
