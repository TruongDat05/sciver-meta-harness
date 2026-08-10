import copy
import json

import pytest

from meta_harness.prompt_family import TEMPLATE_KEYS
from meta_harness.schemas import (
    Candidate,
    CandidateBatch,
    CandidateValidationError,
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_MAX_PROMPT_CHARS,
    load_candidate_batch_schema,
    template_source_sha256,
    validate_candidate,
)


def _templates(suffix=""):
    return {
        "direct": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            f"Assess support carefully{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "analytical": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            f"Check each inference carefully{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "parallel": (
            "Claim: $claim\nContext: $context\nFirst: $caption1\n"
            f"Second: $caption2\nCompare both evidence paths{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "sequential": (
            "Claim: $claim\nContext: $context\nFirst: $caption1\n"
            f"Second: $caption2\nCheck the evidence in order{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
    }


def _candidate(candidate_id="iter_001_candidate_01", **overrides):
    templates = overrides.pop("templates", _templates())
    value = {
        "candidate_id": candidate_id,
        "parent_id": "baseline_cot",
        "search_axis": "exploitation",
        "hypothesis": (
            "Explicit contradiction checks will reduce false support predictions."
        ),
        "templates": templates,
        "expected_tradeoff": "Reasoning may be slightly longer.",
        "source_sha256": template_source_sha256(templates),
    }
    value.update(overrides)
    return value


def _batch(**overrides):
    value = {
        "iteration": 1,
        "candidates": [
            _candidate("iter_001_candidate_01"),
            _candidate(
                "iter_001_candidate_02",
                search_axis="exploration",
                templates=_templates(" using an independent verification pass"),
            ),
        ],
    }
    value.update(overrides)
    return value


def test_valid_candidate_and_batch_round_trip_canonically():
    candidate = validate_candidate(_candidate())
    batch = CandidateBatch.from_mapping(_batch())
    restored = CandidateBatch.from_json(batch.canonical_json())

    assert isinstance(candidate, Candidate)
    assert tuple(candidate.templates) == TEMPLATE_KEYS
    assert restored == batch
    assert restored.canonical_json() == batch.canonical_json()
    assert len(candidate.sha256()) == 64

    with pytest.raises(TypeError):
        candidate.templates["direct"] = "mutated"


def test_json_schema_requires_fixed_batch_and_candidate_shape():
    schema = load_candidate_batch_schema()
    candidate_schema = schema["$defs"]["candidate"]

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"iteration", "candidates"}
    assert schema["properties"]["candidates"]["minItems"] == DEFAULT_CANDIDATE_COUNT
    assert schema["properties"]["candidates"]["maxItems"] == DEFAULT_CANDIDATE_COUNT
    assert candidate_schema["additionalProperties"] is False
    assert set(candidate_schema["required"]) == {
        "candidate_id",
        "parent_id",
        "search_axis",
        "hypothesis",
        "templates",
        "expected_tradeoff",
    }
    assert "source_sha256" not in candidate_schema["properties"]
    assert set(candidate_schema["properties"]["templates"]["required"]) == set(
        TEMPLATE_KEYS
    )


@pytest.mark.parametrize(
    "malformed",
    [
        {"iteration": 1},
        {"iteration": 1, "candidates": [], "unknown": True},
        {"iteration": True, "candidates": []},
    ],
)
def test_malformed_batch_schema_is_rejected(malformed):
    with pytest.raises(CandidateValidationError):
        CandidateBatch.from_mapping(malformed)


def test_unknown_candidate_field_and_duplicate_json_keys_are_rejected():
    candidate = _candidate()
    candidate["prompt_code"] = "not allowed"
    with pytest.raises(CandidateValidationError, match="unexpected"):
        Candidate.from_mapping(candidate)

    serialized = json.dumps(_batch())
    serialized = serialized.replace('"iteration": 1', '"iteration": 1, "iteration": 2')
    with pytest.raises(CandidateValidationError, match="unique keys"):
        CandidateBatch.from_json(serialized)


def test_invalid_placeholders_and_source_hash_mismatch_are_rejected():
    invalid_templates = _templates()
    invalid_templates["direct"] = invalid_templates["direct"].replace(
        "$caption",
        "$caption1",
    )
    with pytest.raises(CandidateValidationError, match="placeholder"):
        Candidate.from_mapping(
            _candidate(
                templates=invalid_templates,
                source_sha256=template_source_sha256(_templates()),
            )
        )

    candidate = _candidate()
    candidate["source_sha256"] = "0" * 64
    with pytest.raises(CandidateValidationError, match="does not match"):
        Candidate.from_mapping(candidate)


def test_duplicate_ids_and_missing_parent_are_rejected():
    batch = _batch()
    batch["candidates"][1]["candidate_id"] = batch["candidates"][0]["candidate_id"]
    with pytest.raises(CandidateValidationError, match="unique"):
        CandidateBatch.from_mapping(batch)

    with pytest.raises(CandidateValidationError, match="existing candidate"):
        CandidateBatch.from_mapping(
            _batch(),
            existing_parent_ids={"different_parent"},
        )


def test_prompt_length_limit_and_terminal_answer_contract_are_enforced():
    with pytest.raises(CandidateValidationError, match="character limit"):
        Candidate.from_mapping(
            _candidate(),
            max_prompt_chars=100,
        )

    templates = _templates()
    templates["direct"] = templates["direct"].replace(
        "Conclude with exactly Answer: yes or Answer: no.",
        "Give a concise conclusion.",
    )
    with pytest.raises(CandidateValidationError, match="Answer: yes/no"):
        Candidate.from_mapping(_candidate(templates=templates))

    assert DEFAULT_MAX_PROMPT_CHARS == 12_000


@pytest.mark.parametrize(
    ("field", "unsafe_text", "message"),
    [
        ("hypothesis", "The paper_id will reduce false predictions.", "identifiers"),
        (
            "expected_tradeoff",
            "The ground-truth label may bias the output.",
            "gold-answer",
        ),
        (
            "expected_tradeoff",
            "Read Authorization: [REDACTED].",
            "credentials",
        ),
        ("expected_tradeoff", "Read API_URL before answering.", "endpoint"),
        (
            "expected_tradeoff",
            "Embed base64 image material.",
            "base64",
        ),
    ],
)
def test_metadata_rejects_ids_gold_hints_secrets_endpoints_and_base64(
    field,
    unsafe_text,
    message,
):
    value = _candidate()
    value[field] = unsafe_text
    with pytest.raises(CandidateValidationError, match=message):
        Candidate.from_mapping(value)


@pytest.mark.parametrize(
    "instruction",
    [
        "Change the model before answering.",
        "Override the parser after answering.",
        "Replace the scoring metric.",
        "Reverse the evidence order.",
        "Make an additional solver call.",
    ],
)
def test_templates_cannot_change_fixed_solver_behavior(instruction):
    templates = _templates()
    templates["direct"] += " " + instruction
    candidate = _candidate(templates=templates)
    with pytest.raises(CandidateValidationError, match="forbidden"):
        Candidate.from_mapping(candidate)


def test_falsifiable_hypothesis_and_exact_four_templates_are_required():
    with pytest.raises(CandidateValidationError, match="falsifiable"):
        Candidate.from_mapping(_candidate(hypothesis="A nicer prompt"))

    templates = copy.deepcopy(_templates())
    del templates["parallel"]
    candidate = _candidate()
    candidate["templates"] = templates
    candidate["source_sha256"] = "0" * 64
    with pytest.raises(CandidateValidationError, match="exactly"):
        Candidate.from_mapping(candidate)


@pytest.mark.parametrize(
    ("effect", "target"),
    [
        ("reduce", "false-positive errors"),
        ("reduces", "false-positive errors"),
        ("reduced", "false-positive errors"),
        ("reducing", "false-positive errors"),
        ("increase", "validation Macro-F1"),
        ("increases", "validation Macro-F1"),
        ("increased", "validation Macro-F1"),
        ("increasing", "validation Macro-F1"),
        ("improve", "validation accuracy"),
        ("improves", "validation accuracy"),
        ("improved", "validation accuracy"),
        ("improving", "validation accuracy"),
        ("decrease", "false-negative errors"),
        ("decreases", "false-negative errors"),
        ("decreased", "false-negative errors"),
        ("decreasing", "false-negative errors"),
    ],
)
def test_hypothesis_accepts_measurable_effect_inflections(effect, target):
    hypothesis = f"A structured evidence check {effect} {target}."

    candidate = Candidate.from_mapping(_candidate(hypothesis=hypothesis))

    assert candidate.hypothesis == hypothesis


def test_hypothesis_accepts_rejected_structured_entailment_claim():
    hypothesis = (
        "A structured claim-to-evidence entailment check reduces both "
        "unsupported positive predictions and overly strict negatives."
    )

    candidate = Candidate.from_mapping(_candidate(hypothesis=hypothesis))

    assert candidate.hypothesis == hypothesis


@pytest.mark.parametrize(
    "hypothesis",
    [
        "A nicer prompt will produce better results.",
        "Structured checks improve results.",
        "Structured checks should affect validation accuracy.",
        "Structured checks will help the model.",
    ],
)
def test_hypothesis_rejects_vague_or_directionless_effects(hypothesis):
    with pytest.raises(CandidateValidationError, match="falsifiable"):
        Candidate.from_mapping(_candidate(hypothesis=hypothesis))
