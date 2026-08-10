import pytest

from utils.answer_parser import parse_answer


@pytest.mark.parametrize(
    ("response", "expected_prediction"),
    [
        ("Therefore, the final answer is: Answer: yes", "yes"),
        ("Therefore, the final answer is: Answer: no", "no"),
        ("FINAL ANSWER: YES", "yes"),
        ("Therefore,  the final answer  is :  Answer :   no  ", "no"),
        ("yes", "yes"),
        ("no", "no"),
        ("The claim is yes.", "yes"),
        ("The claim is no.", "no"),
        ("The claim is **yes**.", "yes"),
        ("The claim is **no**.", "no"),
    ],
)
def test_parses_supported_answer_formats(response, expected_prediction):
    result = parse_answer(response)

    assert result["prediction"] == expected_prediction
    assert result["parse_status"] == "parsed"
    assert result["matched_text"] is not None
    assert result["parse_reason"] is None
    assert result["raw_response"] == response


def test_long_reasoning_followed_by_clear_answer():
    response = (
        "The evidence describes the same intervention and outcome. "
        "Several methodological limitations do not contradict the claim.\n\n"
        "Final answer: yes."
    )

    result = parse_answer(response)

    assert result["prediction"] == "yes"
    assert result["parse_status"] == "parsed"


def test_reasoning_with_yes_and_no_uses_clear_final_conclusion():
    response = (
        "Some observations suggest yes, while one control might suggest no. "
        "After weighing the evidence, Answer: yes."
    )

    result = parse_answer(response)

    assert result["prediction"] == "yes"
    assert result["matched_text"] == "Answer: yes"


def test_standalone_answer_line_is_preferred_over_rationale_words():
    response = (
        "yes\n\n"
        "**Rationale:** One reading initially suggests no, but that objection "
        "does not contradict the evidence."
    )

    result = parse_answer(response)

    assert result["prediction"] == "yes"
    assert result["parse_status"] == "parsed"
    assert result["matched_text"] == "yes"


def test_clear_marker_is_not_overridden_by_answer_word_in_explanation():
    response = "Answer: yes, because no contradictory result was reported."

    result = parse_answer(response)

    assert result["prediction"] == "yes"
    assert result["parse_status"] == "parsed"


def test_ambiguous_response_is_invalid():
    response = "Answer: yes or no depending on the interpretation."

    result = parse_answer(response)

    assert result["prediction"] == "invalid"
    assert result["parse_status"] == "invalid"
    assert result["matched_text"] is None
    assert "conflicting" in result["parse_reason"]
    assert result["raw_response"] == response


@pytest.mark.parametrize(
    "response",
    [
        "yes\nno",
        "The claim is yes.\nThe claim is no.",
        "The claim is yes, but another reading suggests no.",
    ],
)
def test_conflicting_new_conclusion_formats_are_invalid(response):
    result = parse_answer(response)

    assert result["prediction"] == "invalid"
    assert result["parse_status"] == "invalid"


def test_empty_response_is_invalid():
    result = parse_answer("")

    assert result == {
        "prediction": "invalid",
        "parse_status": "invalid",
        "matched_text": None,
        "parse_reason": "response is empty",
        "raw_response": "",
    }


@pytest.mark.parametrize("response", [None, 42, ["Answer: yes"]])
def test_non_string_content_is_invalid_and_preserved(response):
    result = parse_answer(response)

    assert result["prediction"] == "invalid"
    assert result["parse_status"] == "invalid"
    assert result["parse_reason"] == "response is not a string"
    assert result["raw_response"] is response


def test_reasoning_only_answer_words_are_not_classified():
    response = "The evidence could support yes, but another reading suggests no."

    result = parse_answer(response)

    assert result["prediction"] == "invalid"
    assert result["parse_reason"] == "no explicit answer marker found"


@pytest.mark.parametrize(
    "response",
    [
        "This rationale discusses yes without stating a conclusion.",
        "The analysis asks whether the claim is yes or no.",
        "A yes answer would require evidence that is not available.",
    ],
)
def test_rationale_mentions_are_not_classified(response):
    result = parse_answer(response)

    assert result["prediction"] == "invalid"
    assert result["parse_reason"] == "no explicit answer marker found"
