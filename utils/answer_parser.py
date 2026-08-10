"""Parse a scientific claim verification answer from a model response."""

from __future__ import annotations

import re
from typing import Literal, TypedDict, cast


Prediction = Literal["yes", "no", "invalid"]
ParseStatus = Literal["parsed", "invalid"]
PARSER_VERSION = "answer_parser_v2"


class AnswerParseResult(TypedDict):
    """Structured result returned by :func:`parse_answer`."""

    prediction: Prediction
    parse_status: ParseStatus
    matched_text: str | None
    parse_reason: str | None
    raw_response: object


_SEPARATOR = r"(?:\s+is\s*[:=\-\u2013\u2014]?|\s*[:=\-\u2013\u2014])"
_ANSWER = r'''(?:[\s*_`~"'(\[])*(?P<answer>yes|no)\b'''
_LINE_ANSWER = r'''(?:[ \t*_`~"'(\[])*(?P<answer>yes|no)\b'''
_LINE_END = r'''[ \t.,;:!?*_"'`~)\]}\-\u2013\u2014]*$'''
_PATTERNS = (
    # Keep the original SciVer conclusion format as the most specific match.
    re.compile(
        rf"\btherefore\s*[,;:]?\s*the\s+final\s+answer\s+is\s*"
        rf"[:=\-\u2013\u2014]?\s*answer{_SEPARATOR}\s*{_ANSWER}",
        re.IGNORECASE,
    ),
    re.compile(rf"\bfinal\s+answer{_SEPARATOR}\s*{_ANSWER}", re.IGNORECASE),
    re.compile(rf"\banswer{_SEPARATOR}\s*{_ANSWER}", re.IGNORECASE),
    # These formats are only conclusions when they occupy a complete line.
    # Anchoring prevents ordinary mentions such as "whether the claim is yes"
    # in a rationale from becoming predictions.
    re.compile(
        rf"^[ \t]*(?:therefore\s*[,;:]?\s*)?"
        rf"the\s+claim\s+is\s*[:=\-\u2013\u2014]?\s*"
        rf"{_LINE_ANSWER}{_LINE_END}",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        rf"^[ \t]*{_LINE_ANSWER}{_LINE_END}",
        re.IGNORECASE | re.MULTILINE,
    ),
)
_TERMINAL_SUFFIX = re.compile(r'''^[\s.,;:!?*_"'`~)\]}\-\u2013\u2014]*$''')
_AMBIGUOUS_SUFFIX = re.compile(
    r"^\s*(?:[/|]\s*|[,;]?\s*(?:or|and)\s+)"
    r"(?:answer\s*[:=\-\u2013\u2014]\s*)?(?P<answer>yes|no)\b",
    re.IGNORECASE,
)
_AMBIGUOUS_CONNECTOR = re.compile(
    r"^\s*[,;]?\s*(?:or|and|[/|])\s*$", re.IGNORECASE
)


class _Candidate(TypedDict):
    start: int
    end: int
    answer_end: int
    prediction: Literal["yes", "no"]
    matched_text: str
    final_marker: bool


def _find_candidates(response: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for pattern_index, pattern in enumerate(_PATTERNS):
        for match in pattern.finditer(response):
            start, end = match.span()
            if any(
                start >= existing["start"] and end <= existing["end"]
                for existing in candidates
            ):
                # A more specific pattern already captured this same conclusion.
                continue
            prediction = cast(Literal["yes", "no"], match.group("answer").lower())
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "answer_end": match.end("answer"),
                    "prediction": prediction,
                    "matched_text": match.group(0),
                    # Preserve the historical rule that a terminal Answer/
                    # Final Answer marker resolves earlier marked conflicts.
                    # Newly accepted standalone lines do not get that power.
                    "final_marker": pattern_index < 3,
                }
            )

    return sorted(candidates, key=lambda candidate: candidate["start"])


def _invalid(raw_response: object, reason: str) -> AnswerParseResult:
    return {
        "prediction": "invalid",
        "parse_status": "invalid",
        "matched_text": None,
        "parse_reason": reason,
        "raw_response": raw_response,
    }


def _parsed(raw_response: str, candidate: _Candidate) -> AnswerParseResult:
    return {
        "prediction": candidate["prediction"],
        "parse_status": "parsed",
        "matched_text": candidate["matched_text"],
        "parse_reason": None,
        "raw_response": raw_response,
    }


def _has_ambiguous_alternative(
    response: str, candidates: list[_Candidate]
) -> bool:
    for index, candidate in enumerate(candidates):
        suffix_match = _AMBIGUOUS_SUFFIX.match(response[candidate["answer_end"] :])
        if suffix_match is not None:
            alternative = suffix_match.group("answer").lower()
            if alternative != candidate["prediction"]:
                return True

        if index + 1 < len(candidates):
            next_candidate = candidates[index + 1]
            connector = response[candidate["answer_end"] : next_candidate["start"]]
            if (
                next_candidate["prediction"] != candidate["prediction"]
                and _AMBIGUOUS_CONNECTOR.fullmatch(connector)
            ):
                return True
    return False


def parse_answer(raw_response: object) -> AnswerParseResult:
    """Normalize an explicitly marked model conclusion to yes, no, or invalid.

    A bare ``yes`` or ``no`` is accepted only when it occupies a complete
    answer line.  Likewise, ``The claim is yes/no`` must be a complete line.
    Ordinary occurrences of either word inside rationale prose are ignored.
    When explicit markers conflict, a marker at the end of the response is
    treated as the final conclusion; otherwise the response is ambiguous.
    """
    if not isinstance(raw_response, str):
        return _invalid(raw_response, "response is not a string")
    if not raw_response.strip():
        return _invalid(raw_response, "response is empty")

    candidates = _find_candidates(raw_response)
    if not candidates:
        return _invalid(raw_response, "no explicit answer marker found")
    if _has_ambiguous_alternative(raw_response, candidates):
        return _invalid(
            raw_response, "conflicting answers without a clear final conclusion"
        )

    explicit_predictions = {candidate["prediction"] for candidate in candidates}
    terminal_candidates = [
        candidate
        for candidate in candidates
        if _TERMINAL_SUFFIX.fullmatch(raw_response[candidate["answer_end"] :])
    ]
    if terminal_candidates and (
        len(explicit_predictions) == 1
        or terminal_candidates[-1]["final_marker"]
    ):
        # At most the last textual candidate can have only punctuation after it.
        return _parsed(raw_response, terminal_candidates[-1])

    if len(explicit_predictions) > 1:
        return _invalid(
            raw_response, "conflicting answers without a clear final conclusion"
        )

    # Repeated, consistent markers are harmless. Prefer the one nearest the end.
    return _parsed(raw_response, candidates[-1])


def parse_answer_response(raw_response: object) -> AnswerParseResult:
    """Descriptive alias for :func:`parse_answer`."""
    return parse_answer(raw_response)
