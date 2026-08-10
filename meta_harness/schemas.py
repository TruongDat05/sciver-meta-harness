"""Strict, data-only schemas for Meta-Harness prompt candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from meta_harness.prompt_family import (
    InvalidPromptFamilyError,
    PromptFamily,
    TEMPLATE_KEYS,
)


DEFAULT_CANDIDATE_COUNT = 2
DEFAULT_MAX_PROMPT_CHARS = 12_000
DEFAULT_ROOT_CANDIDATE_IDS = frozenset({"baseline_cot"})
SEARCH_AXES = frozenset({"exploitation", "exploration"})

_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "parent_id",
        "search_axis",
        "hypothesis",
        "templates",
        "expected_tradeoff",
        "source_sha256",
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FALSIFIABLE_LANGUAGE = re.compile(
    r"^"
    r"(?=.*\b(?:"
    r"reduc(?:e|es|ed|ing)|"
    r"increas(?:e|es|ed|ing)|"
    r"improv(?:e|es|ed|ing)|"
    r"decreas(?:e|es|ed|ing)|"
    r"lower(?:s|ed|ing)?|"
    r"rais(?:e|es|ed|ing)|"
    r"prevent(?:s|ed|ing)?|"
    r"preserv(?:e|es|ed|ing)|"
    r"fewer|more|less|higher"
    r")\b)"
    r"(?=.*\b(?:"
    r"(?:macro[- ]?)?f1|accuracy|precision|recall|"
    r"(?:request|parse|validation)\s+coverage|"
    r"(?:validation\s+)?(?:metric|score)|"
    r"(?:error|failure)\s+rate|"
    r"(?:false|unsupported|incorrect|unresolved|overly[- ]strict|"
    r"parse|parsing|api|exact)"
    r"(?:[-\s]+(?:positive|negative|yes|no|support|api|model)){0,2}"
    r"[-\s]+"
    r"(?:predictions?|decisions?|answers?|classifications?|"
    r"positives?|negatives?|errors?|failures?|inputs?)|"
    r"model\s+inputs"
    r")\b)",
    re.IGNORECASE | re.DOTALL,
)
_TERMINAL_ANSWER_PATTERNS = (
    re.compile(
        r"answer\s*:\s*yes\s*(?:/|\bor\b)\s*(?:answer\s*:\s*)?no\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"answer\s*:\s*no\s*(?:/|\bor\b)\s*(?:answer\s*:\s*)?yes\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"answer\s*:\s*\$\$answer\b",
        re.IGNORECASE,
    ),
)
_YES_NO_VOCABULARY = re.compile(
    r"\b(?:yes\s*/\s*no|yes\s+or\s+no|no\s+or\s+yes)\b",
    re.IGNORECASE,
)
_FORBIDDEN_CONTENT = (
    (
        "example, request, or paper identifiers",
        re.compile(
            r"\b(?:example|request|paper)[ _-]?(?:id|identifier)\b|"
            r"\b(?:example|request|paper)[_-][A-Za-z0-9][A-Za-z0-9._-]*|"
            r"\b(?:example|request|paper)\s*(?:#|number\s*)?\d+\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gold-answer hints",
        re.compile(
            r"\b(?:gold(?:en)?[ _-]?(?:answer|label|explanation)|"
            r"ground[ _-]?truth|known[ _-]?answer|expected[ _-]?answer|"
            r"correct[ _-]?answer\s+is|target[ _-]?label|"
            r"answer\s+is\s+(?:yes|no)|"
            r"always\s+(?:answer|respond)\s+(?:with\s+)?(?:yes|no))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credentials or authorization data",
        re.compile(
            r"\bapi[ _-]?key\b|\bauthorization\s*:|\bbearer\s+"
            r"[A-Za-z0-9._-]+",
            re.IGNORECASE,
        ),
    ),
    (
        "endpoint data",
        re.compile(
            r"\bapi[ _-]?url\b|\bendpoint\b|https?://",
            re.IGNORECASE,
        ),
    ),
    (
        "base64 or image bytes",
        re.compile(
            r"\bbase64\b|data:image/|"
            r"(?:[A-Za-z0-9+/]{128,}={0,2})",
            re.IGNORECASE,
        ),
    ),
    (
        "model changes",
        re.compile(
            r"\b(?:change|switch|replace|override|select|use\s+(?:a\s+)?"
            r"different)\b.{0,40}\bmodel\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "parser or metric changes",
        re.compile(
            r"\b(?:change|switch|replace|override|bypass|ignore|disable)\b"
            r".{0,40}\b(?:parser|metric|scoring)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "evidence reordering",
        re.compile(
            r"\b(?:reorder|reverse|shuffle|swap|ignore|omit|drop)\b"
            r".{0,40}\b(?:evidence|image|caption)\b|"
            r"\b(?:evidence|image|caption)\b.{0,40}\b"
            r"(?:reorder|reverse|shuffle|swap)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "solver-call changes",
        re.compile(
            r"\b(?:additional|extra|multiple|one|two|three|more|\d+)\s+"
            r"solver\s+calls?\b|"
            r"\b(?:call|invoke|query)\b.{0,30}\bsolver\b.{0,30}\b"
            r"(?:again|once|twice|multiple|additional|extra|\d+\s+times?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


class CandidateValidationError(ValueError):
    """Raised when candidate data violates the prompt-only contract."""


@dataclass(frozen=True)
class Candidate:
    """An immutable, validated set of four prompt-template sources."""

    candidate_id: str
    parent_id: str
    search_axis: str
    hypothesis: str
    templates: Mapping[str, str]
    expected_tradeoff: str
    source_sha256: str

    def __post_init__(self) -> None:
        _validate_identifier(self.candidate_id, "candidate_id")
        _validate_identifier(self.parent_id, "parent_id")
        if self.parent_id == self.candidate_id:
            raise CandidateValidationError(
                "parent_id must identify a different candidate"
            )
        if self.search_axis not in SEARCH_AXES:
            raise CandidateValidationError(
                "search_axis must be exploitation or exploration"
            )
        _validate_hypothesis(self.hypothesis)
        _validate_text(self.expected_tradeoff, "expected_tradeoff")

        try:
            family = PromptFamily(self.templates)
        except InvalidPromptFamilyError as exc:
            raise CandidateValidationError(
                f"templates violate the prompt-family contract: {exc}"
            ) from exc
        sources = {
            method: family[method].template for method in TEMPLATE_KEYS
        }
        _validate_candidate_text(
            self.hypothesis,
            self.expected_tradeoff,
            sources,
        )
        object.__setattr__(self, "templates", MappingProxyType(sources))

        if not isinstance(self.source_sha256, str) or not _SHA256.fullmatch(
            self.source_sha256
        ):
            raise CandidateValidationError(
                "source_sha256 must be a lowercase hexadecimal SHA-256"
            )
        expected_source_sha256 = template_source_sha256(sources)
        if self.source_sha256 != expected_source_sha256:
            raise CandidateValidationError(
                "source_sha256 does not match the canonical template sources"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    ) -> "Candidate":
        _validate_exact_mapping(value, _CANDIDATE_FIELDS, "candidate")
        candidate = cls(
            candidate_id=value["candidate_id"],
            parent_id=value["parent_id"],
            search_axis=value["search_axis"],
            hypothesis=value["hypothesis"],
            templates=value["templates"],
            expected_tradeoff=value["expected_tradeoff"],
            source_sha256=value["source_sha256"],
        )
        _validate_prompt_length(candidate.templates, max_prompt_chars)
        return candidate

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_id": self.parent_id,
            "search_axis": self.search_axis,
            "hypothesis": self.hypothesis,
            "templates": {
                method: self.templates[method] for method in TEMPLATE_KEYS
            },
            "expected_tradeoff": self.expected_tradeoff,
            "source_sha256": self.source_sha256,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateBatch:
    """A validated proposer batch with a fixed candidate count."""

    iteration: int
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or self.iteration < 0
        ):
            raise CandidateValidationError(
                "iteration must be a non-negative integer"
            )
        if (
            isinstance(self.candidates, (str, bytes))
            or not isinstance(self.candidates, Sequence)
            or any(
                not isinstance(candidate, Candidate)
                for candidate in self.candidates
            )
        ):
            raise CandidateValidationError(
                "candidates must contain validated Candidate objects"
            )
        candidates = tuple(self.candidates)
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise CandidateValidationError(
                "candidate IDs must be unique within a batch"
            )
        object.__setattr__(self, "candidates", candidates)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        candidate_count: int = DEFAULT_CANDIDATE_COUNT,
        max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
        existing_parent_ids: Iterable[str] = DEFAULT_ROOT_CANDIDATE_IDS,
    ) -> "CandidateBatch":
        _validate_exact_mapping(
            value,
            frozenset({"iteration", "candidates"}),
            "candidate batch",
        )
        iteration = value["iteration"]
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise CandidateValidationError(
                "iteration must be a non-negative integer"
            )
        if iteration < 0:
            raise CandidateValidationError(
                "iteration must be a non-negative integer"
            )
        _validate_candidate_count(candidate_count)
        raw_candidates = value["candidates"]
        if (
            isinstance(raw_candidates, (str, bytes))
            or not isinstance(raw_candidates, Sequence)
        ):
            raise CandidateValidationError("candidates must be an array")
        if len(raw_candidates) != candidate_count:
            raise CandidateValidationError(
                f"candidate batch must contain exactly {candidate_count} candidates"
            )

        candidates = tuple(
            Candidate.from_mapping(
                raw_candidate,
                max_prompt_chars=max_prompt_chars,
            )
            for raw_candidate in raw_candidates
        )
        candidate_ids = [candidate.candidate_id for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise CandidateValidationError(
                "candidate IDs must be unique within a batch"
            )
        known_parents = _validated_parent_ids(existing_parent_ids)
        for candidate in candidates:
            if candidate.parent_id not in known_parents:
                raise CandidateValidationError(
                    f"parent_id for candidate {candidate.candidate_id!r} "
                    "does not identify an existing candidate"
                )
        return cls(iteration=iteration, candidates=candidates)

    @classmethod
    def from_json(
        cls,
        serialized: str,
        **validation_options: Any,
    ) -> "CandidateBatch":
        if not isinstance(serialized, str):
            raise CandidateValidationError("candidate batch must be JSON text")
        try:
            payload = json.loads(
                serialized,
                object_pairs_hook=_unique_json_object,
            )
        except (json.JSONDecodeError, CandidateValidationError) as exc:
            raise CandidateValidationError(
                "candidate batch must contain valid JSON with unique keys"
            ) from exc
        return cls.from_mapping(payload, **validation_options)

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "candidates": [
                candidate.as_dict() for candidate in self.candidates
            ],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8-safe JSON with no insignificant whitespace."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CandidateValidationError(
            "candidate data must be canonically JSON serializable"
        ) from exc


def template_source_sha256(templates: Mapping[str, Any]) -> str:
    """Hash exactly the four validated template source strings."""

    try:
        family = PromptFamily(templates)
    except InvalidPromptFamilyError as exc:
        raise CandidateValidationError(
            f"templates violate the prompt-family contract: {exc}"
        ) from exc
    payload = {
        method: family[method].template for method in TEMPLATE_KEYS
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_candidate(
    value: Candidate | Mapping[str, Any],
    *,
    existing_parent_ids: Iterable[str] = DEFAULT_ROOT_CANDIDATE_IDS,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> Candidate:
    """Validate candidate content and its parent reference."""

    candidate = (
        value
        if isinstance(value, Candidate)
        else Candidate.from_mapping(
            value,
            max_prompt_chars=max_prompt_chars,
        )
    )
    _validate_prompt_length(candidate.templates, max_prompt_chars)
    if candidate.parent_id not in _validated_parent_ids(existing_parent_ids):
        raise CandidateValidationError(
            "parent_id does not identify an existing candidate"
        )
    return candidate


def validate_candidate_batch(
    value: CandidateBatch | Mapping[str, Any],
    *,
    candidate_count: int = DEFAULT_CANDIDATE_COUNT,
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS,
    existing_parent_ids: Iterable[str] = DEFAULT_ROOT_CANDIDATE_IDS,
) -> CandidateBatch:
    """Validate a complete proposer batch."""

    if isinstance(value, CandidateBatch):
        if len(value.candidates) != candidate_count:
            raise CandidateValidationError(
                f"candidate batch must contain exactly {candidate_count} candidates"
            )
        candidate_ids = [candidate.candidate_id for candidate in value.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise CandidateValidationError(
                "candidate IDs must be unique within a batch"
            )
        for candidate in value.candidates:
            validate_candidate(
                candidate,
                existing_parent_ids=existing_parent_ids,
                max_prompt_chars=max_prompt_chars,
            )
        return value
    return CandidateBatch.from_mapping(
        value,
        candidate_count=candidate_count,
        max_prompt_chars=max_prompt_chars,
        existing_parent_ids=existing_parent_ids,
    )


def load_candidate_batch_schema() -> dict[str, Any]:
    """Load the bundled, non-executable JSON Schema document."""

    schema_path = (
        Path(__file__).resolve().parent
        / "proposer"
        / "candidate_batch.schema.json"
    )
    try:
        schema = json.loads(
            schema_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateValidationError(
            "candidate batch schema must be readable valid JSON"
        ) from exc
    if not isinstance(schema, dict):
        raise CandidateValidationError(
            "candidate batch schema must be a JSON object"
        )
    return schema


def _validate_candidate_text(
    hypothesis: str,
    expected_tradeoff: str,
    templates: Mapping[str, str],
) -> None:
    fields = {
        "hypothesis": hypothesis,
        "expected_tradeoff": expected_tradeoff,
        **{f"templates.{name}": text for name, text in templates.items()},
    }
    for field, text in fields.items():
        for description, pattern in _FORBIDDEN_CONTENT:
            if pattern.search(text):
                raise CandidateValidationError(
                    f"{field} contains forbidden {description}"
                )
    for method, text in templates.items():
        tail = text[-800:]
        if not any(pattern.search(tail) for pattern in _TERMINAL_ANSWER_PATTERNS):
            raise CandidateValidationError(
                f"template {method!r} must end with an explicit "
                "Answer: yes/no instruction"
            )
        if re.search(r"answer\s*:\s*\$\$answer\b", tail, re.IGNORECASE):
            if not _YES_NO_VOCABULARY.search(tail):
                raise CandidateValidationError(
                    f"template {method!r} must restrict the terminal answer "
                    "to yes or no"
                )


def _validate_hypothesis(value: Any) -> None:
    _validate_text(value, "hypothesis")
    if len(value.strip()) < 12 or not _FALSIFIABLE_LANGUAGE.search(value):
        raise CandidateValidationError(
            "hypothesis must state a falsifiable expected effect with an "
            "explicit measurable direction and validation metric or error "
            "category"
        )


def _validate_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CandidateValidationError(f"{field} must be non-empty text")
    if "\x00" in value:
        raise CandidateValidationError(f"{field} must not contain NUL bytes")


def _validate_identifier(value: Any, field: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise CandidateValidationError(
            f"{field} must be a safe local identifier"
        )


def _validate_exact_mapping(
    value: Any,
    expected_fields: frozenset[str],
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise CandidateValidationError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise CandidateValidationError(f"{label} field names must be strings")
    actual_fields = set(value)
    if actual_fields == expected_fields:
        return
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields)
    details = []
    if missing:
        details.append("missing: " + ", ".join(missing))
    if extra:
        details.append("unexpected: " + ", ".join(extra))
    raise CandidateValidationError(
        f"{label} fields are invalid ({'; '.join(details)})"
    )


def _validate_prompt_length(
    templates: Mapping[str, str],
    max_prompt_chars: int,
) -> None:
    if (
        isinstance(max_prompt_chars, bool)
        or not isinstance(max_prompt_chars, int)
        or max_prompt_chars <= 0
    ):
        raise CandidateValidationError(
            "max_prompt_chars must be a positive integer"
        )
    total = sum(len(templates[method]) for method in TEMPLATE_KEYS)
    if total > max_prompt_chars:
        raise CandidateValidationError(
            f"candidate templates exceed the configured {max_prompt_chars} "
            "character limit"
        )


def _validate_candidate_count(candidate_count: int) -> None:
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise CandidateValidationError(
            "candidate_count must be a positive integer"
        )


def _validated_parent_ids(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise CandidateValidationError(
            "existing_parent_ids must be an iterable of identifiers"
        )
    try:
        normalized = frozenset(values)
    except TypeError as exc:
        raise CandidateValidationError(
            "existing_parent_ids must be an iterable of identifiers"
        ) from exc
    for value in normalized:
        _validate_identifier(value, "existing parent ID")
    return normalized


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = [
    "Candidate",
    "CandidateBatch",
    "CandidateValidationError",
    "DEFAULT_CANDIDATE_COUNT",
    "DEFAULT_MAX_PROMPT_CHARS",
    "DEFAULT_ROOT_CANDIDATE_IDS",
    "SEARCH_AXES",
    "canonical_json",
    "load_candidate_batch_schema",
    "template_source_sha256",
    "validate_candidate",
    "validate_candidate_batch",
]
