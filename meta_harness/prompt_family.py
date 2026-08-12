"""Validated, data-only prompt families for SciVer reasoning methods."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import hashlib
import json
from pathlib import Path
from string import Template
from types import MappingProxyType
from typing import Any

from utils.constant import COT_PROMPT


TEMPLATE_KEYS = ("direct", "analytical", "parallel", "sequential")
REQUIRED_PLACEHOLDERS = MappingProxyType(
    {
        "direct": frozenset({"claim", "context", "caption"}),
        "analytical": frozenset({"claim", "context", "caption"}),
        "parallel": frozenset({"claim", "context", "caption1", "caption2"}),
        "sequential": frozenset({"claim", "context", "caption1", "caption2"}),
    }
)


class InvalidPromptFamilyError(ValueError):
    """Raised when prompt-family data violates the fixed SciVer contract."""


class PromptFamily(Mapping[str, Template]):
    """Immutable validated mapping of reasoning methods to templates."""

    def __init__(self, templates: Mapping[str, Template | str]) -> None:
        if not isinstance(templates, Mapping):
            raise InvalidPromptFamilyError("prompt family must be a mapping")

        actual_keys = set(templates)
        required_keys = set(TEMPLATE_KEYS)
        if actual_keys != required_keys:
            missing = sorted(required_keys - actual_keys)
            extra = sorted(actual_keys - required_keys, key=str)
            details = []
            if missing:
                details.append(f"missing templates: {', '.join(missing)}")
            if extra:
                details.append(
                    "unexpected templates: " + ", ".join(map(str, extra))
                )
            raise InvalidPromptFamilyError(
                "prompt family requires exactly direct, analytical, parallel, "
                f"and sequential ({'; '.join(details)})"
            )

        validated: dict[str, Template] = {}
        for method in TEMPLATE_KEYS:
            value = templates[method]
            if isinstance(value, Template):
                template = value
            elif isinstance(value, str):
                template = Template(value)
            else:
                raise InvalidPromptFamilyError(
                    f"template {method!r} must be a string or string.Template"
                )
            _validate_template(method, template)
            validated[method] = template
        self._templates = MappingProxyType(validated)

    def __getitem__(self, key: str) -> Template:
        return self._templates[key]

    def __iter__(self) -> Iterator[str]:
        return iter(TEMPLATE_KEYS)

    def __len__(self) -> int:
        return len(TEMPLATE_KEYS)

    def to_json(self) -> str:
        """Serialize in one stable, human-auditable order."""

        payload = {
            method: self._templates[method].template for method in TEMPLATE_KEYS
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @classmethod
    def from_json(cls, serialized: str) -> "PromptFamily":
        """Load a prompt family from JSON data without executing code."""

        if not isinstance(serialized, str):
            raise InvalidPromptFamilyError("serialized prompt family must be text")
        try:
            payload = json.loads(serialized, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, InvalidPromptFamilyError) as exc:
            raise InvalidPromptFamilyError(
                "serialized prompt family must be a valid JSON object"
            ) from exc
        if not isinstance(payload, Mapping):
            raise InvalidPromptFamilyError(
                "serialized prompt family must be a JSON object"
            )
        return cls(payload)

    @classmethod
    def load(cls, path: str | Path) -> "PromptFamily":
        """Load and validate a UTF-8 JSON prompt-family file."""

        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def validate_prompt_family(
    templates: Mapping[str, Template | str],
) -> PromptFamily:
    """Return an immutable validated prompt family."""

    return PromptFamily(templates)


def serialize_prompt_family(
    templates: Mapping[str, Template | str],
) -> str:
    """Validate and deterministically serialize a prompt-family mapping."""

    return PromptFamily(templates).to_json()


def deserialize_prompt_family(serialized: str) -> PromptFamily:
    """Reconstruct validated Template objects from serialized JSON."""

    return PromptFamily.from_json(serialized)


def load_prompt_family(path: str | Path) -> PromptFamily:
    """Load a validated prompt family from a local JSON file."""

    return PromptFamily.load(path)


def canonical_baseline_sources() -> Mapping[str, str]:
    """Return immutable source text for the canonical ``cot`` prompt family."""

    return MappingProxyType(
        {method: COT_PROMPT[method].template for method in TEMPLATE_KEYS}
    )


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
        raise InvalidPromptFamilyError(
            "prompt-family data must be canonically JSON serializable"
        ) from exc


def template_source_sha256(templates: Mapping[str, Any]) -> str:
    """Hash exactly the four validated template source strings."""

    family = PromptFamily(templates)
    payload = {method: family[method].template for method in TEMPLATE_KEYS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _validate_template(method: str, template: Template) -> None:
    source = template.template
    if not isinstance(source, str) or not source.strip():
        raise InvalidPromptFamilyError(f"template {method!r} must not be empty")

    placeholders: set[str] = set()
    for match in template.pattern.finditer(source):
        if match.group("invalid") is not None:
            raise InvalidPromptFamilyError(
                f"template {method!r} contains malformed Template syntax"
            )
        placeholder = match.group("named") or match.group("braced")
        if placeholder is not None:
            placeholders.add(placeholder)

    required = REQUIRED_PLACEHOLDERS[method]
    if placeholders != required:
        missing = sorted(required - placeholders)
        extra = sorted(placeholders - required)
        details = []
        if missing:
            details.append(f"missing placeholders: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected placeholders: {', '.join(extra)}")
        raise InvalidPromptFamilyError(
            f"template {method!r} has an invalid placeholder set "
            f"({'; '.join(details)})"
        )

    sentinels = {name: f"__{name.upper()}__" for name in required}
    try:
        template.substitute(**sentinels)
    except (KeyError, ValueError) as exc:
        raise InvalidPromptFamilyError(
            f"template {method!r} cannot be safely substituted"
        ) from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPromptFamilyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = [
    "InvalidPromptFamilyError",
    "PromptFamily",
    "REQUIRED_PLACEHOLDERS",
    "TEMPLATE_KEYS",
    "canonical_baseline_sources",
    "canonical_json",
    "deserialize_prompt_family",
    "load_prompt_family",
    "serialize_prompt_family",
    "template_source_sha256",
    "validate_prompt_family",
]
