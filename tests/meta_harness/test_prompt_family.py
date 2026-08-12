from string import Template

import pytest

from meta_harness.prompt_family import (
    InvalidPromptFamilyError,
    PromptFamily,
    REQUIRED_PLACEHOLDERS,
    TEMPLATE_KEYS,
    canonical_baseline_sources,
    deserialize_prompt_family,
    serialize_prompt_family,
    template_source_sha256,
)
from utils.constant import COT_PROMPT


def _sources():
    return {name: template.template for name, template in COT_PROMPT.items()}


def test_valid_family_accepts_strings_and_templates():
    sources = _sources()
    sources["direct"] = Template(sources["direct"])

    family = PromptFamily(sources)

    assert tuple(family) == TEMPLATE_KEYS
    assert all(isinstance(family[name], Template) for name in TEMPLATE_KEYS)
    assert family["direct"] is sources["direct"]


def test_serialization_is_deterministic_and_round_trips_templates():
    first = serialize_prompt_family(_sources())
    second = PromptFamily(_sources()).to_json()

    assert first == second
    restored = deserialize_prompt_family(first)
    assert {
        name: restored[name].template for name in TEMPLATE_KEYS
    } == _sources()


def test_canonical_sources_and_hash_are_derived_from_unchanged_cot():
    sources = canonical_baseline_sources()

    assert sources == _sources()
    assert template_source_sha256(sources) == template_source_sha256(COT_PROMPT)
    with pytest.raises(TypeError):
        sources["direct"] = "changed"


@pytest.mark.parametrize("missing", TEMPLATE_KEYS)
def test_missing_template_is_rejected(missing):
    sources = _sources()
    del sources[missing]

    with pytest.raises(InvalidPromptFamilyError, match="missing templates"):
        PromptFamily(sources)


def test_extra_template_is_rejected():
    sources = _sources()
    sources["extra"] = "Claim: $claim"

    with pytest.raises(InvalidPromptFamilyError, match="unexpected templates"):
        PromptFamily(sources)


@pytest.mark.parametrize("method", TEMPLATE_KEYS)
def test_missing_placeholder_is_rejected(method):
    sources = _sources()
    missing = next(iter(REQUIRED_PLACEHOLDERS[method]))
    sources[method] = sources[method].replace(f"${missing}", "fixed text")

    with pytest.raises(InvalidPromptFamilyError, match="missing placeholders"):
        PromptFamily(sources)


@pytest.mark.parametrize("method", TEMPLATE_KEYS)
def test_extra_placeholder_is_rejected(method):
    sources = _sources()
    sources[method] += "\nUnexpected: $extra"

    with pytest.raises(InvalidPromptFamilyError, match="unexpected placeholders"):
        PromptFamily(sources)


@pytest.mark.parametrize(
    "invalid_source",
    [
        "Claim: $claim\nContext: $context\nCaption: $caption\nMalformed: $",
        "Claim: $claim\nContext: $context\nCaption: $caption\nMalformed: ${open",
    ],
)
def test_malformed_template_syntax_is_rejected(invalid_source):
    sources = _sources()
    sources["direct"] = invalid_source

    with pytest.raises(InvalidPromptFamilyError, match="malformed Template syntax"):
        PromptFamily(sources)


@pytest.mark.parametrize("empty_source", ["", " \n\t"])
def test_empty_prompt_text_is_rejected(empty_source):
    sources = _sources()
    sources["direct"] = empty_source

    with pytest.raises(InvalidPromptFamilyError, match="must not be empty"):
        PromptFamily(sources)


def test_non_string_template_value_is_rejected():
    sources = _sources()
    sources["direct"] = object()

    with pytest.raises(InvalidPromptFamilyError, match="string or string.Template"):
        PromptFamily(sources)


def test_duplicate_json_template_key_is_rejected():
    serialized = (
        '{"direct":"$claim $context $caption",'
        '"direct":"$claim $context $caption",'
        '"analytical":"$claim $context $caption",'
        '"parallel":"$claim $context $caption1 $caption2",'
        '"sequential":"$claim $context $caption1 $caption2"}'
    )

    with pytest.raises(InvalidPromptFamilyError, match="valid JSON object"):
        deserialize_prompt_family(serialized)
