import json
from pathlib import Path

import pytest

from meta_harness.config import (
    DEFAULT_MODEL,
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_PROPOSER_REASONING_EFFORT,
    DEFAULT_SEED,
    DEFAULT_SPLIT_RATIOS,
    MetaHarnessConfig,
    MetaHarnessConfigError,
)


EXAMPLE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "meta_harness"
    / "sciver_gemma_codex.example.json"
)
ACTIVE_CONFIG = EXAMPLE_CONFIG.with_name("sciver_gemma_codex.json")


def test_defaults_are_frozen_for_sciver_search():
    config = MetaHarnessConfig.from_mapping({})

    assert config.model == DEFAULT_MODEL == "gemma-4-26B-A4B-it"
    assert config.proposer_model == DEFAULT_PROPOSER_MODEL == "gpt-5.6-terra"
    assert (
        config.proposer_reasoning_effort
        == DEFAULT_PROPOSER_REASONING_EFFORT
        == "medium"
    )
    assert config.seed == DEFAULT_SEED == 42
    assert config.split_ratios.as_dict() == DEFAULT_SPLIT_RATIOS
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.generation_request_options() == {}


def test_example_config_loads_and_null_generation_values_mean_omit():
    config = MetaHarnessConfig.load(EXAMPLE_CONFIG)

    assert config.as_dict() == json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    assert config.generation_request_options() == {}
    assert len(config.sha256()) == 64


def test_all_active_meta_harness_configs_select_supported_models():
    for path in (EXAMPLE_CONFIG, ACTIVE_CONFIG):
        config = MetaHarnessConfig.load(path)

        assert config.model == "gemma-4-26B-A4B-it"
        assert config.proposer_model in {"gpt-5.6-sol", "gpt-5.6-terra"}
        assert config.proposer_reasoning_effort == "medium"
        assert "qwen" not in path.name.lower()

    with pytest.raises(MetaHarnessConfigError, match="model must be fixed"):
        MetaHarnessConfig.from_mapping(
            {"model": "Qwen2.5-VL-7B-Instruct"}
        )


def test_explicit_optional_generation_values_are_returned():
    config = MetaHarnessConfig.from_mapping(
        {
            "generation": {
                "temperature": 0.25,
                "max_tokens": 512,
            }
        }
    )

    assert config.generation_request_options() == {
        "temperature": 0.25,
        "max_tokens": 512,
    }


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"model": "different-model"}, "model must be fixed"),
        (
            {"proposer": {"model": "different-model"}},
            "proposer.model must be fixed",
        ),
        (
            {"proposer": {"reasoning_effort": "unsupported"}},
            "proposer.reasoning_effort must be one of",
        ),
        ({"proposer": {"unknown": 1}}, "unexpected fields"),
        ({"proposer": "invalid"}, "proposer must be a JSON object"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        (
            {
                "split_ratios": {
                    "search": 0.5,
                    "validation": 0.2,
                    "final_test": 0.4,
                }
            },
            "sum to 1.0",
        ),
        (
            {
                "split_ratios": {
                    "search": 0.0,
                    "validation": 0.5,
                    "final_test": 0.5,
                }
            },
            "greater than zero",
        ),
        ({"generation": {"temperature": -0.1}}, "temperature"),
        ({"generation": {"max_tokens": 0}}, "max_tokens"),
        ({"generation": {"unknown": 1}}, "unexpected fields"),
        ({"unknown": 1}, "unexpected fields"),
    ],
)
def test_invalid_configuration_is_rejected(values, message):
    with pytest.raises(MetaHarnessConfigError, match=message):
        MetaHarnessConfig.from_mapping(values)


def test_duplicate_json_keys_are_rejected():
    serialized = (
        '{"seed":42,"seed":7,"model":"gemma-4-26B-A4B-it"}'
    )

    with pytest.raises(MetaHarnessConfigError, match="valid JSON"):
        MetaHarnessConfig.from_json(serialized)


def test_staged_identity_hash_covers_model_proposer_effort_and_protocol():
    base = json.loads(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    hashes = {MetaHarnessConfig.from_mapping(base).sha256()}
    for mutation in (
        lambda value: value.update(model="gemma-4-31B-it"),
        lambda value: value["proposer"].update(reasoning_effort="high"),
        lambda value: value["search_protocol"].update(promotion_top_k=2),
    ):
        changed = json.loads(json.dumps(base))
        mutation(changed)
        hashes.add(MetaHarnessConfig.from_mapping(changed).sha256())

    assert len(hashes) == 4
