"""Validated local configuration for SciVer Meta-Harness experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from utils.answer_parser import PARSER_VERSION


DEFAULT_MODEL = "gemma-4-26B-A4B-it"
DEFAULT_PROPOSER_MODEL = "gpt-5.6-terra"
DEFAULT_PROPOSER_REASONING_EFFORT = "medium"
DEFAULT_SEED = 42
SUPPORTED_SEARCH_MODELS = (
    "gemma-4-26B-A4B-it",
    "gemma-4-31B-it",
)
SUPPORTED_PROPOSER_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
)
SUPPORTED_REASONING_EFFORTS = (
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "ultra",
)
STAGED_SEARCH_PROTOCOL_VERSION = "search_smoke_promote_validation_v1"
STAGED_SCORING_PROTOCOL_VERSION = "binary_macro_f1_v1"
STAGED_EVALUATOR_CONTRACT_VERSION = "meta_harness_evaluator_v1"
DEFAULT_SPLIT_RATIOS = {
    "search": 0.20,
    "validation": 0.20,
    "final_test": 0.60,
}


class MetaHarnessConfigError(ValueError):
    """Raised when local experiment configuration is invalid."""


@dataclass(frozen=True)
class SearchProtocol:
    """Immutable staged-search settings included in every new config hash."""

    candidates_per_iteration: int = 2
    min_iterations: int = 10
    target_iterations: int = 10
    max_iterations: int = 15
    promotion_top_k: int = 3
    search_examples: int = 80
    smoke_examples: int = 10
    guard_fraction: float = 0.25
    early_stopping_patience: int | None = None
    estimated_tokens_per_solver_call: int = 4096

    def __post_init__(self) -> None:
        for field in (
            "candidates_per_iteration",
            "min_iterations",
            "target_iterations",
            "max_iterations",
            "promotion_top_k",
            "search_examples",
            "smoke_examples",
            "estimated_tokens_per_solver_call",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MetaHarnessConfigError(
                    f"search_protocol.{field} must be a positive integer"
                )
        if not (
            self.min_iterations
            <= self.target_iterations
            <= self.max_iterations
        ):
            raise MetaHarnessConfigError(
                "search_protocol requires min_iterations <= "
                "target_iterations <= max_iterations"
            )
        if not 50 <= self.search_examples <= 100:
            raise MetaHarnessConfigError(
                "search_protocol.search_examples must be between 50 and 100"
            )
        if self.smoke_examples > self.search_examples:
            raise MetaHarnessConfigError(
                "search_protocol.smoke_examples must not exceed search_examples"
            )
        if not 0.20 <= float(self.guard_fraction) <= 0.30:
            raise MetaHarnessConfigError(
                "search_protocol.guard_fraction must be between 0.20 and 0.30"
            )
        if self.early_stopping_patience is not None and (
            isinstance(self.early_stopping_patience, bool)
            or not isinstance(self.early_stopping_patience, int)
            or self.early_stopping_patience <= 0
        ):
            raise MetaHarnessConfigError(
                "search_protocol.early_stopping_patience must be a positive "
                "integer or null"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SearchProtocol":
        if not isinstance(values, Mapping):
            raise MetaHarnessConfigError("search_protocol must be a JSON object")
        expected = set(cls().as_dict())
        extra = sorted(set(values) - expected, key=str)
        if extra:
            raise MetaHarnessConfigError(
                "search_protocol contains unexpected fields: "
                + ", ".join(map(str, extra))
            )
        return cls(**{**cls().as_dict(), **dict(values)})

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidates_per_iteration": self.candidates_per_iteration,
            "min_iterations": self.min_iterations,
            "target_iterations": self.target_iterations,
            "max_iterations": self.max_iterations,
            "promotion_top_k": self.promotion_top_k,
            "search_examples": self.search_examples,
            "smoke_examples": self.smoke_examples,
            "guard_fraction": float(self.guard_fraction),
            "early_stopping_patience": self.early_stopping_patience,
            "estimated_tokens_per_solver_call": (
                self.estimated_tokens_per_solver_call
            ),
        }


@dataclass(frozen=True)
class SplitRatios:
    search: float = DEFAULT_SPLIT_RATIOS["search"]
    validation: float = DEFAULT_SPLIT_RATIOS["validation"]
    final_test: float = DEFAULT_SPLIT_RATIOS["final_test"]

    def __post_init__(self) -> None:
        values = self.as_dict()
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise MetaHarnessConfigError("all split ratios must be finite numbers")
        if any(value <= 0 for value in values.values()):
            raise MetaHarnessConfigError("all split ratios must be greater than zero")
        if not math.isclose(
            sum(values.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MetaHarnessConfigError("split ratios must sum to 1.0")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SplitRatios":
        if not isinstance(values, Mapping):
            raise MetaHarnessConfigError("split_ratios must be a JSON object")
        expected = set(DEFAULT_SPLIT_RATIOS)
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected, key=str)
            details = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if extra:
                details.append("unexpected: " + ", ".join(map(str, extra)))
            raise MetaHarnessConfigError(
                "split_ratios requires exactly search, validation, and "
                f"final_test ({'; '.join(details)})"
            )

        normalized = {
            name: _finite_number(values[name], f"split_ratios.{name}")
            for name in DEFAULT_SPLIT_RATIOS
        }
        if any(value <= 0 for value in normalized.values()):
            raise MetaHarnessConfigError("all split ratios must be greater than zero")
        if not math.isclose(
            sum(normalized.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise MetaHarnessConfigError("split ratios must sum to 1.0")
        return cls(**normalized)

    def as_dict(self) -> dict[str, float]:
        return {
            "search": self.search,
            "validation": self.validation,
            "final_test": self.final_test,
        }


@dataclass(frozen=True)
class MetaHarnessConfig:
    model: str = DEFAULT_MODEL
    proposer_model: str = DEFAULT_PROPOSER_MODEL
    proposer_reasoning_effort: str = DEFAULT_PROPOSER_REASONING_EFFORT
    seed: int = DEFAULT_SEED
    split_ratios: SplitRatios = SplitRatios()
    temperature: float | None = None
    max_tokens: int | None = None
    search_protocol: SearchProtocol = SearchProtocol()
    search_protocol_explicit: bool = False

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_SEARCH_MODELS:
            raise MetaHarnessConfigError(
                "model must be fixed to one of: "
                + ", ".join(SUPPORTED_SEARCH_MODELS)
            )
        if self.proposer_model not in SUPPORTED_PROPOSER_MODELS:
            raise MetaHarnessConfigError(
                "proposer.model must be fixed to one of: "
                + ", ".join(SUPPORTED_PROPOSER_MODELS)
            )
        if self.proposer_reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise MetaHarnessConfigError(
                "proposer.reasoning_effort must be one of: "
                + ", ".join(SUPPORTED_REASONING_EFFORTS)
            )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise MetaHarnessConfigError("seed must be a non-negative integer")
        if not isinstance(self.split_ratios, SplitRatios):
            raise MetaHarnessConfigError("split_ratios must be SplitRatios")
        if self.temperature is not None:
            temperature = _finite_number(
                self.temperature,
                "generation.temperature",
            )
            if temperature < 0:
                raise MetaHarnessConfigError(
                    "generation.temperature must be non-negative or null"
                )
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise MetaHarnessConfigError(
                "generation.max_tokens must be a positive integer or null"
            )
        if not isinstance(self.search_protocol, SearchProtocol):
            raise MetaHarnessConfigError(
                "search_protocol must be a SearchProtocol"
            )
        if not isinstance(self.search_protocol_explicit, bool):
            raise MetaHarnessConfigError(
                "search_protocol_explicit must be boolean"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "MetaHarnessConfig":
        if not isinstance(values, Mapping):
            raise MetaHarnessConfigError("configuration must be a JSON object")
        allowed = {
            "model",
            "proposer",
            "seed",
            "split_ratios",
            "generation",
            "search_protocol",
        }
        extra = sorted(set(values) - allowed, key=str)
        if extra:
            raise MetaHarnessConfigError(
                "configuration contains unexpected fields: "
                + ", ".join(map(str, extra))
            )

        model = values.get("model", DEFAULT_MODEL)
        if model not in SUPPORTED_SEARCH_MODELS:
            raise MetaHarnessConfigError(
                "model must be fixed to one of: "
                + ", ".join(SUPPORTED_SEARCH_MODELS)
            )

        proposer = values.get("proposer", {})
        if not isinstance(proposer, Mapping):
            raise MetaHarnessConfigError("proposer must be a JSON object")
        proposer_extra = sorted(
            set(proposer) - {"model", "reasoning_effort"},
            key=str,
        )
        if proposer_extra:
            raise MetaHarnessConfigError(
                "proposer contains unexpected fields: "
                + ", ".join(map(str, proposer_extra))
            )
        proposer_model = proposer.get("model", DEFAULT_PROPOSER_MODEL)
        if proposer_model not in SUPPORTED_PROPOSER_MODELS:
            raise MetaHarnessConfigError(
                "proposer.model must be fixed to one of: "
                + ", ".join(SUPPORTED_PROPOSER_MODELS)
            )
        proposer_reasoning_effort = proposer.get(
            "reasoning_effort",
            DEFAULT_PROPOSER_REASONING_EFFORT,
        )
        if proposer_reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            raise MetaHarnessConfigError(
                "proposer.reasoning_effort must be one of: "
                + ", ".join(SUPPORTED_REASONING_EFFORTS)
            )

        seed = values.get("seed", DEFAULT_SEED)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise MetaHarnessConfigError("seed must be a non-negative integer")

        split_values = values.get("split_ratios", DEFAULT_SPLIT_RATIOS)
        split_ratios = SplitRatios.from_mapping(split_values)

        generation = values.get("generation", {})
        if not isinstance(generation, Mapping):
            raise MetaHarnessConfigError("generation must be a JSON object")
        generation_extra = sorted(
            set(generation) - {"temperature", "max_tokens"},
            key=str,
        )
        if generation_extra:
            raise MetaHarnessConfigError(
                "generation contains unexpected fields: "
                + ", ".join(map(str, generation_extra))
            )

        raw_temperature = generation.get("temperature")
        temperature = (
            None
            if raw_temperature is None
            else _finite_number(raw_temperature, "generation.temperature")
        )
        if temperature is not None and temperature < 0:
            raise MetaHarnessConfigError(
                "generation.temperature must be non-negative or null"
            )

        max_tokens = generation.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise MetaHarnessConfigError(
                "generation.max_tokens must be a positive integer or null"
            )
        search_protocol = SearchProtocol.from_mapping(
            values.get("search_protocol", {})
        )

        return cls(
            model=model,
            proposer_model=proposer_model,
            proposer_reasoning_effort=proposer_reasoning_effort,
            seed=seed,
            split_ratios=split_ratios,
            temperature=temperature,
            max_tokens=max_tokens,
            search_protocol=search_protocol,
            search_protocol_explicit="search_protocol" in values,
        )

    @classmethod
    def from_json(cls, serialized: str) -> "MetaHarnessConfig":
        if not isinstance(serialized, str):
            raise MetaHarnessConfigError("serialized configuration must be text")
        try:
            values = json.loads(serialized, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, MetaHarnessConfigError) as exc:
            raise MetaHarnessConfigError(
                "configuration must contain valid JSON"
            ) from exc
        return cls.from_mapping(values)

    @classmethod
    def load(cls, path: str | Path) -> "MetaHarnessConfig":
        try:
            serialized = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise MetaHarnessConfigError(
                "configuration file must be readable UTF-8 text"
            ) from exc
        return cls.from_json(serialized)

    def as_dict(self) -> dict[str, Any]:
        result = {
            "model": self.model,
            "proposer": {
                "model": self.proposer_model,
                "reasoning_effort": self.proposer_reasoning_effort,
            },
            "seed": self.seed,
            "split_ratios": self.split_ratios.as_dict(),
            "generation": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }
        if self.search_protocol_explicit:
            result["search_protocol"] = self.search_protocol.as_dict()
        return result

    def generation_request_options(self) -> dict[str, int | float]:
        """Return only explicitly configured request fields."""

        options: dict[str, int | float] = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.max_tokens is not None:
            options["max_tokens"] = self.max_tokens
        return options

    def sha256(self) -> str:
        payload: Any = self.as_dict()
        if self.search_protocol_explicit:
            payload = {
                "configuration": payload,
                "immutable_contract": {
                    "search_protocol_version": (
                        STAGED_SEARCH_PROTOCOL_VERSION
                    ),
                    "scoring_protocol_version": (
                        STAGED_SCORING_PROTOCOL_VERSION
                    ),
                    "evaluator_contract_version": (
                        STAGED_EVALUATOR_CONTRACT_VERSION
                    ),
                    "parser_version": PARSER_VERSION,
                },
            }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_meta_harness_config(path: str | Path) -> MetaHarnessConfig:
    return MetaHarnessConfig.load(path)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetaHarnessConfigError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise MetaHarnessConfigError(f"{field} must be a finite number")
    return normalized


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MetaHarnessConfigError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROPOSER_MODEL",
    "DEFAULT_PROPOSER_REASONING_EFFORT",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT_RATIOS",
    "MetaHarnessConfig",
    "MetaHarnessConfigError",
    "SearchProtocol",
    "SplitRatios",
    "SUPPORTED_PROPOSER_MODELS",
    "SUPPORTED_REASONING_EFFORTS",
    "SUPPORTED_SEARCH_MODELS",
    "load_meta_harness_config",
]
