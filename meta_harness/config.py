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


FULL_SEARCH_V3_PROTOCOL_ID = "sciver_full_search_v3"
FULL_SEARCH_V3_SEARCH_SIZE = 1000
FULL_SEARCH_V3_FINAL_SIZE = 1000
FULL_SEARCH_V3_SPLIT_SEED = 42
FULL_SEARCH_V3_CANDIDATES_PER_ITERATION = 1
FULL_SEARCH_V3_MIN_ITERATIONS = 15
FULL_SEARCH_V3_MAX_ITERATIONS = 40
FULL_SEARCH_V3_PATIENCE = 8
FULL_SEARCH_V3_PROPOSAL_ATTEMPTS = 3
FULL_SEARCH_V3_SOLVER_MODEL = "Qwen/Qwen3.5-35B-A3B"
FULL_SEARCH_V3_SOLVER_TEMPERATURE = 0
FULL_SEARCH_V3_SOLVER_TOP_P = 1
FULL_SEARCH_V3_SOLVER_SEED = 42
FULL_SEARCH_V3_SOLVER_N = 1
FULL_SEARCH_V3_SOLVER_STREAM = False
FULL_SEARCH_V3_SOLVER_MAX_TOKENS = 8192


@dataclass(frozen=True)
class FullSearchV3Config:
    """The locked, non-staged configuration for ``sciver_full_search_v3``.

    This is deliberately separate from :class:`MetaHarnessConfig`, whose
    fields describe the legacy staged protocol.  Every value here contributes
    to an immutable V3 run identity and must equal the approved contract.
    """

    protocol_id: str = FULL_SEARCH_V3_PROTOCOL_ID
    search_size: int = FULL_SEARCH_V3_SEARCH_SIZE
    final_size: int = FULL_SEARCH_V3_FINAL_SIZE
    split_seed: int = FULL_SEARCH_V3_SPLIT_SEED
    candidate_count_per_iteration: int = (
        FULL_SEARCH_V3_CANDIDATES_PER_ITERATION
    )
    min_iterations: int = FULL_SEARCH_V3_MIN_ITERATIONS
    max_iterations: int = FULL_SEARCH_V3_MAX_ITERATIONS
    patience: int = FULL_SEARCH_V3_PATIENCE
    proposal_attempts: int = FULL_SEARCH_V3_PROPOSAL_ATTEMPTS
    solver_model: str = FULL_SEARCH_V3_SOLVER_MODEL
    solver_temperature: int = FULL_SEARCH_V3_SOLVER_TEMPERATURE
    solver_top_p: int = FULL_SEARCH_V3_SOLVER_TOP_P
    solver_seed: int = FULL_SEARCH_V3_SOLVER_SEED
    solver_n: int = FULL_SEARCH_V3_SOLVER_N
    solver_stream: bool = FULL_SEARCH_V3_SOLVER_STREAM
    solver_max_tokens: int = FULL_SEARCH_V3_SOLVER_MAX_TOKENS

    def __post_init__(self) -> None:
        expected = self._canonical_values()
        for field, value in expected.items():
            actual = getattr(self, field)
            if type(actual) is not type(value) or actual != value:
                raise MetaHarnessConfigError(
                    f"full-search V3 {field} is locked to {value!r}"
                )

    @staticmethod
    def _canonical_values() -> dict[str, Any]:
        return {
            "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
            "search_size": FULL_SEARCH_V3_SEARCH_SIZE,
            "final_size": FULL_SEARCH_V3_FINAL_SIZE,
            "split_seed": FULL_SEARCH_V3_SPLIT_SEED,
            "candidate_count_per_iteration": (
                FULL_SEARCH_V3_CANDIDATES_PER_ITERATION
            ),
            "min_iterations": FULL_SEARCH_V3_MIN_ITERATIONS,
            "max_iterations": FULL_SEARCH_V3_MAX_ITERATIONS,
            "patience": FULL_SEARCH_V3_PATIENCE,
            "proposal_attempts": FULL_SEARCH_V3_PROPOSAL_ATTEMPTS,
            "solver_model": FULL_SEARCH_V3_SOLVER_MODEL,
            "solver_temperature": FULL_SEARCH_V3_SOLVER_TEMPERATURE,
            "solver_top_p": FULL_SEARCH_V3_SOLVER_TOP_P,
            "solver_seed": FULL_SEARCH_V3_SOLVER_SEED,
            "solver_n": FULL_SEARCH_V3_SOLVER_N,
            "solver_stream": FULL_SEARCH_V3_SOLVER_STREAM,
            "solver_max_tokens": FULL_SEARCH_V3_SOLVER_MAX_TOKENS,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FullSearchV3Config":
        """Validate one complete V3 configuration object without defaults."""

        if not isinstance(values, Mapping):
            raise MetaHarnessConfigError("full-search V3 configuration must be an object")
        expected = {
            "protocol_id",
            "search_size",
            "final_size",
            "split_seed",
            "candidate_count_per_iteration",
            "min_iterations",
            "max_iterations",
            "patience",
            "proposal_attempts",
            "solver",
        }
        _require_exact_fields(values, expected, "full-search V3 configuration")
        solver = values["solver"]
        if not isinstance(solver, Mapping):
            raise MetaHarnessConfigError("full-search V3 solver must be an object")
        solver_expected = {
            "model",
            "temperature",
            "top_p",
            "seed",
            "n",
            "stream",
            "max_tokens",
        }
        _require_exact_fields(solver, solver_expected, "full-search V3 solver")
        return cls(
            protocol_id=values["protocol_id"],
            search_size=values["search_size"],
            final_size=values["final_size"],
            split_seed=values["split_seed"],
            candidate_count_per_iteration=values["candidate_count_per_iteration"],
            min_iterations=values["min_iterations"],
            max_iterations=values["max_iterations"],
            patience=values["patience"],
            proposal_attempts=values["proposal_attempts"],
            solver_model=solver["model"],
            solver_temperature=solver["temperature"],
            solver_top_p=solver["top_p"],
            solver_seed=solver["seed"],
            solver_n=solver["n"],
            solver_stream=solver["stream"],
            solver_max_tokens=solver["max_tokens"],
        )

    @classmethod
    def from_json(cls, serialized: str) -> "FullSearchV3Config":
        if not isinstance(serialized, str):
            raise MetaHarnessConfigError(
                "full-search V3 serialized configuration must be text"
            )
        try:
            values = json.loads(serialized, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, MetaHarnessConfigError) as exc:
            raise MetaHarnessConfigError(
                "full-search V3 configuration must contain valid JSON"
            ) from exc
        return cls.from_mapping(values)

    @classmethod
    def load(cls, path: str | Path) -> "FullSearchV3Config":
        try:
            serialized = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise MetaHarnessConfigError(
                "full-search V3 configuration file must be readable UTF-8 text"
            ) from exc
        return cls.from_json(serialized)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "search_size": self.search_size,
            "final_size": self.final_size,
            "split_seed": self.split_seed,
            "candidate_count_per_iteration": self.candidate_count_per_iteration,
            "min_iterations": self.min_iterations,
            "max_iterations": self.max_iterations,
            "patience": self.patience,
            "proposal_attempts": self.proposal_attempts,
            "solver": {
                "model": self.solver_model,
                "temperature": self.solver_temperature,
                "top_p": self.solver_top_p,
                "seed": self.solver_seed,
                "n": self.solver_n,
                "stream": self.solver_stream,
                "max_tokens": self.solver_max_tokens,
            },
        }

    def sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def canonical_full_search_v3_config() -> FullSearchV3Config:
    """Return the only permitted ``sciver_full_search_v3`` configuration."""

    return FullSearchV3Config()


def load_full_search_v3_config(path: str | Path) -> FullSearchV3Config:
    """Load only the locked V3 configuration, never the staged config."""

    return FullSearchV3Config.load(path)


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


def _require_exact_fields(
    values: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(values)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise MetaHarnessConfigError(
            f"{context} fields must exactly match the V3 contract "
            f"({'; '.join(details)})"
        )


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
    "FULL_SEARCH_V3_CANDIDATES_PER_ITERATION",
    "FULL_SEARCH_V3_FINAL_SIZE",
    "FULL_SEARCH_V3_MAX_ITERATIONS",
    "FULL_SEARCH_V3_MIN_ITERATIONS",
    "FULL_SEARCH_V3_PATIENCE",
    "FULL_SEARCH_V3_PROPOSAL_ATTEMPTS",
    "FULL_SEARCH_V3_PROTOCOL_ID",
    "FULL_SEARCH_V3_SEARCH_SIZE",
    "FULL_SEARCH_V3_SOLVER_MAX_TOKENS",
    "FULL_SEARCH_V3_SOLVER_MODEL",
    "FULL_SEARCH_V3_SOLVER_N",
    "FULL_SEARCH_V3_SOLVER_SEED",
    "FULL_SEARCH_V3_SOLVER_STREAM",
    "FULL_SEARCH_V3_SOLVER_TEMPERATURE",
    "FULL_SEARCH_V3_SOLVER_TOP_P",
    "FULL_SEARCH_V3_SPLIT_SEED",
    "FullSearchV3Config",
    "MetaHarnessConfig",
    "MetaHarnessConfigError",
    "SearchProtocol",
    "SplitRatios",
    "SUPPORTED_PROPOSER_MODELS",
    "SUPPORTED_REASONING_EFFORTS",
    "SUPPORTED_SEARCH_MODELS",
    "canonical_full_search_v3_config",
    "load_full_search_v3_config",
    "load_meta_harness_config",
]
