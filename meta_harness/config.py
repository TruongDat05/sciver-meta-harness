"""Locked configuration for the SciVer Meta-Harness workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_PROPOSER_MODEL = "gpt-5.6-terra"
DEFAULT_PROPOSER_REASONING_EFFORT = "medium"
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

EXPERIMENT_PROTOCOL_ID = "sciver_full_search_v3"
EXPERIMENT_SEARCH_SIZE = 1000
EXPERIMENT_FINAL_SIZE = 1000
EXPERIMENT_SPLIT_SEED = 42
EXPERIMENT_CANDIDATES_PER_ITERATION = 1
EXPERIMENT_MIN_ITERATIONS = 15
EXPERIMENT_MAX_ITERATIONS = 40
EXPERIMENT_PATIENCE = 8
EXPERIMENT_PROPOSAL_ATTEMPTS = 3
EXPERIMENT_SOLVER_MODEL = "Qwen2.5-VL-7B-Instruct"
EXPERIMENT_SOLVER_TEMPERATURE = 0
EXPERIMENT_SOLVER_TOP_P = 1
EXPERIMENT_SOLVER_SEED = 42
EXPERIMENT_SOLVER_N = 1
EXPERIMENT_SOLVER_STREAM = False
EXPERIMENT_SOLVER_MAX_TOKENS = 8192


class MetaHarnessConfigError(ValueError):
    """Raised when the locked Meta-Harness configuration is invalid."""


@dataclass(frozen=True)
class Config:
    """The locked configuration for ``sciver_full_search_v3``.

    Every value contributes to an immutable run identity and must equal
    the approved protocol contract.
    """

    protocol_id: str = EXPERIMENT_PROTOCOL_ID
    search_size: int = EXPERIMENT_SEARCH_SIZE
    final_size: int = EXPERIMENT_FINAL_SIZE
    split_seed: int = EXPERIMENT_SPLIT_SEED
    candidate_count_per_iteration: int = EXPERIMENT_CANDIDATES_PER_ITERATION
    min_iterations: int = EXPERIMENT_MIN_ITERATIONS
    max_iterations: int = EXPERIMENT_MAX_ITERATIONS
    patience: int = EXPERIMENT_PATIENCE
    proposal_attempts: int = EXPERIMENT_PROPOSAL_ATTEMPTS
    solver_model: str = EXPERIMENT_SOLVER_MODEL
    solver_temperature: int = EXPERIMENT_SOLVER_TEMPERATURE
    solver_top_p: int = EXPERIMENT_SOLVER_TOP_P
    solver_seed: int = EXPERIMENT_SOLVER_SEED
    solver_n: int = EXPERIMENT_SOLVER_N
    solver_stream: bool = EXPERIMENT_SOLVER_STREAM
    solver_max_tokens: int = EXPERIMENT_SOLVER_MAX_TOKENS

    def __post_init__(self) -> None:
        for field, expected in self._canonical_values().items():
            actual = getattr(self, field)
            if type(actual) is not type(expected) or actual != expected:
                raise MetaHarnessConfigError(
                    f"meta-harness {field} is locked to {expected!r}"
                )

    @staticmethod
    def _canonical_values() -> dict[str, Any]:
        return {
            "protocol_id": EXPERIMENT_PROTOCOL_ID,
            "search_size": EXPERIMENT_SEARCH_SIZE,
            "final_size": EXPERIMENT_FINAL_SIZE,
            "split_seed": EXPERIMENT_SPLIT_SEED,
            "candidate_count_per_iteration": EXPERIMENT_CANDIDATES_PER_ITERATION,
            "min_iterations": EXPERIMENT_MIN_ITERATIONS,
            "max_iterations": EXPERIMENT_MAX_ITERATIONS,
            "patience": EXPERIMENT_PATIENCE,
            "proposal_attempts": EXPERIMENT_PROPOSAL_ATTEMPTS,
            "solver_model": EXPERIMENT_SOLVER_MODEL,
            "solver_temperature": EXPERIMENT_SOLVER_TEMPERATURE,
            "solver_top_p": EXPERIMENT_SOLVER_TOP_P,
            "solver_seed": EXPERIMENT_SOLVER_SEED,
            "solver_n": EXPERIMENT_SOLVER_N,
            "solver_stream": EXPERIMENT_SOLVER_STREAM,
            "solver_max_tokens": EXPERIMENT_SOLVER_MAX_TOKENS,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "Config":
        """Validate one complete configuration object without defaults."""

        if not isinstance(values, Mapping):
            raise MetaHarnessConfigError("meta-harness configuration must be an object")
        _require_exact_fields(
            values,
            {
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
            },
            "meta-harness configuration",
        )
        solver = values["solver"]
        if not isinstance(solver, Mapping):
            raise MetaHarnessConfigError("meta-harness solver must be an object")
        _require_exact_fields(
            solver,
            {"model", "temperature", "top_p", "seed", "n", "stream", "max_tokens"},
            "meta-harness solver",
        )
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
    def from_json(cls, serialized: str) -> "Config":
        if not isinstance(serialized, str):
            raise MetaHarnessConfigError(
                "meta-harness serialized configuration must be text"
            )
        try:
            values = json.loads(serialized, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, MetaHarnessConfigError) as exc:
            raise MetaHarnessConfigError(
                "meta-harness configuration must contain valid JSON"
            ) from exc
        return cls.from_mapping(values)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        try:
            serialized = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise MetaHarnessConfigError(
                "meta-harness configuration file must be readable UTF-8 text"
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


def canonical_experiment_config() -> Config:
    """Return the only permitted ``sciver_full_search_v3`` configuration."""

    return Config()


def load_experiment_config(path: str | Path) -> Config:
    """Load and validate the complete locked configuration."""

    return Config.load(path)


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
            f"{context} fields must exactly match the contract "
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
    "DEFAULT_PROPOSER_MODEL",
    "DEFAULT_PROPOSER_REASONING_EFFORT",
    "EXPERIMENT_CANDIDATES_PER_ITERATION",
    "EXPERIMENT_FINAL_SIZE",
    "EXPERIMENT_MAX_ITERATIONS",
    "EXPERIMENT_MIN_ITERATIONS",
    "EXPERIMENT_PATIENCE",
    "EXPERIMENT_PROPOSAL_ATTEMPTS",
    "EXPERIMENT_PROTOCOL_ID",
    "EXPERIMENT_SEARCH_SIZE",
    "EXPERIMENT_SOLVER_MAX_TOKENS",
    "EXPERIMENT_SOLVER_MODEL",
    "EXPERIMENT_SOLVER_N",
    "EXPERIMENT_SOLVER_SEED",
    "EXPERIMENT_SOLVER_STREAM",
    "EXPERIMENT_SOLVER_TEMPERATURE",
    "EXPERIMENT_SOLVER_TOP_P",
    "EXPERIMENT_SPLIT_SEED",
    "Config",
    "MetaHarnessConfigError",
    "SUPPORTED_PROPOSER_MODELS",
    "SUPPORTED_REASONING_EFFORTS",
    "canonical_experiment_config",
    "load_experiment_config",
]
