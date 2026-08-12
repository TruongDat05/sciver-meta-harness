"""Locked configuration for the SciVer Full-Search V3 workflow."""

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


class MetaHarnessConfigError(ValueError):
    """Raised when the locked Full-Search V3 configuration is invalid."""


@dataclass(frozen=True)
class FullSearchV3Config:
    """The locked configuration for ``sciver_full_search_v3``.

    Every value contributes to an immutable V3 run identity and must equal
    the approved protocol contract.
    """

    protocol_id: str = FULL_SEARCH_V3_PROTOCOL_ID
    search_size: int = FULL_SEARCH_V3_SEARCH_SIZE
    final_size: int = FULL_SEARCH_V3_FINAL_SIZE
    split_seed: int = FULL_SEARCH_V3_SPLIT_SEED
    candidate_count_per_iteration: int = FULL_SEARCH_V3_CANDIDATES_PER_ITERATION
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
        for field, expected in self._canonical_values().items():
            actual = getattr(self, field)
            if type(actual) is not type(expected) or actual != expected:
                raise MetaHarnessConfigError(
                    f"full-search V3 {field} is locked to {expected!r}"
                )

    @staticmethod
    def _canonical_values() -> dict[str, Any]:
        return {
            "protocol_id": FULL_SEARCH_V3_PROTOCOL_ID,
            "search_size": FULL_SEARCH_V3_SEARCH_SIZE,
            "final_size": FULL_SEARCH_V3_FINAL_SIZE,
            "split_seed": FULL_SEARCH_V3_SPLIT_SEED,
            "candidate_count_per_iteration": FULL_SEARCH_V3_CANDIDATES_PER_ITERATION,
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
            "full-search V3 configuration",
        )
        solver = values["solver"]
        if not isinstance(solver, Mapping):
            raise MetaHarnessConfigError("full-search V3 solver must be an object")
        _require_exact_fields(
            solver,
            {"model", "temperature", "top_p", "seed", "n", "stream", "max_tokens"},
            "full-search V3 solver",
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
    """Load and validate the complete locked V3 configuration."""

    return FullSearchV3Config.load(path)


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
    "DEFAULT_PROPOSER_MODEL",
    "DEFAULT_PROPOSER_REASONING_EFFORT",
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
    "MetaHarnessConfigError",
    "SUPPORTED_PROPOSER_MODELS",
    "SUPPORTED_REASONING_EFFORTS",
    "canonical_full_search_v3_config",
    "load_full_search_v3_config",
]
