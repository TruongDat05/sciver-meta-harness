"""Pure deterministic SEARCH ranking and patience semantics for Meta-Harness.

This module deliberately has no solver, proposer, persistence, or iteration
orchestration dependency.  It accepts only complete, rankable SEARCH reports
and ignores every non-metric resource field.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

from meta_harness.config import (
    EXPERIMENT_PROTOCOL_ID,
    EXPERIMENT_SEARCH_SIZE,
)


EXPERIMENT_RANKING_STAGE = "SEARCH"
_CANDIDATE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class RankingError(ValueError):
    """Raised when an otherwise eligible ranking input is inconsistent."""


@dataclass(frozen=True)
class RankedCandidate:
    """The only fields permitted to influence a SEARCH rank."""

    candidate_id: str
    prompt_sha256: str
    macro_f1: float
    accuracy: float

    def __post_init__(self) -> None:
        _candidate_id(self.candidate_id)
        _sha256(self.prompt_sha256)
        _rate(self.macro_f1, "macro_f1")
        _rate(self.accuracy, "accuracy")


@dataclass(frozen=True)
class PatienceState:
    """Metric-only state carried by a future trusted iteration loop."""

    best_macro_f1: float | None = None
    best_accuracy: float | None = None
    consecutive_non_improving: int = 0

    def __post_init__(self) -> None:
        if (self.best_macro_f1 is None) != (self.best_accuracy is None):
            raise RankingError(
                "best Macro-F1 and Accuracy must be set together"
            )
        if self.best_macro_f1 is not None:
            _rate(self.best_macro_f1, "best_macro_f1")
            _rate(self.best_accuracy, "best_accuracy")
        if (
            isinstance(self.consecutive_non_improving, bool)
            or not isinstance(self.consecutive_non_improving, int)
            or self.consecutive_non_improving < 0
        ):
            raise RankingError(
                "consecutive_non_improving must be a non-negative integer"
            )


@dataclass(frozen=True)
class PatienceUpdate:
    """The pure result of comparing one current winner to prior metrics."""

    state: PatienceState
    winner: RankedCandidate | None
    metric_improved: bool


def experiment_rank_key(
    candidate: RankedCandidate,
) -> tuple[float, float, str, str]:
    """Return the locked ascending ranking key, without resource criteria."""

    if not isinstance(candidate, RankedCandidate):
        raise TypeError("candidate must be RankedCandidate")
    return (
        -candidate.macro_f1,
        -candidate.accuracy,
        candidate.prompt_sha256,
        candidate.candidate_id,
    )


def rank_experiment_candidates(
    candidates: Iterable[RankedCandidate],
) -> tuple[RankedCandidate, ...]:
    """Rank eligible candidates deterministically, independent of input order."""

    normalized = tuple(candidates)
    if any(not isinstance(candidate, RankedCandidate) for candidate in normalized):
        raise TypeError("candidates must contain RankedCandidate objects")
    by_id: dict[str, RankedCandidate] = {}
    for candidate in normalized:
        existing = by_id.get(candidate.candidate_id)
        if existing is not None and existing != candidate:
            raise RankingError(
                "candidate_id cannot have multiple eligible SEARCH results"
            )
        by_id[candidate.candidate_id] = candidate
    return tuple(sorted(by_id.values(), key=experiment_rank_key))


def eligible_experiment_candidate(
    report: Mapping[str, Any],
) -> RankedCandidate | None:
    """Return a rankable candidate only for a complete eligible SEARCH report.

    A missing, incomplete, ineligible, or incompatible report returns ``None``
    so it cannot enter ranking.  This boundary intentionally does not accept
    alternate stages or partial/full-search subsets.
    """

    if not isinstance(report, Mapping):
        return None
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    if (
        report.get("protocol_id") != EXPERIMENT_PROTOCOL_ID
        or report.get("stage") != EXPERIMENT_RANKING_STAGE
        or report.get("total_records") != EXPERIMENT_SEARCH_SIZE
        or report.get("completed_solver_responses") != EXPERIMENT_SEARCH_SIZE
        or report.get("parsed_predictions") != EXPERIMENT_SEARCH_SIZE
        or report.get("abstentions_or_parse_failures") != 0
        or report.get("infrastructure_failures") != 0
        or metrics.get("rankable") is not True
    ):
        return None
    try:
        macro_f1 = _rate(metrics.get("macro_f1"), "metrics.macro_f1")
        accuracy = _rate(metrics.get("accuracy"), "metrics.accuracy")
        if _rate(metrics.get("parse_coverage"), "metrics.parse_coverage") != 1.0:
            return None
        return RankedCandidate(
            candidate_id=_candidate_id(report.get("candidate_id")),
            prompt_sha256=_sha256(report.get("prompt_sha256")),
            macro_f1=macro_f1,
            accuracy=accuracy,
        )
    except RankingError:
        return None


def rank_eligible_experiment_reports(
    reports: Iterable[Mapping[str, Any]],
) -> tuple[RankedCandidate, ...]:
    """Filter incomplete reports then apply the locked ranking key."""

    candidates = [
        candidate
        for report in reports
        if (candidate := eligible_experiment_candidate(report)) is not None
    ]
    return rank_experiment_candidates(candidates)


def is_experiment_metric_improvement(
    candidate: RankedCandidate,
    *,
    prior_macro_f1: float | None,
    prior_accuracy: float | None,
) -> bool:
    """Return whether metrics improve, excluding prompt-hash and ID ties."""

    if not isinstance(candidate, RankedCandidate):
        raise TypeError("candidate must be RankedCandidate")
    if (prior_macro_f1 is None) != (prior_accuracy is None):
        raise RankingError(
            "prior Macro-F1 and Accuracy must be set together"
        )
    if prior_macro_f1 is None:
        return True
    macro_f1 = _rate(prior_macro_f1, "prior_macro_f1")
    accuracy = _rate(prior_accuracy, "prior_accuracy")
    return candidate.macro_f1 > macro_f1 or (
        candidate.macro_f1 == macro_f1 and candidate.accuracy > accuracy
    )


def advance_experiment_patience(
    state: PatienceState,
    ranking: Iterable[RankedCandidate],
) -> PatienceUpdate:
    """Advance patience from the current rank winner using metrics only.

    A hash/ID-only winner change has identical metrics and therefore increments
    non-improving patience rather than resetting it.  Empty rankings leave the
    state unchanged; an iteration loop may decide separately whether it counts
    as completed.
    """

    if not isinstance(state, PatienceState):
        raise TypeError("state must be PatienceState")
    ordered = rank_experiment_candidates(ranking)
    if not ordered:
        return PatienceUpdate(
            state=state,
            winner=None,
            metric_improved=False,
        )
    winner = ordered[0]
    improved = is_experiment_metric_improvement(
        winner,
        prior_macro_f1=state.best_macro_f1,
        prior_accuracy=state.best_accuracy,
    )
    if improved:
        next_state = PatienceState(
            best_macro_f1=winner.macro_f1,
            best_accuracy=winner.accuracy,
            consecutive_non_improving=0,
        )
    else:
        next_state = PatienceState(
            best_macro_f1=state.best_macro_f1,
            best_accuracy=state.best_accuracy,
            consecutive_non_improving=state.consecutive_non_improving + 1,
        )
    return PatienceUpdate(
        state=next_state,
        winner=winner,
        metric_improved=improved,
    )


def _candidate_id(value: Any) -> str:
    if not isinstance(value, str) or not _CANDIDATE_ID.fullmatch(value):
        raise RankingError("candidate_id must be a safe identifier")
    return value


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise RankingError(
            "prompt_sha256 must be a lowercase hexadecimal SHA-256"
        )
    return value


def _rate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RankingError(f"{field} must be a finite rate")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise RankingError(f"{field} must be a finite rate")
    return normalized


__all__ = [
    "EXPERIMENT_RANKING_STAGE",
    "PatienceState",
    "PatienceUpdate",
    "RankedCandidate",
    "RankingError",
    "advance_experiment_patience",
    "eligible_experiment_candidate",
    "experiment_rank_key",
    "is_experiment_metric_improvement",
    "rank_eligible_experiment_reports",
    "rank_experiment_candidates",
]
