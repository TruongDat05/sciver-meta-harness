"""Durable, resumable orchestration for validation-only prompt search."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from types import MappingProxyType
from typing import Any, Iterator, Protocol

from meta_harness.baseline import canonical_baseline_sources
from meta_harness.candidate_store import CandidateStore
from meta_harness.config import MetaHarnessConfig
from meta_harness.evaluator import evaluate_candidate
from meta_harness.prompt_family import TEMPLATE_KEYS
from meta_harness.proposer.feedback import (
    current_search_feedback,
    merge_search_feedback,
    normalize_search_feedback,
    search_feedback_candidate_count,
    search_feedback_sha256,
)
from meta_harness.schemas import (
    Candidate,
    CandidateBatch,
    DEFAULT_CANDIDATE_COUNT,
    canonical_json,
    template_source_sha256,
)
from meta_harness.split_manager import verify_split_manifest


RUN_STATE_SCHEMA_VERSION = 1
ORCHESTRATOR_VERSION = "meta_harness_orchestrator_v1"
EVALUATION_PROCEDURE = "fixed_validation_split_v1"
BASELINE_CANDIDATE_ID = "baseline_cot"
_STATE_FILE = "run_state.json"
_LOCK_FILE = ".run.lock"
_SAFE_RESUME_FIELDS = frozenset(
    {
        "max_iterations",
        "max_candidates",
        "max_solver_calls",
        "max_tokens",
        "max_wall_time_seconds",
        "max_consecutive_failures",
    }
)
_TERMINAL_RUN_STATUSES = frozenset(
    {
        "completed",
        "early_stopped",
        "budget_exhausted",
        "failure_limit",
        "baseline_incomplete",
    }
)
_TERMINAL_CANDIDATE_STATUSES = frozenset({"evaluated", "failed"})
_SENSITIVE_TEXT = (
    re.compile(r"(?i)\bauthorization\s*:\s*[^\s,;]+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(?:api[_ -]?key|api[_ -]?url)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)https?://[^\s\"'<>]+"),
    re.compile(r"(?i)data:image/[^,;\s]+,[A-Za-z0-9+/=]+"),
    re.compile(r"\b[A-Za-z0-9+/]{128,}={0,2}\b"),
)


class OrchestratorError(RuntimeError):
    """Raised when a search run cannot safely advance."""


class ResumeConfigurationError(OrchestratorError):
    """Raised when resume inputs do not match the frozen run identity."""


class RunLockedError(OrchestratorError):
    """Raised when another writer already holds the run lock."""


class FinalizedRunError(OrchestratorError):
    """Raised when evolution is requested after winner freezing."""


class BudgetExhausted(OrchestratorError):
    """Raised internally when no further transition fits a hard budget."""


@dataclass(frozen=True)
class BudgetLimits:
    """Hard run ceilings; ``None`` means that resource is not bounded."""

    max_iterations: int = 10
    max_candidates: int | None = 20
    max_solver_calls: int | None = None
    max_tokens: int | None = None
    max_wall_time_seconds: float | None = None
    max_consecutive_failures: int = 3

    def __post_init__(self) -> None:
        _positive_integer(self.max_iterations, "max_iterations")
        _optional_positive_integer(self.max_candidates, "max_candidates")
        _optional_positive_integer(
            self.max_solver_calls,
            "max_solver_calls",
        )
        _optional_positive_integer(self.max_tokens, "max_tokens")
        if self.max_wall_time_seconds is not None:
            _positive_number(
                self.max_wall_time_seconds,
                "max_wall_time_seconds",
            )
        _positive_integer(
            self.max_consecutive_failures,
            "max_consecutive_failures",
        )

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "max_iterations": self.max_iterations,
            "max_candidates": self.max_candidates,
            "max_solver_calls": self.max_solver_calls,
            "max_tokens": self.max_tokens,
            "max_wall_time_seconds": (
                None
                if self.max_wall_time_seconds is None
                else float(self.max_wall_time_seconds)
            ),
            "max_consecutive_failures": self.max_consecutive_failures,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BudgetLimits":
        if not isinstance(value, Mapping):
            raise OrchestratorError("budget limits must be a mapping")
        expected = set(cls().as_dict())
        if set(value) != expected:
            raise OrchestratorError("stored budget limits have invalid fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Validation Macro-F1 early-stopping policy."""

    patience: int | None = 3
    min_delta: float = 0.005
    min_iterations: int = 0

    def __post_init__(self) -> None:
        if self.patience is not None:
            _positive_integer(self.patience, "patience")
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float))
            or not math.isfinite(float(self.min_delta))
            or self.min_delta < 0
        ):
            raise ValueError("min_delta must be a non-negative finite number")
        if (
            isinstance(self.min_iterations, bool)
            or not isinstance(self.min_iterations, int)
            or self.min_iterations < 0
        ):
            raise ValueError("min_iterations must be a non-negative integer")

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "patience": self.patience,
            "min_delta": float(self.min_delta),
            "min_iterations": self.min_iterations,
        }


@dataclass(frozen=True)
class FrontierEntry:
    """One eligible validation result used by deterministic Pareto ranking."""

    candidate_id: str
    macro_f1: float
    solver_calls: int
    tokens: int
    latency_seconds: float
    accuracy: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("candidate_id must be non-empty text")
        _unit_interval(self.macro_f1, "macro_f1")
        if self.accuracy is not None:
            _unit_interval(self.accuracy, "accuracy")
        _non_negative_integer(self.solver_calls, "solver_calls")
        _non_negative_integer(self.tokens, "tokens")
        _non_negative_number(self.latency_seconds, "latency_seconds")

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "macro_f1": float(self.macro_f1),
            "accuracy": (
                None if self.accuracy is None else float(self.accuracy)
            ),
            "solver_calls": self.solver_calls,
            "tokens": self.tokens,
            "latency_seconds": float(self.latency_seconds),
        }


class Proposer(Protocol):
    def propose(
        self,
        candidate_store: CandidateStore,
        *,
        iteration: int,
        parent_candidate_ids: Sequence[str],
        validation_scores: Mapping[str, Mapping[str, Any]],
        aggregate_metrics: Mapping[str, Any],
        failure_summaries: Mapping[str, Any],
        search_feedback: Mapping[str, Any],
    ) -> Any: ...


Evaluator = Callable[..., Mapping[str, Any]]
TransitionHook = Callable[[str, Mapping[str, Any]], None]


class _RunCandidateStore(CandidateStore):
    """Expose the byte-preserved baseline as a virtual immutable root."""

    def __init__(self, repository_root: str | Path, run_id: str) -> None:
        super().__init__(repository_root, run_id)
        self._baseline = _baseline_candidate()

    def load(self, candidate_id: str) -> Candidate:
        if candidate_id == BASELINE_CANDIDATE_ID:
            return self._baseline
        return super().load(candidate_id)

    def update_status(self, candidate_id: str, status: str) -> str:
        if candidate_id == BASELINE_CANDIDATE_ID:
            return status
        return super().update_status(candidate_id, status)


def dominates(left: FrontierEntry, right: FrontierEntry) -> bool:
    """Return whether ``left`` Pareto-dominates ``right``.

    Exact objective ties are resolved by lexicographically smaller
    ``candidate_id`` so frontier output is stable across input order.
    """

    no_worse = (
        left.macro_f1 >= right.macro_f1
        and left.solver_calls <= right.solver_calls
        and left.tokens <= right.tokens
        and left.latency_seconds <= right.latency_seconds
    )
    if not no_worse:
        return False
    strictly_better = (
        left.macro_f1 > right.macro_f1
        or left.solver_calls < right.solver_calls
        or left.tokens < right.tokens
        or left.latency_seconds < right.latency_seconds
    )
    return strictly_better or left.candidate_id < right.candidate_id


def pareto_frontier(entries: Iterable[FrontierEntry]) -> tuple[FrontierEntry, ...]:
    """Return the deterministic non-dominated eligible candidate set."""

    normalized = tuple(entries)
    if any(not isinstance(entry, FrontierEntry) for entry in normalized):
        raise TypeError("frontier entries must be FrontierEntry objects")
    by_id: dict[str, FrontierEntry] = {}
    for entry in normalized:
        existing = by_id.get(entry.candidate_id)
        if existing is not None and existing != entry:
            raise ValueError("one candidate_id cannot have multiple scores")
        by_id[entry.candidate_id] = entry
    unique = tuple(by_id.values())
    frontier = [
        entry
        for entry in unique
        if not any(
            other is not entry and dominates(other, entry)
            for other in unique
        )
    ]
    return tuple(sorted(frontier, key=frontier_rank_key))


def frontier_rank_key(entry: FrontierEntry) -> tuple[Any, ...]:
    """Primary validation rank with resource use and ID as stable tie-breakers."""

    return (
        -entry.macro_f1,
        entry.solver_calls,
        entry.tokens,
        entry.latency_seconds,
        entry.candidate_id,
    )


def winner_rank_key(entry: FrontierEntry) -> tuple[Any, ...]:
    """Match the frozen winner's deterministic validation ordering."""

    accuracy = -1.0 if entry.accuracy is None else entry.accuracy
    return (
        -entry.macro_f1,
        -accuracy,
        entry.tokens,
        entry.candidate_id,
    )


@contextmanager
def run_lock(path: str | Path) -> Iterator[None]:
    """Hold a non-blocking advisory lock for one run writer."""

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLockedError(
                "another process is already writing this run"
            ) from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class MetaHarnessOrchestrator:
    """Evaluate baseline, propose, validate, evaluate, and checkpoint a search."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        run_id: str,
        config: MetaHarnessConfig,
        split_manifest: Mapping[str, Any],
        validation_samples: Sequence[Any],
        solver: Any,
        proposer: Proposer,
        limits: BudgetLimits | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
        evaluator: Evaluator = evaluate_candidate,
        candidates_per_iteration: int = DEFAULT_CANDIDATE_COUNT,
        prior_search_feedback: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        transition_hook: TransitionHook | None = None,
    ) -> None:
        if not isinstance(config, MetaHarnessConfig):
            raise TypeError("config must be a MetaHarnessConfig")
        if config.search_protocol_explicit:
            raise ValueError(
                "search_protocol configs require StagedMetaHarnessOrchestrator; "
                "full validation per candidate is disabled"
            )
        if config.generation_request_options():
            raise ValueError(
                "Meta-Harness search requires the existing omitted generation "
                "settings; temperature and max_tokens must remain null"
            )
        verify_split_manifest(split_manifest)
        if split_manifest["config_sha256"] != config.sha256():
            raise ValueError(
                "split manifest configuration does not match the search "
                "configuration"
            )
        if (
            isinstance(validation_samples, (str, bytes))
            or not isinstance(validation_samples, Sequence)
        ):
            raise TypeError("validation_samples must be a sequence")
        if not validation_samples:
            raise ValueError("validation_samples must not be empty")
        if not callable(getattr(proposer, "propose", None)):
            raise TypeError("proposer must provide a propose method")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        if not callable(clock):
            raise TypeError("clock must be callable")
        _positive_integer(
            candidates_per_iteration,
            "candidates_per_iteration",
        )

        self.repository_root = Path(repository_root)
        self.run_id = _safe_identifier(run_id, "run_id")
        self.config = config
        self.split_manifest = split_manifest
        self.validation_samples = tuple(validation_samples)
        self.validation_sample_ids = tuple(
            split_manifest["splits"]["validation"]["sample_ids"]
        )
        if not self.validation_sample_ids:
            raise ValueError("validation split must not be empty")
        self.solver = solver
        self.proposer = proposer
        self.limits = limits or BudgetLimits()
        self.early_stopping = early_stopping or EarlyStoppingConfig()
        self.evaluator = evaluator
        self.candidates_per_iteration = candidates_per_iteration
        self.prior_search_feedback = normalize_search_feedback(
            prior_search_feedback
        )
        for history in self.prior_search_feedback["histories"]:
            if history["split_sha256"] != split_manifest["split_sha256"]:
                raise ValueError(
                    "proposer history must use the fixed search split"
                )
            if history["model"] != config.model:
                raise ValueError(
                    "proposer history must use the fixed solver model"
                )
        self.clock = clock
        self.transition_hook = transition_hook
        self.candidate_store = _RunCandidateStore(
            self.repository_root,
            self.run_id,
        )
        self.run_directory = self.candidate_store.run_directory
        self.state_path = self.run_directory / _STATE_FILE
        self.lock_path = self.run_directory / _LOCK_FILE
        self._session_started = 0.0
        self._elapsed_before_session = 0.0
        self._state: dict[str, Any] = {}

        _validate_loaded_samples(
            self.validation_samples,
            self.validation_sample_ids,
        )

    def planned_workload(self) -> dict[str, Any]:
        """Return a side-effect-free, final-test-free workload summary."""

        candidate_limit = self.limits.max_candidates
        iteration_capacity = self.limits.max_iterations
        if candidate_limit is not None:
            iteration_capacity = min(
                iteration_capacity,
                candidate_limit // self.candidates_per_iteration,
            )
        planned_candidates = iteration_capacity * self.candidates_per_iteration
        planned_evaluations = 1 + planned_candidates
        planned_solver_calls = (
            planned_evaluations * len(self.validation_sample_ids)
        )
        if self.limits.max_solver_calls is not None:
            planned_solver_calls = min(
                planned_solver_calls,
                self.limits.max_solver_calls,
            )
        return {
            "run_id": self.run_id,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "evaluation_procedure": EVALUATION_PROCEDURE,
            "model": self.config.model,
            "config_sha256": self.config.sha256(),
            "split_sha256": self.split_manifest["split_sha256"],
            "validation_sample_count": len(self.validation_sample_ids),
            "validation_sample_ids_sha256": _json_sha256(
                self.validation_sample_ids
            ),
            "baseline_evaluations": 1,
            "planned_iterations": iteration_capacity,
            "planned_candidates": planned_candidates,
            "planned_evaluations": planned_evaluations,
            "planned_solver_calls_upper_bound": planned_solver_calls,
            "limits": self.limits.as_dict(),
            "early_stopping": self.early_stopping.as_dict(),
            "prior_search_candidate_count": search_feedback_candidate_count(
                self.prior_search_feedback
            ),
            "prior_search_feedback_sha256": search_feedback_sha256(
                self.prior_search_feedback
            ),
        }

    def run(
        self,
        *,
        resume: bool = False,
        safe_resume_changes: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Run or resume search while holding the single-writer lock."""

        with run_lock(self.lock_path):
            self._session_started = _clock_value(self.clock)
            if resume:
                self._state = _load_state(self.state_path)
                if (
                    self.run_directory
                    / "finalization"
                    / "frozen_winner.json"
                ).is_file():
                    raise FinalizedRunError(
                        "a frozen winner exists; finalized runs cannot resume "
                        "prompt evolution"
                    )
                self._elapsed_before_session = float(
                    self._state["budgets"]["consumed"]["wall_time_seconds"]
                )
                self._validate_resume(tuple(safe_resume_changes))
            else:
                if self.state_path.exists():
                    raise OrchestratorError(
                        "run state already exists; use resume with this run_id"
                    )
                self._state = self._initial_state()
                self._persist_transition("initialized")

            if self._state["status"] in _TERMINAL_RUN_STATUSES:
                return _detached(self._state)

            self._ensure_baseline()
            self._reconcile_all_usage()
            try:
                self._advance()
            finally:
                self._refresh_wall_time()
                _atomic_write_json(self.state_path, self._state)
            return _detached(self._state)

    def _advance(self) -> None:
        baseline = self._state["candidates"][BASELINE_CANDIDATE_ID]
        if baseline["status"] not in _TERMINAL_CANDIDATE_STATUSES:
            if not self._can_evaluate(BASELINE_CANDIDATE_ID):
                self._stop("budget_exhausted", "baseline_budget")
                return
            self._evaluate_one(BASELINE_CANDIDATE_ID)
            if self._state["status"] in _TERMINAL_RUN_STATUSES:
                return
        if not self._baseline_has_full_coverage():
            self._stop("baseline_incomplete", "baseline_coverage")
            return

        while True:
            iteration = self._state["next_iteration"]
            iteration_state = self._state["iterations"].get(str(iteration))
            if iteration_state is None:
                stop_reason = self._stop_reason_before_iteration()
                if stop_reason is not None:
                    status = (
                        "completed"
                        if stop_reason == "max_iterations"
                        else "failure_limit"
                        if stop_reason == "consecutive_failures"
                        else "budget_exhausted"
                    )
                    self._stop(status, stop_reason)
                    return
                iteration_state = {
                    "iteration": iteration,
                    "status": "pending",
                    "candidate_ids": [],
                    "failures": [],
                    "best_macro_f1_before": self._best_macro_f1(),
                    "proposer_metadata": None,
                    "proposer_call_counted": False,
                }
                self._state["iterations"][str(iteration)] = iteration_state
                self._persist_transition("iteration_started")

            if iteration_state["status"] in {"pending", "proposing"}:
                if not self._propose_iteration(iteration_state):
                    self._finish_failed_iteration(iteration_state)
                    if self._failure_limit_reached():
                        self._stop(
                            "failure_limit",
                            "consecutive_failures",
                        )
                        return
                    continue

            for candidate_id in tuple(iteration_state["candidate_ids"]):
                candidate_state = self._state["candidates"][candidate_id]
                if candidate_state["status"] in _TERMINAL_CANDIDATE_STATUSES:
                    continue
                self._reconcile_candidate_usage(candidate_id)
                if not self._can_evaluate(candidate_id):
                    self._stop("budget_exhausted", "evaluation_budget")
                    return
                self._evaluate_one(candidate_id)
                if self._state["status"] in _TERMINAL_RUN_STATUSES:
                    return
                if self._failure_limit_reached():
                    self._stop("failure_limit", "consecutive_failures")
                    return

            self._complete_iteration(iteration_state)
            if self._should_early_stop():
                self._stop("early_stopped", "patience")
                return

    def _ensure_baseline(self) -> None:
        if BASELINE_CANDIDATE_ID not in self._state["candidates"]:
            baseline = self.candidate_store.load(BASELINE_CANDIDATE_ID)
            self._state["candidates"][BASELINE_CANDIDATE_ID] = (
                self._candidate_state(baseline, iteration=0, role="baseline")
            )
            self._persist_transition("baseline_validated")
            return
        self.candidate_store.load(BASELINE_CANDIDATE_ID)

    def _baseline_has_full_coverage(self) -> bool:
        baseline = self._state["candidates"][BASELINE_CANDIDATE_ID]
        score = baseline.get("score")
        return bool(
            baseline.get("status") == "evaluated"
            and score
            and score.get("request_coverage") == 1.0
            and score.get("parse_coverage") == 1.0
            and score.get("unresolved_api_failures") == 0
        )

    def _propose_iteration(self, iteration_state: dict[str, Any]) -> bool:
        iteration = iteration_state["iteration"]
        recovered = self._recover_proposal(iteration)
        if recovered is not None:
            batch, metadata = recovered
        else:
            iteration_state["status"] = "proposing"
            self._persist_transition("proposal_started")
            try:
                result = self.proposer.propose(
                    self.candidate_store,
                    iteration=iteration,
                    parent_candidate_ids=self._parent_candidate_ids(),
                    validation_scores=self._proposer_scores(),
                    aggregate_metrics=self._aggregate_feedback(),
                    failure_summaries=self._failure_feedback(),
                    search_feedback=self._proposer_search_feedback(),
                )
                batch, metadata = _proposal_parts(result)
                self._write_proposal_checkpoint(batch, metadata)
            except Exception as exc:
                self._record_failure(
                    scope="proposer",
                    iteration=iteration,
                    candidate_id=None,
                    error=exc,
                )
                iteration_state["status"] = "failed"
                iteration_state["failures"].append("proposer")
                if not iteration_state["proposer_call_counted"]:
                    self._state["budgets"]["consumed"]["proposer_calls"] += 1
                    iteration_state["proposer_call_counted"] = True
                self._persist_transition("proposal_failed")
                return False

        if not iteration_state["proposer_call_counted"]:
            self._state["budgets"]["consumed"]["proposer_calls"] += 1
            iteration_state["proposer_call_counted"] = True
            self._persist_transition("proposal_returned")
        if (
            batch.iteration != iteration
            or len(batch.candidates) != self.candidates_per_iteration
        ):
            error = OrchestratorError(
                "proposer returned a batch that violates the frozen workload"
            )
            self._record_failure(
                scope="proposer",
                iteration=iteration,
                candidate_id=None,
                error=error,
            )
            iteration_state["status"] = "failed"
            iteration_state["failures"].append("proposer_contract")
            self._persist_transition("proposal_failed")
            return False

        candidate_ids: list[str] = []
        try:
            known_prompt_hashes = {
                candidate["prompt_sha256"]
                for candidate in self._state["candidates"].values()
            }
            for candidate in batch.candidates:
                existing_state = self._state["candidates"].get(
                    candidate.candidate_id
                )
                if existing_state is not None and (
                    existing_state["candidate_sha256"] != candidate.sha256()
                    or existing_state["prompt_sha256"]
                    != candidate.source_sha256
                ):
                    raise OrchestratorError(
                        "candidate ID conflicts with its durable run state"
                    )
                if (
                    existing_state is None
                    and candidate.source_sha256 in known_prompt_hashes
                ):
                    raise OrchestratorError(
                        "proposed prompt content must be new within the run"
                    )
                stored = self.candidate_store.create(
                    candidate,
                    status="validated",
                )
                self.candidate_store.update_status(
                    stored.candidate_id,
                    "validated",
                )
                candidate_ids.append(stored.candidate_id)
                known_prompt_hashes.add(stored.source_sha256)
                self._state["candidates"].setdefault(
                    stored.candidate_id,
                    self._candidate_state(
                        stored,
                        iteration=iteration,
                        role="candidate",
                    ),
                )
                self._persist_transition("candidate_validated")
        except Exception as exc:
            self._record_failure(
                scope="storage",
                iteration=iteration,
                candidate_id=None,
                error=exc,
            )
            iteration_state["status"] = "failed"
            iteration_state["failures"].append("storage")
            self._persist_transition("proposal_storage_failed")
            return False

        iteration_state["candidate_ids"] = candidate_ids
        iteration_state["proposer_metadata"] = _metadata_copy(metadata)
        iteration_state["status"] = "proposed"
        self._state["proposer_metadata"][str(iteration)] = _metadata_copy(
            metadata
        )
        self._state["budgets"]["consumed"]["candidates"] = sum(
            candidate["role"] == "candidate"
            for candidate in self._state["candidates"].values()
        )
        self._reset_consecutive_failures()
        self._persist_transition("proposal_completed")
        return True

    def _evaluate_one(self, candidate_id: str) -> None:
        candidate_state = self._state["candidates"][candidate_id]
        result_path, metrics_path = self._evaluation_paths(candidate_id)
        candidate_state["result_path"] = str(
            result_path.relative_to(self.run_directory)
        )
        candidate_state["metrics_path"] = str(
            metrics_path.relative_to(self.run_directory)
        )

        report: Mapping[str, Any] | None = None
        candidate_state["status"] = "evaluating"
        self._persist_transition("evaluation_started")
        try:
            if metrics_path.is_file():
                report = _load_json_mapping(metrics_path, "evaluation report")
            else:
                report = self.evaluator(
                    run_id=self.run_id,
                    candidate_id=candidate_id,
                    candidate_store=self.candidate_store,
                    split_manifest=self.split_manifest,
                    split_name="validation",
                    sample_ids=self.validation_sample_ids,
                    samples=self.validation_samples,
                    solver=self.solver,
                    output_path=result_path,
                    metrics_path=metrics_path,
                    config=self.config,
                )
                if not isinstance(report, Mapping):
                    raise OrchestratorError(
                        "evaluator must return a metrics mapping"
                    )
                _atomic_write_json(metrics_path, report)
            score = _score_from_report(
                report,
                result_path=result_path,
                candidate_id=candidate_id,
            )
        except (KeyboardInterrupt, SystemExit):
            self._reconcile_candidate_usage(candidate_id)
            self._persist_transition("evaluation_interrupted")
            raise
        except Exception as exc:
            usage = _resource_usage(
                None,
                result_path=result_path,
                candidate_id=candidate_id,
            )
            candidate_state["usage"] = usage
            candidate_state["status"] = "failed"
            candidate_state["failure"] = _safe_error(exc)
            self.candidate_store.update_status(candidate_id, "rejected")
            self._record_failure(
                scope="evaluator",
                iteration=candidate_state["iteration"],
                candidate_id=candidate_id,
                error=exc,
            )
            self._recompute_budget_usage()
            self._persist_transition("evaluation_failed")
            return

        candidate_state["score"] = score
        candidate_state["usage"] = {
            key: score[key]
            for key in ("solver_calls", "tokens", "latency_seconds")
        }
        candidate_state["status"] = "evaluated"
        candidate_state["failure"] = None
        self.candidate_store.update_status(candidate_id, "evaluated")
        self._recompute_budget_usage()
        self._reset_consecutive_failures()
        self._persist_transition("evaluation_completed")
        self._update_frontier()

    def _complete_iteration(self, iteration_state: dict[str, Any]) -> None:
        # A process may stop after an evaluation checkpoint and before its
        # separate frontier transition. Rebuilding from all durable scores is
        # deterministic and keeps that resume path idempotent.
        self._update_frontier()
        candidate_states = [
            self._state["candidates"][candidate_id]
            for candidate_id in iteration_state["candidate_ids"]
        ]
        failed = any(
            candidate["status"] == "failed" for candidate in candidate_states
        )
        if failed:
            iteration_state["status"] = "partial_failure"
            self._state["early_stopping"]["failed_iterations"] += 1
        else:
            iteration_state["status"] = "completed"
            self._state["early_stopping"]["completed_iterations"] += 1
            before = iteration_state["best_macro_f1_before"]
            after = self._best_macro_f1()
            improved = (
                after is not None
                and (
                    before is None
                    or after - before >= self.early_stopping.min_delta
                )
            )
            if improved:
                self._state["early_stopping"][
                    "non_improving_iterations"
                ] = 0
            else:
                self._state["early_stopping"][
                    "non_improving_iterations"
                ] += 1
        self._state["next_iteration"] = iteration_state["iteration"] + 1
        self._persist_transition("iteration_completed")

    def _finish_failed_iteration(self, iteration_state: dict[str, Any]) -> None:
        self._state["early_stopping"]["failed_iterations"] += 1
        self._state["next_iteration"] = iteration_state["iteration"] + 1
        self._persist_transition("iteration_failed")

    def _update_frontier(self) -> None:
        entries = []
        for candidate_id, candidate in self._state["candidates"].items():
            score = candidate.get("score")
            if not score or not score["eligible"]:
                continue
            entries.append(
                FrontierEntry(
                    candidate_id=candidate_id,
                    macro_f1=score["macro_f1"],
                    accuracy=score["accuracy"],
                    solver_calls=score["solver_calls"],
                    tokens=score["tokens"],
                    latency_seconds=score["latency_seconds"],
                )
            )
        frontier = pareto_frontier(entries)
        self._state["frontier"] = [entry.as_dict() for entry in frontier]
        self._state["best_candidate_id"] = (
            min(entries, key=winner_rank_key).candidate_id
            if entries
            else None
        )
        self._persist_transition("frontier_updated")

    def _parent_candidate_ids(self) -> list[str]:
        frontier = self._state["frontier"]
        if frontier:
            return [entry["candidate_id"] for entry in frontier]
        return [BASELINE_CANDIDATE_ID]

    def _proposer_scores(self) -> dict[str, dict[str, Any]]:
        scores: dict[str, dict[str, Any]] = {}
        for candidate_id in self._parent_candidate_ids():
            score = self._state["candidates"][candidate_id].get("score")
            if score is None:
                continue
            scores[candidate_id] = {
                "macro_f1": score["macro_f1"],
                "accuracy": score["accuracy"],
                "parse_coverage": score["parse_coverage"],
                "request_coverage": score["request_coverage"],
                "unresolved_api_failures": score[
                    "unresolved_api_failures"
                ],
            }
        return scores

    def _aggregate_feedback(self) -> dict[str, int | float]:
        consumed = self._state["budgets"]["consumed"]
        evaluated = sum(
            candidate["status"] == "evaluated"
            for candidate in self._state["candidates"].values()
        )
        eligible = sum(
            bool((candidate.get("score") or {}).get("eligible"))
            for candidate in self._state["candidates"].values()
        )
        return {
            "completed_candidates": evaluated,
            "eligible_rate": eligible / evaluated if evaluated else 0.0,
            "solver_calls": consumed["solver_calls"],
            "tokens": consumed["tokens"],
        }

    def _failure_feedback(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for failure in self._state["failures"]["records"]:
            key = f"{failure['scope']}_failures"
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _proposer_search_feedback(self) -> dict[str, Any]:
        ordered_candidates = sorted(
            (
                candidate
                for candidate in self._state["candidates"].values()
                if candidate.get("status") == "evaluated"
            ),
            key=lambda candidate: (
                candidate["iteration"],
                candidate["candidate_id"],
            ),
        )
        current = current_search_feedback(
            self.candidate_store,
            self.run_directory / "evaluations",
            candidate_ids=[
                candidate["candidate_id"]
                for candidate in ordered_candidates
            ],
        )
        return merge_search_feedback(
            self.prior_search_feedback,
            current,
        )

    def _candidate_state(
        self,
        candidate: Candidate,
        *,
        iteration: int,
        role: str,
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "candidate_sha256": candidate.sha256(),
            "prompt_sha256": candidate.source_sha256,
            "role": role,
            "iteration": iteration,
            "status": "validated",
            "score": None,
            "usage": {
                "solver_calls": 0,
                "tokens": 0,
                "latency_seconds": 0.0,
            },
            "result_path": None,
            "metrics_path": None,
            "failure": None,
        }

    def _initial_state(self) -> dict[str, Any]:
        snapshot = self._configuration_snapshot()
        return {
            "schema_version": RUN_STATE_SCHEMA_VERSION,
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "run_id": self.run_id,
            "status": "running",
            "stop_reason": None,
            "transition": 0,
            "last_transition": None,
            "next_iteration": 1,
            "configuration": snapshot,
            "configuration_sha256": _json_sha256(snapshot),
            "budgets": {
                "limits": self.limits.as_dict(),
                "consumed": {
                    "iterations": 0,
                    "candidates": 0,
                    "proposer_calls": 0,
                    "solver_calls": 0,
                    "tokens": 0,
                    "wall_time_seconds": 0.0,
                },
            },
            "candidates": {},
            "scores": {},
            "frontier": [],
            "best_candidate_id": None,
            "iterations": {},
            "early_stopping": {
                **self.early_stopping.as_dict(),
                "completed_iterations": 0,
                "failed_iterations": 0,
                "non_improving_iterations": 0,
            },
            "failures": {
                "total": 0,
                "consecutive": 0,
                "records": [],
            },
            "proposer_metadata": {},
        }

    def _configuration_snapshot(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "config_sha256": self.config.sha256(),
            "split_sha256": self.split_manifest["split_sha256"],
            "dataset_sha256": self.split_manifest["dataset_sha256"],
            "validation_sample_count": len(self.validation_sample_ids),
            "validation_sample_ids_sha256": _json_sha256(
                self.validation_sample_ids
            ),
            "evaluation_procedure": EVALUATION_PROCEDURE,
            "candidates_per_iteration": self.candidates_per_iteration,
            "limits": self.limits.as_dict(),
            "early_stopping": self.early_stopping.as_dict(),
            "prior_search_feedback_sha256": search_feedback_sha256(
                self.prior_search_feedback
            ),
        }

    def _validate_resume(self, safe_changes: tuple[str, ...]) -> None:
        if self._state.get("schema_version") != RUN_STATE_SCHEMA_VERSION:
            raise ResumeConfigurationError(
                "run state schema_version is not supported"
            )
        if self._state.get("run_id") != self.run_id:
            raise ResumeConfigurationError("run_id does not match run state")
        if any(field not in _SAFE_RESUME_FIELDS for field in safe_changes):
            raise ResumeConfigurationError(
                "safe resume changes may include resource limits only"
            )
        if len(safe_changes) != len(set(safe_changes)):
            raise ResumeConfigurationError(
                "safe resume changes must not contain duplicates"
            )

        previous = self._state["configuration"]
        current = self._configuration_snapshot()
        immutable_fields = set(current) - {"limits"}
        changed_immutable = sorted(
            field
            for field in immutable_fields
            if previous.get(field) != current.get(field)
        )
        if changed_immutable:
            raise ResumeConfigurationError(
                "resume configuration changed immutable fields: "
                + ", ".join(changed_immutable)
            )

        previous_limits = previous["limits"]
        current_limits = current["limits"]
        changed_limits = {
            field
            for field in current_limits
            if previous_limits.get(field) != current_limits.get(field)
        }
        unapproved = sorted(changed_limits - set(safe_changes))
        if unapproved:
            raise ResumeConfigurationError(
                "resume resource limits changed without explicit safe marking: "
                + ", ".join(unapproved)
            )
        for field in changed_limits:
            _validate_non_decreasing_limit(
                field,
                previous_limits[field],
                current_limits[field],
            )
        if changed_limits:
            self._state["configuration"] = current
            self._state["configuration_sha256"] = _json_sha256(current)
            self._state["budgets"]["limits"] = self.limits.as_dict()
            self._persist_transition("safe_configuration_extended")
            if self._state["status"] in {
                "completed",
                "budget_exhausted",
                "failure_limit",
            }:
                self._state["status"] = "running"
                self._state["stop_reason"] = None
                self._persist_transition("run_reopened")

    def _stop_reason_before_iteration(self) -> str | None:
        consumed = self._state["budgets"]["consumed"]
        if self._failure_limit_reached():
            return "consecutive_failures"
        if self._state["next_iteration"] > self.limits.max_iterations:
            return "max_iterations"
        if (
            self.limits.max_candidates is not None
            and consumed["candidates"] + self.candidates_per_iteration
            > self.limits.max_candidates
        ):
            return "candidate_budget"
        if (
            self.limits.max_solver_calls is not None
            and consumed["solver_calls"] >= self.limits.max_solver_calls
        ):
            return "solver_call_budget"
        if (
            self.limits.max_tokens is not None
            and consumed["tokens"] >= self.limits.max_tokens
        ):
            return "token_budget"
        if self._wall_time_exhausted():
            return "wall_time_budget"
        return None

    def _proposal_fits_budget(self) -> bool:
        consumed = self._state["budgets"]["consumed"]["candidates"]
        return (
            self.limits.max_candidates is None
            or consumed + self.candidates_per_iteration
            <= self.limits.max_candidates
        )

    def _can_evaluate(self, candidate_id: str) -> bool:
        _, metrics_path = self._evaluation_paths(candidate_id)
        if metrics_path.is_file():
            return True
        self._refresh_wall_time()
        if self._wall_time_exhausted():
            return False
        consumed = self._state["budgets"]["consumed"]
        required_calls = self._remaining_sample_calls(candidate_id)
        if (
            self.limits.max_solver_calls is not None
            and consumed["solver_calls"] + required_calls
            > self.limits.max_solver_calls
        ):
            return False
        if (
            self.limits.max_tokens is not None
            and consumed["tokens"] >= self.limits.max_tokens
        ):
            return False
        return True

    def _remaining_sample_calls(self, candidate_id: str) -> int:
        result_path, _ = self._evaluation_paths(candidate_id)
        completed = _completed_sample_ids(result_path, candidate_id)
        return max(0, len(self.validation_sample_ids) - len(completed))

    def _failure_limit_reached(self) -> bool:
        return (
            self._state["failures"]["consecutive"]
            >= self.limits.max_consecutive_failures
        )

    def _should_early_stop(self) -> bool:
        state = self._state["early_stopping"]
        return (
            self.early_stopping.patience is not None
            and state["completed_iterations"]
            >= self.early_stopping.min_iterations
            and state["non_improving_iterations"]
            >= self.early_stopping.patience
        )

    def _best_macro_f1(self) -> float | None:
        if not self._state["frontier"]:
            return None
        return max(entry["macro_f1"] for entry in self._state["frontier"])

    def _stop(self, status: str, reason: str) -> None:
        self._state["status"] = status
        self._state["stop_reason"] = reason
        self._persist_transition(status)

    def _record_failure(
        self,
        *,
        scope: str,
        iteration: int,
        candidate_id: str | None,
        error: BaseException,
    ) -> None:
        failures = self._state["failures"]
        failures["total"] += 1
        failures["consecutive"] += 1
        failures["records"].append(
            {
                "scope": scope,
                "iteration": iteration,
                "candidate_id": candidate_id,
                "error_type": type(error).__name__,
                "message": _safe_error(error),
            }
        )

    def _reset_consecutive_failures(self) -> None:
        self._state["failures"]["consecutive"] = 0

    def _evaluation_paths(self, candidate_id: str) -> tuple[Path, Path]:
        directory = (
            self.run_directory / "evaluations" / _safe_identifier(
                candidate_id,
                "candidate_id",
            )
        )
        return (
            directory / "validation.results.jsonl",
            directory / "validation.metrics.json",
        )

    def _recover_proposal(
        self,
        iteration: int,
    ) -> tuple[CandidateBatch, Mapping[str, Any]] | None:
        checkpoint_path = self._proposal_checkpoint_path(iteration)
        if checkpoint_path.is_file():
            checkpoint = _load_json_mapping(
                checkpoint_path,
                "proposal checkpoint",
            )
            batch = CandidateBatch.from_mapping(
                checkpoint.get("batch"),
                candidate_count=self.candidates_per_iteration,
                existing_parent_ids=self._state["candidates"],
            )
            metadata = checkpoint.get("metadata")
            if not isinstance(metadata, Mapping):
                raise OrchestratorError(
                    "proposal checkpoint metadata must be a mapping"
                )
            return batch, metadata

        directory = (
            self.run_directory / "proposer" / f"iteration_{iteration:04d}"
        )
        if not directory.is_dir():
            return None
        for audit_path in sorted(directory.glob("attempt_*.json"), reverse=True):
            audit = _load_json_mapping(audit_path, "proposer audit")
            if audit.get("status") != "success":
                continue
            candidate_ids = audit.get("candidate_ids")
            if (
                isinstance(candidate_ids, (str, bytes))
                or not isinstance(candidate_ids, Sequence)
                or len(candidate_ids) != self.candidates_per_iteration
            ):
                continue
            candidates = tuple(
                self.candidate_store.load(candidate_id)
                for candidate_id in candidate_ids
            )
            return CandidateBatch(iteration, candidates), audit
        return None

    def _proposal_checkpoint_path(self, iteration: int) -> Path:
        return (
            self.run_directory
            / "iterations"
            / f"iteration_{iteration:04d}"
            / "proposal.json"
        )

    def _write_proposal_checkpoint(
        self,
        batch: CandidateBatch,
        metadata: Mapping[str, Any],
    ) -> None:
        path = self._proposal_checkpoint_path(batch.iteration)
        payload = {
            "batch": batch.as_dict(),
            "metadata": _metadata_copy(metadata),
        }
        if path.is_file():
            existing = _load_json_mapping(path, "proposal checkpoint")
            if canonical_json(existing) != canonical_json(payload):
                raise OrchestratorError(
                    "proposal checkpoint conflicts with an existing batch"
                )
            return
        _atomic_write_json(path, payload)

    def _reconcile_all_usage(self) -> None:
        changed = False
        for candidate_id in self._state["candidates"]:
            changed = self._reconcile_candidate_usage(
                candidate_id,
                persist=False,
            ) or changed
        if changed:
            self._recompute_budget_usage()
            self._persist_transition("budget_reconciled")

    def _reconcile_candidate_usage(
        self,
        candidate_id: str,
        *,
        persist: bool = True,
    ) -> bool:
        result_path, _ = self._evaluation_paths(candidate_id)
        _, metrics_path = self._evaluation_paths(candidate_id)
        report = (
            _load_json_mapping(metrics_path, "evaluation report")
            if metrics_path.is_file()
            else None
        )
        usage = _resource_usage(
            report,
            result_path=result_path,
            candidate_id=candidate_id,
        )
        candidate = self._state["candidates"][candidate_id]
        changed = candidate["usage"] != usage
        if changed:
            candidate["usage"] = usage
            self._recompute_budget_usage()
            if persist:
                self._persist_transition("candidate_usage_reconciled")
        return changed

    def _recompute_budget_usage(self) -> None:
        consumed = self._state["budgets"]["consumed"]
        consumed["candidates"] = sum(
            candidate["role"] == "candidate"
            for candidate in self._state["candidates"].values()
        )
        consumed["solver_calls"] = sum(
            candidate["usage"]["solver_calls"]
            for candidate in self._state["candidates"].values()
        )
        consumed["tokens"] = sum(
            candidate["usage"]["tokens"]
            for candidate in self._state["candidates"].values()
        )
        consumed["iterations"] = sum(
            iteration["status"] in {"completed", "partial_failure", "failed"}
            for iteration in self._state["iterations"].values()
        )
        self._state["scores"] = {
            candidate_id: candidate["score"]
            for candidate_id, candidate in self._state["candidates"].items()
            if candidate["score"] is not None
        }

    def _refresh_wall_time(self) -> None:
        if not self._session_started:
            return
        now = _clock_value(self.clock)
        session_elapsed = max(0.0, now - self._session_started)
        self._state["budgets"]["consumed"]["wall_time_seconds"] = (
            self._elapsed_before_session + session_elapsed
        )

    def _wall_time_exhausted(self) -> bool:
        maximum = self.limits.max_wall_time_seconds
        return (
            maximum is not None
            and self._state["budgets"]["consumed"]["wall_time_seconds"]
            >= maximum
        )

    def _persist_transition(self, name: str) -> None:
        self._refresh_wall_time()
        self._recompute_budget_usage()
        self._state["transition"] += 1
        self._state["last_transition"] = name
        _atomic_write_json(self.state_path, self._state)
        if self.transition_hook is not None:
            self.transition_hook(name, _detached(self._state))


def run_search(**kwargs: Any) -> dict[str, Any]:
    """Construct and run a :class:`MetaHarnessOrchestrator`."""

    resume = bool(kwargs.pop("resume", False))
    safe_resume_changes = kwargs.pop("safe_resume_changes", ())
    orchestrator = MetaHarnessOrchestrator(**kwargs)
    return orchestrator.run(
        resume=resume,
        safe_resume_changes=safe_resume_changes,
    )


def load_run_state(repository_root: str | Path, run_id: str) -> dict[str, Any]:
    """Load one state snapshot without acquiring write authority."""

    store = CandidateStore(repository_root, run_id)
    return _load_state(store.run_directory / _STATE_FILE)


def load_run_candidate(
    repository_root: str | Path,
    run_id: str,
    candidate_id: str,
) -> Candidate:
    """Load a stored proposal or the byte-preserved virtual baseline."""

    return _RunCandidateStore(repository_root, run_id).load(candidate_id)


def score_evaluation_report(
    report: Mapping[str, Any],
    *,
    result_path: str | Path,
    candidate_id: str,
) -> dict[str, Any]:
    """Return the durable candidate score used by search and targeted retries."""

    return _score_from_report(
        report,
        result_path=Path(result_path),
        candidate_id=candidate_id,
    )


def recompute_selection_state(state: dict[str, Any]) -> None:
    """Rebuild the eligible frontier and winner in an already validated state."""

    candidates = state.get("candidates")
    if not isinstance(candidates, Mapping):
        raise OrchestratorError("run state candidates must be a mapping")
    entries = []
    for candidate_id, candidate in candidates.items():
        if not isinstance(candidate, Mapping):
            raise OrchestratorError("run state candidate entry must be a mapping")
        score = candidate.get("score")
        if not isinstance(score, Mapping) or not score.get("eligible"):
            continue
        entries.append(
            FrontierEntry(
                candidate_id=candidate_id,
                macro_f1=score["macro_f1"],
                accuracy=score["accuracy"],
                solver_calls=score["solver_calls"],
                tokens=score["tokens"],
                latency_seconds=score["latency_seconds"],
            )
        )
    frontier = pareto_frontier(entries)
    state["frontier"] = [entry.as_dict() for entry in frontier]
    state["best_candidate_id"] = (
        min(entries, key=winner_rank_key).candidate_id
        if entries
        else None
    )


def _proposal_parts(result: Any) -> tuple[CandidateBatch, Mapping[str, Any]]:
    batch = getattr(result, "batch", result)
    metadata = getattr(result, "metadata", {})
    if not isinstance(batch, CandidateBatch):
        raise OrchestratorError(
            "proposer must return CandidateBatch or a result containing one"
        )
    if not isinstance(metadata, Mapping):
        raise OrchestratorError("proposer metadata must be a mapping")
    return batch, metadata


def _score_from_report(
    report: Mapping[str, Any],
    *,
    result_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    metrics = report.get("metrics", report)
    if not isinstance(metrics, Mapping):
        raise OrchestratorError("evaluation report metrics must be a mapping")
    macro_f1 = _metric(metrics, "Macro-F1", "macro_f1")
    accuracy = _metric(metrics, "accuracy")
    parse_coverage = _metric(metrics, "coverage", "parse_coverage")
    request_coverage = _request_coverage(
        report,
        metrics,
        parse_coverage=parse_coverage,
    )
    unresolved = report.get(
        "unresolved_api_failures",
        metrics.get("unresolved_api_failures", 0),
    )
    _non_negative_integer(unresolved, "unresolved_api_failures")
    rankable = report.get("rankable")
    eligible = (
        request_coverage == 1.0
        and parse_coverage == 1.0
        and unresolved == 0
        and (rankable is not False)
    )
    usage = _resource_usage(
        report,
        result_path=result_path,
        candidate_id=candidate_id,
    )
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "request_coverage": request_coverage,
        "parse_coverage": parse_coverage,
        "unresolved_api_failures": unresolved,
        "eligible": eligible,
        **usage,
    }


def _resource_usage(
    report: Mapping[str, Any] | None,
    *,
    result_path: Path,
    candidate_id: str,
) -> dict[str, Any]:
    records = _result_records(result_path, candidate_id)
    record_calls = sum(
        record.get("request_status") != "invalid_input" for record in records
    )
    record_tokens = 0
    record_latency = 0.0
    for record in records:
        usage = record.get("usage")
        if isinstance(usage, Mapping):
            if _is_non_negative_int(usage.get("total_tokens")):
                record_tokens += usage["total_tokens"]
            else:
                for field in ("input_tokens", "output_tokens"):
                    if _is_non_negative_int(usage.get(field)):
                        record_tokens += usage[field]
        latency = record.get("latency")
        if _is_non_negative_number(latency):
            record_latency += float(latency)

    source: Mapping[str, Any] = {}
    if report is not None:
        raw = report.get("resource_usage", report)
        if isinstance(raw, Mapping):
            source = raw
    solver_calls = source.get("solver_calls", record_calls)
    tokens = source.get("tokens", source.get("total_tokens", record_tokens))
    latency = source.get(
        "latency_seconds",
        source.get("latency", record_latency),
    )
    _non_negative_integer(solver_calls, "solver_calls")
    _non_negative_integer(tokens, "tokens")
    _non_negative_number(latency, "latency_seconds")
    return {
        "solver_calls": solver_calls,
        "tokens": tokens,
        "latency_seconds": float(latency),
    }


def _result_records(path: Path, candidate_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    result_keys: set[tuple[str, str, str, int]] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if (
                isinstance(value, dict)
                and value.get("candidate_id", value.get("prompt_variant"))
                == candidate_id
            ):
                attempt_count = value.get("attempt_count")
                if (
                    not isinstance(attempt_count, int)
                    or isinstance(attempt_count, bool)
                    or attempt_count < 1
                ):
                    raise OrchestratorError(
                        "evaluation result attempt_count must be a positive integer"
                    )
                key = (
                    candidate_id,
                    _identity(value.get("sample_id")),
                    str(value.get("split_sha256")),
                    attempt_count,
                )
                if key in result_keys:
                    raise OrchestratorError(
                        "duplicate attempt_count for one candidate/sample/split "
                        "evaluation result"
                    )
                result_keys.add(key)
                records.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrchestratorError(
            "evaluation results must be readable valid JSONL"
        ) from exc
    return records


def _completed_sample_ids(path: Path, candidate_id: str) -> set[str]:
    completed = set()
    for record in _result_records(path, candidate_id):
        completed.add(_identity(record.get("sample_id")))
    return completed


def _validate_loaded_samples(
    samples: Sequence[Any],
    validation_ids: Sequence[Any],
) -> None:
    loaded_ids: list[str] = []
    for index, sample in enumerate(samples):
        if hasattr(sample, "sample_id"):
            sample_id = sample.sample_id
        elif isinstance(sample, Mapping):
            sample_id = sample.get("sample_id")
        else:
            raise ValueError(
                f"validation sample at index {index} has no sample identity"
            )
        loaded_ids.append(_identity(sample_id))
    if len(loaded_ids) != len(set(loaded_ids)):
        raise ValueError("validation_samples must not contain duplicate IDs")
    expected_ids = {_identity(sample_id) for sample_id in validation_ids}
    if set(loaded_ids) != expected_ids:
        raise ValueError(
            "validation_samples must contain exactly the fixed validation "
            "split and no search or final-test samples"
        )


def _metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in metrics:
            value = metrics[name]
            _unit_interval(value, name)
            return float(value)
    raise OrchestratorError(
        "evaluation report is missing metric: " + " or ".join(names)
    )


def _request_coverage(
    report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    parse_coverage: float,
) -> float:
    if "request_coverage" in metrics:
        return _metric(metrics, "request_coverage")
    failure_counts = report.get("failure_counts")
    sample_count = report.get("sample_count")
    if (
        isinstance(failure_counts, Mapping)
        and isinstance(sample_count, int)
        and not isinstance(sample_count, bool)
        and sample_count > 0
    ):
        invalid_inputs = failure_counts.get("invalid_input")
        api_failures = failure_counts.get("api_failure")
        if (
            isinstance(invalid_inputs, int)
            and not isinstance(invalid_inputs, bool)
            and invalid_inputs >= 0
            and isinstance(api_failures, int)
            and not isinstance(api_failures, bool)
            and api_failures >= 0
        ):
            coverage = 1.0 - ((invalid_inputs + api_failures) / sample_count)
            _unit_interval(coverage, "request_coverage")
            return coverage
    # Compatibility for injected offline evaluators that predate the additive
    # request-coverage field. A perfect parse rate proves every request also
    # completed; any lower rate cannot satisfy the baseline gate.
    return 1.0 if parse_coverage == 1.0 else 0.0


def _metadata_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = canonical_json(value)
        decoded = json.loads(encoded)
    except Exception as exc:
        raise OrchestratorError(
            "proposer metadata must be safe canonical JSON"
        ) from exc
    if not isinstance(decoded, dict):
        raise OrchestratorError("proposer metadata must be a JSON object")
    return _sanitize_metadata(decoded)


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, str):
        return _safe_error(RuntimeError(value))
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in {
                "api_key",
                "api_url",
                "authorization",
                "authorization_header",
                "image_base64",
                "base64_image",
            }:
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize_metadata(item)
        return sanitized
    return value


def _validate_non_decreasing_limit(
    field: str,
    previous: int | float | None,
    current: int | float | None,
) -> None:
    if previous is None:
        if current is not None:
            raise ResumeConfigurationError(
                f"safe resume change {field} cannot add a tighter limit"
            )
        return
    if current is not None and current < previous:
        raise ResumeConfigurationError(
            f"safe resume change {field} cannot reduce a resource limit"
        )


def _baseline_candidate() -> Candidate:
    """Construct the exact legacy snapshot without applying proposal-only rules."""

    sources = canonical_baseline_sources()
    templates = MappingProxyType(
        {
            method: sources[method]
            for method in TEMPLATE_KEYS
        }
    )
    candidate = object.__new__(Candidate)
    object.__setattr__(candidate, "candidate_id", BASELINE_CANDIDATE_ID)
    object.__setattr__(candidate, "parent_id", BASELINE_CANDIDATE_ID)
    object.__setattr__(candidate, "search_axis", "exploitation")
    object.__setattr__(
        candidate,
        "hypothesis",
        "The original prompt will establish the fixed validation reference.",
    )
    object.__setattr__(candidate, "templates", templates)
    object.__setattr__(
        candidate,
        "expected_tradeoff",
        "The unchanged baseline provides the comparison point.",
    )
    object.__setattr__(
        candidate,
        "source_sha256",
        template_source_sha256(templates),
    )
    return candidate


def _load_state(path: Path) -> dict[str, Any]:
    value = _load_json_mapping(path, "run state")
    if value.get("schema_version") == 2:
        required_v2 = {
            "schema_version",
            "orchestrator_version",
            "run_id",
            "status",
            "configuration",
            "configuration_sha256",
            "budgets",
            "candidates",
            "frontier",
            "iterations",
        }
        if not required_v2.issubset(value):
            raise OrchestratorError("staged run state is missing required fields")
        if _json_sha256(value["configuration"]) != value[
            "configuration_sha256"
        ]:
            raise OrchestratorError("run configuration snapshot hash mismatch")
        return dict(value)
    required = {
        "schema_version",
        "orchestrator_version",
        "run_id",
        "status",
        "configuration",
        "configuration_sha256",
        "budgets",
        "candidates",
        "frontier",
        "iterations",
        "failures",
        "proposer_metadata",
    }
    if not required.issubset(value):
        raise OrchestratorError("run state is missing required fields")
    if _json_sha256(value["configuration"]) != value["configuration_sha256"]:
        raise OrchestratorError("run configuration snapshot hash mismatch")
    return dict(value)


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrchestratorError(f"{label} must be readable valid JSON") from exc
    if not isinstance(value, dict):
        raise OrchestratorError(f"{label} must contain a JSON object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_error(error: BaseException) -> str:
    value = str(error)
    for name in ("API_KEY", "API_URL"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    for pattern in _SENSITIVE_TEXT:
        value = pattern.sub("[REDACTED]", value)
    return value[:1000]


def _detached(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def _identity(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("sample IDs must be JSON serializable") from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _safe_identifier(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value)
    ):
        raise ValueError(
            f"{field} must contain only letters, numbers, dot, underscore, "
            "or hyphen"
        )
    return value


def _clock_value(clock: Callable[[], float]) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise OrchestratorError("clock must return a finite number")
    return float(value)


def _positive_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")


def _optional_positive_integer(value: Any, field: str) -> None:
    if value is not None:
        _positive_integer(value, field)


def _non_negative_integer(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _positive_number(value: Any, field: str) -> None:
    if not _is_non_negative_number(value) or float(value) <= 0:
        raise ValueError(f"{field} must be a positive finite number")


def _non_negative_number(value: Any, field: str) -> None:
    if not _is_non_negative_number(value):
        raise ValueError(f"{field} must be a non-negative finite number")


def _unit_interval(value: Any, field: str) -> None:
    if not _is_non_negative_number(value) or float(value) > 1:
        raise ValueError(f"{field} must be between 0.0 and 1.0")


def _is_non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_non_negative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0
    )


__all__ = [
    "BASELINE_CANDIDATE_ID",
    "BudgetExhausted",
    "BudgetLimits",
    "EVALUATION_PROCEDURE",
    "EarlyStoppingConfig",
    "FrontierEntry",
    "FinalizedRunError",
    "MetaHarnessOrchestrator",
    "ORCHESTRATOR_VERSION",
    "OrchestratorError",
    "ResumeConfigurationError",
    "RunLockedError",
    "dominates",
    "frontier_rank_key",
    "load_run_state",
    "load_run_candidate",
    "pareto_frontier",
    "recompute_selection_state",
    "run_lock",
    "run_search",
    "score_evaluation_report",
    "winner_rank_key",
]
