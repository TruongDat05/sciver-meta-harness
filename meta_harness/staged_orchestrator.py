"""Budget-efficient search, protected-validation promotion, and durable resume."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from meta_harness.config import MetaHarnessConfig
from meta_harness.evaluator import EVALUATOR_VERSION, evaluate_candidate
from meta_harness.experience_store import SearchExperienceStore
from meta_harness.hard_search import verify_hard_search_manifest
from meta_harness.orchestrator import (
    BASELINE_CANDIDATE_ID,
    BudgetExhausted,
    BudgetLimits,
    EarlyStoppingConfig,
    FinalizedRunError,
    FrontierEntry,
    OrchestratorError,
    ResumeConfigurationError,
    _RunCandidateStore,
    _metadata_copy,
    _proposal_parts,
    _resource_usage,
    _safe_error,
    _score_from_report,
    _validate_non_decreasing_limit,
    frontier_rank_key,
    run_lock,
)
from meta_harness.schemas import CandidateBatch, canonical_json
from meta_harness.split_manager import verify_split_manifest
from utils.answer_parser import PARSER_VERSION


STAGED_RUN_STATE_SCHEMA_VERSION = 2
STAGED_ORCHESTRATOR_VERSION = "meta_harness_orchestrator_v2"
STAGED_EVALUATION_PROCEDURE = "search_smoke_promote_validation_v1"
_STATE_FILE = "run_state.json"
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


class StageRetryRequired(OrchestratorError):
    """Raised after checkpointing unresolved API failures for targeted resume."""


class StagedMetaHarnessOrchestrator:
    """Run prompt search without evaluating every candidate on validation."""

    def __init__(
        self,
        *,
        repository_root: str | Path,
        run_id: str,
        config: MetaHarnessConfig,
        split_manifest: Mapping[str, Any],
        hard_search_manifest: Mapping[str, Any],
        search_samples: Sequence[Any],
        validation_samples: Sequence[Any],
        solver: Any,
        proposer: Any,
        limits: BudgetLimits | None = None,
        early_stopping: EarlyStoppingConfig | None = None,
        requested_settings: Mapping[str, Any] | None = None,
        evaluator: Any = evaluate_candidate,
        clock=time.monotonic,
        transition_hook: Any = None,
    ) -> None:
        verify_split_manifest(split_manifest)
        verify_hard_search_manifest(
            hard_search_manifest,
            split_manifest=split_manifest,
        )
        if split_manifest["config_sha256"] != config.sha256():
            raise ValueError(
                "split manifest configuration does not match this run; "
                "create a new split and run ID"
            )
        if not config.search_protocol_explicit:
            raise ValueError(
                "staged runs require an explicit search_protocol in config"
            )
        if config.generation_request_options():
            raise ValueError(
                "staged prompt search preserves omitted generation settings"
            )
        self.repository_root = Path(repository_root)
        self.run_id = run_id
        self.config = config
        self.protocol = config.search_protocol
        self.split_manifest = split_manifest
        self.hard_search_manifest = hard_search_manifest
        self.search_ids = tuple(
            item["sample_id"] for item in hard_search_manifest["items"]
        )
        self.validation_ids = tuple(
            split_manifest["splits"]["validation"]["sample_ids"]
        )
        self.search_samples = tuple(search_samples)
        self.validation_samples = tuple(validation_samples)
        _validate_samples(self.search_samples, self.search_ids, "search")
        _validate_samples(
            self.validation_samples,
            self.validation_ids,
            "protected validation",
        )
        self.solver = solver
        self.proposer = proposer
        self.limits = limits or BudgetLimits(
            max_iterations=self.protocol.target_iterations,
            max_candidates=(
                self.protocol.target_iterations
                * self.protocol.candidates_per_iteration
            ),
        )
        self.early_stopping = early_stopping or EarlyStoppingConfig(
            patience=self.protocol.early_stopping_patience,
            min_delta=0.0,
            min_iterations=self.protocol.min_iterations,
        )
        self.requested_settings = _requested_settings_copy(requested_settings)
        self.evaluator = evaluator
        self.clock = clock
        self.transition_hook = transition_hook
        self.candidate_store = _RunCandidateStore(
            self.repository_root,
            self.run_id,
        )
        self.run_directory = self.candidate_store.run_directory
        self.state_path = self.run_directory / _STATE_FILE
        self.lock_path = self.run_directory / ".run.lock"
        self.experience_store = SearchExperienceStore(
            self.repository_root,
            self.run_id,
            self.candidate_store,
        )
        self._session_started: float | None = None
        self._elapsed_before_session = 0.0
        self._state: dict[str, Any] = {}
        self._validate_effective_settings()

    def planned_workload(self) -> dict[str, Any]:
        iterations = self._effective_iteration_capacity()
        effective_plan = self._workload_for_iterations(iterations)
        minimum_plan = self._workload_for_iterations(
            self.protocol.min_iterations
        )
        maximum_plan = self._workload_for_iterations(
            self.protocol.max_iterations
        )
        stages = effective_plan["stages"]
        return {
            "run_id": self.run_id,
            "orchestrator_version": STAGED_ORCHESTRATOR_VERSION,
            "evaluation_procedure": STAGED_EVALUATION_PROCEDURE,
            "model": self.config.model,
            "proposer_model": self.config.proposer_model,
            "proposer_reasoning_effort": (
                self.config.proposer_reasoning_effort
            ),
            "config_sha256": self.config.sha256(),
            "split_sha256": self.split_manifest["split_sha256"],
            "hard_search_sha256": self.hard_search_manifest[
                "hard_search_sha256"
            ],
            "requested_settings": self.requested_settings,
            "requested_limits": self.requested_settings["limits"],
            "effective_limits": self.limits.as_dict(),
            "effective_early_stopping": self.early_stopping.as_dict(),
            "iteration_plan": {
                "configured_minimum": self.protocol.min_iterations,
                "configured_target": self.protocol.target_iterations,
                "configured_maximum": self.protocol.max_iterations,
                "effective_minimum": self.protocol.min_iterations,
                "effective_maximum": iterations,
            },
            "plans": {
                "minimum": minimum_plan,
                "effective": effective_plan,
                "maximum": maximum_plan,
            },
            "planned_iterations": iterations,
            "planned_candidates": effective_plan["candidates"],
            "planned_proposer_calls": iterations,
            "search_sample_count": len(self.search_ids),
            "validation_sample_count": len(self.validation_ids),
            "stages": stages,
            "total_solver_calls_before_final": effective_plan[
                "total_solver_calls_before_final"
            ],
            "total_estimated_tokens_before_final": effective_plan[
                "total_estimated_tokens_before_final"
            ],
        }

    def _workload_for_iterations(self, iterations: int) -> dict[str, Any]:
        candidates = iterations * self.protocol.candidates_per_iteration
        smoke_calls = candidates * self.protocol.smoke_examples
        search_calls = (
            len(self.search_ids)
            + candidates
            * (len(self.search_ids) - self.protocol.smoke_examples)
        )
        validation_calls = (
            self.protocol.promotion_top_k * len(self.validation_ids)
        )
        final_calls = len(
            self.split_manifest["splits"]["final_test"]["sample_ids"]
        )
        per_call = self.protocol.estimated_tokens_per_solver_call
        stages = {
            "search": {
                "baseline_calls": len(self.search_ids),
                "smoke_calls": smoke_calls,
                "full_search_remaining_calls": search_calls
                - len(self.search_ids),
                "solver_calls": smoke_calls + search_calls,
                "estimated_tokens": (smoke_calls + search_calls) * per_call,
            },
            "protected_validation": {
                "promoted_candidates": self.protocol.promotion_top_k,
                "solver_calls": validation_calls,
                "estimated_tokens": validation_calls * per_call,
            },
            "final_test": {
                "frozen_winners": 1,
                "solver_calls": final_calls,
                "estimated_tokens": final_calls * per_call,
                "included_in_search_execution": False,
            },
        }
        return {
            "iterations": iterations,
            "candidates": candidates,
            "proposer_calls": iterations,
            "stages": stages,
            "total_solver_calls_before_final": (
                stages["search"]["solver_calls"] + validation_calls
            ),
            "total_estimated_tokens_before_final": (
                stages["search"]["estimated_tokens"]
                + stages["protected_validation"]["estimated_tokens"]
            ),
        }

    def run(
        self,
        *,
        resume: bool = False,
        safe_resume_changes: Sequence[str] = (),
    ) -> dict[str, Any]:
        with run_lock(self.lock_path):
            self._session_started = _clock_value(self.clock)
            if (self.run_directory / "finalization" / "frozen_winner.json").exists():
                raise FinalizedRunError(
                    "finalized runs are immutable; use a new run ID"
                )
            if resume:
                self._state = _load_state(self.state_path)
                self._elapsed_before_session = float(
                    self._state["budgets"]["consumed"]["wall_time_seconds"]
                )
                self._validate_resume(tuple(safe_resume_changes))
            else:
                if self.state_path.exists():
                    raise OrchestratorError(
                        "run state already exists; pass --resume or use a new run ID"
                    )
                self._state = self._initial_state()
                self._persist("initialized")
            if self._state["status"] == "completed":
                return _copy(self._state)
            try:
                self._ensure_baseline()
                self._search()
                if self._state["status"] == "search_complete":
                    self._promote()
            except BudgetExhausted as exc:
                self._state["status"] = "budget_exhausted"
                self._state["stop_reason"] = str(exc)
                self._persist("budget_exhausted")
            except StageRetryRequired:
                self._record_failure("stage_retry_required")
                if self._failure_limit_reached():
                    self._state["status"] = "failure_limit"
                    self._state["stop_reason"] = "consecutive_failures"
                    self._persist("failure_limit")
                else:
                    self._state["status"] = "retry_required"
                    self._state["stop_reason"] = "stage_retry_required"
                    self._persist("targeted_retry_required")
            finally:
                self._refresh_wall_time()
                _atomic_replace_json(self.state_path, self._state)
            return _copy(self._state)

    def _search(self) -> None:
        while self._state["next_iteration"] <= self.limits.max_iterations:
            iteration = self._state["next_iteration"]
            iteration_key = str(iteration)
            iteration_state = self._state["iterations"].get(iteration_key)
            if iteration_state is None:
                stop_reason = self._stop_reason_before_iteration()
                if stop_reason is not None:
                    self._state["search_stop_reason"] = stop_reason
                    break
                iteration_state = {
                    "iteration": iteration,
                    "status": "proposing",
                    "candidate_ids": [],
                    "proposer_metadata": None,
                }
                self._state["iterations"][iteration_key] = iteration_state
                self._persist("iteration_started")
            if not iteration_state["candidate_ids"]:
                self._propose(iteration_state)
            for candidate_id in iteration_state["candidate_ids"]:
                candidate = self._state["candidates"][candidate_id]
                if candidate["stages"]["search"]["status"] == "completed":
                    continue
                self._evaluate_search_candidate(candidate_id)
            iteration_state["status"] = "completed"
            self._state["next_iteration"] = iteration + 1
            self._persist("iteration_completed")
            if self._performance_stop_allowed():
                self._state["search_stop_reason"] = "performance_patience"
                break
        if (
            self._state["next_iteration"] - 1
            < self.protocol.min_iterations
        ):
            raise BudgetExhausted(
                self._state["search_stop_reason"]
                or "minimum_protocol_budget"
            )
        self._state["status"] = "search_complete"
        self._state["search_stop_reason"] = (
            self._state["search_stop_reason"] or "max_iterations"
        )
        self._persist("search_completed")

    def _ensure_baseline(self) -> None:
        if BASELINE_CANDIDATE_ID not in self._state["candidates"]:
            baseline = self.candidate_store.load(BASELINE_CANDIDATE_ID)
            self._state["candidates"][BASELINE_CANDIDATE_ID] = _candidate_state(
                baseline.candidate_id,
                baseline.sha256(),
                baseline.source_sha256,
                parent_id=BASELINE_CANDIDATE_ID,
                iteration=0,
                role="baseline",
            )
            self._persist("baseline_registered")
        baseline_state = self._state["candidates"][BASELINE_CANDIDATE_ID]
        if baseline_state["stages"]["search"]["status"] != "completed":
            self._evaluate_stage(
                BASELINE_CANDIDATE_ID,
                stage="search",
                split_name="search",
                sample_ids=self.search_ids,
                samples=self.search_samples,
            )
            if not baseline_state["search_score"]["eligible"]:
                raise OrchestratorError(
                    "baseline search evaluation is incomplete or unrankable"
                )
            self.experience_store.persist_candidate(
                candidate_id=BASELINE_CANDIDATE_ID,
                samples=self.search_samples,
                results_path=self._paths(
                    BASELINE_CANDIDATE_ID, "search"
                )[0],
                parent_id=BASELINE_CANDIDATE_ID,
            )

    def _propose(self, iteration_state: dict[str, Any]) -> None:
        iteration = iteration_state["iteration"]
        checkpoint = self._proposal_path(iteration)
        if checkpoint.exists():
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            batch = CandidateBatch.from_mapping(
                payload["batch"],
                candidate_count=self.protocol.candidates_per_iteration,
                existing_parent_ids=self._state["candidates"],
            )
            metadata = payload["metadata"]
        else:
            self._require_wall_time()
            try:
                result = self.proposer.propose(
                    self.candidate_store,
                    iteration=iteration,
                    parent_candidate_ids=self._parents(),
                    validation_scores=self._search_scores_for_parents(),
                    aggregate_metrics=self._aggregate_search_metrics(),
                    failure_summaries={},
                    search_feedback=None,
                )
            except Exception as exc:
                iteration_state["failure"] = _safe_error(exc)
                self._persist("proposer_failed")
                raise StageRetryRequired(
                    "proposer failed; resume retries the incomplete iteration"
                ) from exc
            batch, metadata = _proposal_parts(result)
            if (
                batch.iteration != iteration
                or len(batch.candidates)
                != self.protocol.candidates_per_iteration
            ):
                raise OrchestratorError(
                    "proposer batch violates the immutable search protocol"
                )
            _atomic_create_json(
                checkpoint,
                {"batch": batch.as_dict(), "metadata": _metadata_copy(metadata)},
            )
        candidate_ids = []
        for candidate in batch.candidates:
            stored = self.candidate_store.create(candidate, status="validated")
            self._state["candidates"].setdefault(
                stored.candidate_id,
                _candidate_state(
                    stored.candidate_id,
                    stored.sha256(),
                    stored.source_sha256,
                    parent_id=stored.parent_id,
                    iteration=iteration,
                    role="candidate",
                ),
            )
            candidate_ids.append(stored.candidate_id)
        iteration_state["candidate_ids"] = candidate_ids
        iteration_state["proposer_metadata"] = _metadata_copy(metadata)
        iteration_state["status"] = "proposed"
        self._state["failures"]["consecutive"] = 0
        self._persist("proposal_completed")

    def _evaluate_search_candidate(self, candidate_id: str) -> None:
        state = self._state["candidates"][candidate_id]
        smoke_ids = self.search_ids[: self.protocol.smoke_examples]
        if state["stages"]["smoke"]["status"] != "completed":
            self._evaluate_stage(
                candidate_id,
                stage="smoke",
                split_name="search",
                sample_ids=smoke_ids,
                samples=self.search_samples,
            )
        if not state["stages"]["smoke"]["score"]["eligible"]:
            state["status"] = "failed"
            state["stages"]["search"]["status"] = "skipped"
            self.experience_store.persist_candidate(
                candidate_id=candidate_id,
                samples=self.search_samples,
                results_path=self._paths(candidate_id, "search")[0],
                parent_id=state["parent_id"],
            )
            self._persist("smoke_rejected")
            return
        self._evaluate_stage(
            candidate_id,
            stage="search",
            split_name="search",
            sample_ids=self.search_ids,
            samples=self.search_samples,
        )
        state["status"] = "search_evaluated"
        self.experience_store.persist_candidate(
            candidate_id=candidate_id,
            samples=self.search_samples,
            results_path=self._paths(candidate_id, "search")[0],
            parent_id=state["parent_id"],
        )
        self._persist("search_candidate_completed")

    def _promote(self) -> None:
        eligible = [
            candidate
            for candidate in self._state["candidates"].values()
            if candidate["role"] == "candidate"
            and candidate.get("search_score", {}).get("eligible")
        ]
        eligible.sort(
            key=lambda candidate: frontier_rank_key(
                _entry(candidate["candidate_id"], candidate["search_score"])
            )
        )
        promoted = eligible[: self.protocol.promotion_top_k]
        if len(promoted) < self.protocol.promotion_top_k:
            raise OrchestratorError(
                "fewer rankable search candidates than promotion_top_k"
            )
        self._state["promotion"] = {
            "top_k": self.protocol.promotion_top_k,
            "candidate_ids": [
                candidate["candidate_id"] for candidate in promoted
            ],
            "status": "evaluating",
        }
        self._state["status"] = "protected_validation"
        self._persist("promotion_selected")
        for candidate in promoted:
            candidate_id = candidate["candidate_id"]
            if candidate["stages"]["protected_validation"]["status"] == "completed":
                continue
            self._evaluate_stage(
                candidate_id,
                stage="protected_validation",
                split_name="validation",
                sample_ids=self.validation_ids,
                samples=self.validation_samples,
            )
            candidate["score"] = candidate["validation_score"]
            candidate["status"] = "evaluated"
            self.candidate_store.update_status(candidate_id, "evaluated")
            self._persist("protected_validation_candidate_completed")
        ranked = sorted(
            promoted,
            key=lambda candidate: (
                -candidate["validation_score"]["macro_f1"],
                -candidate["validation_score"]["accuracy"],
                candidate["validation_score"]["tokens"],
                candidate["candidate_id"],
            ),
        )
        self._state["best_candidate_id"] = ranked[0]["candidate_id"]
        self._state["frontier"] = [
            _entry(candidate["candidate_id"], candidate["validation_score"]).as_dict()
            for candidate in ranked
        ]
        self._state["promotion"]["status"] = "completed"
        self._state["status"] = "completed"
        self._persist("protected_validation_completed")

    def _evaluate_stage(
        self,
        candidate_id: str,
        *,
        stage: str,
        split_name: str,
        sample_ids: Sequence[Any],
        samples: Sequence[Any],
    ) -> None:
        self._require_wall_time()
        candidate = self._state["candidates"][candidate_id]
        stage_state = candidate["stages"][stage]
        result_path, metrics_path = self._paths(candidate_id, stage)
        stage_state["result_path"] = str(result_path.relative_to(self.run_directory))
        stage_state["metrics_path"] = str(
            metrics_path.relative_to(self.run_directory)
        )
        retrying = stage_state["status"] == "retry_pending"
        if (
            metrics_path.exists()
            and not retrying
        ):
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            report = self.evaluator(
                run_id=self.run_id,
                candidate_id=candidate_id,
                candidate_store=self.candidate_store,
                split_manifest=self.split_manifest,
                split_name=split_name,
                sample_ids=tuple(sample_ids),
                samples=tuple(samples),
                solver=self.solver,
                output_path=result_path,
                metrics_path=metrics_path,
                config=self.config,
                before_solver_call=self._before_solver_call,
            )
            if retrying:
                _atomic_replace_json(metrics_path, report)
            elif not metrics_path.exists():
                _atomic_create_json(metrics_path, report)
        score = _score_from_report(
            report,
            result_path=result_path,
            candidate_id=candidate_id,
        )
        stage_state["score"] = score
        stage_state["status"] = (
            "retry_pending"
            if score["unresolved_api_failures"] > 0
            else "completed"
        )
        if stage == "search":
            candidate["search_score"] = score
        elif stage == "protected_validation":
            candidate["validation_score"] = score
        self._recompute_budgets()
        if stage_state["status"] == "retry_pending":
            self._persist(f"{stage}_retry_pending")
            raise StageRetryRequired(
                f"{stage} has unresolved API failures; resume with explicit "
                "live access to retry only those samples"
            )
        self._state["failures"]["consecutive"] = 0
        self._persist(f"{stage}_completed")

    def _paths(self, candidate_id: str, stage: str) -> tuple[Path, Path]:
        if stage in {"smoke", "search"}:
            directory = self.run_directory / "evaluations" / "search" / candidate_id
            result = directory / "search.results.jsonl"
            metrics = directory / f"{stage}.metrics.json"
        else:
            directory = (
                self.run_directory
                / "evaluations"
                / "protected_validation"
                / candidate_id
            )
            result = directory / "validation.results.jsonl"
            metrics = directory / "validation.metrics.json"
        return result, metrics

    def _parents(self) -> list[str]:
        evaluated = [
            candidate
            for candidate in self._state["candidates"].values()
            if candidate.get("search_score", {}).get("eligible")
        ]
        if not evaluated:
            return [BASELINE_CANDIDATE_ID]
        evaluated.sort(
            key=lambda candidate: frontier_rank_key(
                _entry(candidate["candidate_id"], candidate["search_score"])
            )
        )
        return [candidate["candidate_id"] for candidate in evaluated[:3]]

    def _search_scores_for_parents(self) -> dict[str, dict[str, Any]]:
        return {
            candidate_id: {
                key: score[key]
                for key in (
                    "macro_f1",
                    "accuracy",
                    "parse_coverage",
                    "request_coverage",
                    "unresolved_api_failures",
                )
            }
            for candidate_id in self._parents()
            if (score := self._state["candidates"][candidate_id].get("search_score"))
        }

    def _aggregate_search_metrics(self) -> dict[str, Any]:
        evaluated = [
            candidate
            for candidate in self._state["candidates"].values()
            if candidate.get("search_score")
        ]
        return {
            "completed_candidates": len(evaluated),
            "eligible_rate": (
                sum(candidate["search_score"]["eligible"] for candidate in evaluated)
                / len(evaluated)
                if evaluated
                else 0.0
            ),
            "solver_calls": self._state["budgets"]["search"]["solver_calls"],
            "tokens": self._state["budgets"]["search"]["tokens"],
        }

    def _performance_stop_allowed(self) -> bool:
        patience = self.early_stopping.patience
        completed = self._state["next_iteration"] - 1
        if patience is None or completed < self.early_stopping.min_iterations:
            return False
        history = []
        for index in range(1, completed + 1):
            candidates = self._state["iterations"].get(str(index), {}).get(
                "candidate_ids", []
            )
            values = [
                self._state["candidates"][candidate_id]["search_score"]["macro_f1"]
                for candidate_id in candidates
                if self._state["candidates"][candidate_id].get("search_score")
            ]
            history.append(max(values) if values else -1.0)
        best = -1.0
        non_improving = 0
        for value in history:
            if (
                value > best
                and value - best >= self.early_stopping.min_delta
            ):
                best = value
                non_improving = 0
            else:
                non_improving += 1
        return non_improving >= patience

    def _stop_reason_before_iteration(self) -> str | None:
        self._refresh_wall_time()
        if self._failure_limit_reached():
            return "consecutive_failures"
        if self._state["next_iteration"] > self.limits.max_iterations:
            return "max_iterations"
        consumed = self._state["budgets"]["consumed"]
        per_iteration_candidates = self.protocol.candidates_per_iteration
        if (
            self.limits.max_candidates is not None
            and consumed["candidates"] + per_iteration_candidates
            > self.limits.max_candidates
        ):
            return "candidate_budget"
        if self._wall_time_exhausted():
            return "wall_time_budget"
        validation_calls = self._remaining_validation_calls()
        iteration_calls = (
            per_iteration_candidates * len(self.search_ids)
        )
        required_calls = iteration_calls + validation_calls
        if (
            self.limits.max_solver_calls is not None
            and consumed["solver_calls"] + required_calls
            > self.limits.max_solver_calls
        ):
            return "solver_call_budget"
        if (
            self.limits.max_tokens is not None
            and consumed["tokens"]
            + required_calls
            * self.protocol.estimated_tokens_per_solver_call
            > self.limits.max_tokens
        ):
            return "token_budget"
        return None

    def _before_solver_call(self) -> None:
        self._recompute_budgets()
        self._refresh_wall_time()
        if self._wall_time_exhausted():
            raise BudgetExhausted("wall_time_budget")
        consumed = self._state["budgets"]["consumed"]
        if (
            self.limits.max_solver_calls is not None
            and consumed["solver_calls"] >= self.limits.max_solver_calls
        ):
            raise BudgetExhausted("solver_call_budget")
        if (
            self.limits.max_tokens is not None
            and consumed["tokens"]
            + self.protocol.estimated_tokens_per_solver_call
            > self.limits.max_tokens
        ):
            raise BudgetExhausted("token_budget")

    def _remaining_validation_calls(self) -> int:
        promoted = (
            self._state.get("promotion", {}).get("candidate_ids", ())
            if isinstance(self._state.get("promotion"), Mapping)
            else ()
        )
        completed = sum(
            self._state["candidates"][candidate_id]["stages"][
                "protected_validation"
            ]["status"]
            == "completed"
            for candidate_id in promoted
        )
        return (
            self.protocol.promotion_top_k - completed
        ) * len(self.validation_ids)

    def _require_wall_time(self) -> None:
        self._refresh_wall_time()
        if self._wall_time_exhausted():
            raise BudgetExhausted("wall_time_budget")

    def _failure_limit_reached(self) -> bool:
        return (
            self._state["failures"]["consecutive"]
            >= self.limits.max_consecutive_failures
        )

    def _record_failure(
        self,
        scope: str,
        *,
        error: BaseException | None = None,
    ) -> None:
        failures = self._state["failures"]
        failures["total"] += 1
        failures["consecutive"] += 1
        failures["records"].append(
            {
                "scope": scope,
                "error": _safe_error(error) if error is not None else None,
            }
        )
        self._persist("failure_recorded")

    def _initial_state(self) -> dict[str, Any]:
        configuration = self._configuration()
        return {
            "schema_version": STAGED_RUN_STATE_SCHEMA_VERSION,
            "orchestrator_version": STAGED_ORCHESTRATOR_VERSION,
            "run_id": self.run_id,
            "status": "searching",
            "stop_reason": None,
            "search_stop_reason": None,
            "next_iteration": 1,
            "configuration": configuration,
            "configuration_sha256": _sha256_json(configuration),
            "budgets": _empty_budgets(self.limits),
            "candidates": {},
            "scores": {},
            "frontier": [],
            "best_candidate_id": None,
            "iterations": {},
            "promotion": None,
            "failures": {
                "total": 0,
                "consecutive": 0,
                "records": [],
            },
            "transition": 0,
            "last_transition": None,
        }

    def _configuration(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "config_sha256": self.config.sha256(),
            "split_sha256": self.split_manifest["split_sha256"],
            "dataset_sha256": self.split_manifest["dataset_sha256"],
            "hard_search_sha256": self.hard_search_manifest[
                "hard_search_sha256"
            ],
            "search_sample_count": len(self.search_ids),
            "search_sample_ids_sha256": _sha256_json(self.search_ids),
            "validation_sample_count": len(self.validation_ids),
            "validation_sample_ids_sha256": _sha256_json(self.validation_ids),
            "evaluation_procedure": STAGED_EVALUATION_PROCEDURE,
            "parser_contract": PARSER_VERSION,
            "evaluator_contract": EVALUATOR_VERSION,
            "limits": self.limits.as_dict(),
            "early_stopping": self.early_stopping.as_dict(),
        }

    def _validate_resume(self, safe_changes: tuple[str, ...]) -> None:
        if self._state.get("schema_version") != STAGED_RUN_STATE_SCHEMA_VERSION:
            raise ResumeConfigurationError(
                "legacy and staged runs cannot be resumed interchangeably"
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
        current = self._configuration()
        previous = self._state.get("configuration")
        if not isinstance(previous, Mapping):
            raise ResumeConfigurationError(
                "staged run state has no immutable configuration"
            )
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
        previous_limits = previous.get("limits")
        current_limits = current["limits"]
        if not isinstance(previous_limits, Mapping):
            raise ResumeConfigurationError(
                "staged run state has no persisted effective limits"
            )
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
            self._state["configuration_sha256"] = _sha256_json(current)
            self._state["budgets"]["limits"] = self.limits.as_dict()
            self._persist("safe_configuration_extended")
            if self._state["status"] in {
                "budget_exhausted",
                "failure_limit",
            }:
                self._state["status"] = "searching"
                self._state["stop_reason"] = None
                self._persist("run_reopened")
        elif _sha256_json(current) != self._state.get("configuration_sha256"):
            raise ResumeConfigurationError("run configuration hash mismatch")

    def _recompute_budgets(self) -> None:
        previous = self._state.get("budgets", {})
        budgets = _empty_budgets(self.limits)
        for candidate in self._state["candidates"].values():
            search_path = self._paths(
                candidate["candidate_id"], "search"
            )[0]
            search_usage = _resource_usage(
                None,
                result_path=search_path,
                candidate_id=candidate["candidate_id"],
            )
            _add_usage(budgets["search"], search_usage)
            validation_path = self._paths(
                candidate["candidate_id"], "protected_validation"
            )[0]
            validation_usage = _resource_usage(
                None,
                result_path=validation_path,
                candidate_id=candidate["candidate_id"],
            )
            _add_usage(budgets["protected_validation"], validation_usage)
        budgets["final_test"] = previous.get(
            "final_test", budgets["final_test"]
        )
        budgets["total"] = {
            field: sum(
                budgets[stage][field]
                for stage in ("search", "protected_validation", "final_test")
            )
            for field in ("solver_calls", "tokens", "latency_seconds")
        }
        budgets["consumed"] = {
            "iterations": sum(
                iteration.get("status") == "completed"
                for iteration in self._state["iterations"].values()
            ),
            "candidates": sum(
                candidate["role"] == "candidate"
                for candidate in self._state["candidates"].values()
            ),
            "proposer_calls": sum(
                bool(iteration.get("candidate_ids"))
                for iteration in self._state["iterations"].values()
            ),
            "solver_calls": budgets["total"]["solver_calls"],
            "tokens": budgets["total"]["tokens"],
            "wall_time_seconds": float(
                previous.get("consumed", {}).get(
                    "wall_time_seconds",
                    0.0,
                )
            ),
        }
        self._state["budgets"] = budgets
        self._state["scores"] = {
            candidate_id: candidate["score"]
            for candidate_id, candidate in self._state["candidates"].items()
            if candidate.get("score") is not None
        }

    def _proposal_path(self, iteration: int) -> Path:
        return (
            self.run_directory
            / "iterations"
            / f"iteration_{iteration:04d}"
            / "proposal.json"
        )

    def _persist(self, transition: str) -> None:
        self._recompute_budgets()
        self._refresh_wall_time()
        self._state["transition"] += 1
        self._state["last_transition"] = transition
        _atomic_replace_json(self.state_path, self._state)
        if self.transition_hook is not None:
            self.transition_hook(transition, _copy(self._state))

    def _refresh_wall_time(self) -> None:
        if self._session_started is None or not self._state:
            return
        now = _clock_value(self.clock)
        elapsed = max(0.0, now - self._session_started)
        self._state["budgets"]["consumed"]["wall_time_seconds"] = (
            self._elapsed_before_session + elapsed
        )

    def _wall_time_exhausted(self) -> bool:
        maximum = self.limits.max_wall_time_seconds
        return (
            maximum is not None
            and self._state["budgets"]["consumed"]["wall_time_seconds"]
            >= maximum
        )

    def _effective_iteration_capacity(self) -> int:
        capacity = min(
            self.limits.max_iterations,
            self.protocol.max_iterations,
        )
        if self.limits.max_candidates is not None:
            capacity = min(
                capacity,
                self.limits.max_candidates
                // self.protocol.candidates_per_iteration,
            )
        fixed_calls = (
            len(self.search_ids)
            + self.protocol.promotion_top_k * len(self.validation_ids)
        )
        calls_per_iteration = (
            self.protocol.candidates_per_iteration * len(self.search_ids)
        )
        if self.limits.max_solver_calls is not None:
            capacity = min(
                capacity,
                max(
                    0,
                    (self.limits.max_solver_calls - fixed_calls)
                    // calls_per_iteration,
                ),
            )
        if self.limits.max_tokens is not None:
            available_calls = (
                self.limits.max_tokens
                // self.protocol.estimated_tokens_per_solver_call
            )
            capacity = min(
                capacity,
                max(0, (available_calls - fixed_calls) // calls_per_iteration),
            )
        return capacity

    def _validate_effective_settings(self) -> None:
        if self.limits.max_iterations > self.protocol.max_iterations:
            raise OrchestratorError(
                "max_iterations exceeds search_protocol.max_iterations "
                f"({self.protocol.max_iterations})"
            )
        if self.early_stopping.min_iterations < self.protocol.min_iterations:
            raise OrchestratorError(
                "min_iterations cannot be below search_protocol.min_iterations "
                f"({self.protocol.min_iterations})"
            )
        minimum = self._workload_for_iterations(
            self.protocol.min_iterations
        )
        required_candidates = minimum["candidates"]
        required_calls = minimum["total_solver_calls_before_final"]
        required_tokens = minimum["total_estimated_tokens_before_final"]
        failures = []
        if self.limits.max_iterations < self.protocol.min_iterations:
            failures.append(
                "max_iterations="
                f"{self.limits.max_iterations} < {self.protocol.min_iterations}"
            )
        if (
            self.limits.max_candidates is not None
            and self.limits.max_candidates < required_candidates
        ):
            failures.append(
                "max_candidates="
                f"{self.limits.max_candidates} < {required_candidates}"
            )
        if (
            self.limits.max_solver_calls is not None
            and self.limits.max_solver_calls < required_calls
        ):
            failures.append(
                "max_solver_calls="
                f"{self.limits.max_solver_calls} < {required_calls}"
            )
        if (
            self.limits.max_tokens is not None
            and self.limits.max_tokens < required_tokens
        ):
            failures.append(
                f"max_tokens={self.limits.max_tokens} < {required_tokens}"
            )
        if failures:
            raise OrchestratorError(
                "supplied ceilings cannot fund the configured minimum "
                f"{self.protocol.min_iterations}-iteration protocol: "
                + "; ".join(failures)
            )
        if self.early_stopping.min_iterations > self.limits.max_iterations:
            raise OrchestratorError(
                "min_iterations cannot exceed max_iterations"
            )


def _candidate_state(
    candidate_id: str,
    candidate_sha256: str,
    prompt_sha256: str,
    *,
    parent_id: str,
    iteration: int,
    role: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "prompt_sha256": prompt_sha256,
        "parent_id": parent_id,
        "iteration": iteration,
        "role": role,
        "status": "validated",
        "search_score": None,
        "validation_score": None,
        "score": None,
        "stages": {
            name: {
                "status": "completed" if name == "static" else "pending",
                "score": None,
                "result_path": None,
                "metrics_path": None,
            }
            for name in (
                "static",
                "smoke",
                "search",
                "protected_validation",
            )
        },
    }


def _entry(candidate_id: str, score: Mapping[str, Any]) -> FrontierEntry:
    return FrontierEntry(
        candidate_id=candidate_id,
        macro_f1=score["macro_f1"],
        accuracy=score["accuracy"],
        solver_calls=score["solver_calls"],
        tokens=score["tokens"],
        latency_seconds=score["latency_seconds"],
    )


def _validate_samples(
    samples: Sequence[Any],
    expected_ids: Sequence[Any],
    label: str,
) -> None:
    actual = []
    for sample in samples:
        if hasattr(sample, "sample_id"):
            actual.append(_identity(sample.sample_id))
        elif isinstance(sample, Mapping):
            actual.append(_identity(sample.get("sample_id")))
        else:
            raise ValueError(f"{label} sample has no sample_id")
    expected = {_identity(value) for value in expected_ids}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(
            f"{label} samples must exactly match their immutable manifest"
        )


def _empty_budgets(limits: BudgetLimits) -> dict[str, Any]:
    stages = {
        stage: {
            "solver_calls": 0,
            "tokens": 0,
            "latency_seconds": 0.0,
        }
        for stage in ("search", "protected_validation", "final_test")
    }
    return {
        "limits": limits.as_dict(),
        **stages,
        "total": {
            "solver_calls": 0,
            "tokens": 0,
            "latency_seconds": 0.0,
        },
        "consumed": {
            "iterations": 0,
            "candidates": 0,
            "proposer_calls": 0,
            "solver_calls": 0,
            "tokens": 0,
            "wall_time_seconds": 0.0,
        },
    }


def _add_usage(target: dict[str, Any], score: Mapping[str, Any]) -> None:
    target["solver_calls"] += score["solver_calls"]
    target["tokens"] += score["tokens"]
    target["latency_seconds"] += score["latency_seconds"]


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OrchestratorError("run state must be readable valid JSON") from exc
    if not isinstance(value, dict):
        raise OrchestratorError("run state must contain a JSON object")
    return value


def _atomic_create_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise OrchestratorError("immutable artifact already has different content")
        return
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _atomic_create_json(path, value)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace_json(path: Path, value: Mapping[str, Any]) -> None:
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
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identity(value: Any) -> str:
    return canonical_json(value)


def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json(value))


def _requested_settings_copy(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    empty = {
        "limits": {
            "max_iterations": None,
            "max_candidates": None,
            "max_solver_calls": None,
            "max_tokens": None,
            "max_wall_time_seconds": None,
            "max_consecutive_failures": None,
        },
        "early_stopping": {
            "patience": None,
            "min_delta": None,
            "min_iterations": None,
        },
    }
    if value is None:
        return empty
    if not isinstance(value, Mapping):
        raise ValueError("requested_settings must be a mapping")
    copied = _copy(value)
    if (
        set(copied) != set(empty)
        or not isinstance(copied["limits"], Mapping)
        or set(copied["limits"]) != set(empty["limits"])
        or not isinstance(copied["early_stopping"], Mapping)
        or set(copied["early_stopping"]) != set(empty["early_stopping"])
    ):
        raise ValueError("requested_settings has invalid fields")
    return copied


def _clock_value(clock: Any) -> float:
    value = clock()
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
    ):
        raise OrchestratorError("clock must return a non-negative number")
    return float(value)


__all__ = [
    "STAGED_EVALUATION_PROCEDURE",
    "STAGED_ORCHESTRATOR_VERSION",
    "STAGED_RUN_STATE_SCHEMA_VERSION",
    "StageRetryRequired",
    "StagedMetaHarnessOrchestrator",
]
