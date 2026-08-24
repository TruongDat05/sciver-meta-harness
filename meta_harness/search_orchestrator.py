"""Trusted, resumable full-SEARCH orchestration without FINAL access."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterator

from meta_harness.config import (
    EXPERIMENT_MAX_ITERATIONS,
    EXPERIMENT_MIN_ITERATIONS,
    EXPERIMENT_PATIENCE,
    EXPERIMENT_PROTOCOL_ID,
)
from meta_harness.search_evaluator import (
    EXPERIMENT_P0_CANDIDATE_ID,
    EvaluationIncomplete,
    SearchInput,
    canonical_experiment_p0_prompt_sha256,
    evaluate_experiment_candidate,
    evaluate_experiment_p0,
)
from meta_harness.prompt_proposer import (
    Candidate,
    ProposalExhausted,
    ProposalResult,
    build_prompt_proposer_input,
    load_experiment_accepted_proposal,
)
from meta_harness.run_identity import validate_run_identity
from meta_harness.ranking import (
    PatienceState,
    advance_experiment_patience,
    eligible_experiment_candidate,
    rank_eligible_experiment_reports,
)
from meta_harness.prompt_family import (
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)
from utils.answer_parser import PARSER_VERSION


EXPERIMENT_ORCHESTRATOR_SCHEMA_VERSION = (
    "sciver_full_search_v3_orchestrator_state_v1"
)
EXPERIMENT_ORCHESTRATOR_VERSION = "sciver_full_search_v3_orchestrator_v1"
_STATE_FILENAME = "orchestration_state.json"
_LOCK_FILENAME = ".orchestration.lock"
_TERMINAL_STATUSES = frozenset(
    {"p0_ineligible", "proposal_exhausted", "patience_stopped", "max_stopped"}
)
_SENSITIVE_IDENTITY = re.compile(
    r"(?:api[_ -]?(?:key|url)|authorization|bearer|credential|secret|"
    r"https?://|base64|data:image/)",
    re.IGNORECASE,
)


class OrchestrationError(RuntimeError):
    """Raised when the SEARCH state machine cannot advance safely."""


class ResumeError(OrchestrationError):
    """Raised when existing durable state has a different run identity."""


P0Evaluator = Callable[..., Mapping[str, Any]]
CandidateEvaluator = Callable[..., Mapping[str, Any]]
TransitionHook = Callable[[str, Mapping[str, Any]], None]


class Orchestrator:
    """Run P0 then one fully evaluated candidate per locked iteration.

    The class intentionally accepts the SEARCH-only input object rather than
    any paths or records from another stage.  All remote-capable work remains
    behind injected M3 evaluator functions and their injected M2 boundaries.
    """

    def __init__(
        self,
        *,
        repository_root: str | Path,
        run_id: str,
        search_input: SearchInput,
        solver_identity_sha256: str,
        cache: Any,
        executor: Any,
        proposer: Any,
        proposer_identity: Mapping[str, Any],
        p0_evaluator: P0Evaluator = evaluate_experiment_p0,
        candidate_evaluator: CandidateEvaluator = evaluate_experiment_candidate,
        representative_search_failures: Sequence[Mapping[str, Any]] = (),
        transition_hook: TransitionHook | None = None,
    ) -> None:
        if not isinstance(search_input, SearchInput):
            raise TypeError("search_input must be SearchInput")
        if not callable(getattr(proposer, "propose", None)):
            raise TypeError("proposer must provide a propose method")
        if not callable(p0_evaluator) or not callable(candidate_evaluator):
            raise TypeError("Meta-Harness evaluators must be callable")
        if transition_hook is not None and not callable(transition_hook):
            raise TypeError("transition_hook must be callable or None")
        _sha256(solver_identity_sha256, "solver_identity_sha256")
        self.repository_root = Path(repository_root)
        self.run_id = validate_run_identity(run_id)
        self.search_input = search_input
        self.solver_identity_sha256 = solver_identity_sha256
        self.cache = cache
        self.executor = executor
        self.proposer = proposer
        self.p0_evaluator = p0_evaluator
        self.candidate_evaluator = candidate_evaluator
        self.representative_search_failures = tuple(
            build_prompt_proposer_input(
                iteration=0,
                parent_id=EXPERIMENT_P0_CANDIDATE_ID,
                parent_templates=canonical_baseline_sources(),
                aggregate_search_metrics={},
                lineage=(),
                representative_search_failures=representative_search_failures,
            )["representative_search_failures"]
        )
        self.transition_hook = transition_hook
        self.run_directory = (
            self.repository_root
            / "workspace"
            / "meta_harness"
            / "full_search_v3"
            / self.run_id
        )
        self.state_path = self.run_directory / _STATE_FILENAME
        self.lock_path = self.run_directory / _LOCK_FILENAME
        self.identity = _run_identity(
            search_input=search_input,
            solver_identity_sha256=solver_identity_sha256,
            proposer_identity=proposer_identity,
        )
        self._state: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        """Advance until a durable terminal state or retryable interruption."""

        with _run_lock(self.lock_path):
            self._state = self._load_or_initialize_state()
            if self._state["status"] in _TERMINAL_STATUSES:
                return _copy_json(self._state)
            if not self._ensure_p0():
                return _copy_json(self._state)
            while self._state["status"] == "running":
                unfinished = next(
                    (
                        entry
                        for entry in self._state["iterations"]
                        if entry["status"] != "complete"
                    ),
                    None,
                )
                if unfinished is not None:
                    if not self._advance_iteration(unfinished["iteration"]):
                        break
                    continue
                completed = len(self._state["iterations"])
                if completed >= EXPERIMENT_MAX_ITERATIONS:
                    self._state["status"] = "max_stopped"
                    self._state["stop_reason"] = "maximum_completed_iterations"
                    self._persist("max_stopped")
                    break
                if (
                    completed >= EXPERIMENT_MIN_ITERATIONS
                    and self._state["patience"]["consecutive_non_improving"]
                    >= EXPERIMENT_PATIENCE
                ):
                    self._state["status"] = "patience_stopped"
                    self._state["stop_reason"] = "metric_patience"
                    self._persist("patience_stopped")
                    break
                if not self._advance_iteration(completed + 1):
                    break
            return _copy_json(self._state)

    def state(self) -> dict[str, Any]:
        """Return a detached durable state snapshot without advancing work."""

        with _run_lock(self.lock_path):
            self._state = self._load_or_initialize_state()
            return _copy_json(self._state)

    def _ensure_p0(self) -> bool:
        p0 = self._state["p0"]
        if p0["status"] == "complete":
            return True
        if p0["status"] == "ineligible":
            self._state["status"] = "p0_ineligible"
            self._persist("p0_ineligible")
            return False
        resume = p0["status"] != "pending"
        p0["status"] = "evaluating"
        self._persist("p0_evaluating")
        try:
            report = self.p0_evaluator(
                search_input=self.search_input,
                solver_identity_sha256=self.solver_identity_sha256,
                cache=self.cache,
                executor=self.executor,
                checkpoint_path=self.run_directory / "evaluations" / "cot.checkpoint.json",
                result_path=self.run_directory / "evaluations" / "cot.metrics.json",
                resume=resume,
            )
        except EvaluationIncomplete:
            p0["status"] = "incomplete"
            self._persist("p0_incomplete")
            return False
        p0["report"] = _safe_report(report, EXPERIMENT_P0_CANDIDATE_ID)
        eligible = eligible_experiment_candidate(p0["report"])
        if eligible is None:
            p0["status"] = "ineligible"
            self._state["status"] = "p0_ineligible"
            self._persist("p0_ineligible")
            return False
        p0["status"] = "complete"
        self._state["patience"] = _patience_dict(
            PatienceState(
                best_macro_f1=eligible.macro_f1,
                best_accuracy=eligible.accuracy,
            )
        )
        self._update_ranking()
        self._persist("p0_complete")
        self._notify("p0_complete")
        return True

    def _advance_iteration(self, iteration: int) -> bool:
        entry = _iteration_entry(self._state, iteration)
        if entry is None:
            entry = {
                "iteration": iteration,
                "status": "proposal_pending",
                "candidate": None,
                "report": None,
                "metric_improved": None,
            }
            self._state["iterations"].append(entry)
            self._persist("proposal_pending")
        if entry["status"] == "proposal_exhausted":
            self._state["status"] = "proposal_exhausted"
            self._persist("proposal_exhausted")
            return False
        if entry["status"] == "proposal_pending":
            proposal = self._recover_or_propose(iteration)
            if proposal is None:
                return False
            entry["candidate"] = proposal.candidate.as_dict()
            entry["status"] = "proposed"
            self._persist("proposal_accepted")
            self._notify("proposal_accepted")
        if entry["status"] in {"proposed", "evaluating", "incomplete"}:
            resume = entry["status"] in {"evaluating", "incomplete"}
            entry["status"] = "evaluating"
            self._persist("candidate_evaluating")
            candidate = _candidate_from_state(entry["candidate"])
            try:
                report = self.candidate_evaluator(
                    search_input=self.search_input,
                    candidate_id=candidate.candidate_id,
                    prompt=candidate.templates,
                    solver_identity_sha256=self.solver_identity_sha256,
                    cache=self.cache,
                    executor=self.executor,
                    checkpoint_path=(
                        self.run_directory
                        / "evaluations"
                        / f"{candidate.candidate_id}.checkpoint.json"
                    ),
                    result_path=(
                        self.run_directory
                        / "evaluations"
                        / f"{candidate.candidate_id}.metrics.json"
                    ),
                    resume=resume,
                )
            except EvaluationIncomplete:
                entry["status"] = "incomplete"
                self._persist("candidate_incomplete")
                return False
            entry["report"] = _safe_report(report, candidate.candidate_id)
            entry["status"] = "complete"
            self._apply_completed_iteration(entry)
            self._persist("candidate_complete")
            self._notify("candidate_complete")
        return entry["status"] == "complete"

    def _recover_or_propose(self, iteration: int) -> ProposalResult | None:
        recovered = self._recover_accepted_proposal(iteration)
        if recovered is not None:
            return recovered
        parent_id, parent_templates = self._parent_for_next_proposal()
        try:
            return self.proposer.propose(
                proposal_directory=self.repository_root,
                run_id=self.run_id,
                iteration=iteration,
                parent_id=parent_id,
                parent_templates=parent_templates,
                aggregate_search_metrics=self._aggregate_search_metrics(),
                lineage=self._candidate_lineage(),
                representative_search_failures=self.representative_search_failures,
                existing_candidate_ids=self._existing_candidate_ids(),
                existing_source_sha256=self._existing_prompt_hashes(),
            )
        except ProposalExhausted:
            entry = _iteration_entry(self._state, iteration)
            assert entry is not None
            entry["status"] = "proposal_exhausted"
            self._state["status"] = "proposal_exhausted"
            self._persist("proposal_exhausted")
            return None

    def _recover_accepted_proposal(
        self, iteration: int
    ) -> ProposalResult | None:
        directory = self.run_directory / "proposals" / f"iteration_{iteration:04d}"
        if not directory.is_dir():
            return None
        accepted: list[ProposalResult] = []
        for path in sorted(directory.glob("attempt_*.json")):
            try:
                proposal = load_experiment_accepted_proposal(
                    path, expected_iteration=iteration
                )
            except Exception as exc:
                if _receipt_declares_accepted(path):
                    raise OrchestrationError(
                        "accepted proposal receipt is corrupt"
                    ) from exc
                continue
            accepted.append(proposal)
        if not accepted:
            return None
        if len(accepted) != 1 or accepted[0].candidate.candidate_id in self._existing_candidate_ids():
            raise OrchestrationError(
                "proposal receipts are incompatible with resumable state"
            )
        return accepted[0]

    def _apply_completed_iteration(self, entry: Mapping[str, Any]) -> None:
        reports = [self._state["p0"]["report"]] + [
            value["report"]
            for value in self._state["iterations"]
            if value["status"] == "complete"
        ]
        ranking = rank_eligible_experiment_reports(reports)
        previous = PatienceState(**self._state["patience"])
        update = advance_experiment_patience(previous, ranking)
        self._state["patience"] = _patience_dict(update.state)
        entry["metric_improved"] = update.metric_improved
        self._update_ranking(ranking)

    def _update_ranking(self, ranking: Sequence[Any] | None = None) -> None:
        if ranking is None:
            reports = [self._state["p0"]["report"]] + [
                value["report"]
                for value in self._state["iterations"]
                if value["status"] == "complete"
            ]
            ranking = rank_eligible_experiment_reports(reports)
        self._state["ranking"] = [candidate.candidate_id for candidate in ranking]
        self._state["winner_id"] = None if not ranking else ranking[0].candidate_id

    def _parent_for_next_proposal(self) -> tuple[str, Mapping[str, str]]:
        winner_id = self._state.get("winner_id")
        if winner_id == EXPERIMENT_P0_CANDIDATE_ID:
            return winner_id, canonical_baseline_sources()
        for entry in self._state["iterations"]:
            candidate = entry.get("candidate")
            if candidate is not None and candidate.get("candidate_id") == winner_id:
                restored = _candidate_from_state(candidate)
                return restored.candidate_id, restored.templates
        raise OrchestrationError(
            "Meta-Harness ranking winner has no durable prompt family"
        )

    def _aggregate_search_metrics(self) -> dict[str, Any]:
        winner_id = self._state.get("winner_id")
        reports = [self._state["p0"]["report"]] + [
            item["report"] for item in self._state["iterations"] if item["status"] == "complete"
        ]
        for report in reports:
            if report and report.get("candidate_id") == winner_id:
                metrics = report["metrics"]
                return {
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "parse_coverage": metrics["parse_coverage"],
                    "total_records": report["total_records"],
                    "parsed_predictions": report["parsed_predictions"],
                    "abstentions_or_parse_failures": report[
                        "abstentions_or_parse_failures"
                    ],
                    "infrastructure_failures": report["infrastructure_failures"],
                }
        raise OrchestrationError("Meta-Harness winner metrics are unavailable")

    def _candidate_lineage(self) -> list[dict[str, str]]:
        lineage = []
        for entry in self._state["iterations"]:
            candidate = entry.get("candidate")
            if candidate is None:
                continue
            lineage.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "parent_id": candidate["parent_id"],
                    "source_sha256": candidate["source_sha256"],
                }
            )
        return lineage

    def _existing_candidate_ids(self) -> list[str]:
        return [EXPERIMENT_P0_CANDIDATE_ID] + [
            entry["candidate"]["candidate_id"]
            for entry in self._state["iterations"]
            if entry.get("candidate") is not None
        ]

    def _existing_prompt_hashes(self) -> list[str]:
        values = [canonical_experiment_p0_prompt_sha256()]
        values.extend(
            entry["candidate"]["source_sha256"]
            for entry in self._state["iterations"]
            if entry.get("candidate") is not None
        )
        return values

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            state = {
                "schema_version": EXPERIMENT_ORCHESTRATOR_SCHEMA_VERSION,
                "orchestrator_version": EXPERIMENT_ORCHESTRATOR_VERSION,
                "identity": self.identity,
                "status": "running",
                "stop_reason": None,
                "p0": {"status": "pending", "report": None},
                "iterations": [],
                "patience": _patience_dict(PatienceState()),
                "ranking": [],
                "winner_id": None,
            }
            _atomic_write(self.state_path, state)
            return state
        state = _load_state(self.state_path)
        if state.get("schema_version") != EXPERIMENT_ORCHESTRATOR_SCHEMA_VERSION:
            raise ResumeError("Meta-Harness orchestration state schema is incompatible")
        if state.get("identity") != self.identity:
            raise ResumeError("Meta-Harness orchestration run identity is incompatible")
        _validate_state_shape(state)
        return state

    def _persist(self, _transition: str) -> None:
        _validate_state_shape(self._state)
        _atomic_write(self.state_path, self._state)

    def _notify(self, transition: str) -> None:
        if self.transition_hook is not None:
            self.transition_hook(transition, _copy_json(self._state))


def _run_identity(
    *,
    search_input: SearchInput,
    solver_identity_sha256: str,
    proposer_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(proposer_identity, Mapping):
        raise TypeError("proposer_identity must be a safe JSON object")
    try:
        normalized_proposer_identity = json.loads(canonical_json(dict(proposer_identity)))
    except (TypeError, ValueError) as exc:
        raise ValueError("proposer_identity must be canonical JSON data") from exc
    if _SENSITIVE_IDENTITY.search(canonical_json(normalized_proposer_identity)):
        raise ValueError("proposer_identity must not contain sensitive configuration")
    manifest = search_input.manifest
    required = {"split_sha256", "search_membership_sha256"}
    if not isinstance(manifest, Mapping) or any(name not in manifest for name in required):
        raise OrchestrationError("SEARCH input lacks immutable manifest identity")
    return {
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "config_sha256": _config_sha256(),
        "split_sha256": manifest["split_sha256"],
        "search_membership_sha256": manifest["search_membership_sha256"],
        "sample_ids_sha256": _sha256_json(list(search_input.sample_ids)),
        "canonical_p0_prompt_sha256": canonical_experiment_p0_prompt_sha256(),
        "parser_version": PARSER_VERSION,
        "solver_identity_sha256": solver_identity_sha256,
        "proposer_identity": normalized_proposer_identity,
    }


def _config_sha256() -> str:
    from meta_harness.config import canonical_experiment_config

    return canonical_experiment_config().sha256()


def _safe_report(report: Mapping[str, Any], expected_candidate_id: str) -> dict[str, Any]:
    if not isinstance(report, Mapping) or report.get("candidate_id") != expected_candidate_id:
        raise OrchestrationError("Meta-Harness evaluator returned an incompatible report")
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise OrchestrationError("Meta-Harness evaluator report lacks metrics")
    fields = (
        "protocol_id",
        "stage",
        "candidate_id",
        "prompt_sha256",
        "total_records",
        "completed_solver_responses",
        "parsed_predictions",
        "abstentions_or_parse_failures",
        "infrastructure_failures",
    )
    metric_fields = ("macro_f1", "accuracy", "parse_coverage", "rankable")
    if any(field not in report for field in fields) or any(
        field not in metrics for field in metric_fields
    ):
        raise OrchestrationError("Meta-Harness evaluator report is incomplete")
    return {
        **{field: report[field] for field in fields},
        "metrics": {field: metrics[field] for field in metric_fields},
    }


def _candidate_from_state(value: Any) -> Candidate:
    if not isinstance(value, Mapping) or set(value) != {
        "candidate_id", "parent_id", "hypothesis", "expected_tradeoff", "templates", "source_sha256"
    }:
        raise ResumeError("stored candidate is invalid")
    candidate = Candidate(
        candidate_id=_identifier(value["candidate_id"], "candidate_id"),
        parent_id=_identifier(value["parent_id"], "parent_id"),
        hypothesis=value["hypothesis"],
        expected_tradeoff=value["expected_tradeoff"],
        templates=value["templates"],
        source_sha256=_sha256(value["source_sha256"], "source_sha256"),
    )
    try:
        source_sha256 = template_source_sha256(candidate.templates)
    except Exception as exc:
        raise ResumeError("stored candidate templates are invalid") from exc
    if source_sha256 != candidate.source_sha256:
        raise ResumeError("stored candidate hash is invalid")
    return candidate


def _iteration_entry(state: Mapping[str, Any], iteration: int) -> dict[str, Any] | None:
    for entry in state["iterations"]:
        if entry["iteration"] == iteration:
            return entry
    return None


def _patience_dict(state: PatienceState) -> dict[str, Any]:
    return {
        "best_macro_f1": state.best_macro_f1,
        "best_accuracy": state.best_accuracy,
        "consecutive_non_improving": state.consecutive_non_improving,
    }


def _validate_state_shape(state: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "orchestrator_version", "identity", "status", "stop_reason",
        "p0", "iterations", "patience", "ranking", "winner_id",
    }
    if not isinstance(state, Mapping) or set(state) != required:
        raise ResumeError("Meta-Harness orchestration state fields are invalid")
    if not isinstance(state["iterations"], list):
        raise ResumeError("Meta-Harness orchestration iterations are invalid")
    if not isinstance(state["p0"], Mapping) or set(state["p0"]) != {"status", "report"}:
        raise ResumeError("Meta-Harness orchestration P0 state is invalid")
    if state["p0"]["status"] not in {"pending", "evaluating", "incomplete", "complete", "ineligible"}:
        raise ResumeError("Meta-Harness orchestration P0 status is invalid")
    expected = list(range(1, len(state["iterations"]) + 1))
    observed = [entry.get("iteration") for entry in state["iterations"] if isinstance(entry, Mapping)]
    if observed != expected:
        raise ResumeError("Meta-Harness orchestration iteration order is invalid")
    for entry in state["iterations"]:
        if not isinstance(entry, Mapping) or set(entry) != {
            "iteration", "status", "candidate", "report", "metric_improved"
        }:
            raise ResumeError("Meta-Harness orchestration iteration state is invalid")
        if entry["status"] not in {
            "proposal_pending", "proposal_exhausted", "proposed", "evaluating", "incomplete", "complete"
        }:
            raise ResumeError("Meta-Harness orchestration iteration status is invalid")
        if entry["status"] in {"proposed", "evaluating", "incomplete", "complete"}:
            _candidate_from_state(entry["candidate"])
        if entry["status"] == "complete" and not isinstance(entry["report"], Mapping):
            raise ResumeError("completed iteration lacks a report")
    PatienceState(**state["patience"])


def _load_state(path: Path) -> dict[str, Any]:
    try:
        encoded = path.read_bytes()
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, ResumeError) as exc:
        raise ResumeError("Meta-Harness orchestration state is unreadable") from exc
    if not isinstance(value, dict) or encoded != canonical_json(value).encode("utf-8"):
        raise ResumeError("Meta-Harness orchestration state is not canonical")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json(value).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".state.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise OrchestrationError("another process owns this run") from exc
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json(value))


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "/" in value or ".." in value:
        raise ValueError(f"{field} must be a safe local identifier")
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ResumeError("duplicate state JSON key")
        result[key] = value
    return result


def _receipt_declares_accepted(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(value, Mapping) and value.get("status") == "accepted"


def experiment_orchestration_state_path(
    repository_root: str | Path, run_id: str
) -> Path:
    """Return the durable SEARCH state path for one safe run identifier."""

    return (
        Path(repository_root)
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / validate_run_identity(run_id)
        / _STATE_FILENAME
    )


def load_experiment_orchestration_state(path: str | Path) -> dict[str, Any]:
    """Load an existing SEARCH-only state without creating or advancing a run."""

    state = _load_state(Path(path))
    _validate_state_shape(state)
    return _copy_json(state)


__all__ = [
    "EXPERIMENT_ORCHESTRATOR_SCHEMA_VERSION",
    "EXPERIMENT_ORCHESTRATOR_VERSION",
    "OrchestrationError",
    "Orchestrator",
    "ResumeError",
    "experiment_orchestration_state_path",
    "load_experiment_orchestration_state",
]
