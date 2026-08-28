"""Offline isolation and identity coverage for full-SEARCH FINAL planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import MappingProxyType

import pytest

from meta_harness.config import (
    EXPERIMENT_PROTOCOL_ID,
    EXPERIMENT_SEARCH_SIZE,
    canonical_experiment_config,
)
from meta_harness.records import (
    build_experiment_split,
    validate_sciver_records,
)
from meta_harness.search_evaluator import (
    SearchInput,
    canonical_experiment_p0_prompt_sha256,
)
from meta_harness.final_evaluation import (
    FinalError,
    FinalSolverContract,
    account_final_evaluation_outcomes,
    canonical_final_evaluation_solver_contract,
    execute_final_evaluation,
    final_evaluation_state_path,
    initialize_final_evaluation,
    load_final_evaluation_state,
    preflight_final_evaluation,
)
from meta_harness.winner_freeze import freeze_experiment_winner
from meta_harness.search_orchestrator import Orchestrator
from meta_harness.preparation import (
    build_experiment_private_manifest,
    derive_experiment_search_safe_manifest,
    save_experiment_private_manifest,
    save_experiment_search_safe_manifest,
    source_dataset_sha256,
)
from meta_harness.prompt_proposer import (
    Candidate,
    ProposalResult,
)
from meta_harness.retry import SolverRetryPolicy
from meta_harness.solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
)
from model_inference.remote_client import RateLimitError
from meta_harness.prompt_family import (
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)


DATASET_PATH = Path(__file__).resolve().parents[2] / "data" / "sciver" / "testset.json"
RUN_ID = "offline_final"


def test_final_outcome_accounting_reports_metrics_without_private_values():
    report = account_final_evaluation_outcomes(
        gold_labels=("yes", "no", "yes"),
        parsed_predictions=("yes", "no", None),
        infrastructure_failures=(False, False, False),
    )

    assert report["confusion"] == {"yes": {"yes": 1, "no": 0}, "no": {"yes": 0, "no": 1}}
    assert report["parsed_predictions"] == 2
    assert report["abstentions_or_parse_failures"] == 1
    assert report["infrastructure_failures"] == 0
    assert report["metrics"] == pytest.approx(
        {"accuracy": 2 / 3, "macro_f1": 5 / 6, "parse_coverage": 2 / 3}
    )
    assert "sample_id" not in canonical_json(report)
    assert "Answer:" not in canonical_json(report)


def test_final_outcome_accounting_counts_infrastructure_failure_in_macro_f1_support():
    report = account_final_evaluation_outcomes(
        gold_labels=("yes", "no", "yes"),
        parsed_predictions=("yes", "no", None),
        infrastructure_failures=(False, False, True),
    )

    assert report["infrastructure_failures"] == 1
    assert report["metrics"]["macro_f1"] == pytest.approx(5 / 6)


def test_final_outcome_accounting_uses_search_zero_denominator_semantics():
    report = account_final_evaluation_outcomes(
        gold_labels=(), parsed_predictions=(), infrastructure_failures=()
    )

    assert report["metrics"] == {
        "accuracy": None,
        "macro_f1": 0.0,
        "parse_coverage": None,
    }


def _search_input(split_sha256="a" * 64, search_membership_sha256="b" * 64):
    return SearchInput(
        _manifest=MappingProxyType(
            {
                "split_sha256": split_sha256,
                "search_membership_sha256": search_membership_sha256,
            }
        ),
        _records=(),
    )


def _report(candidate_id, prompt_sha256, macro_f1, accuracy):
    return {
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "stage": "SEARCH",
        "candidate_id": candidate_id,
        "prompt_sha256": prompt_sha256,
        "total_records": EXPERIMENT_SEARCH_SIZE,
        "completed_solver_responses": EXPERIMENT_SEARCH_SIZE,
        "parsed_predictions": EXPERIMENT_SEARCH_SIZE,
        "abstentions_or_parse_failures": 0,
        "infrastructure_failures": 0,
        "metrics": {
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "parse_coverage": 1.0,
            "rankable": True,
        },
    }


class _Proposer:
    def propose(self, **kwargs):
        iteration = kwargs["iteration"]
        templates = {
            method: source + f"\nOffline FINAL candidate {iteration}."
            for method, source in canonical_baseline_sources().items()
        }
        candidate = Candidate(
            candidate_id=f"candidate_{iteration:03d}",
            parent_id=kwargs["parent_id"],
            hypothesis="Independent checks reduce SEARCH classification errors.",
            expected_tradeoff="The additional check may increase response length.",
            templates=MappingProxyType(templates),
            source_sha256=template_source_sha256(templates),
        )
        return ProposalResult(
            candidate=candidate,
            receipt_path=kwargs["proposal_directory"] / "offline-receipt.json",
            attempt=1,
        )


class _Evaluator:
    def p0(self, **_kwargs):
        return _report("cot", canonical_experiment_p0_prompt_sha256(), 0.9, 0.9)

    def candidate(self, **kwargs):
        return _report(
            kwargs["candidate_id"], template_source_sha256(kwargs["prompt"]), 0.4, 0.4
        )


@pytest.fixture(scope="module")
def preparation_material():
    records = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    validated = validate_sciver_records(records, source_path=DATASET_PATH)
    split = build_experiment_split(validated)
    private = build_experiment_private_manifest(
        split, source_dataset_sha256=source_dataset_sha256(DATASET_PATH)
    )
    normalized = {
        item.sample_id: {
            **{
                key: (
                    str(DATASET_PATH.parent / value.removeprefix("./SciVer/"))
                    if key in {"paper_path", "image_path", "item1_path", "item2_path"}
                    and isinstance(value, str)
                    and value.startswith("./SciVer/")
                    else value
                )
                for key, value in item.record.items()
            },
            "sample_id": item.sample_id,
            "gold_label": "yes" if item.record["label"] else "no",
        }
        for item in validated
    }
    return private, derive_experiment_search_safe_manifest(private), normalized


@pytest.fixture
def final_inputs(tmp_path, preparation_material):
    private, safe, normalized = preparation_material
    private_path = tmp_path / "trusted" / "private_split_manifest.json"
    safe_path = tmp_path / "search" / "search_safe_manifest.json"
    save_experiment_private_manifest(private_path, private)
    save_experiment_search_safe_manifest(safe_path, safe)
    evaluator = _Evaluator()
    state = Orchestrator(
        repository_root=tmp_path,
        run_id=RUN_ID,
        search_input=_search_input(
            private["split"]["split_sha256"], safe["search_membership_sha256"]
        ),
        solver_identity_sha256="c" * 64,
        cache=object(),
        executor=object(),
        proposer=_Proposer(),
        proposer_identity={"kind": "offline_fake", "version": 1},
        p0_evaluator=evaluator.p0,
        candidate_evaluator=evaluator.candidate,
    ).run()
    assert state["status"] == "patience_stopped"
    freeze_experiment_winner(repository_root=tmp_path, run_id=RUN_ID)
    return {
        "root": tmp_path,
        "private_path": private_path,
        "safe_path": safe_path,
        "records": [normalized[sample_id] for sample_id in private["split"]["FINAL"]["sample_ids"]],
        "contract": canonical_final_evaluation_solver_contract(
            solver_identity_sha256="c" * 64
        ),
    }


def _preflight(inputs, **overrides):
    values = {
        "repository_root": inputs["root"],
        "run_id": RUN_ID,
        "frozen_winner_path": None,
        "private_manifest_path": inputs["private_path"],
        "search_safe_manifest_path": inputs["safe_path"],
        "final_records": inputs["records"],
        "solver_contract": inputs["contract"],
    }
    values.update(overrides)
    return preflight_final_evaluation(**values)


def test_preflight_and_authorized_initialization_are_isolated_and_resumable(
    final_inputs, monkeypatch
):
    monkeypatch.setattr(
        "meta_harness.solver.create_live_solver_client",
        lambda *_args, **_kwargs: pytest.fail("FINAL planning constructed a live client"),
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("FINAL planning started a subprocess"),
    )
    destination = final_evaluation_state_path(final_inputs["root"], RUN_ID)

    dry_run = _preflight(final_inputs)

    assert not destination.exists()
    assert dry_run["run_id"] == RUN_ID
    assert set(dry_run) == {
        "schema_version", "protocol_id", "run_id", "execution_identity_sha256",
        "p0_prompt_sha256", "p_star_prompt_sha256",
    }
    state = initialize_final_evaluation(
        repository_root=final_inputs["root"],
        run_id=RUN_ID,
        frozen_winner_path=None,
        private_manifest_path=final_inputs["private_path"],
        search_safe_manifest_path=final_inputs["safe_path"],
        final_records=final_inputs["records"],
        solver_contract=final_inputs["contract"],
        authorize_final_execution=True,
    )
    resumed = initialize_final_evaluation(
        repository_root=final_inputs["root"],
        run_id=RUN_ID,
        frozen_winner_path=None,
        private_manifest_path=final_inputs["private_path"],
        search_safe_manifest_path=final_inputs["safe_path"],
        final_records=final_inputs["records"],
        solver_contract=final_inputs["contract"],
        authorize_final_execution=True,
    )

    assert destination.exists()
    assert state == resumed
    assert state["status"] == "planned"
    assert "sample_id" not in canonical_json(state)


def test_final_state_requires_its_own_authorization(final_inputs):
    with pytest.raises(FinalError, match="authorization"):
        initialize_final_evaluation(
            repository_root=final_inputs["root"],
            run_id=RUN_ID,
            frozen_winner_path=None,
            private_manifest_path=final_inputs["private_path"],
            search_safe_manifest_path=final_inputs["safe_path"],
            final_records=final_inputs["records"],
            solver_contract=final_inputs["contract"],
            authorize_final_execution=False,
        )


def test_execution_migrates_zero_completion_legacy_planned_state_before_dispatch():
    """A pre-accounting planned artifact must resume without a KeyError."""

    import meta_harness.final_evaluation as final_module

    legacy = {
        "status": "planned",
        "variants": [
            {"completed_request_sha256": []},
            {"completed_request_sha256": []},
        ],
    }

    assert final_module._migrate_zero_completion_legacy_state(legacy) is True
    assert legacy["status"] == "planned"
    assert all(
        variant["completed_request_sha256"] == []
        and variant["outcomes"] == {
            "confusion": {"yes": {"yes": 0, "no": 0}, "no": {"yes": 0, "no": 0}},
            "label_support": {"yes": 0, "no": 0},
            "parsed_predictions": 0,
            "abstentions_or_parse_failures": 0,
            "infrastructure_failures": 0,
        }
        for variant in legacy["variants"]
    )


def test_final_execution_retries_and_persists_only_aggregate_metrics(final_inputs, monkeypatch):
    class _RetryingSolver:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, _request):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("offline retry", http_status_code=429)
            return SolverResult(content="Answer: yes")

    solver = _RetryingSolver()
    import meta_harness.final_evaluation as final_module

    request = SolverRequest(
        model="Qwen2.5-VL-7B-Instruct",
        messages=({"role": "user", "content": "offline FINAL request"},),
        generation=SolverGenerationSettings.from_config(canonical_experiment_config()),
    )
    monkeypatch.setattr(final_module, "build_solver_request", lambda *_args: request)
    atomic_replace = final_module._atomic_replace_json
    monkeypatch.setattr(
        final_module,
        "_atomic_replace_json",
        lambda path, value: atomic_replace(path, value) if value["status"] == "complete" else None,
    )
    receipt = execute_final_evaluation(
        repository_root=final_inputs["root"],
        run_id=RUN_ID,
        frozen_winner_path=None,
        private_manifest_path=final_inputs["private_path"],
        search_safe_manifest_path=final_inputs["safe_path"],
        final_records=final_inputs["records"],
        solver_contract=final_inputs["contract"],
        solver=solver,
        authorize_final_execution=True,
        retry_policy=SolverRetryPolicy(
            maximum_attempts=2,
            initial_backoff_seconds=0,
            maximum_backoff_seconds=0,
        ),
    )

    state = load_final_evaluation_state(
        final_evaluation_state_path(final_inputs["root"], RUN_ID)
    )
    assert solver.calls == 2001
    for variant in state["variants"]:
        assert variant["metrics"]["parse_coverage"] == 1.0
        assert variant["metrics"]["accuracy"] is not None
        assert "sample_id" not in canonical_json(variant)
        assert "Answer:" not in canonical_json(variant)
    assert all("metrics" in variant for variant in receipt["variants"])
    assert all(
        variant["outcomes"]["parsed_predictions"] == 1000
        and variant["outcomes"]["infrastructure_failures"] == 0
        for variant in receipt["variants"]
    )


def test_preflight_rejects_missing_freeze(final_inputs):
    with pytest.raises(FinalError, match="frozen winner"):
        _preflight(final_inputs, frozen_winner_path=final_inputs["root"] / "missing.json")


def test_preflight_rejects_wrong_final_commitment(final_inputs):
    safe = json.loads(final_inputs["safe_path"].read_text(encoding="utf-8"))
    safe["final_membership_commitment"] = "0" * 64
    safe["search_safe_manifest_sha256"] = hashlib.sha256(
        canonical_json(dict(safe, search_safe_manifest_sha256="")).encode("utf-8")
    ).hexdigest()
    final_inputs["safe_path"].write_text(canonical_json(safe), encoding="utf-8")

    with pytest.raises(FinalError, match="manifests are incompatible"):
        _preflight(final_inputs)


@pytest.mark.parametrize("size", [999, 1001])
def test_preflight_rejects_wrong_final_record_count(final_inputs, size):
    records = final_inputs["records"][:size]
    if size == 1001:
        records = [*final_inputs["records"], {"sample_id": "foreign-final-row"}]

    with pytest.raises(FinalError, match="exactly 1,000"):
        _preflight(final_inputs, final_records=records)


def test_preflight_rejects_duplicate_final_ids(final_inputs):
    records = list(final_inputs["records"])
    records[-1] = records[0]

    with pytest.raises(FinalError, match="duplicate"):
        _preflight(final_inputs, final_records=records)


def test_preflight_rejects_wrong_final_order(final_inputs):
    with pytest.raises(FinalError, match="order"):
        _preflight(final_inputs, final_records=list(reversed(final_inputs["records"])))


def test_preflight_rejects_incompatible_search_run_identity(final_inputs):
    state_path = (
        final_inputs["root"]
        / "workspace"
        / "meta_harness"
        / "full_search_v3"
        / RUN_ID
        / "orchestration_state.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["identity"]["solver_identity_sha256"] = "d" * 64
    state_path.write_text(canonical_json(state), encoding="utf-8")

    with pytest.raises(FinalError, match="frozen winner and SEARCH run identity"):
        _preflight(final_inputs)


def test_preflight_rejects_wrong_solver_or_generation_contract(final_inputs):
    wrong_solver = canonical_final_evaluation_solver_contract(
        solver_identity_sha256="d" * 64
    )
    with pytest.raises(FinalError, match="solver identity"):
        _preflight(final_inputs, solver_contract=wrong_solver)

    values = final_inputs["contract"].as_dict()
    values["model"] = "wrong-model"
    with pytest.raises(FinalError, match="model"):
        FinalSolverContract(**values)

    values = final_inputs["contract"].as_dict()
    values["generation"]["max_tokens"] = 1
    with pytest.raises(FinalError, match="generation"):
        FinalSolverContract(**values)


def test_search_modules_do_not_import_the_private_final_loader():
    root = Path(__file__).resolve().parents[2]
    for relative_path in (
        "meta_harness/search_evaluator.py",
        "meta_harness/search_orchestrator.py",
        "meta_harness/prompt_proposer.py",
        "meta_harness/winner_freeze.py",
    ):
        source = (root / relative_path).read_text(encoding="utf-8")
        assert "load_trusted_experiment_private_manifest" not in source
