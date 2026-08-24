"""Deterministic offline end-to-end coverage for full-SEARCH Milestone 5."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import MappingProxyType

import pytest
from PIL import Image

from meta_harness.records import validate_sciver_records
from meta_harness.search_evaluator import SearchInput
from meta_harness.final_evaluation import (
    canonical_final_evaluation_solver_contract,
    execute_final_evaluation,
    final_evaluation_completion_receipt_path,
    load_final_evaluation_state,
    preflight_final_evaluation,
)
from meta_harness.winner_freeze import freeze_experiment_winner
from meta_harness.search_orchestrator import Orchestrator
from meta_harness.preparation import (
    load_trusted_experiment_private_manifest,
    prepare_experiment,
)
from meta_harness.prompt_proposer import (
    Candidate,
    ProposalResult,
)
from meta_harness.solver import SolverResult, solver_request_payload_sha256
from meta_harness.prompt_family import (
    canonical_baseline_sources,
    canonical_json,
    template_source_sha256,
)
from utils.dataset_adapters import get_dataset_adapter


class _FakeCodex:
    def __init__(self) -> None:
        self.calls = []

    def propose(self, **kwargs):
        self.calls.append(kwargs)
        iteration = kwargs["iteration"]
        templates = {
            method: source + f"\nOffline candidate verification {iteration}."
            for method, source in canonical_baseline_sources().items()
        }
        candidate = Candidate(
            candidate_id=f"candidate_{iteration:03d}",
            parent_id=kwargs["parent_id"],
            hypothesis="Independent checks reduce SEARCH classification errors.",
            expected_tradeoff="The check may increase response length.",
            templates=MappingProxyType(templates),
            source_sha256=template_source_sha256(templates),
        )
        return ProposalResult(
            candidate=candidate,
            receipt_path=kwargs["proposal_directory"] / "offline-receipt.json",
            attempt=1,
        )


class _SearchEvaluator:
    def __init__(self, *, p0_metrics, candidate_metrics):
        self.p0_metrics = p0_metrics
        self.candidate_metrics = candidate_metrics

    @staticmethod
    def _report(candidate_id, prompt_sha256, metrics):
        return {
            "protocol_id": "sciver_full_search_v3",
            "stage": "SEARCH",
            "candidate_id": candidate_id,
            "prompt_sha256": prompt_sha256,
            "total_records": 1000,
            "completed_solver_responses": 1000,
            "parsed_predictions": 1000,
            "abstentions_or_parse_failures": 0,
            "infrastructure_failures": 0,
            "metrics": {
                "macro_f1": metrics[0],
                "accuracy": metrics[1],
                "parse_coverage": 1.0,
                "rankable": True,
            },
        }

    def p0(self, **_kwargs):
        from meta_harness.search_evaluator import canonical_experiment_p0_prompt_sha256

        return self._report("cot", canonical_experiment_p0_prompt_sha256(), self.p0_metrics)

    def candidate(self, **kwargs):
        iteration = int(kwargs["candidate_id"].rsplit("_", 1)[1])
        return self._report(
            kwargs["candidate_id"],
            template_source_sha256(kwargs["prompt"]),
            self.candidate_metrics(iteration),
        )


class _FakeFinalSolver:
    def __init__(self, *, interrupt_after: int | None = None) -> None:
        self.interrupt_after = interrupt_after
        self.successful = []
        self.attempts = 0

    def complete(self, request):
        self.attempts += 1
        if self.interrupt_after is not None and self.attempts == self.interrupt_after:
            raise KeyboardInterrupt("offline interruption")
        serialized = canonical_json(list(request.messages))
        assert "gold_label" not in serialized
        self.successful.append(
            {
                "request_sha256": solver_request_payload_sha256(request),
                "messages": serialized,
            }
        )
        return SolverResult(content="Answer: yes")


@pytest.fixture(scope="module")
def synthetic_preparation(tmp_path_factory):
    root = tmp_path_factory.mktemp("m5_synthetic")
    source = root / "testset.json"
    papers = root / "papers"
    papers.mkdir()
    image_path = root / "evidence.png"
    Image.new("RGB", (1, 1), color="white").save(image_path, format="PNG")
    rows = []
    for index in range(2000):
        paper_id = f"paper_{index:04d}"
        (papers / f"{paper_id}.json").write_text(
            json.dumps(
                {
                    "sections": [{"section_id": "1", "section_name": "Synthetic", "text": f"Context {index}"}],
                    "tables": {
                        "1": {
                            "caption": f"Caption {index}",
                            "capture": f"Caption {index}",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "paperid": paper_id,
                "paper_path": f"papers/{paper_id}.json",
                "claim_type": "direct",
                "type": "table",
                "item": "1",
                "section": ["1"],
                "image_path": "evidence.png",
                "request_id": index,
                "origin_statement": f"Origin {index}",
                "perturbed_statement": f"Perturbed {index}",
                "perturbed_explanation": f"Explanation {index}",
                "claim": f"Claim {index}",
                "label": bool(index % 2),
            }
        )
    source.write_text(json.dumps(rows), encoding="utf-8")
    artifacts = prepare_experiment(
        source_path=source,
        private_directory=root / "trusted",
        search_directory=root / "search",
    )
    private = load_trusted_experiment_private_manifest(artifacts.private_manifest_path)
    checked = validate_sciver_records(rows, source_path=source)
    adapted = get_dataset_adapter("SciVer").load(source)
    normalized = {
        record.sample_id: {
            **item.record,
            "sample_id": record.sample_id,
            "gold_label": item.record["gold_label"],
        }
        for item, record in zip(adapted, checked)
    }
    final_records = [normalized[sample_id] for sample_id in private["split"]["FINAL"]["sample_ids"]]
    return root, artifacts, final_records


@pytest.mark.parametrize("p0_wins", [False, True])
def test_prepare_search_freeze_paired_final_resume_and_isolation(
    tmp_path, synthetic_preparation, p0_wins, monkeypatch
):
    _source_root, artifacts, final_records = synthetic_preparation
    repository_root = tmp_path
    search_input = SearchInput(
        _manifest=MappingProxyType(
            json.loads(artifacts.search_safe_manifest_path.read_text(encoding="utf-8"))
        ),
        _records=tuple(
            json.loads(artifacts.search_dataset_path.read_text(encoding="utf-8"))
        ),
    )
    proposer = _FakeCodex()
    evaluator = _SearchEvaluator(
        p0_metrics=(0.9, 0.9) if p0_wins else (0.5, 0.5),
        candidate_metrics=(lambda iteration: (0.4, 0.4)) if p0_wins else (
            lambda iteration: (0.6, 0.6) if iteration == 1 else (0.5, 0.5)
        ),
    )
    run_id = f"m5_{'p0' if p0_wins else 'candidate'}"
    search = Orchestrator(
        repository_root=repository_root,
        run_id=run_id,
        search_input=search_input,
        solver_identity_sha256="c" * 64,
        cache=object(),
        executor=object(),
        proposer=proposer,
        proposer_identity={"kind": "offline_fake", "version": 1},
        p0_evaluator=evaluator.p0,
        candidate_evaluator=evaluator.candidate,
    ).run()
    assert search["status"] == "patience_stopped"
    assert len(search["iterations"]) == 15
    assert len(proposer.calls) == 15
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("M5 integration started a subprocess"),
    )
    search_bytes = (
        repository_root / "workspace" / "meta_harness" / "full_search_v3" / run_id / "orchestration_state.json"
    ).read_bytes()

    import meta_harness.final_evaluation as final_module

    with monkeypatch.context() as guarded:
        guarded.setattr(
            final_module,
            "load_trusted_experiment_private_manifest",
            lambda *_args, **_kwargs: pytest.fail("FINAL membership loaded before freeze"),
        )
        with pytest.raises(Exception):
            preflight_final_evaluation(
                repository_root=repository_root,
                run_id=run_id,
                frozen_winner_path=repository_root / "missing-freeze.json",
                private_manifest_path=artifacts.private_manifest_path,
                search_safe_manifest_path=artifacts.search_safe_manifest_path,
                final_records=final_records,
                solver_contract=canonical_final_evaluation_solver_contract(solver_identity_sha256="c" * 64),
            )

    frozen = freeze_experiment_winner(repository_root=repository_root, run_id=run_id)
    frozen_path = repository_root / "workspace" / "meta_harness" / "full_search_v3" / run_id / "freeze" / "frozen_winner.json"
    frozen_bytes = frozen_path.read_bytes()
    contract = canonical_final_evaluation_solver_contract(solver_identity_sha256="c" * 64)
    interrupted = _FakeFinalSolver(interrupt_after=137)
    with pytest.raises(KeyboardInterrupt, match="interruption"):
        execute_final_evaluation(
            repository_root=repository_root,
            run_id=run_id,
            frozen_winner_path=None,
            private_manifest_path=artifacts.private_manifest_path,
            search_safe_manifest_path=artifacts.search_safe_manifest_path,
            final_records=final_records,
            solver_contract=contract,
            solver=interrupted,
            authorize_final_execution=True,
        )
    resumed = _FakeFinalSolver()
    receipt = execute_final_evaluation(
        repository_root=repository_root,
        run_id=run_id,
        frozen_winner_path=None,
        private_manifest_path=artifacts.private_manifest_path,
        search_safe_manifest_path=artifacts.search_safe_manifest_path,
        final_records=final_records,
        solver_contract=contract,
        solver=resumed,
        authorize_final_execution=True,
    )

    all_successful = [*interrupted.successful, *resumed.successful]
    assert len(all_successful) == 2000
    assert len({item["request_sha256"] for item in all_successful}) == 2000 if not p0_wins else 1000
    assert receipt["logical_calls"] == 2000
    assert final_evaluation_completion_receipt_path(repository_root, run_id).is_file()
    final_state = load_final_evaluation_state(
        repository_root / "workspace" / "meta_harness" / "full_search_v3" / run_id / "final" / "final_state.json"
    )
    assert final_state["status"] == "complete"
    assert [len(item["completed_request_sha256"]) for item in final_state["variants"]] == [1000, 1000]
    assert all(len(set(item["completed_request_sha256"])) == 1000 for item in final_state["variants"])
    assert frozen_path.read_bytes() == frozen_bytes
    assert (
        repository_root / "workspace" / "meta_harness" / "full_search_v3" / run_id / "orchestration_state.json"
    ).read_bytes() == search_bytes
    assert all("gold_label" not in item["messages"] for item in all_successful)
    persisted = canonical_json(final_state) + frozen_path.read_text(encoding="utf-8")
    assert all(value not in persisted for value in ("API_KEY", "API_URL", "Authorization", "base64"))
    if p0_wins:
        assert receipt["variants"][0]["prompt_sha256"] == receipt["variants"][1]["prompt_sha256"]
    else:
        assert receipt["variants"][0]["prompt_sha256"] != receipt["variants"][1]["prompt_sha256"]
