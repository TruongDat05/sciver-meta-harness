"""Synthetic offline end-to-end verification of the thin M6 server interface."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest
from PIL import Image

import meta_harness.final_evaluation as final_module
import meta_harness.server_run as server
from meta_harness.config import canonical_experiment_config
from meta_harness.search_evaluator import (
    SearchInput,
    canonical_experiment_p0_prompt_sha256,
)
from meta_harness.final_evaluation import FinalError
from meta_harness.search_orchestrator import (
    Orchestrator,
    ResumeError,
)
from meta_harness.prompt_proposer import (
    Candidate,
    ProposalResult,
)
from meta_harness.solver import (
    SolverGenerationSettings,
    SolverRequest,
    SolverResult,
)
from meta_harness.prompt_family import canonical_baseline_sources, template_source_sha256
from utils.constant import COT_PROMPT


RUN_ID = "server-e2e"
COMMIT = "a" * 40
SOLVER_IDENTITY = "c" * 64


class _FakeProposer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def propose(self, **kwargs):
        iteration = kwargs["iteration"]
        self.calls.append(iteration)
        templates = {
            method: source + f"\nOffline server candidate {iteration}."
            for method, source in canonical_baseline_sources().items()
        }
        candidate = Candidate(
            candidate_id=f"candidate_{iteration:03d}",
            parent_id=kwargs["parent_id"],
            hypothesis="Independent checks may reduce aggregate SEARCH errors.",
            expected_tradeoff="The check may add response text.",
            templates=MappingProxyType(templates),
            source_sha256=template_source_sha256(templates),
        )
        return ProposalResult(
            candidate=candidate,
            receipt_path=Path(kwargs["proposal_directory"]) / "offline-receipt.json",
            attempt=1,
        )


def _report(candidate_id: str, prompt_sha256: str, macro_f1: float) -> dict[str, object]:
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
            "macro_f1": macro_f1,
            "accuracy": macro_f1,
            "parse_coverage": 1.0,
            "rankable": True,
        },
    }


def _p0(**_kwargs):
    return _report("cot", canonical_experiment_p0_prompt_sha256(), 0.9)


def _candidate(**kwargs):
    return _report(
        kwargs["candidate_id"], template_source_sha256(kwargs["prompt"]), 0.4
    )


class _FakeFinalSolver:
    def __init__(self, interrupt_after: int | None = None) -> None:
        self.interrupt_after = interrupt_after
        self.calls: list[str] = []

    def list_model_ids(self):
        return ("gemma-4-26B-A4B-it",)

    def complete(self, request: SolverRequest) -> SolverResult:
        if self.interrupt_after is not None and len(self.calls) == self.interrupt_after:
            raise RuntimeError("simulated offline interruption")
        self.calls.append(str(request.messages[0]["content"]))
        return SolverResult(content="Answer: yes")


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("server_e2e_source")
    papers = root / "papers"
    papers.mkdir()
    Image.new("RGB", (1, 1), color="white").save(root / "evidence.png", format="PNG")
    rows = []
    for index in range(2000):
        paper_id = f"paper_{index:04d}"
        (papers / f"{paper_id}.json").write_text(
            json.dumps(
                {
                    "sections": [{"section_id": "1", "section_name": "Synthetic", "text": f"Context {index}"}],
                    "tables": {"1": {"caption": f"Caption {index}", "capture": f"Caption {index}"}},
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
                "origin_statement": "Origin",
                "perturbed_statement": "Perturbed",
                "perturbed_explanation": "Explanation",
                "claim": "Synthetic claim",
                "label": bool(index % 2),
            }
        )
    source = root / "testset.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    return source


def test_server_interface_end_to_end_with_fake_boundaries(
    monkeypatch, tmp_path, synthetic_source
):
    prepared = server.prepare_run(
        dataset_path=synthetic_source,
        repository_root=tmp_path,
        run_id=RUN_ID,
    )
    safe_manifest = json.loads(
        Path(prepared["search_safe_manifest_path"]).read_text(encoding="utf-8")
    )
    search_input = SearchInput(
        _manifest=MappingProxyType(safe_manifest),
        _records=tuple(
            json.loads(Path(prepared["search_dataset_path"]).read_text(encoding="utf-8"))
        ),
    )
    final_records = [
        {"sample_id": sample_id, "gold_label": "yes"}
        for sample_id in json.loads(
            Path(prepared["private_manifest_path"]).read_text(encoding="utf-8")
        )["split"]["FINAL"]["sample_ids"]
    ]
    proposer = _FakeProposer()
    monkeypatch.setattr(server, "load_experiment_search_input", lambda **_kwargs: search_input)
    monkeypatch.setattr(server.shutil, "which", lambda _name: "/offline/codex")
    monkeypatch.setattr(server, "_source_commit", lambda *_args: COMMIT)
    monkeypatch.setattr(server, "solver_identity_from_api_url", lambda _value: SOLVER_IDENTITY)
    monkeypatch.setattr(server, "SearchCache", lambda _path: object())
    monkeypatch.setattr(server, "RequestExecutor", lambda **_kwargs: object())
    monkeypatch.setattr(server, "Proposer", lambda **_kwargs: proposer)
    monkeypatch.setattr(server, "_trusted_final_records", lambda *_args: final_records)

    def engine(**kwargs):
        kwargs.pop("proposer")
        return Orchestrator(
            **kwargs,
            proposer=proposer,
            p0_evaluator=_p0,
            candidate_evaluator=_candidate,
        )

    monkeypatch.setattr(server, "Orchestrator", engine)
    live_clients = []
    monkeypatch.setattr(
        server,
        "_construct_live_solver",
        lambda *_args: live_clients.append(_FakeFinalSolver()) or live_clients[-1],
    )

    # The default stage gates must not construct a solver client.
    with pytest.raises(server.ServerAuthorizationError):
        server.start_or_resume_search_run(
            repository_root=tmp_path,
            run_id=RUN_ID,
            search_safe_manifest_path=prepared["search_safe_manifest_path"],
            search_records_path=prepared["search_dataset_path"],
            authorize_search_execution=False,
        )
    with pytest.raises(server.ServerAuthorizationError):
        server.start_or_resume_server_run_final(
            repository_root=tmp_path,
            run_id=RUN_ID,
            dataset_path=synthetic_source,
            private_manifest_path=prepared["private_manifest_path"],
            search_safe_manifest_path=prepared["search_safe_manifest_path"],
            authorize_final_execution=False,
        )
    assert live_clients == []

    preflight = server.preflight_search_run(
        repository_root=tmp_path,
        run_id=RUN_ID,
        search_safe_manifest_path=prepared["search_safe_manifest_path"],
        search_records_path=prepared["search_dataset_path"],
        source_commit=COMMIT,
        solver_identity_sha256=SOLVER_IDENTITY,
    )
    assert preflight["resume"] is False
    assert preflight["workload"]["maximum_search_logical_calls"] == 41000

    with pytest.raises(FinalError, match="frozen winner"):
        server.preflight_server_run_final(
            repository_root=tmp_path,
            run_id=RUN_ID,
            dataset_path=synthetic_source,
            private_manifest_path=prepared["private_manifest_path"],
            search_safe_manifest_path=prepared["search_safe_manifest_path"],
            solver_identity_sha256=SOLVER_IDENTITY,
        )

    smoke = server.run_meta_harness_smoke(
        repository_root=tmp_path,
        run_id=RUN_ID,
        search_safe_manifest_path=prepared["search_safe_manifest_path"],
        search_records_path=prepared["search_dataset_path"],
        authorize_smoke_execution=True,
        api_url="https://invalid.example.test/base",
        api_key="unmistakably-fake-key",
        source_commit=COMMIT,
    )
    assert smoke["status"] == "complete" and smoke["logical_calls"] == 1
    assert len(live_clients) == 1 and len(live_clients[0].calls) == 1
    reused_smoke = server.run_meta_harness_smoke(
        repository_root=tmp_path,
        run_id=RUN_ID,
        search_safe_manifest_path=prepared["search_safe_manifest_path"],
        search_records_path=prepared["search_dataset_path"],
        authorize_smoke_execution=True,
        api_url="https://invalid.example.test/base",
        api_key="unmistakably-fake-key",
        source_commit=COMMIT,
    )
    assert reused_smoke["reused"] is True and len(live_clients) == 1

    search = server.start_or_resume_search_run(
        repository_root=tmp_path,
        run_id=RUN_ID,
        search_safe_manifest_path=prepared["search_safe_manifest_path"],
        search_records_path=prepared["search_dataset_path"],
        authorize_search_execution=True,
        api_url="https://invalid.example.test/base",
        api_key="unmistakably-fake-key",
        source_commit=COMMIT,
    )
    assert search["search"]["status"] == "patience_stopped"
    assert proposer.calls == list(range(1, 16))
    assert server.inspect_server_run_status(
        repository_root=tmp_path, run_id=RUN_ID
    )["status"] == "patience_stopped"
    with pytest.raises(ResumeError, match="identity"):
        server.preflight_search_run(
            repository_root=tmp_path,
            run_id=RUN_ID,
            search_safe_manifest_path=prepared["search_safe_manifest_path"],
            search_records_path=prepared["search_dataset_path"],
            source_commit=COMMIT,
            solver_identity_sha256="d" * 64,
        )

    frozen = server.freeze_server_run_winner(repository_root=tmp_path, run_id=RUN_ID)
    final_preflight = server.preflight_server_run_final(
        repository_root=tmp_path,
        run_id=RUN_ID,
        dataset_path=synthetic_source,
        private_manifest_path=prepared["private_manifest_path"],
        search_safe_manifest_path=prepared["search_safe_manifest_path"],
        solver_identity_sha256=SOLVER_IDENTITY,
    )
    assert frozen["prompt_variant"] == "meta_cot"
    assert set(final_preflight) == {
        "schema_version", "protocol_id", "run_id", "execution_identity_sha256",
        "p0_prompt_sha256", "p_star_prompt_sha256",
    }

    generation = SolverGenerationSettings.from_config(canonical_experiment_config())
    monkeypatch.setattr(
        final_module,
        "build_solver_request",
        lambda record, prompt: SolverRequest(
            model=canonical_experiment_config().solver_model,
            messages=(
                {
                    "role": "user",
                    "content": (
                        ("cot:" if prompt is COT_PROMPT else "meta_cot:")
                        + record["sample_id"]
                    ),
                },
            ),
            generation=generation,
        ),
    )
    interrupted = _FakeFinalSolver(interrupt_after=37)
    monkeypatch.setattr(server, "_construct_live_solver", lambda *_args: interrupted)
    final = server.start_or_resume_server_run_final(
        repository_root=tmp_path,
        run_id=RUN_ID,
        dataset_path=synthetic_source,
        private_manifest_path=prepared["private_manifest_path"],
        search_safe_manifest_path=prepared["search_safe_manifest_path"],
        authorize_final_execution=True,
        api_url="https://invalid.example.test/base",
        api_key="unmistakably-fake-key",
    )
    # A terminal fake transport failure is accounted as FINAL infrastructure
    # failure; it must not abort the remaining paired logical workload.
    assert len(interrupted.calls) == 37
    assert len(set(interrupted.calls)) == 37
    assert final["final"]["logical_calls"] == 2000
    assert "sample_id" not in json.dumps(final)
    assert "unmistakably-fake-key" not in json.dumps({"prepared": prepared, **final})
