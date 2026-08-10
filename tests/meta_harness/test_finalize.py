from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

import main as benchmark_cli
import scripts.finalize_meta_harness as finalize_cli
import scripts.export_meta_cot as export_cli
import scripts.run_meta_cot_transfer as transfer_cli
import utils.prompt_registry as prompt_registry
from meta_harness.finalize import (
    META_COT_VARIANT,
    SEARCH_MODEL,
    TRANSFER_MODELS,
    CompletionReceiptError,
    FinalTestConfirmationError,
    FinalizationError,
    execute_final_test,
    execute_final_test_pair,
    execute_transfer_matrix,
    execute_transfers,
    freeze_winner,
    frozen_winner_path,
    load_frozen_winner,
    select_global_best,
)
from meta_harness.orchestrator import (
    BudgetLimits,
    FinalizedRunError,
)
from meta_harness.split_manager import build_split_manifest
from model_inference.remote_api import generate_remote_responses
from tests.meta_harness.test_orchestrator import (
    FakeEvaluator,
    FakeProposer,
    _orchestrator,
)
from utils.constant import COT_PROMPT
from utils.dataset_adapters import AdaptedSample
from utils.result_writer import iter_result_records


METHODS = ("direct", "analytical", "parallel", "sequential")


def test_search_and_transfer_model_roles_are_frozen_without_qwen_default():
    assert SEARCH_MODEL == "gemma-4-26B-A4B-it"
    assert SEARCH_MODEL not in TRANSFER_MODELS
    assert TRANSFER_MODELS == (
        "Qwen2.5-VL-7B-Instruct",
        "gemma-4-31B-it",
        "gemma-3-27b-it",
    )


class RecordingSolver:
    def __init__(self, answer: str = "yes") -> None:
        self.answer = answer
        self.calls = []
        self.last_usage = None

    def create_chat_completion(self, model, messages):
        self.calls.append((model, messages))
        self.last_usage = {"input_tokens": 7, "output_tokens": 2}
        return (
            "Reasoning remains fixed. Therefore, the final answer is: "
            f"Answer: {self.answer}"
        )


def _samples_and_manifest(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), "red").save(first)
    Image.new("RGB", (2, 2), "green").save(second)

    samples = []
    records = []
    for paper_index in range(5):
        for method in METHODS:
            sample_id = f"paper-{paper_index}-{method}"
            record = {
                "sample_id": sample_id,
                "paper_id": f"paper-{paper_index}",
                "claim_type": method,
                "claim": f"claim-{paper_index}-{method}",
                "context": f"context-{paper_index}",
                "gold_label": "yes",
            }
            if method in {"direct", "analytical"}:
                record.update(
                    {
                        "caption": f"caption-{method}",
                        "image_path": str(first),
                    }
                )
            else:
                record.update(
                    {
                        "caption1": f"caption-1-{method}",
                        "caption2": f"caption-2-{method}",
                        "item1_path": str(first),
                        "item2_path": str(second),
                    }
                )
            samples.append(AdaptedSample(sample_id, record))
            records.append(record)
    manifest = build_split_manifest(records)
    return samples, manifest


def _completed_search(tmp_path, samples, manifest):
    validation_ids = set(manifest["splits"]["validation"]["sample_ids"])
    validation_samples = tuple(
        sample for sample in samples if sample.sample_id in validation_ids
    )
    proposer = FakeProposer()
    evaluator = FakeEvaluator(
        {
            "baseline_cot": 0.5,
            "iter_001_candidate_01": 0.8,
            "iter_001_candidate_02": 0.7,
        },
        tokens=6,
    )
    state = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=evaluator,
        manifest=manifest,
        samples=validation_samples,
        limits=BudgetLimits(max_iterations=1, max_candidates=2),
    ).run()
    assert state["status"] == "completed"
    return state, proposer


def _freeze(tmp_path, samples, manifest):
    state, proposer = _completed_search(tmp_path, samples, manifest)
    winner = freeze_winner(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        code_revision="0" * 40,
    )
    return state, proposer, winner


def _final_samples(samples, manifest):
    final_ids = set(manifest["splits"]["final_test"]["sample_ids"])
    return tuple(sample for sample in samples if sample.sample_id in final_ids)


def _prompt_texts(solver):
    return [
        messages[0]["content"][-1]["text"]
        for _, messages in solver.calls
    ]


def test_deterministic_winner_uses_macro_f1_accuracy_tokens_and_id():
    state = {
        "candidates": {
            "lower_score": {
                "status": "evaluated",
                "score": {
                    "eligible": True,
                    "macro_f1": 0.7,
                    "accuracy": 1.0,
                    "parse_coverage": 1.0,
                    "unresolved_api_failures": 0,
                    "solver_calls": 1,
                    "tokens": 1,
                    "latency_seconds": 0.1,
                },
            },
            "z_tied": {
                "status": "evaluated",
                "score": {
                    "eligible": True,
                    "macro_f1": 0.8,
                    "accuracy": 0.1,
                    "parse_coverage": 1.0,
                    "unresolved_api_failures": 0,
                    "solver_calls": 2,
                    "tokens": 8,
                    "latency_seconds": 1.0,
                },
            },
            "a_tied": {
                "status": "evaluated",
                "score": {
                    "eligible": True,
                    "macro_f1": 0.8,
                    "accuracy": 0.0,
                    "parse_coverage": 1.0,
                    "unresolved_api_failures": 0,
                    "solver_calls": 2,
                    "tokens": 8,
                    "latency_seconds": 1.0,
                },
            },
            "ineligible": {
                "status": "evaluated",
                "score": {
                    "eligible": False,
                    "macro_f1": 1.0,
                    "solver_calls": 0,
                    "tokens": 0,
                    "latency_seconds": 0.0,
                },
            },
        }
    }

    assert select_global_best(state)["candidate_id"] == "z_tied"


def test_winner_rechecks_coverage_and_unresolved_failures():
    state = {
        "candidates": {
            "unsafe_high_score": {
                "status": "evaluated",
                "score": {
                    "eligible": True,
                    "macro_f1": 1.0,
                    "accuracy": 1.0,
                    "parse_coverage": 0.5,
                    "unresolved_api_failures": 1,
                    "solver_calls": 2,
                    "tokens": 2,
                    "latency_seconds": 0.1,
                },
            },
            "safe": {
                "status": "evaluated",
                "score": {
                    "eligible": True,
                    "macro_f1": 0.7,
                    "accuracy": 0.7,
                    "parse_coverage": 1.0,
                    "unresolved_api_failures": 0,
                    "solver_calls": 2,
                    "tokens": 4,
                    "latency_seconds": 0.2,
                },
            },
        }
    }

    assert select_global_best(state)["candidate_id"] == "safe"


def test_freeze_is_immutable_and_verifies_candidate_prompt_and_artifact_hashes(
    tmp_path,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    path = frozen_winner_path(tmp_path, state["run_id"])
    original = path.read_bytes()

    assert winner["candidate_id"] == "iter_001_candidate_01"
    assert winner["validation"]["macro_f1"] == 0.8
    assert winner["search_solver_configuration"]["model"] == SEARCH_MODEL
    assert winner["run_id"] == state["run_id"]
    assert winner["code_revision"] == "0" * 40
    assert path.stat().st_mode & 0o222 == 0

    retry = freeze_winner(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        code_revision="0" * 40,
    )
    assert retry == winner
    assert path.read_bytes() == original

    tampered = tmp_path / "tampered_winner.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["templates"]["direct"] += " changed"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalizationError, match="hash mismatch"):
        load_frozen_winner(tampered)


def test_frozen_run_cannot_resume_evolution(tmp_path):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, proposer, _ = _freeze(tmp_path, samples, manifest)
    validation_ids = set(manifest["splits"]["validation"]["sample_ids"])
    validation_samples = tuple(
        sample for sample in samples if sample.sample_id in validation_ids
    )
    resumed = _orchestrator(
        tmp_path,
        proposer=proposer,
        evaluator=FakeEvaluator(),
        manifest=manifest,
        samples=validation_samples,
        limits=BudgetLimits(max_iterations=2, max_candidates=4),
    )

    with pytest.raises(FinalizedRunError, match="cannot resume"):
        resumed.run(
            resume=True,
            safe_resume_changes={"max_iterations", "max_candidates"},
        )


def test_frozen_registration_keeps_cot_unchanged_and_variants_independent(
    tmp_path,
    monkeypatch,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    cot_sources = {
        method: COT_PROMPT[method].template for method in COT_PROMPT
    }
    cot_objects = dict(COT_PROMPT)
    monkeypatch.setattr(
        prompt_registry,
        "_PROMPT_FAMILIES",
        {"cot": COT_PROMPT},
    )
    monkeypatch.setattr(prompt_registry, "_META_COT_ARTIFACT_SHA256", None)

    meta_cot = prompt_registry.register_frozen_prompt_family(
        frozen_winner_path(tmp_path, state["run_id"])
    )

    assert prompt_registry.get_prompt_family("cot") is COT_PROMPT
    assert {
        method: COT_PROMPT[method].template for method in COT_PROMPT
    } == cot_sources
    assert dict(COT_PROMPT) == cot_objects
    assert {
        method: meta_cot[method].template for method in meta_cot
    } == winner["templates"]
    assert meta_cot is prompt_registry.get_prompt_family(META_COT_VARIANT)
    assert meta_cot is not COT_PROMPT

    cot_solver = RecordingSolver()
    meta_solver = RecordingSolver()
    record = samples[0].record
    generate_remote_responses(
        SEARCH_MODEL,
        [record],
        prompt_registry.get_prompt_family("cot"),
        client=cot_solver,
    )
    generate_remote_responses(
        SEARCH_MODEL,
        [record],
        prompt_registry.get_prompt_family(META_COT_VARIANT),
        client=meta_solver,
    )
    assert len(cot_solver.calls) == len(meta_solver.calls) == 1
    assert _prompt_texts(cot_solver)[0] == COT_PROMPT["direct"].substitute(
        claim=record["claim"],
        context=record["context"],
        caption=record["caption"],
    )
    assert _prompt_texts(meta_solver)[0] != _prompt_texts(cot_solver)[0]


def test_export_meta_cot_writes_exact_immutable_prompt_family(tmp_path):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    output_dir = tmp_path / "optimized"

    assert export_cli.cli(
        [
            "--frozen-winner",
            str(frozen_winner_path(tmp_path, state["run_id"])),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    prompts_path = output_dir / "meta_cot.prompts.json"
    metadata_path = output_dir / "meta_cot.metadata.json"
    assert json.loads(prompts_path.read_text(encoding="utf-8")) == winner[
        "templates"
    ]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["method"] == META_COT_VARIANT
    assert metadata["prompt_sha256"] == winner["prompt_sha256"]
    assert prompts_path.stat().st_mode & 0o222 == 0
    assert metadata_path.stat().st_mode & 0o222 == 0

    assert export_cli.cli(
        [
            "--frozen-winner",
            str(frozen_winner_path(tmp_path, state["run_id"])),
            "--output-dir",
            str(output_dir),
        ]
    ) == 0


def test_fresh_remote_cli_registers_meta_cot_from_explicit_frozen_artifact(
    tmp_path,
    monkeypatch,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    artifact_path = frozen_winner_path(tmp_path, state["run_id"])
    monkeypatch.setattr(
        prompt_registry,
        "_PROMPT_FAMILIES",
        {"cot": COT_PROMPT},
    )
    monkeypatch.setattr(prompt_registry, "_META_COT_ARTIFACT_SHA256", None)
    import utils.dataset_adapters as dataset_adapters

    adapter = SimpleNamespace(
        adapter_name="offline-test",
        load=Mock(return_value=[samples[0]]),
    )
    monkeypatch.setattr(
        dataset_adapters,
        "get_dataset_adapter",
        Mock(return_value=adapter),
    )
    client_factory = Mock(side_effect=AssertionError("dry-run created client"))
    monkeypatch.setattr(benchmark_cli, "_create_remote_client", client_factory)
    arguments = SimpleNamespace(
        dataset="SciVer",
        prompt=META_COT_VARIANT,
        meta_cot_artifact=artifact_path,
        method="cot",
        n=1,
        model=SEARCH_MODEL,
        dry_run=True,
        live_api=False,
        output_dir=str(tmp_path / "outputs"),
        data_path=str(tmp_path / "unused.json"),
        max_num=1,
        resume=False,
        request_delay=0.0,
    )

    assert benchmark_cli._run_remote(arguments) == 0
    client_factory.assert_not_called()
    registered = prompt_registry.get_prompt_family(META_COT_VARIANT)
    assert {
        method: registered[method].template for method in registered
    } == winner["templates"]
    assert prompt_registry.get_prompt_family("cot") is COT_PROMPT


def test_confirmation_one_time_receipt_resume_and_reasoning_identity(
    tmp_path,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, proposer, winner = _freeze(tmp_path, samples, manifest)
    final_samples = _final_samples(samples, manifest)
    solver = RecordingSolver()

    with pytest.raises(FinalTestConfirmationError):
        execute_final_test(
            repository_root=tmp_path,
            run_id=state["run_id"],
            split_manifest=manifest,
            final_samples=final_samples,
            solver=solver,
            confirm_final_test=False,
        )
    assert solver.calls == []

    receipt = execute_final_test(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=solver,
        confirm_final_test=True,
    )
    assert len(solver.calls) == len(final_samples)
    assert {model for model, _ in solver.calls} == {SEARCH_MODEL}
    assert receipt["prompt_sha256"] == winner["prompt_sha256"]

    second_solver = RecordingSolver(answer="no")
    resumed = execute_final_test(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=second_solver,
        confirm_final_test=False,
    )
    assert resumed == receipt
    assert second_solver.calls == []
    assert len(proposer.calls) == 1

    results_path = (
        frozen_winner_path(tmp_path, state["run_id"]).parent
        / "executions"
        / SEARCH_MODEL
        / "results.jsonl"
    )
    records = list(iter_result_records(results_path))
    assert {record["prompt_variant"] for record in records} == {
        META_COT_VARIANT
    }
    assert {record["candidate_id"] for record in records} == {
        winner["candidate_id"]
    }
    assert {record["method"] for record in records} == set(METHODS)
    assert {record["reasoning_method"] for record in records} == set(METHODS)
    assert all(
        record["method"] == record["reasoning_method"] for record in records
    )
    assert results_path.with_name("results.csv").is_file()
    assert results_path.with_name("metrics.json").is_file()
    assert results_path.with_name("manifest.json").is_file()


def test_gemma_search_solver_final_pair_runs_cot_and_meta_cot_once(tmp_path):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    final_samples = _final_samples(samples, manifest)
    solver = RecordingSolver()

    receipts = execute_final_test_pair(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=solver,
        confirm_final_test=True,
    )

    assert set(receipts) == {"cot", META_COT_VARIANT}
    assert len(solver.calls) == 2 * len(final_samples)
    assert receipts["cot"]["candidate_id"] == "baseline_cot"
    assert receipts["cot"]["prompt_variant"] == "cot"
    assert receipts[META_COT_VARIANT]["candidate_id"] == winner["candidate_id"]
    assert (
        receipts[META_COT_VARIANT]["prompt_sha256"]
        == winner["prompt_sha256"]
    )

    retry_solver = RecordingSolver(answer="no")
    assert execute_final_test_pair(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=retry_solver,
        confirm_final_test=False,
    ) == receipts
    assert retry_solver.calls == []


@pytest.mark.parametrize(
    "gate_flag",
    ["--live-api", "--confirm-final-test"],
)
def test_finalize_cli_rejects_incomplete_live_gate_before_freezing(
    tmp_path,
    monkeypatch,
    gate_flag,
):
    loader = Mock(side_effect=AssertionError("gate loaded split"))
    monkeypatch.setattr(finalize_cli, "load_split_manifest", loader)
    monkeypatch.setattr(finalize_cli, "REPOSITORY_ROOT", tmp_path)

    result = finalize_cli.cli(
        [
            "--run-id",
            "gated_run",
            "--split-manifest",
            str(tmp_path / "split.json"),
            gate_flag,
        ]
    )

    assert result == 1
    loader.assert_not_called()
    assert not (tmp_path / "workspace" / "meta_harness").exists()


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        (
            finalize_cli,
            [
                "--run-id",
                "missing_credentials",
                "--split-manifest",
                "split.json",
                "--confirm-final-test",
                "--live-api",
            ],
        ),
        (
            transfer_cli,
            [
                "--run-id",
                "missing_credentials",
                "--split-manifest",
                "split.json",
                "--dataset-path",
                "dataset.json",
                "--confirm-final-test",
                "--live-api",
            ],
        ),
    ],
)
def test_final_live_clis_require_credentials_before_artifact_access(
    monkeypatch,
    capsys,
    module,
    arguments,
):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setattr(module, "load_cli_environment", Mock(return_value=False))

    result = module.cli(arguments)

    assert result == 1
    assert "API_URL and API_KEY" in capsys.readouterr().err


def test_exact_frozen_prompt_bytes_transfer_once_without_selection_feedback(
    tmp_path,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, proposer, winner = _freeze(tmp_path, samples, manifest)
    final_samples = _final_samples(samples, manifest)
    original_solver = RecordingSolver()
    execute_final_test(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=original_solver,
        confirm_final_test=True,
    )
    frozen_before = frozen_winner_path(
        tmp_path,
        state["run_id"],
    ).read_bytes()
    transfer_solvers = {
        model: RecordingSolver(answer="no") for model in TRANSFER_MODELS
    }

    receipts = execute_transfers(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solvers=transfer_solvers,
        confirm_final_test=True,
    )

    original_prompts = _prompt_texts(original_solver)
    assert set(receipts) == set(TRANSFER_MODELS)
    for model, solver in transfer_solvers.items():
        assert len(solver.calls) == len(final_samples)
        assert {called_model for called_model, _ in solver.calls} == {model}
        assert _prompt_texts(solver) == original_prompts
        assert receipts[model]["prompt_sha256"] == winner["prompt_sha256"]
    assert frozen_winner_path(
        tmp_path,
        state["run_id"],
    ).read_bytes() == frozen_before
    assert len(proposer.calls) == 1

    retry_solvers = {model: RecordingSolver() for model in TRANSFER_MODELS}
    assert execute_transfers(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solvers=retry_solvers,
        confirm_final_test=False,
    ) == receipts
    assert all(not solver.calls for solver in retry_solvers.values())

    aggregate = (
        frozen_winner_path(tmp_path, state["run_id"]).parent
        / "results_manifest.json"
    )
    aggregate_payload = json.loads(aggregate.read_text(encoding="utf-8"))
    assert aggregate_payload["prompt_sha256"] == winner["prompt_sha256"]
    assert set(aggregate_payload["models"]) == {
        SEARCH_MODEL,
        *TRANSFER_MODELS,
    }
    assert {
        configuration["model"]
        for configuration in (
            entry["model_configuration"]
            for entry in aggregate_payload["models"].values()
        )
    } == {SEARCH_MODEL, *TRANSFER_MODELS}
    fixed_configurations = []
    for entry in aggregate_payload["models"].values():
        configuration = dict(entry["model_configuration"])
        configuration.pop("model")
        fixed_configurations.append(configuration)
    assert all(
        configuration == fixed_configurations[0]
        for configuration in fixed_configurations[1:]
    )


def test_four_model_final_matrix_contains_cot_and_exact_meta_cot(tmp_path):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    final_samples = _final_samples(samples, manifest)
    search_solver = RecordingSolver()
    search_receipts = execute_final_test_pair(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=search_solver,
        confirm_final_test=True,
    )
    transfer_solvers = {
        model: RecordingSolver() for model in TRANSFER_MODELS
    }

    receipts = execute_transfer_matrix(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solvers=transfer_solvers,
        confirm_final_test=True,
    )

    assert set(receipts) == set(TRANSFER_MODELS)
    for model, model_receipts in receipts.items():
        assert set(model_receipts) == {"cot", META_COT_VARIANT}
        assert model_receipts["cot"]["prompt_sha256"] == search_receipts[
            "cot"
        ]["prompt_sha256"]
        assert model_receipts[META_COT_VARIANT]["prompt_sha256"] == winner[
            "prompt_sha256"
        ]
        assert len(transfer_solvers[model].calls) == 2 * len(final_samples)
    aggregate = json.loads(
        (
            frozen_winner_path(tmp_path, state["run_id"]).parent
            / "results_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert all(
        set(entry["prompt_variants"]) == {"cot", META_COT_VARIANT}
        for entry in aggregate["models"].values()
    )


def test_transfer_refuses_before_original_final_test_and_rejects_split_change(
    tmp_path,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, _ = _freeze(tmp_path, samples, manifest)
    final_samples = _final_samples(samples, manifest)

    with pytest.raises(CompletionReceiptError, match="must complete"):
        execute_transfers(
            repository_root=tmp_path,
            run_id=state["run_id"],
            split_manifest=manifest,
            final_samples=final_samples,
            solvers={model: RecordingSolver() for model in TRANSFER_MODELS},
            confirm_final_test=True,
        )

    changed = json.loads(json.dumps(manifest))
    changed["splits"]["final_test"]["sample_ids"].reverse()
    without_hash = {
        key: value for key, value in changed.items() if key != "split_sha256"
    }
    changed["split_sha256"] = __import__("hashlib").sha256(
        json.dumps(
            without_hash,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(FinalizationError, match="does not match"):
        execute_final_test(
            repository_root=tmp_path,
            run_id=state["run_id"],
            split_manifest=changed,
            final_samples=tuple(reversed(final_samples)),
            solver=RecordingSolver(),
            confirm_final_test=True,
        )


def test_final_results_never_change_frozen_selection(tmp_path):
    samples, manifest = _samples_and_manifest(tmp_path)
    state, _, winner = _freeze(tmp_path, samples, manifest)
    path = frozen_winner_path(tmp_path, state["run_id"])
    before = path.read_bytes()
    solver = RecordingSolver(answer="no")

    execute_final_test(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=_final_samples(samples, manifest),
        solver=solver,
        confirm_final_test=True,
    )

    assert load_frozen_winner(path)["candidate_id"] == winner["candidate_id"]
    assert path.read_bytes() == before
    serialized_prompts = json.dumps(_prompt_texts(solver))
    assert "gold_label" not in serialized_prompts
    assert "final_test" not in json.dumps(winner["validation"])
