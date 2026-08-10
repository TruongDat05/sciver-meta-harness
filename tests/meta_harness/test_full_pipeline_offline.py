from __future__ import annotations

import json
from unittest.mock import Mock

from meta_harness.config import MetaHarnessConfig
from meta_harness.finalize import (
    SEARCH_MODEL,
    TRANSFER_MODELS,
    execute_final_test,
    execute_transfers,
    freeze_winner,
    frozen_winner_path,
)
from meta_harness.orchestrator import (
    BudgetLimits,
    MetaHarnessOrchestrator,
)
from meta_harness.proposer.codex_cli import CodexCLIProposer
from tests.meta_harness.test_codex_cli import (
    FIXED_TIME,
    FakeRunner,
    _batch,
)
from tests.meta_harness.test_finalize import (
    RecordingSolver,
    _final_samples,
    _prompt_texts,
    _samples_and_manifest,
)
import scripts.prepare_meta_harness_data as prepare_cli
import scripts.run_meta_harness as search_cli


class SearchSolver:
    """Deterministic fake solver whose answer depends only on prompt text."""

    def __init__(self) -> None:
        self.calls = []
        self.last_usage = None

    def create_chat_completion(self, model, messages):
        self.calls.append((model, messages))
        self.last_usage = {"input_tokens": 5, "output_tokens": 1}
        prompt = messages[0]["content"][-1]["text"]
        answer = "yes" if "before deciding" in prompt else "no"
        return (
            "Offline fake reasoning. Therefore, the final answer is: "
            f"Answer: {answer}"
        )


def test_prepare_and_dry_run_cli_are_offline_and_validation_only(
    tmp_path,
    monkeypatch,
    capsys,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    config_path = tmp_path / "config.json"
    dataset_path = tmp_path / "all_samples.json"
    split_path = tmp_path / "prepared" / "split_manifest.json"
    validation_path = tmp_path / "prepared" / "validation.json"
    config_path.write_text(
        json.dumps(MetaHarnessConfig().as_dict()),
        encoding="utf-8",
    )
    dataset_path.write_text(
        json.dumps([dict(sample.record) for sample in samples]),
        encoding="utf-8",
    )

    assert prepare_cli.cli(
        [
            "--config",
            str(config_path),
            "--dataset-path",
            str(dataset_path),
            "--split-manifest",
            str(split_path),
            "--validation-output",
            str(validation_path),
        ]
    ) == 0
    capsys.readouterr()
    validation_records = json.loads(validation_path.read_text(encoding="utf-8"))
    assert {
        record["sample_id"] for record in validation_records
    } == set(manifest["splits"]["validation"]["sample_ids"])
    assert {
        record["sample_id"] for record in validation_records
    }.isdisjoint(manifest["splits"]["final_test"]["sample_ids"])

    proposer_boundary = Mock(
        side_effect=AssertionError("dry-run created proposer")
    )
    monkeypatch.setattr(search_cli, "CodexCLIProposer", proposer_boundary)
    monkeypatch.setattr(search_cli, "REPOSITORY_ROOT", tmp_path)
    assert search_cli.cli(
        [
            "--config",
            str(config_path),
            "--split-manifest",
            str(split_path),
            "--dataset-path",
            str(validation_path),
            "--run-id",
            "offline_dry_run",
            "--dry-run",
            "--max-iterations",
            "2",
            "--max-candidates",
            "4",
        ]
    ) == 0

    plan = json.loads(capsys.readouterr().out)
    assert plan["validation_sample_count"] == len(validation_records)
    assert plan["planned_candidates"] == 4
    proposer_boundary.assert_not_called()
    assert not (tmp_path / "workspace" / "meta_harness").exists()


def test_complete_search_freeze_final_and_transfer_pipeline_is_offline(
    tmp_path,
):
    samples, manifest = _samples_and_manifest(tmp_path)
    validation_ids = set(manifest["splits"]["validation"]["sample_ids"])
    validation_samples = tuple(
        sample for sample in samples if sample.sample_id in validation_ids
    )
    runner = FakeRunner(json.dumps(_batch()))
    proposer = CodexCLIProposer(
        runner=runner,
        timestamp_factory=lambda: FIXED_TIME,
    )
    search_solver = SearchSolver()
    orchestrator = MetaHarnessOrchestrator(
        repository_root=tmp_path,
        run_id="offline_full_pipeline",
        config=MetaHarnessConfig(),
        split_manifest=manifest,
        validation_samples=validation_samples,
        solver=search_solver,
        proposer=proposer,
        limits=BudgetLimits(max_iterations=1, max_candidates=2),
    )

    state = orchestrator.run()

    assert state["status"] == "completed"
    assert state["best_candidate_id"] == "iter_001_candidate_01"
    assert len(runner.calls) == 2
    proposer_input = runner.calls[1][1]["input"]
    assert b"final_test" not in proposer_input
    assert b"gold_label" not in proposer_input
    assert {model for model, _ in search_solver.calls} == {SEARCH_MODEL}
    assert len(search_solver.calls) == 3 * len(validation_samples)

    calls_before_resume = (len(runner.calls), len(search_solver.calls))
    assert orchestrator.run(resume=True) == state
    assert (len(runner.calls), len(search_solver.calls)) == calls_before_resume

    winner = freeze_winner(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        code_revision="0" * 40,
    )
    frozen_bytes = frozen_winner_path(
        tmp_path,
        state["run_id"],
    ).read_bytes()
    final_samples = _final_samples(samples, manifest)
    original_solver = RecordingSolver()
    original_receipt = execute_final_test(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=original_solver,
        confirm_final_test=True,
    )
    transfer_solvers = {
        model: RecordingSolver() for model in TRANSFER_MODELS
    }
    transfer_receipts = execute_transfers(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solvers=transfer_solvers,
        confirm_final_test=True,
    )

    assert original_receipt["prompt_sha256"] == winner["prompt_sha256"]
    assert set(transfer_receipts) == set(TRANSFER_MODELS)
    for model, solver in transfer_solvers.items():
        assert _prompt_texts(solver) == _prompt_texts(original_solver)
        assert transfer_receipts[model]["prompt_sha256"] == winner[
            "prompt_sha256"
        ]
    assert frozen_winner_path(
        tmp_path,
        state["run_id"],
    ).read_bytes() == frozen_bytes

    retry_solver = RecordingSolver(answer="no")
    assert execute_final_test(
        repository_root=tmp_path,
        run_id=state["run_id"],
        split_manifest=manifest,
        final_samples=final_samples,
        solver=retry_solver,
        confirm_final_test=False,
    ) == original_receipt
    assert retry_solver.calls == []
