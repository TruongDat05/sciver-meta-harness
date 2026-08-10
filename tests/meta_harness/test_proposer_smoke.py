from __future__ import annotations

import json
from types import SimpleNamespace

import scripts.smoke_meta_harness_proposer as smoke_cli
from meta_harness.config import MetaHarnessConfig


def test_smoke_invokes_production_proposer_contract_without_solver(
    tmp_path,
    monkeypatch,
    capsys,
):
    calls = []
    search_feedback = {"schema_version": 1, "histories": []}

    class FakeProposer:
        def __init__(self, *, config):
            assert config.model == "gpt-5.6-terra"
            assert config.reasoning_effort == "medium"

        def propose(self, store, **kwargs):
            calls.append((store, kwargs))
            return SimpleNamespace(
                batch=SimpleNamespace(
                    candidates=(
                        SimpleNamespace(candidate_id="candidate_01"),
                        SimpleNamespace(candidate_id="candidate_02"),
                    )
                ),
                metadata={
                    "cli_version": "codex-cli 0.145.0",
                    "return_code": 0,
                },
                audit_path=tmp_path / "attempt.json",
            )

    monkeypatch.setattr(
        smoke_cli,
        "load_meta_harness_config",
        lambda _path: MetaHarnessConfig(),
    )
    monkeypatch.setattr(smoke_cli, "CodexCLIProposer", FakeProposer)
    monkeypatch.setattr(
        smoke_cli,
        "load_reparsed_search_feedback",
        lambda repository_root, path: search_feedback,
    )
    monkeypatch.setattr(smoke_cli, "REPOSITORY_ROOT", tmp_path)

    result = smoke_cli.cli(
        [
            "--config",
            str(tmp_path / "config.json"),
            "--run-id",
            "proposer_smoke_v1",
            "--proposer-history-reparse",
            str(tmp_path / "parser-v2-reparse"),
        ]
    )

    assert result == 0
    assert len(calls) == 1
    _store, kwargs = calls[0]
    assert kwargs["iteration"] == 1
    assert kwargs["parent_candidate_ids"] == ["baseline_cot"]
    assert kwargs["search_feedback"] is search_feedback
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "success"
    assert output["return_code"] == 0
