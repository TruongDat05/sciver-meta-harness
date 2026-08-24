"""Shared run-ID contract tests for every durable boundary."""

from __future__ import annotations

import pytest

from meta_harness.final_evaluation import (
    FinalError,
    final_evaluation_state_path,
)
from meta_harness.winner_freeze import (
    FreezeError,
    experiment_frozen_winner_path,
)
from meta_harness.search_orchestrator import (
    experiment_orchestration_state_path,
)
from meta_harness.server_run import ServerError, _run_id


@pytest.mark.parametrize("run_id", [".", "..", "-leading", "space run"])
def test_all_durable_boundaries_reject_the_same_invalid_run_ids(tmp_path, run_id):
    with pytest.raises(ValueError, match="run_id"):
        experiment_orchestration_state_path(tmp_path, run_id)
    with pytest.raises(FreezeError, match="run_id"):
        experiment_frozen_winner_path(tmp_path, run_id)
    with pytest.raises(FinalError, match="run_id"):
        final_evaluation_state_path(tmp_path, run_id)
    with pytest.raises(ServerError, match="run_id"):
        _run_id(run_id)


def test_all_durable_boundaries_accept_a_canonical_run_id(tmp_path):
    run_id = "offline_run-42.1"

    assert experiment_orchestration_state_path(tmp_path, run_id).parent.name == run_id
    assert experiment_frozen_winner_path(tmp_path, run_id).parents[1].name == run_id
    assert final_evaluation_state_path(tmp_path, run_id).parent.parent.name == run_id
    assert _run_id(run_id) == run_id
