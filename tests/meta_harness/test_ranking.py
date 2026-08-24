"""Pure offline tests for locked full SEARCH rank and patience semantics."""

from __future__ import annotations

import pytest

from meta_harness.config import (
    EXPERIMENT_PROTOCOL_ID,
    EXPERIMENT_SEARCH_SIZE,
)
from meta_harness.ranking import (
    PatienceState,
    RankedCandidate,
    advance_experiment_patience,
    eligible_experiment_candidate,
    rank_eligible_experiment_reports,
    rank_experiment_candidates,
)


def _candidate(candidate_id, prompt_letter, macro_f1, accuracy):
    return RankedCandidate(
        candidate_id=candidate_id,
        prompt_sha256=prompt_letter * 64,
        macro_f1=macro_f1,
        accuracy=accuracy,
    )


def _report(candidate, *, rankable=True, complete=True):
    total = EXPERIMENT_SEARCH_SIZE if complete else 999
    return {
        "protocol_id": EXPERIMENT_PROTOCOL_ID,
        "stage": "SEARCH",
        "candidate_id": candidate.candidate_id,
        "prompt_sha256": candidate.prompt_sha256,
        "total_records": total,
        "completed_solver_responses": total,
        "parsed_predictions": total,
        "abstentions_or_parse_failures": 0,
        "infrastructure_failures": 0,
        "metrics": {
            "macro_f1": candidate.macro_f1,
            "accuracy": candidate.accuracy,
            "parse_coverage": 1.0,
            "rankable": rankable,
        },
    }


def test_macro_f1_has_priority_over_accuracy():
    higher_f1 = _candidate("higher_f1", "b", 0.81, 0.60)
    higher_accuracy = _candidate("higher_accuracy", "a", 0.80, 1.00)

    ranking = rank_experiment_candidates([higher_accuracy, higher_f1])

    assert ranking == (higher_f1, higher_accuracy)


def test_accuracy_breaks_a_macro_f1_tie():
    lower_accuracy = _candidate("lower_accuracy", "a", 0.80, 0.70)
    higher_accuracy = _candidate("higher_accuracy", "b", 0.80, 0.71)

    assert rank_experiment_candidates([lower_accuracy, higher_accuracy]) == (
        higher_accuracy,
        lower_accuracy,
    )


def test_prompt_sha256_breaks_an_exact_metric_tie():
    earlier_hash = _candidate("later_id", "a", 0.80, 0.70)
    later_hash = _candidate("earlier_id", "b", 0.80, 0.70)

    assert rank_experiment_candidates([later_hash, earlier_hash]) == (
        earlier_hash,
        later_hash,
    )


def test_candidate_id_breaks_an_exact_hash_and_metric_tie():
    earlier_id = _candidate("candidate_a", "a", 0.80, 0.70)
    later_id = _candidate("candidate_b", "a", 0.80, 0.70)

    assert rank_experiment_candidates([later_id, earlier_id]) == (
        earlier_id,
        later_id,
    )


@pytest.mark.parametrize(
    "candidates",
    [
        [_candidate("candidate_z", "b", 0.80, 0.70), _candidate("candidate_a", "a", 0.80, 0.70)],
        [_candidate("candidate_z", "a", 0.80, 0.70), _candidate("candidate_a", "a", 0.80, 0.70)],
    ],
    ids=["hash", "candidate_id"],
)
def test_hash_or_id_only_winner_change_does_not_reset_patience(candidates):
    state = PatienceState(
        best_macro_f1=0.80,
        best_accuracy=0.70,
        consecutive_non_improving=4,
    )

    update = advance_experiment_patience(state, candidates)

    assert update.winner == _candidate("candidate_a", "a", 0.80, 0.70)
    assert update.metric_improved is False
    assert update.state.consecutive_non_improving == 5


def test_true_metric_improvement_resets_patience():
    state = PatienceState(
        best_macro_f1=0.80,
        best_accuracy=0.70,
        consecutive_non_improving=4,
    )
    improved = _candidate("candidate_improved", "a", 0.80, 0.71)

    update = advance_experiment_patience(state, [improved])

    assert update.metric_improved is True
    assert update.state.best_macro_f1 == 0.80
    assert update.state.best_accuracy == 0.71
    assert update.state.consecutive_non_improving == 0


def test_canonical_p0_can_beat_all_worse_candidates():
    p0 = _candidate("cot", "a", 0.90, 0.90)
    worse = _candidate("candidate_worse", "b", 0.89, 1.00)

    ranking = rank_eligible_experiment_reports([_report(worse), _report(p0)])

    assert ranking[0].candidate_id == "cot"


def test_incomplete_candidate_is_ineligible_and_never_ranked():
    eligible = _candidate("candidate_eligible", "a", 0.70, 0.70)
    incomplete = _candidate("candidate_incomplete", "b", 1.00, 1.00)

    assert eligible_experiment_candidate(_report(incomplete, complete=False)) is None
    assert rank_eligible_experiment_reports(
        [_report(incomplete, complete=False), _report(eligible)]
    ) == (eligible,)


def test_ranking_is_deterministic_independent_of_input_order():
    candidates = (
        _candidate("candidate_c", "c", 0.80, 0.70),
        _candidate("candidate_a", "a", 0.80, 0.70),
        _candidate("candidate_b", "b", 0.81, 0.60),
    )

    forward = rank_experiment_candidates(candidates)
    backward = rank_experiment_candidates(reversed(candidates))

    assert forward == backward
