"""Offline progress-bar behavior for paired FINAL evaluation (narrow seam)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import meta_harness.final_evaluation as final_module
from meta_harness.retry import SolverRetryPolicy
from meta_harness.solver import SolverResult


class _Bar:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.updates = 0
        self.closed = False
        self.postfix = {}
        type(self).instances.append(self)

    def update(self, n=1):
        self.updates += n

    def set_postfix(self, **kwargs):
        self.postfix = kwargs

    def close(self):
        self.closed = True


def _records(n):
    return [{"sample_id": f"offline-final-{i}", "gold_label": "yes"} for i in range(n)]


def _expected_hashes(n):
    return [f"{i:064x}" for i in range(n)]


def _call(monkeypatch, tmp_path, *, n, completed=0, atomic=None, execute=None):
    _Bar.instances = []
    calls = {"build": 0}

    monkeypatch.setattr(final_module, "tqdm", _Bar)
    calls = {"build": 0}

    def counting_build(record, prompt):
        calls["build"] += 1
        return object()

    monkeypatch.setattr(final_module, "build_solver_request", counting_build)
    monkeypatch.setattr(
        final_module,
        "execute_solver_request_with_retry",
        (lambda *_a, **_k: SolverResult(content="Answer: yes"))
        if execute is None
        else execute,
    )
    if atomic is None:
        monkeypatch.setattr(final_module, "_atomic_replace_json", lambda path, value: None)
    else:
        monkeypatch.setattr(final_module, "_atomic_replace_json", atomic)

    accounting = final_module._empty_final_accounting()
    variant = {
        "prompt_variant": "cot",
        "candidate_id": "cot",
        "prompt_sha256": "a" * 64,
        "completed_request_sha256": list(_expected_hashes(completed)),
        "outcomes": accounting,
        "metrics": final_module._metrics_from_final_accounting(accounting),
    }
    state = {"status": "planned", "identity": {}, "variants": [variant]}
    destination = tmp_path / "final_state.json"

    final_module._complete_final_variant(
        variant=variant,
        expected_hashes=_expected_hashes(n),
        records=_records(n),
        prompt={"direct": "x"},
        solver=object(),
        retry_policy=SolverRetryPolicy(),
        state=state,
        destination=destination,
    )
    return SimpleNamespace(bar=_Bar.instances[-1], calls=calls, variant=variant, state=state)


def test_fresh_variant_starts_at_zero(monkeypatch, tmp_path):
    h = _call(monkeypatch, tmp_path, n=3, completed=0)
    assert h.bar.kwargs["total"] == 3
    assert h.bar.kwargs["initial"] == 0
    assert h.bar.kwargs["desc"] == "FINAL cot"
    assert h.calls["build"] == 3
    assert h.bar.updates == 3
    assert h.bar.closed is True


def test_resumed_variant_uses_completed_count_as_initial_and_does_not_redispatch(
    monkeypatch, tmp_path,
):
    h = _call(monkeypatch, tmp_path, n=5, completed=2)
    assert h.bar.kwargs["initial"] == 2
    assert h.calls["build"] == 3
    assert h.bar.updates == 3
    assert h.bar.closed is True


def test_fully_completed_variant_does_not_redispatch(monkeypatch, tmp_path):
    h = _call(monkeypatch, tmp_path, n=5, completed=5)
    assert h.bar.kwargs["initial"] == 5
    assert h.calls["build"] == 0
    assert h.bar.updates == 0
    assert h.bar.closed is True


def test_progress_advances_once_per_durable_checkpoint(monkeypatch, tmp_path):
    h = _call(monkeypatch, tmp_path, n=4, completed=1)
    assert h.bar.updates == 3
    assert h.calls["build"] == 3


def test_progress_does_not_advance_when_atomic_persistence_fails(monkeypatch, tmp_path):
    def fail_atomic(path, value):
        raise final_module.FinalError("unable to atomically update FINAL state")

    with pytest.raises(final_module.FinalError):
        _call(monkeypatch, tmp_path, n=3, completed=0, atomic=fail_atomic)
    bar = _Bar.instances[-1]
    assert bar.updates == 0
    assert bar.closed is True


def test_progress_bar_uses_stderr(monkeypatch, tmp_path):
    h = _call(monkeypatch, tmp_path, n=1, completed=0)
    assert h.bar.kwargs["file"] is sys.stderr
    assert h.bar.kwargs["total"] == 1


def test_progress_closes_on_interruption(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _call(monkeypatch, tmp_path, n=3, completed=0, execute=boom)
    bar = _Bar.instances[-1]
    assert bar.closed is True
    assert bar.updates == 0


def test_progress_reports_pass_and_fail_counts(monkeypatch, tmp_path):
    h = _call(monkeypatch, tmp_path, n=3, completed=0)
    assert h.bar.postfix == {"passed": 3, "failed": 0}
