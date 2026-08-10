import json
import os

import pytest

import meta_harness.candidate_store as candidate_store_module
from meta_harness.candidate_store import (
    CANDIDATE_STATUSES,
    CandidateConflictError,
    CandidateIntegrityError,
    CandidateStore,
    CandidateStoreError,
)
from meta_harness.schemas import canonical_json, template_source_sha256


def _templates(suffix=""):
    return {
        "direct": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            f"Check support{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "analytical": (
            "Claim: $claim\nContext: $context\nCaption: $caption\n"
            f"Analyze support{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "parallel": (
            "Claim: $claim\nContext: $context\nCaption 1: $caption1\n"
            f"Caption 2: $caption2\nCompare evidence{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "sequential": (
            "Claim: $claim\nContext: $context\nCaption 1: $caption1\n"
            f"Caption 2: $caption2\nFollow evidence order{suffix}. "
            "Conclude with exactly Answer: yes or Answer: no."
        ),
    }


def _candidate(candidate_id="iter_001_candidate_01", **overrides):
    templates = overrides.pop("templates", _templates())
    value = {
        "candidate_id": candidate_id,
        "parent_id": "baseline_cot",
        "search_axis": "exploitation",
        "hypothesis": "Explicit checks will reduce unsupported yes predictions.",
        "templates": templates,
        "expected_tradeoff": "The reasoning may become longer.",
        "source_sha256": template_source_sha256(templates),
    }
    value.update(overrides)
    return value


def test_valid_round_trip_uses_required_layout_and_canonical_hash(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    created = store.create(_candidate())
    path = (
        tmp_path
        / "workspace"
        / "meta_harness"
        / "runs"
        / "run_001"
        / "candidates"
        / created.candidate_id
        / "candidate.json"
    )

    assert store.candidate_path(created.candidate_id) == path
    assert path.read_text(encoding="utf-8") == created.canonical_json()
    assert store.load(created.candidate_id) == created
    registry = store.read_registry()
    assert registry["candidates"][created.candidate_id] == {
        "sha256": created.sha256(),
        "status": "proposed",
    }
    assert store.registry_path != path


def test_identical_retry_is_safe_but_different_content_with_same_id_fails(
    tmp_path,
):
    store = CandidateStore(tmp_path, "run_001")
    first = store.create(_candidate())
    original_bytes = store.candidate_path(first.candidate_id).read_bytes()

    assert store.create(_candidate()) == first
    assert store.candidate_path(first.candidate_id).read_bytes() == original_bytes

    changed_templates = _templates(" with a stricter contradiction check")
    with pytest.raises(CandidateConflictError, match="new candidate ID"):
        store.create(
            _candidate(
                templates=changed_templates,
                expected_tradeoff="The output may be more conservative.",
            )
        )
    assert store.candidate_path(first.candidate_id).read_bytes() == original_bytes


def test_parent_must_exist_and_repair_uses_a_new_candidate_id(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    parent = store.create(_candidate())
    child = store.create(
        _candidate(
            "iter_002_candidate_01",
            parent_id=parent.candidate_id,
            templates=_templates(" with a repair"),
        )
    )
    assert store.load(child.candidate_id) == child

    with pytest.raises(CandidateStoreError, match="existing candidate"):
        store.create(
            _candidate(
                "iter_002_candidate_02",
                parent_id="missing_parent",
            )
        )


def test_load_rejects_hash_mismatch(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    created = store.create(_candidate())
    path = store.candidate_path(created.candidate_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["expected_tradeoff"] = "Reasoning will use more text."
    os.chmod(path, 0o600)
    path.write_text(canonical_json(payload), encoding="utf-8")

    with pytest.raises(CandidateIntegrityError, match="hash"):
        store.load(created.candidate_id)


def test_candidate_content_cannot_be_mutated_through_store(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    created = store.create(_candidate())
    before = store.candidate_path(created.candidate_id).read_bytes()

    replacement = _candidate(expected_tradeoff="A changed immutable field.")
    with pytest.raises(CandidateConflictError):
        store.save_candidate(replacement)

    assert store.candidate_path(created.candidate_id).read_bytes() == before


def test_atomic_write_failure_leaves_no_partial_candidate_or_temporary_file(
    tmp_path,
    monkeypatch,
):
    store = CandidateStore(tmp_path, "run_001")

    def fail_replace(source, destination):
        raise OSError("simulated local write failure")

    monkeypatch.setattr(candidate_store_module.os, "replace", fail_replace)
    with pytest.raises(CandidateStoreError, match="atomic"):
        store.create(_candidate())

    candidate_directory = (
        store.candidates_directory / "iter_001_candidate_01"
    )
    assert not store.candidate_path("iter_001_candidate_01").exists()
    assert not list(candidate_directory.glob("*.tmp"))
    assert (candidate_directory / ".create.lock").is_file()


def test_stale_lock_file_does_not_block_failure_recovery(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    lock_path = (
        store.candidates_directory
        / "iter_001_candidate_01"
        / ".create.lock"
    )
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("stale-process-marker\n", encoding="utf-8")

    created = store.create(_candidate())

    assert store.load(created.candidate_id) == created


def test_registry_status_updates_do_not_change_candidate_bytes(tmp_path):
    store = CandidateStore(tmp_path, "run_001")
    created = store.create(_candidate())
    path = store.candidate_path(created.candidate_id)
    candidate_bytes = path.read_bytes()

    for status in sorted(CANDIDATE_STATUSES):
        assert store.update_status(created.candidate_id, status) == status
        assert store.get_status(created.candidate_id) == status
        assert path.read_bytes() == candidate_bytes

    with pytest.raises(CandidateStoreError, match="status"):
        store.update_status(created.candidate_id, "unknown")


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Read API_KEY before deciding.",
        "Include a base64 image.",
    ],
)
def test_store_never_persists_secrets_or_image_bytes(tmp_path, unsafe_text):
    store = CandidateStore(tmp_path, "run_001")
    candidate = _candidate(expected_tradeoff=unsafe_text)

    with pytest.raises(CandidateStoreError, match="forbidden"):
        store.create(candidate)

    assert not store.candidate_path(candidate["candidate_id"]).exists()
