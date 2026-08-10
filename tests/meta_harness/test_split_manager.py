import copy
import json

import pytest

from meta_harness.config import MetaHarnessConfig
from meta_harness.split_manager import (
    SplitManagerError,
    build_split_manifest,
    load_split_manifest,
    proposer_split_summary,
    save_split_manifest,
    verify_split_manifest,
)


def _paper_id_records(paper_count=10, claims_per_paper=2):
    return [
        {
            "sample_id": f"paper-{paper_index}-claim-{claim_index}",
            "paper_id": f"released-paper-{paper_index}",
        }
        for paper_index in range(paper_count)
        for claim_index in range(claims_per_paper)
    ]


def _sample_to_split(manifest):
    return {
        sample_id: split_name
        for split_name, split in manifest["splits"].items()
        for sample_id in split["sample_ids"]
    }


def test_default_split_is_deterministic_paper_disjoint_and_20_20_60():
    records = _paper_id_records()

    first = build_split_manifest(records)
    second = build_split_manifest(list(reversed(records)))

    assert first == second
    assert first["seed"] == 42
    assert first["total_papers"] == 10
    assert first["total_samples"] == 20
    assert {
        name: split["paper_count"] for name, split in first["splits"].items()
    } == {
        "search": 2,
        "validation": 2,
        "final_test": 6,
    }
    assert {
        name: split["sample_count"] for name, split in first["splits"].items()
    } == {
        "search": 4,
        "validation": 4,
        "final_test": 12,
    }
    assert {
        name: split["achieved_sample_ratio"]
        for name, split in first["splits"].items()
    } == {
        "search": 0.2,
        "validation": 0.2,
        "final_test": 0.6,
    }
    assert len(first["dataset_sha256"]) == 64
    assert len(first["split_sha256"]) == 64

    sample_splits = _sample_to_split(first)
    for paper_index in range(10):
        assert (
            sample_splits[f"paper-{paper_index}-claim-0"]
            == sample_splits[f"paper-{paper_index}-claim-1"]
        )
    verify_split_manifest(first)


def test_seed_changes_configuration_and_split_identity():
    records = _paper_id_records()

    first = build_split_manifest(records, MetaHarnessConfig(seed=42))
    second = build_split_manifest(records, MetaHarnessConfig(seed=7))

    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["config_sha256"] != second["config_sha256"]
    assert first["split_sha256"] != second["split_sha256"]


def test_dataset_hash_changes_when_record_content_changes():
    records = _paper_id_records()
    changed_records = copy.deepcopy(records)
    changed_records[0]["claim"] = "changed scientific claim"

    original = build_split_manifest(records)
    changed = build_split_manifest(changed_records)

    assert original["dataset_sha256"] != changed["dataset_sha256"]
    assert original["split_sha256"] != changed["split_sha256"]


def test_duplicate_paper_bytes_at_different_paths_stay_together(tmp_path):
    shared_content = {"sections": [{"text": "same paper"}]}
    first_copy = tmp_path / "first.json"
    second_copy = tmp_path / "second.json"
    first_copy.write_text(json.dumps(shared_content), encoding="utf-8")
    second_copy.write_text(json.dumps(shared_content), encoding="utf-8")
    other_paths = []
    for index in range(4):
        path = tmp_path / f"other-{index}.json"
        path.write_text(
            json.dumps({"sections": [{"text": f"paper {index}"}]}),
            encoding="utf-8",
        )
        other_paths.append(path)
    records = [
        {"sample_id": "shared-a", "paper_path": str(first_copy)},
        {"sample_id": "shared-b", "paper_path": str(second_copy)},
        *[
            {"sample_id": f"other-{index}", "paper_path": str(path)}
            for index, path in enumerate(other_paths)
        ],
    ]

    manifest = build_split_manifest(records)

    sample_splits = _sample_to_split(manifest)
    assert sample_splits["shared-a"] == sample_splits["shared-b"]
    assert manifest["total_papers"] == 5
    assert all(
        split["paper_identity_sources"]["paper_sha256"] == split["paper_count"]
        for split in manifest["splits"].values()
    )


def test_released_paper_id_takes_precedence_over_unreadable_path():
    records = [
        {
            "sample_id": f"sample-{index}",
            "paper_id": f"paper-{index}",
            "paper_path": "/path/that/is/not/read.json",
        }
        for index in range(3)
    ]

    manifest = build_split_manifest(records)

    assert manifest["total_papers"] == 3
    assert all(
        split["paper_identity_sources"]["paper_id"] == split["paper_count"]
        for split in manifest["splits"].values()
    )


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"sample_id": "missing-paper"}, "sample_id fallback is forbidden"),
        (
            {"sample_id": "missing-file", "paper_path": "/missing/paper.json"},
            "readable valid JSON",
        ),
    ],
)
def test_missing_paper_identity_fails_clearly(record, message):
    records = [
        record,
        {"sample_id": "paper-2", "paper_id": "paper-2"},
        {"sample_id": "paper-3", "paper_id": "paper-3"},
    ]

    with pytest.raises(SplitManagerError, match=message):
        build_split_manifest(records)


def test_invalid_paper_json_is_rejected(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not-json", encoding="utf-8")
    records = [
        {"sample_id": "bad", "paper_path": str(invalid)},
        {"sample_id": "paper-2", "paper_id": "paper-2"},
        {"sample_id": "paper-3", "paper_id": "paper-3"},
    ]

    with pytest.raises(SplitManagerError, match="readable valid JSON"):
        build_split_manifest(records)


def test_proposer_summary_excludes_final_test_ids_and_all_paths(tmp_path):
    records = _paper_id_records()
    manifest = build_split_manifest(records)
    final_sample_id = manifest["splits"]["final_test"]["sample_ids"][0]

    summary = proposer_split_summary(manifest)
    serialized = json.dumps(summary)

    assert set(summary["splits"]) == {"search", "validation"}
    assert "final_test" not in serialized
    assert "sample_ids" not in serialized
    assert "paper_group_ids" not in serialized
    assert "dataset_sha256" not in serialized
    assert "split_sha256" not in serialized
    assert "achieved_sample_ratio" not in serialized
    assert final_sample_id not in serialized
    assert "paper_path" not in serialized


def test_manifest_save_load_is_immutable_and_hash_checked(tmp_path):
    manifest = build_split_manifest(_paper_id_records())
    path = tmp_path / "splits" / "manifest.json"

    assert save_split_manifest(path, manifest) == path
    assert load_split_manifest(path) == manifest
    assert save_split_manifest(path, manifest) == path

    changed = build_split_manifest(
        _paper_id_records(),
        MetaHarnessConfig(seed=7),
    )
    with pytest.raises(SplitManagerError, match="refusing to replace"):
        save_split_manifest(path, changed)

    tampered = copy.deepcopy(manifest)
    tampered["splits"]["search"]["sample_count"] += 1
    with pytest.raises(SplitManagerError, match="sample_count"):
        verify_split_manifest(tampered)
