import copy
import json
from pathlib import Path

import pytest

from meta_harness.full_search_v3 import (
    build_full_search_v3_split,
    validate_sciver_full_search_v3_records,
)
from meta_harness.full_search_v3_preparation import (
    FULL_SEARCH_V3_FINAL_COMMITMENT_SCHEMA,
    FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA,
    FULL_SEARCH_V3_PRIVATE_MANIFEST_SCHEMA,
    FULL_SEARCH_V3_SEARCH_DATASET_SCHEMA,
    FULL_SEARCH_V3_SEARCH_SAFE_MANIFEST_SCHEMA,
    FullSearchV3PreparationError,
    build_full_search_v3_private_manifest,
    compute_full_search_v3_final_commitment,
    derive_full_search_v3_search_safe_manifest,
    load_full_search_v3_search_dataset,
    load_full_search_v3_search_safe_manifest,
    load_trusted_full_search_v3_private_manifest,
    materialize_full_search_v3_search_records,
    prepare_full_search_v3,
    save_full_search_v3_private_manifest,
    source_dataset_sha256,
    verify_full_search_v3_private_manifest,
    verify_full_search_v3_search_dataset,
    verify_full_search_v3_search_safe_manifest,
)


TESTSET_PATH = Path(__file__).resolve().parents[2] / "data" / "sciver" / "testset.json"


@pytest.fixture(scope="module")
def source_records():
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def validated_records(source_records):
    return validate_sciver_full_search_v3_records(
        source_records,
        source_path=TESTSET_PATH,
    )


@pytest.fixture(scope="module")
def split(validated_records):
    return build_full_search_v3_split(validated_records)


@pytest.fixture(scope="module")
def private_manifest(split):
    return build_full_search_v3_private_manifest(
        split,
        source_dataset_sha256=source_dataset_sha256(TESTSET_PATH),
    )


def test_private_manifest_binds_exact_v3_split_and_rejects_conflicts(
    tmp_path, validated_records, private_manifest
):
    path = tmp_path / "trusted" / "private.json"
    assert save_full_search_v3_private_manifest(path, private_manifest) == path
    loaded = load_trusted_full_search_v3_private_manifest(path)
    verify_full_search_v3_private_manifest(
        loaded,
        records=validated_records,
        source_path=TESTSET_PATH,
    )

    assert loaded["schema_version"] == FULL_SEARCH_V3_PRIVATE_MANIFEST_SCHEMA
    assert loaded["protocol_id"] == "sciver_full_search_v3"
    assert loaded["split_seed"] == 42
    assert loaded["search_size"] == 1000
    assert loaded["final_size"] == 1000
    assert loaded["source_record_count"] == 2000
    assert loaded["source_paper_count"] == 786
    assert loaded["source_dataset_sha256"] == source_dataset_sha256(TESTSET_PATH)
    assert (
        loaded["preparation_identity_schema_version"]
        == FULL_SEARCH_V3_PREPARATION_IDENTITY_SCHEMA
    )
    assert len(loaded["preparation_identity_sha256"]) == 64
    assert loaded["split"]["SEARCH"]["sample_count"] == 1000
    assert loaded["split"]["FINAL"]["sample_count"] == 1000
    assert not set(loaded["split"]["SEARCH"]["sample_ids"]) & set(
        loaded["split"]["FINAL"]["sample_ids"]
    )
    assert not set(loaded["split"]["SEARCH"]["paper_identities"]) & set(
        loaded["split"]["FINAL"]["paper_identities"]
    )

    conflicting = build_full_search_v3_private_manifest(
        loaded["split"],
        source_dataset_sha256="0" * 64,
    )
    with pytest.raises(FullSearchV3PreparationError, match="refusing to replace"):
        save_full_search_v3_private_manifest(path, conflicting)


def test_search_safe_manifest_is_opaque_about_final_membership(private_manifest):
    safe = derive_full_search_v3_search_safe_manifest(private_manifest)
    serialized = json.dumps(safe, sort_keys=True)

    assert safe["schema_version"] == FULL_SEARCH_V3_SEARCH_SAFE_MANIFEST_SCHEMA
    assert safe["SEARCH"] == private_manifest["split"]["SEARCH"]
    assert safe["final_membership_commitment"] == private_manifest[
        "final_membership_commitment"
    ]
    assert "FINAL" not in safe
    assert "final_size" not in safe
    assert safe["preparation_identity_sha256"] == private_manifest[
        "preparation_identity_sha256"
    ]
    assert "private_manifest_path" not in serialized
    forbidden_keys = {
        "claim",
        "label",
        "gold_label",
        "image_path",
        "item1_path",
        "item2_path",
        "records",
    }
    assert forbidden_keys.isdisjoint(_all_mapping_keys(safe))
    for final_sample_id in private_manifest["split"]["FINAL"]["sample_ids"]:
        assert final_sample_id not in serialized
    for final_paper_id in private_manifest["split"]["FINAL"]["paper_identities"]:
        assert final_paper_id not in serialized
    verify_full_search_v3_search_safe_manifest(safe, private_manifest)

    changed_split = copy.deepcopy(private_manifest["split"])
    changed_split["FINAL"]["sample_ids"].reverse()
    assert compute_full_search_v3_final_commitment(changed_split) != safe[
        "final_membership_commitment"
    ]


def test_final_commitment_binds_split_identity(private_manifest):
    changed_split = copy.deepcopy(private_manifest["split"])
    changed_split["split_sha256"] = "0" * 64

    assert FULL_SEARCH_V3_FINAL_COMMITMENT_SCHEMA.startswith(
        "sciver_full_search_v3_"
    )
    assert compute_full_search_v3_final_commitment(changed_split) != private_manifest[
        "final_membership_commitment"
    ]


def test_materialized_search_dataset_is_manifest_ordered_and_evidence_ordered(
    validated_records, private_manifest
):
    safe = derive_full_search_v3_search_safe_manifest(private_manifest)
    records = materialize_full_search_v3_search_records(
        validated_records,
        private_manifest,
        source_path=TESTSET_PATH,
    )

    assert len(records) == 1000
    assert [record["sample_id"] for record in records] == safe["SEARCH"]["sample_ids"]
    assert len({record["sample_id"] for record in records}) == 1000
    assert not set(record["sample_id"] for record in records) & set(
        private_manifest["split"]["FINAL"]["sample_ids"]
    )
    verify_full_search_v3_search_dataset(records, safe)

    source_by_id = {
        checked.sample_id: checked.record for checked in validated_records
    }
    paired = next(
        record
        for record in records
        if record["claim_type"] in {"parallel", "sequential"}
    )
    source = source_by_id[paired["sample_id"]]
    assert Path(paired["item1_path"]).name == Path(source["item1_path"]).name
    assert Path(paired["item2_path"]).name == Path(source["item2_path"]).name


def test_preparation_is_deterministic_and_binds_source_content(tmp_path):
    first = prepare_full_search_v3(
        source_path=TESTSET_PATH,
        private_directory=tmp_path / "one" / "trusted",
        search_directory=tmp_path / "one" / "search",
    )
    second = prepare_full_search_v3(
        source_path=TESTSET_PATH,
        private_directory=tmp_path / "two" / "trusted",
        search_directory=tmp_path / "two" / "search",
    )
    first_private = load_trusted_full_search_v3_private_manifest(
        first.private_manifest_path
    )
    second_private = load_trusted_full_search_v3_private_manifest(
        second.private_manifest_path
    )
    first_safe = load_full_search_v3_search_safe_manifest(
        first.search_safe_manifest_path
    )
    second_safe = load_full_search_v3_search_safe_manifest(
        second.search_safe_manifest_path
    )
    first_records = load_full_search_v3_search_dataset(first.search_dataset_path)
    second_records = load_full_search_v3_search_dataset(second.search_dataset_path)

    assert first_private == second_private
    assert first_safe == second_safe
    assert first_records == second_records

    changed_source = tmp_path / "changed_testset.json"
    changed_source.write_bytes(TESTSET_PATH.read_bytes() + b"\n")
    assert source_dataset_sha256(changed_source) != first_private["source_dataset_sha256"]
    conflicting = build_full_search_v3_private_manifest(
        first_private["split"],
        source_dataset_sha256=source_dataset_sha256(changed_source),
    )
    assert conflicting["preparation_identity_sha256"] != first_private[
        "preparation_identity_sha256"
    ]
    with pytest.raises(FullSearchV3PreparationError, match="refusing to replace"):
        save_full_search_v3_private_manifest(
            first.private_manifest_path,
            conflicting,
        )


def test_reordered_source_has_identical_split_and_commitment(
    source_records, validated_records, split
):
    reordered_records = validate_sciver_full_search_v3_records(
        list(reversed(source_records)),
        source_path=TESTSET_PATH,
    )
    reordered_split = build_full_search_v3_split(reordered_records)

    assert {record.sample_id for record in reordered_records} == {
        record.sample_id for record in validated_records
    }
    assert {record.paper_identity for record in reordered_records} == {
        record.paper_identity for record in validated_records
    }
    assert reordered_split == split
    assert compute_full_search_v3_final_commitment(
        reordered_split
    ) == compute_full_search_v3_final_commitment(split)


def test_full_preparation_is_input_order_independent(tmp_path, source_records):
    reordered_root = tmp_path / "reordered" / "SciVer"
    reordered_root.mkdir(parents=True)
    (reordered_root / "papers").symlink_to(
        TESTSET_PATH.parent / "papers",
        target_is_directory=True,
    )
    (reordered_root / "images").symlink_to(
        TESTSET_PATH.parent / "images",
        target_is_directory=True,
    )
    reordered_source = reordered_root / "testset.json"
    reordered_source.write_text(
        json.dumps(list(reversed(source_records))),
        encoding="utf-8",
    )

    original = prepare_full_search_v3(
        source_path=TESTSET_PATH,
        private_directory=tmp_path / "original" / "private",
        search_directory=tmp_path / "original" / "search",
    )
    reordered = prepare_full_search_v3(
        source_path=reordered_source,
        private_directory=tmp_path / "reordered-artifacts" / "private",
        search_directory=tmp_path / "reordered-artifacts" / "search",
    )
    original_private = load_trusted_full_search_v3_private_manifest(
        original.private_manifest_path
    )
    reordered_private = load_trusted_full_search_v3_private_manifest(
        reordered.private_manifest_path
    )
    original_safe = load_full_search_v3_search_safe_manifest(
        original.search_safe_manifest_path
    )
    reordered_safe = load_full_search_v3_search_safe_manifest(
        reordered.search_safe_manifest_path
    )

    assert reordered_private["split"] == original_private["split"]
    assert reordered_private["final_membership_commitment"] == original_private[
        "final_membership_commitment"
    ]
    assert reordered_safe["SEARCH"] == original_safe["SEARCH"]
    assert reordered_safe["split_sha256"] == original_safe["split_sha256"]
    assert reordered_safe["final_membership_commitment"] == original_safe[
        "final_membership_commitment"
    ]
    assert reordered_private["source_dataset_sha256"] != original_private[
        "source_dataset_sha256"
    ]


def test_search_safe_loader_never_returns_private_final_membership(
    tmp_path, private_manifest
):
    safe = derive_full_search_v3_search_safe_manifest(private_manifest)
    path = tmp_path / "search" / "safe.json"
    from meta_harness.full_search_v3_preparation import (
        save_full_search_v3_search_safe_manifest,
    )

    save_full_search_v3_search_safe_manifest(path, safe)
    loaded = load_full_search_v3_search_safe_manifest(path)

    assert set(loaded) == set(safe)
    assert "FINAL" not in loaded
    assert set(loaded["SEARCH"]["sample_ids"]).isdisjoint(
        private_manifest["split"]["FINAL"]["sample_ids"]
    )


def test_search_dataset_verification_rejects_missing_duplicate_and_reordered_rows(
    validated_records, private_manifest
):
    safe = derive_full_search_v3_search_safe_manifest(private_manifest)
    records = materialize_full_search_v3_search_records(
        validated_records,
        private_manifest,
        source_path=TESTSET_PATH,
    )

    with pytest.raises(FullSearchV3PreparationError, match="count"):
        verify_full_search_v3_search_dataset(records[:-1], safe)
    duplicate = [*records[:-1], records[0]]
    with pytest.raises(FullSearchV3PreparationError, match="duplicate"):
        verify_full_search_v3_search_dataset(duplicate, safe)
    reordered = list(reversed(records))
    with pytest.raises(FullSearchV3PreparationError, match="order"):
        verify_full_search_v3_search_dataset(reordered, safe)


def _all_mapping_keys(value):
    if isinstance(value, dict):
        return set(value).union(
            *(_all_mapping_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_all_mapping_keys(item) for item in value))
    return set()
