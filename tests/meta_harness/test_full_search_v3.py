import copy
import json
from pathlib import Path

import pytest

from meta_harness.config import (
    FullSearchV3Config,
    MetaHarnessConfigError,
    canonical_full_search_v3_config,
)
from meta_harness.full_search_v3 import (
    FullSearchV3DataError,
    FullSearchV3SplitError,
    ValidatedSciVerV3Record,
    build_full_search_v3_split,
    validate_sciver_full_search_v3_records,
    verify_full_search_v3_split,
)


TESTSET_PATH = Path(__file__).resolve().parents[2] / "data" / "sciver" / "testset.json"


@pytest.fixture(scope="module")
def released_source_records():
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def released_records(released_source_records):
    return validate_sciver_full_search_v3_records(
        released_source_records,
        source_path=TESTSET_PATH,
    )


def test_canonical_v3_configuration_is_complete_and_locked():
    config = canonical_full_search_v3_config()

    assert config == FullSearchV3Config.from_mapping(config.as_dict())
    assert config.as_dict() == {
        "protocol_id": "sciver_full_search_v3",
        "search_size": 1000,
        "final_size": 1000,
        "split_seed": 42,
        "candidate_count_per_iteration": 1,
        "min_iterations": 15,
        "max_iterations": 40,
        "patience": 8,
        "proposal_attempts": 3,
        "solver": {
            "model": "Qwen/Qwen3.5-35B-A3B",
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "n": 1,
            "stream": False,
            "max_tokens": 8192,
        },
    }


def test_changed_v3_configuration_invariant_is_rejected():
    values = canonical_full_search_v3_config().as_dict()
    values["search_size"] = 999

    with pytest.raises(MetaHarnessConfigError, match="search_size is locked"):
        FullSearchV3Config.from_mapping(values)


def test_released_sample_identity_is_unique_order_independent_and_label_free(
    released_source_records, released_records
):
    assert len(released_records) == 2000
    assert len({record.sample_id for record in released_records}) == 2000

    reordered = validate_sciver_full_search_v3_records(
        list(reversed(released_source_records)),
        source_path=TESTSET_PATH,
    )
    assert {record.sample_id for record in reordered} == {
        record.sample_id for record in released_records
    }

    changed_label = copy.deepcopy(released_source_records[0])
    changed_label["label"] = not changed_label["label"]
    label_variant = validate_sciver_full_search_v3_records(
        [changed_label], source_path=TESTSET_PATH
    )
    original = validate_sciver_full_search_v3_records(
        [released_source_records[0]], source_path=TESTSET_PATH
    )
    assert label_variant[0].sample_id == original[0].sample_id


def test_malformed_or_unresolvable_v3_source_record_is_rejected(released_source_records):
    hybrid = copy.deepcopy(released_source_records[0])
    hybrid["item1_type"] = "chart"

    with pytest.raises(FullSearchV3DataError, match="unexpected or hybrid"):
        validate_sciver_full_search_v3_records([hybrid], source_path=TESTSET_PATH)

    missing_paper = copy.deepcopy(released_source_records[0])
    missing_paper["paper_path"] = "./SciVer/papers/does-not-exist.json"

    with pytest.raises(FullSearchV3DataError, match="does not resolve"):
        validate_sciver_full_search_v3_records(
            [missing_paper], source_path=TESTSET_PATH
        )


def test_paper_identity_uses_verified_paperid_or_json_bytes(released_source_records):
    verified = validate_sciver_full_search_v3_records(
        [released_source_records[0]], source_path=TESTSET_PATH
    )[0]
    fallback_source = copy.deepcopy(released_source_records[0])
    fallback_source["paperid"] = "unverified-paper-id"
    fallback = validate_sciver_full_search_v3_records(
        [fallback_source], source_path=TESTSET_PATH
    )[0]

    assert verified.paper_identity_source == "verified_paperid"
    assert verified.paper_identity.startswith("paper_sha256:")
    assert fallback.paper_identity_source == "paper_sha256"
    assert fallback.paper_identity.startswith("paper_sha256:")
    assert verified.paper_identity == fallback.paper_identity


def test_released_v3_split_is_exact_disjoint_and_order_independent(released_records):
    first = build_full_search_v3_split(released_records)
    second = build_full_search_v3_split(tuple(reversed(released_records)))

    assert first == second
    assert first["eligible_sample_count"] == 2000
    assert first["eligible_paper_count"] == 786
    assert first["SEARCH"]["sample_count"] == 1000
    assert first["FINAL"]["sample_count"] == 1000
    assert first["SEARCH"]["paper_count"] == 392
    assert first["FINAL"]["paper_count"] == 394
    assert not set(first["SEARCH"]["sample_ids"]) & set(first["FINAL"]["sample_ids"])
    assert not set(first["SEARCH"]["paper_identities"]) & set(
        first["FINAL"]["paper_identities"]
    )
    verify_full_search_v3_split(first, released_records)


def test_infeasible_paper_exclusive_allocation_fails_explicitly():
    records = tuple(
        ValidatedSciVerV3Record(
            sample_id=f"sample-{index}",
            paper_identity="paperid:only-paper",
            paper_identity_source="verified_paperid",
            record={},
        )
        for index in range(2000)
    )

    with pytest.raises(FullSearchV3SplitError, match="cannot allocate exact"):
        build_full_search_v3_split(records)
