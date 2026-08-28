import copy
import json
from pathlib import Path

import pytest

from meta_harness.config import (
    Config,
    MetaHarnessConfigError,
    canonical_experiment_config,
)
from meta_harness.records import (
    DataError,
    SplitError,
    ValidatedSciVerRecord,
    build_experiment_split,
    validate_sciver_records,
    verify_experiment_split,
)


TESTSET_PATH = Path(__file__).resolve().parents[2] / "data" / "sciver" / "testset.json"


@pytest.fixture(scope="module")
def released_source_records():
    return json.loads(TESTSET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def released_records(released_source_records):
    return validate_sciver_records(
        released_source_records,
        source_path=TESTSET_PATH,
    )


def test_canonical_v3_configuration_is_complete_and_locked():
    config = canonical_experiment_config()

    assert config == Config.from_mapping(config.as_dict())
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
            "model": "Qwen2.5-VL-7B-Instruct",
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "n": 1,
            "stream": False,
            "max_tokens": 8192,
        },
    }


def test_changed_v3_configuration_invariant_is_rejected():
    values = canonical_experiment_config().as_dict()
    values["search_size"] = 999

    with pytest.raises(MetaHarnessConfigError, match="search_size is locked"):
        Config.from_mapping(values)


def test_previous_solver_model_is_not_resume_compatible():
    values = canonical_experiment_config().as_dict()
    current_identity = canonical_experiment_config().sha256()
    values["solver"]["model"] = "Qwen/Qwen3.5-35B-A3B"

    with pytest.raises(MetaHarnessConfigError, match="solver_model is locked"):
        Config.from_mapping(values)

    assert current_identity == "2f1120f81f48b4c5dba837e2a6ca287c7a199bfaabdea65c1781e85fe04d542e"
    assert current_identity != "9cd31a36b1763de32ed3e3878176aee5d7521c645d8ca4bfe4e4f91dc5019517"


def test_released_sample_identity_is_unique_order_independent_and_label_free(
    released_source_records, released_records
):
    assert len(released_records) == 2000
    assert len({record.sample_id for record in released_records}) == 2000

    reordered = validate_sciver_records(
        list(reversed(released_source_records)),
        source_path=TESTSET_PATH,
    )
    assert {record.sample_id for record in reordered} == {
        record.sample_id for record in released_records
    }

    changed_label = copy.deepcopy(released_source_records[0])
    changed_label["label"] = not changed_label["label"]
    label_variant = validate_sciver_records(
        [changed_label], source_path=TESTSET_PATH
    )
    original = validate_sciver_records(
        [released_source_records[0]], source_path=TESTSET_PATH
    )
    assert label_variant[0].sample_id == original[0].sample_id


def test_malformed_or_unresolvable_v3_source_record_is_rejected(released_source_records):
    hybrid = copy.deepcopy(released_source_records[0])
    hybrid["item1_type"] = "chart"

    with pytest.raises(DataError, match="unexpected or hybrid"):
        validate_sciver_records([hybrid], source_path=TESTSET_PATH)

    missing_paper = copy.deepcopy(released_source_records[0])
    missing_paper["paper_path"] = "./SciVer/papers/does-not-exist.json"

    with pytest.raises(DataError, match="does not resolve"):
        validate_sciver_records(
            [missing_paper], source_path=TESTSET_PATH
        )


def test_paper_identity_uses_verified_paperid_or_json_bytes(released_source_records):
    verified = validate_sciver_records(
        [released_source_records[0]], source_path=TESTSET_PATH
    )[0]
    fallback_source = copy.deepcopy(released_source_records[0])
    fallback_source["paperid"] = "unverified-paper-id"
    fallback = validate_sciver_records(
        [fallback_source], source_path=TESTSET_PATH
    )[0]

    assert verified.paper_identity_source == "verified_paperid"
    assert verified.paper_identity.startswith("paper_sha256:")
    assert fallback.paper_identity_source == "paper_sha256"
    assert fallback.paper_identity.startswith("paper_sha256:")
    assert verified.paper_identity == fallback.paper_identity


def test_released_v3_split_is_exact_disjoint_and_order_independent(released_records):
    first = build_experiment_split(released_records)
    second = build_experiment_split(tuple(reversed(released_records)))

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
    verify_experiment_split(first, released_records)


def test_infeasible_paper_exclusive_allocation_fails_explicitly():
    records = tuple(
        ValidatedSciVerRecord(
            sample_id=f"sample-{index}",
            paper_identity="paperid:only-paper",
            paper_identity_source="verified_paperid",
            record={},
        )
        for index in range(2000)
    )

    with pytest.raises(SplitError, match="cannot allocate exact"):
        build_experiment_split(records)
