from __future__ import annotations

import copy

import pytest

from meta_harness.config import MetaHarnessConfig, SearchProtocol
from meta_harness.hard_search import (
    HardSearchError,
    build_hard_search_manifest,
    materialize_hard_search_records,
)
from meta_harness.split_manager import build_split_manifest


def _fixture():
    config = MetaHarnessConfig(
        search_protocol=SearchProtocol(
            min_iterations=1,
            target_iterations=1,
            max_iterations=1,
            promotion_top_k=1,
            search_examples=50,
        ),
        search_protocol_explicit=True,
    )
    records = [
        {
            "sample_id": f"sample-{index:03d}",
            "paper_id": f"paper-{index:03d}",
            "gold_label": "yes" if index % 2 else "no",
            "type": ("direct", "analytical", "parallel", "sequential")[
                index % 4
            ],
            "claim": f"claim {index}",
            "section": f"evidence {index}",
            "image_path": f"images/{index}.jpg" if index % 3 == 0 else None,
        }
        for index in range(300)
    ]
    split = build_split_manifest(records, config)
    search_ids = set(split["splits"]["search"]["sample_ids"])
    search_records = [
        record for record in records if record["sample_id"] in search_ids
    ]
    baseline = [
        {
            "sample_id": record["sample_id"],
            "gold_label": record["gold_label"],
            "prediction": (
                record["gold_label"]
                if index % 5 == 0
                else ("no" if record["gold_label"] == "yes" else "yes")
            ),
            "request_status": "success",
            "reasoning_method": record["type"],
        }
        for index, record in enumerate(search_records)
    ]
    return config, records, search_records, baseline, split


def test_hard_search_manifest_is_deterministic_diverse_and_paper_disjoint():
    _, _, records, baseline, split = _fixture()

    first = build_hard_search_manifest(
        records,
        baseline,
        split,
        target_size=50,
    )
    second = build_hard_search_manifest(
        list(reversed(records)),
        list(reversed(baseline)),
        split,
        target_size=50,
    )

    assert first == second
    assert first["baseline_error_count"] == 38
    assert first["guard_count"] == 12
    assert first["diversity"]["labels"] == ["no", "yes"]
    assert set(first["diversity"]["reasoning_methods"]) == {
        "direct",
        "analytical",
        "parallel",
        "sequential",
    }
    selected_papers = {
        item["paper_group_id"] for item in first["items"]
    }
    assert selected_papers.isdisjoint(
        split["splits"]["validation"]["paper_group_ids"]
    )
    assert selected_papers.isdisjoint(
        split["splits"]["final_test"]["paper_group_ids"]
    )
    selected = materialize_hard_search_records(records, first)
    assert len(selected) == 50


def test_hard_search_refuses_protected_records_and_tampered_sources():
    _, all_records, search_records, baseline, split = _fixture()
    protected_id = split["splits"]["validation"]["sample_ids"][0]
    protected = next(
        record for record in all_records if record["sample_id"] == protected_id
    )
    contaminated = copy.deepcopy(search_records)
    contaminated[0] = protected

    with pytest.raises(HardSearchError, match="reserved search"):
        build_hard_search_manifest(
            contaminated,
            baseline,
            split,
            target_size=50,
        )
