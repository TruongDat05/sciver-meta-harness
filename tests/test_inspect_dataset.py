import csv
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY_ROOT / "scripts" / "inspect_dataset.py"
SPEC = importlib.util.spec_from_file_location("inspect_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INSPECT_DATASET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECT_DATASET)
InspectionError = INSPECT_DATASET.InspectionError
inspect_dataset = INSPECT_DATASET.inspect_dataset
write_report = INSPECT_DATASET.write_report


def _columns(section):
    return {column["name"]: column for column in section["columns"]}


def _candidate_fields(possible_fields, role):
    return {candidate["field"] for candidate in possible_fields[role]}


def test_cli_inspects_json_and_resolves_images_without_modifying_dataset(tmp_path):
    dataset_path = tmp_path / "input" / "synthetic"
    dataset_path.mkdir(parents=True)
    image_path = dataset_path / "images" / "figure.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"synthetic-image-bytes")
    metadata_path = dataset_path / "records.json"
    metadata_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": 1,
                    "claim_text": "A short synthetic claim.",
                    "context": "Synthetic context.",
                    "figure_caption": "A synthetic caption.",
                    "image_paths": ["images/figure.png"],
                    "gold_label": "supported",
                },
                {
                    "sample_id": 2,
                    "claim_text": "Another claim.",
                    "context": None,
                    "figure_caption": "Another caption.",
                    "image_paths": ["images/missing.png", "https://example.test/x.png"],
                    "gold_label": "not_supported",
                },
            ]
        ),
        encoding="utf-8",
    )
    before = {path.relative_to(dataset_path): path.read_bytes() for path in dataset_path.rglob("*") if path.is_file()}
    output_path = tmp_path / "working" / "schema-report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            "Synthetic",
            "--dataset-path",
            str(dataset_path),
            "--output",
            str(output_path),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["dataset"] == "Synthetic"
    assert report["discovered_metadata_files"] == ["records.json"]
    assert report["file_formats"] == ["json"]
    assert report["mapping_status"] == "not_selected"
    assert report["image_files"] == {
        "count": 1,
        "formats": {"png": 1},
        "examples": ["images/figure.png"],
        "images_were_opened": False,
    }
    section = report["metadata_files"][0]["sections"][0]
    assert section["row_count"] == 2
    columns = _columns(section)
    assert columns["sample_id"]["types"] == ["integer"]
    assert columns["context"]["types"] == ["null", "string"]
    assert _candidate_fields(section["possible_fields"], "claim") == {"claim_text"}
    assert _candidate_fields(section["possible_fields"], "label") == {"gold_label"}
    assert section["image_statistics"]["possible_image_count_per_sample"] == {
        "minimum": 1,
        "maximum": 2,
        "distribution": {"1": 1, "2": 1},
    }
    assert section["image_statistics"]["path_resolution"] == {
        "references_checked": 3,
        "resolved": 1,
        "missing": 1,
        "remote_reference_not_accessed": 1,
        "embedded_data_omitted": 0,
        "invalid_reference": 0,
    }
    after = {path.relative_to(dataset_path): path.read_bytes() for path in dataset_path.rglob("*") if path.is_file()}
    assert after == before


def test_reports_every_plausible_metadata_file_and_format(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "records.jsonl").write_text(
        '{"id": 1, "statement": "first"}\n'
        '{"id": 2, "statement": "second"}\n',
        encoding="utf-8",
    )
    with (dataset_path / "alternate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "evidence", "label"])
        writer.writeheader()
        writer.writerow({"uid": "a", "evidence": "one", "label": "yes"})
    (dataset_path / "notes.txt").write_text("not metadata", encoding="utf-8")

    report = inspect_dataset("Synthetic", dataset_path)

    assert report["discovered_metadata_files"] == ["alternate.csv", "records.jsonl"]
    assert report["file_formats"] == ["csv", "jsonl"]
    reports = {item["path"]: item for item in report["metadata_files"]}
    assert reports["alternate.csv"]["sections"][0]["row_count"] == 1
    assert reports["records.jsonl"]["sections"][0]["row_count"] == 2
    aggregate_claims = {
        (candidate["metadata_file"], candidate["field"])
        for candidate in report["possible_fields"]["claim"]
    }
    assert aggregate_claims == {("records.jsonl", "statement")}


def test_json_object_reports_all_plausible_record_collections(tmp_path):
    metadata_path = tmp_path / "collections.json"
    metadata_path.write_text(
        json.dumps(
            {
                "train": [{"claim": "train claim", "image": "train.png"}],
                "validation": [
                    {"claim": "validation claim one"},
                    {"claim": "validation claim two"},
                ],
                "description": "synthetic fixture",
            }
        ),
        encoding="utf-8",
    )

    report = inspect_dataset("Synthetic", metadata_path)

    metadata_report = report["metadata_files"][0]
    assert metadata_report["top_level_keys"] == ["description", "train", "validation"]
    sections = {section["location"]: section for section in metadata_report["sections"]}
    assert set(sections) == {"$", "$.train", "$.validation"}
    assert sections["$.train"]["row_count"] == 1
    assert sections["$.validation"]["row_count"] == 2
    assert report["mapping_status"] == "not_selected"


def test_examples_are_truncated_and_base64_content_is_omitted(tmp_path):
    encoded = "A" * 128
    long_context = "synthetic context with spaces " * 25
    metadata_path = tmp_path / "records.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "long_context": long_context,
                "image": f"data:image/png;base64,{encoded}",
                "encoded_value": encoded,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_dataset("Synthetic", metadata_path)
    serialized = json.dumps(report)
    columns = _columns(report["metadata_files"][0]["sections"][0])

    assert columns["image"]["examples"] == ["<base64 data omitted>"]
    assert columns["encoded_value"]["examples"] == ["<possible base64 data omitted>"]
    assert len(columns["long_context"]["examples"][0]) == 160
    assert columns["long_context"]["examples"][0].endswith("…")
    assert encoded not in serialized


def test_parquet_uses_footer_row_count_and_bounded_examples(tmp_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    metadata_path = tmp_path / "records.parquet"
    parquet.write_table(
        pyarrow.table(
            {
                "claim_id": list(range(25)),
                "claim": [f"claim {index}" for index in range(25)],
                "image_path": [f"missing-{index}.png" for index in range(25)],
            }
        ),
        metadata_path,
    )

    report = inspect_dataset("Synthetic", metadata_path)

    section = report["metadata_files"][0]["sections"][0]
    assert section["row_count"] == 25
    assert section["image_statistics"]["scope"] == "first_20_rows"
    assert section["image_statistics"]["samples_observed"] == 20
    assert _columns(section)["claim_id"]["types"] == ["integer"]
    assert len(_columns(section)["claim"]["examples"]) == 3


def test_invalid_json_is_reported_without_hiding_other_metadata(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "broken.json").write_text("{invalid", encoding="utf-8")
    (dataset_path / "valid.csv").write_text("id,claim\n1,test\n", encoding="utf-8")

    report = inspect_dataset("Synthetic", dataset_path)

    reports = {item["path"]: item for item in report["metadata_files"]}
    assert "inspection_error" in reports["broken.json"]
    assert reports["broken.json"]["sections"] == []
    assert reports["valid.csv"]["sections"][0]["row_count"] == 1


def test_remote_image_reference_never_opens_a_network_connection(tmp_path, monkeypatch):
    metadata_path = tmp_path / "records.json"
    metadata_path.write_text(
        '[{"image_url": "https://example.test/synthetic.png"}]',
        encoding="utf-8",
    )

    def reject_network(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    report = inspect_dataset("Synthetic", metadata_path)

    resolution = report["metadata_files"][0]["sections"][0][
        "image_statistics"
    ]["path_resolution"]
    assert resolution["remote_reference_not_accessed"] == 1
    assert resolution["references_checked"] == 1


def test_output_inside_dataset_is_rejected(tmp_path):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    metadata_path = dataset_path / "records.json"
    metadata_path.write_text("[]", encoding="utf-8")
    report = inspect_dataset("Synthetic", dataset_path)

    with pytest.raises(InspectionError, match="outside"):
        write_report(report, dataset_path / "schema-report.json", dataset_path)

    assert list(dataset_path.iterdir()) == [metadata_path]
