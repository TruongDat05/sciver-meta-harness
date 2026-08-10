import json
from pathlib import Path

from PIL import Image
import pytest

from evaluation.metrics import EvaluationError, evaluate_records
from utils.benchmark_workflow import (
    BenchmarkWorkflowError,
    build_run_manifest,
    ensure_run_manifest,
    preflight_dataset,
    require_adapter_ready,
)


def _dataset(tmp_path, *, context="Local context.", caption="Local caption.", label="yes"):
    image_path = tmp_path / "figure.png"
    Image.new("RGB", (2, 2), color="blue").save(image_path, format="PNG")
    paper_path = tmp_path / "paper.json"
    paper_path.write_text(
        json.dumps(
            {
                "sections": [
                    {
                        "section_id": "1",
                        "section_name": "Results",
                        "text": context,
                    }
                ],
                "image_paths": [{"caption": caption}],
                "tables": [],
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "sample_id": "sample-1",
            "split": "test",
            "claim_type": "direct",
            "claim": "Synthetic claim.",
            "paper_path": str(paper_path),
            "section": ["1"],
            "type": "chart",
            "item": 0,
            "image_path": str(image_path),
            "gold_label": label,
        }
    ]
    dataset_path = tmp_path / "records.json"
    dataset_path.write_text(json.dumps(records), encoding="utf-8")
    return dataset_path


def test_preflight_uses_native_request_preparation_without_credentials_or_network(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    dataset_path = _dataset(tmp_path)

    summary = preflight_dataset("SciVer", dataset_path, selected_sample_count=1)

    assert summary == {
        "dataset": "SciVer",
        "split": {'"test"': 1},
        "total_samples": 1,
        "selected_samples": 1,
        "label_distribution": {'"yes"': 1},
        "label_support": "binary_evaluation_ready",
        "reasoning_or_claim_type_distribution": {'"direct"': 1},
        "missing_context_count": 0,
        "missing_image_count": 0,
        "adapter_readiness": "ready",
        "invalid_inputs": 0,
        "adapter": "normalized_json",
    }
    require_adapter_ready(summary)


@pytest.mark.parametrize(
    ("context", "caption", "expected_missing_context"),
    [("", "Local caption.", 1), ("Local context.", "", 0)],
)
def test_preflight_rejects_missing_required_context_or_caption(
    tmp_path, context, caption, expected_missing_context
):
    dataset_path = _dataset(tmp_path, context=context, caption=caption)
    summary = preflight_dataset("SciVer", dataset_path, selected_sample_count=1)

    assert summary["adapter_readiness"].startswith("not_ready")
    assert summary["missing_context_count"] == expected_missing_context
    with pytest.raises(BenchmarkWorkflowError, match="native adapter"):
        require_adapter_ready(summary)


def test_manifest_is_credential_free_and_rejects_configuration_changes(tmp_path):
    dataset_path = _dataset(tmp_path)
    summary = preflight_dataset("SciVer", dataset_path, selected_sample_count=1)
    manifest = build_run_manifest(
        dataset_name="SciVer",
        dataset_path=dataset_path,
        summary=summary,
        model_name="Qwen2.5-VL-7B-Instruct",
        method="cot",
        experiment_id="test",
        request_delay=0.0,
        git_sha="0" * 40,
    )
    manifest_path = tmp_path / "results" / "run_manifest.json"

    assert ensure_run_manifest(manifest_path, manifest) == "created"
    assert ensure_run_manifest(manifest_path, manifest) == "matched"
    serialized = manifest_path.read_text(encoding="utf-8")
    assert "API_KEY" not in serialized
    assert "API_URL" not in serialized
    assert str(dataset_path.parent) not in serialized

    changed = {**manifest, "configuration_fingerprint": "f" * 64}
    with pytest.raises(BenchmarkWorkflowError, match="Resume rejected"):
        ensure_run_manifest(manifest_path, changed)


def _result_record(gold_label):
    return {
        "run_id": "run",
        "dataset": "SciClaimEval",
        "model": "Qwen2.5-VL-7B-Instruct",
        "method": "direct",
        "sample_id": "sample",
        "prediction": "yes",
        "parse_status": "parsed",
        "request_status": "success",
        "attempt_count": 1,
        "gold_label": gold_label,
    }


def test_non_binary_native_labels_are_preserved_and_explicitly_unscored():
    record = _result_record("insufficient_evidence")
    with pytest.raises(EvaluationError, match="gold_label"):
        evaluate_records([record])

    summary = evaluate_records([record], allow_unscored_gold_labels=True)[0]

    assert record["gold_label"] == "insufficient_evidence"
    assert summary["successful_requests"] == 1
    assert summary["parse_coverage"]["value"] == 1.0
    assert summary["accuracy_all_labeled"]["value"] is None

