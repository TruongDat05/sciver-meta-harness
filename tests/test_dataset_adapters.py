import json

from PIL import Image
import pytest

from model_inference.remote_api import prepare_remote_requests
from utils.constant import COT_PROMPT
from utils.dataset_adapters import DatasetAdapterError, get_dataset_adapter


def _image(path, color="blue"):
    Image.new("RGB", (8, 8), color=color).save(path, format="PNG")
    return path


def _model_prompt(record):
    messages = prepare_remote_requests([record], COT_PROMPT)[0]
    return messages[-1]["content"][-1]["text"]


def test_sciver_released_schema_preserves_legacy_pointers_without_gold_explanations(
    tmp_path,
):
    image_path = _image(tmp_path / "figure.png")
    paper_path = tmp_path / "paper.json"
    paper_path.write_text(
        json.dumps(
            {
                "sections": [
                    {"section_id": "2.1", "section_name": "Results", "text": "Local context."}
                ],
                "image_paths": [{"caption": "Verified caption."}],
                "tables": [],
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "testset.json"
    marker = "GOLD_EXPLANATION_MUST_NOT_SURVIVE"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "request_id": 7,
                    "claim": "A scientific claim.",
                    "claim_type": "direct",
                    "paper_path": str(paper_path),
                    "section": ["2.1"],
                    "type": "chart",
                    "item": 0,
                    "image_path": str(image_path),
                    "label": True,
                    "perturbed_explanation": marker,
                }
            ]
        ),
        encoding="utf-8",
    )

    sample = get_dataset_adapter("SciVer").load(dataset_path)[0]

    assert sample.sample_id == "SciVer:test:7:0"
    assert sample.record == {
        "sample_id": "SciVer:test:7:0",
        "claim_type": "direct",
        "claim": "A scientific claim.",
        "paper_path": str(paper_path.resolve()),
        "section": ["2.1"],
        "type": "chart",
        "item": 0,
        "image_path": str(image_path.resolve()),
        "gold_label": "yes",
    }
    assert marker not in json.dumps(sample.record)
    assert marker not in _model_prompt(sample.record)


def test_sciver_two_evidence_order_is_preserved_in_legacy_schema(tmp_path):
    first = _image(tmp_path / "first.png", "red")
    second = _image(tmp_path / "second.png", "green")
    paper_path = tmp_path / "paper.json"
    paper_path.write_text(
        json.dumps(
            {
                "sections": [
                    {"section_id": "1", "section_name": "Results", "text": "Context."}
                ],
                "image_paths": {
                    "A": {"caption": "First caption."},
                    "B": {"caption": "Second caption."},
                },
                "tables": {},
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "testset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "request_id": 8,
                    "claim": "A two-image claim.",
                    "claim_type": "sequential",
                    "paper_path": str(paper_path),
                    "section": ["1"],
                    "item1_type": "chart",
                    "item1": "A",
                    "item1_path": str(first),
                    "item2_type": "chart",
                    "item2": "B",
                    "item2_path": str(second),
                    "label": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    record = get_dataset_adapter("SciVer").load(dataset_path)[0].record

    assert record["paper_path"] == str(paper_path.resolve())
    assert record["item1_type"] == "chart"
    assert record["item1"] == "A"
    assert record["item2_type"] == "chart"
    assert record["item2"] == "B"
    assert record["item1_path"] == str(first.resolve())
    assert record["item2_path"] == str(second.resolve())
    messages = prepare_remote_requests([record], COT_PROMPT)[0]
    image_blocks = messages[-1]["content"][:-1]
    assert len(image_blocks) == 2
    assert "item1 Caption: First caption." in messages[-1]["content"][-1]["text"]
    assert "item2 Caption: Second caption." in messages[-1]["content"][-1]["text"]


@pytest.mark.parametrize(
    ("label", "expected"),
    [("support", "yes"), ("refute", "no")],
)
def test_sciatomic_schema_maps_labels_and_renders_readable_png(
    tmp_path, label, expected
):
    dataset_path = tmp_path / "mat.json"
    rationale_marker = "ATOMIC_RATIONALE_MUST_NOT_SURVIVE"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "id": f"sample-{label}",
                    "claim": "The second value is larger.",
                    "label": label,
                    "table_caption": "Table 1: Synthetic measurements.",
                    "table_content": "|| Method | Value ||\n|| A | 1.0 ||\n|| B | 2.0 ||",
                    "reasoning": rationale_marker,
                }
            ]
        ),
        encoding="utf-8",
    )
    evidence_dir = tmp_path / "rendered"

    sample = get_dataset_adapter("SciAtomicBench").load(
        dataset_path, evidence_dir=evidence_dir
    )[0]

    assert sample.sample_id == f"SciAtomicBench:mat:sample-{label}:0"
    assert sample.record["gold_label"] == expected
    assert sample.record["claim_type"] == "analytical"
    assert sample.record["context"] == "No additional context is provided."
    rendered_path = evidence_dir / next(evidence_dir.iterdir()).name
    assert sample.record["image_path"] == str(rendered_path.resolve())
    with Image.open(rendered_path) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width >= 240
        assert rendered.height >= 100
    assert rationale_marker not in json.dumps(sample.record)
    prompt = _model_prompt(sample.record)
    assert "Table 1: Synthetic measurements." in prompt
    assert rationale_marker not in prompt


def test_sciatomic_directory_requires_every_released_domain_file(tmp_path):
    (tmp_path / "fin.json").write_text("[]", encoding="utf-8")
    (tmp_path / "mat.json").write_text("[]", encoding="utf-8")
    (tmp_path / "med.json").write_text("[]", encoding="utf-8")

    with pytest.raises(
        DatasetAdapterError,
        match=r"missing required domain files: ml\.json",
    ):
        get_dataset_adapter("SciAtomicBench").load(tmp_path)


@pytest.mark.parametrize(
    ("label", "expected"),
    [("SUPPORT", "yes"), ("NON_SUPPORT", "no")],
)
def test_musciclaims_verified_schema_uses_local_figure_and_label_2class(
    tmp_path, label, expected
):
    figures = tmp_path / "paper_figures"
    figures.mkdir()
    image_path = _image(figures / "figure.png")
    dataset_path = tmp_path / "test_set.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "base_claim_id": "base",
                "claim_id": f"claim-{label}",
                "claim_text": "A figure-grounded claim.",
                "label_3class": "CONTRADICT",
                "label_2class": label,
                "associated_figure_filepath": "paper_figures/figure.png",
                "caption": "Figure 1: Local evidence.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = get_dataset_adapter("MuSciClaims").load(dataset_path)[0]

    assert sample.sample_id == f"MuSciClaims:test:claim-{label}:0"
    assert sample.record["gold_label"] == expected
    assert sample.record["image_path"] == str(image_path.resolve())
    assert "label_3class" not in sample.record


@pytest.mark.parametrize(
    ("label", "expected"),
    [("Supported", "yes"), ("Refuted", "no")],
)
def test_sciclaimeval_task1_schema_maps_only_supported_and_refuted(
    tmp_path, label, expected
):
    image_path = _image(tmp_path / "evidence.png")
    dataset_path = tmp_path / "dev_task1_release.json"
    marker = "OPERATION_MUST_NOT_SURVIVE"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "claim_id": f"claim-{label}",
                    "claim": "A Task 1 claim.",
                    "label": label,
                    "caption": "Evidence caption.",
                    "context": "Preceding paragraph.",
                    "evi_path": "evidence.png",
                    "evi_type": "figure",
                    "operation": marker,
                }
            ]
        ),
        encoding="utf-8",
    )

    sample = get_dataset_adapter("SciClaimEval").load(dataset_path)[0]

    assert sample.sample_id == f"SciClaimEval:dev:claim-{label}:0"
    assert sample.record["gold_label"] == expected
    assert sample.record["image_path"] == str(image_path.resolve())
    assert marker not in json.dumps(sample.record)
    assert marker not in _model_prompt(sample.record)


def test_raw_schema_errors_do_not_guess_fields_or_labels(tmp_path):
    image_path = _image(tmp_path / "figure.png")
    dataset_path = tmp_path / "test_set.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "claim_id": "claim",
                "claim": "Wrong field name.",
                "label_2class": "MAYBE",
                "associated_figure_filepath": str(image_path),
                "caption": "Caption.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DatasetAdapterError, match="claim_text"):
        get_dataset_adapter("MuSciClaims").load(dataset_path)


def test_duplicate_raw_ids_get_collision_safe_split_and_row_ids(tmp_path):
    image_path = _image(tmp_path / "evidence.png")
    record = {
        "claim_id": "duplicate",
        "claim": "Claim.",
        "label": "Supported",
        "caption": "Caption.",
        "context": "Context.",
        "evi_path": str(image_path),
    }
    dataset_path = tmp_path / "task1.json"
    dataset_path.write_text(json.dumps([record, record]), encoding="utf-8")

    samples = get_dataset_adapter("SciClaimEval").load(dataset_path)

    assert [sample.sample_id for sample in samples] == [
        "SciClaimEval:task1:duplicate:0",
        "SciClaimEval:task1:duplicate:1",
    ]


def test_sciclaimeval_task1_test_records_may_be_unlabeled(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    image_path = _image(data_dir / "test-evidence.png")
    dataset_path = data_dir / "test_task1_release.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "claim_id": "test-claim",
                    "claim": "An unlabeled formal-run claim.",
                    "caption": "Test evidence caption.",
                    "context": None,
                    "evi_path": "test-evidence.png",
                }
            ]
        ),
        encoding="utf-8",
    )

    sample = get_dataset_adapter("SciClaimEval").load(tmp_path)[0]

    assert sample.sample_id == "SciClaimEval:test:test-claim:0"
    assert sample.record["image_path"] == str(image_path.resolve())
    assert sample.record["context"] == "No additional context is provided."
    assert "gold_label" not in sample.record
