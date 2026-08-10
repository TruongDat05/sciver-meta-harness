import json

from PIL import Image

from scripts.normalize_dataset import cli


def test_normalizer_writes_leakage_safe_unified_json(tmp_path, capsys):
    image_path = tmp_path / "evidence.png"
    Image.new("RGB", (4, 4), color="green").save(image_path, format="PNG")
    marker = "GOLD_OPERATION_MUST_NOT_BE_COPIED"
    source = tmp_path / "task1.json"
    source.write_text(
        json.dumps(
            [
                {
                    "claim_id": "one",
                    "claim": "Claim.",
                    "label": "Refuted",
                    "caption": "Caption.",
                    "context": "Context.",
                    "evi_path": str(image_path),
                    "operation": marker,
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "normalized" / "records.json"

    assert cli(
        [
            "--dataset",
            "SciClaimEval",
            "--dataset-path",
            str(source),
            "--output",
            str(output),
        ]
    ) == 0

    records = json.loads(output.read_text(encoding="utf-8"))
    assert records == [
        {
            "sample_id": "SciClaimEval:task1:one:0",
            "claim_type": "analytical",
            "claim": "Claim.",
            "context": "Context.",
            "caption": "Caption.",
            "image_path": str(image_path.resolve()),
            "gold_label": "no",
        }
    ]
    assert marker not in output.read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["normalized_samples"] == 1
