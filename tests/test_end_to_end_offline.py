import json
import base64
import hashlib

from PIL import Image

from meta_harness.candidate_store import CandidateStore
from meta_harness.evaluator import evaluate_candidate
from meta_harness.schemas import template_source_sha256
from meta_harness.split_manager import build_split_manifest
from utils.dataset_adapters import get_dataset_adapter
from utils.result_writer import iter_result_records


METHODS = ("direct", "analytical", "parallel", "sequential")


class RecordingSolver:
    def __init__(self, sentinel):
        self.sentinel = sentinel
        self.calls = []

    def create_chat_completion(self, model, messages):
        self.calls.append((model, messages))
        assert messages[0]["content"][-1]["text"].startswith(self.sentinel)
        return "Therefore, the final answer is: Answer: yes"


def _candidate(store, candidate_id, sentinel):
    templates = {
        "direct": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption: $caption\nConclude with exactly Answer: yes or Answer: no."
        ),
        "analytical": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption: $caption\nConclude with exactly Answer: yes or Answer: no."
        ),
        "parallel": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption 1: $caption1\nCaption 2: $caption2\n"
            "Conclude with exactly Answer: yes or Answer: no."
        ),
        "sequential": (
            f"{sentinel}\nClaim: $claim\nContext: $context\n"
            "Caption 1: $caption1\nCaption 2: $caption2\n"
            "Conclude with exactly Answer: yes or Answer: no."
        ),
    }
    return store.create(
        {
            "candidate_id": candidate_id,
            "parent_id": "baseline_cot",
            "search_axis": "exploration",
            "hypothesis": "The prompt-only prefix will preserve model inputs.",
            "templates": templates,
            "expected_tradeoff": "The request text will have one extra prefix.",
            "source_sha256": template_source_sha256(templates),
        }
    )


def _dataset(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), "blue").save(first)
    Image.new("RGB", (2, 2), "yellow").save(second)
    records = []
    for paper_index in range(5):
        for method in METHODS:
            record = {
                "sample_id": f"paper-{paper_index}-{method}",
                "paper_id": f"paper-{paper_index}",
                "claim_type": method,
                "claim": f"claim-{method}",
                "context": f"context-{paper_index}",
                "gold_label": "yes",
            }
            if method in {"direct", "analytical"}:
                record.update(
                    {
                        "caption": f"caption-{method}",
                        "image_path": str(first),
                    }
                )
            else:
                record.update(
                    {
                        "caption1": f"caption-one-{method}",
                        "caption2": f"caption-two-{method}",
                        "item1_path": str(first),
                        "item2_path": str(second),
                    }
                )
            records.append(record)
    path = tmp_path / "sciver.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    samples = get_dataset_adapter("SciVer").load(path)
    manifest = build_split_manifest([sample.record for sample in samples])
    search_ids = manifest["splits"]["search"]["sample_ids"]
    paper = search_ids[0].rsplit("-", 1)[0]
    selected_ids = [f"{paper}-{method}" for method in METHODS]
    return samples, manifest, selected_ids


def _without_text(messages):
    content = []
    for block in messages[0]["content"]:
        if block["type"] == "image_url":
            encoded = block["image_url"]["url"].split(",", 1)[1]
            content.append(
                {
                    "type": "image_sha256",
                    "sha256": hashlib.sha256(
                        base64.b64decode(encoded)
                    ).hexdigest(),
                }
            )
        else:
            content.append({"type": "text", "text": "[PROMPT]"})
    return [{"role": messages[0]["role"], "content": content}]


def test_prompt_only_candidates_change_text_not_samples_or_evidence(tmp_path):
    samples, manifest, selected_ids = _dataset(tmp_path)
    store = CandidateStore(tmp_path, "run_001")
    _candidate(store, "candidate_alpha", "ALPHA")
    _candidate(store, "candidate_beta", "BETA")
    alpha = RecordingSolver("ALPHA")
    beta = RecordingSolver("BETA")

    alpha_report = evaluate_candidate(
        run_id="offline_evaluation",
        candidate_id="candidate_alpha",
        candidate_store=store,
        split_manifest=manifest,
        split_name="search",
        sample_ids=selected_ids,
        samples=samples,
        solver=alpha,
        output_path=tmp_path / "alpha.jsonl",
    )
    beta_report = evaluate_candidate(
        run_id="offline_evaluation",
        candidate_id="candidate_beta",
        candidate_store=store,
        split_manifest=manifest,
        split_name="search",
        sample_ids=selected_ids,
        samples=samples,
        solver=beta,
        output_path=tmp_path / "beta.jsonl",
    )

    assert len(alpha.calls) == len(beta.calls) == 4
    assert [_without_text(call[1]) for call in alpha.calls] == [
        _without_text(call[1]) for call in beta.calls
    ]
    assert alpha_report["prompt_sha256"] != beta_report["prompt_sha256"]
    assert alpha_report["split_sha256"] == beta_report["split_sha256"]
    assert alpha_report["config_sha256"] == beta_report["config_sha256"]
    assert alpha_report["metrics"] == beta_report["metrics"]
    assert alpha_report["rankable"] is beta_report["rankable"] is True


def test_failure_diagnostics_redact_gold_secret_and_image_bytes(
    tmp_path,
    monkeypatch,
):
    samples, manifest, selected_ids = _dataset(tmp_path)
    store = CandidateStore(tmp_path, "run_001")
    _candidate(store, "candidate_alpha", "ALPHA")
    fake_secret = "UNMISTAKABLY_FAKE_SECRET_FOR_OFFLINE_TEST"
    fake_endpoint = "https://invalid.example.test/v1/chat"
    image_blob = "A" * 256
    monkeypatch.setenv("API_KEY", fake_secret)
    monkeypatch.setenv("API_URL", fake_endpoint)

    class FailingSolver:
        def create_chat_completion(self, model, messages):
            raise RuntimeError(
                f"Authorization: Bearer {fake_secret}; "
                f"endpoint={fake_endpoint}; "
                f"data:image/png;base64,{image_blob}"
            )

    output = tmp_path / "failure.jsonl"
    report = evaluate_candidate(
        run_id="offline_failure",
        candidate_id="candidate_alpha",
        candidate_store=store,
        split_manifest=manifest,
        split_name="search",
        sample_ids=[selected_ids[0]],
        samples=samples,
        solver=FailingSolver(),
        output_path=output,
    )

    serialized = output.read_text(encoding="utf-8")
    assert fake_secret not in serialized
    assert fake_endpoint not in serialized
    assert image_blob not in serialized
    assert "data:image" not in serialized
    record = list(iter_result_records(output))[0]
    assert record["request_status"] == "api_failure"
    assert record["gold_label"] == "yes"
    assert record["raw_response"] is None
    assert report["rankable"] is False
