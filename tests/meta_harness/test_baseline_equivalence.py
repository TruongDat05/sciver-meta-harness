import base64
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image
import pytest

import main as cli_module
from meta_harness.prompt_family import (
    PromptFamily,
    canonical_baseline_sources,
)
from model_inference import remote_api
from model_inference.remote_api import (
    generate_remote_responses,
    prepare_remote_requests,
)
from utils.answer_parser import parse_answer
from utils.constant import COT_PROMPT


MODEL = "gemma-4-26B-A4B-it"
REASONING_CASES = (
    ("direct", {"caption": "Primary caption."}, 1),
    ("analytical", {"caption": "Primary caption."}, 1),
    (
        "parallel",
        {"caption1": "Primary caption.", "caption2": "Secondary caption."},
        2,
    ),
    (
        "sequential",
        {"caption1": "Primary caption.", "caption2": "Secondary caption."},
        2,
    ),
)


@pytest.fixture
def image_paths(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (2, 2), color="red").save(first, format="PNG")
    Image.new("RGB", (2, 2), color="green").save(second, format="PNG")
    return first, second


def _record(method, image_paths):
    first, second = image_paths
    common = {
        "claim_type": method,
        "claim": "The scientific claim is preserved.",
        "context": "The selected scientific context is preserved.",
        "gold_label": "GROUND_TRUTH_MUST_NOT_BE_MODEL_VISIBLE",
    }
    if method in {"direct", "analytical"}:
        return {
            **common,
            "caption": "Primary caption.",
            "image_path": str(first),
        }
    return {
        **common,
        "caption1": "Primary caption.",
        "caption2": "Secondary caption.",
        "item1_path": str(first),
        "item2_path": str(second),
    }


def _called_messages(client):
    return client.create_chat_completion.call_args.args[1]


def _content(messages):
    return messages[0]["content"]


def _prompt_text(messages):
    return _content(messages)[-1]["text"]


def _image_bytes(messages):
    return [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in _content(messages)
        if block["type"] == "image_url"
    ]


def _synthetic_family():
    return PromptFamily(
        {
            method: "SYNTHETIC PROMPT FAMILY\n" + template.template
            for method, template in COT_PROMPT.items()
        }
    )


@pytest.mark.parametrize(
    ("method", "captions", "_expected_image_count"), REASONING_CASES
)
def test_baseline_raw_and_rendered_templates_match_canonical_cot(
    method,
    captions,
    _expected_image_count,
):
    snapshot = PromptFamily(canonical_baseline_sources())
    values = {
        "claim": "Exact claim\nwith preserved punctuation.",
        "context": "Exact context: α, β, and $literal.",
        **captions,
    }

    assert snapshot[method].template == COT_PROMPT[method].template
    assert snapshot[method].substitute(**values) == COT_PROMPT[
        method
    ].substitute(**values)
@pytest.mark.parametrize(
    ("method", "captions", "expected_image_count"), REASONING_CASES
)
def test_cot_model_visible_request_is_exactly_baseline_equivalent(
    image_paths, method, captions, expected_image_count
):
    record = _record(method, image_paths)
    snapshot_family = PromptFamily(canonical_baseline_sources())
    selected_family = COT_PROMPT
    raw_response = "Reasoning. Therefore, the final answer is: Answer: yes"
    baseline_client = Mock()
    selected_client = Mock()
    baseline_client.create_chat_completion.return_value = raw_response
    selected_client.create_chat_completion.return_value = raw_response

    baseline_result = generate_remote_responses(
        MODEL,
        [record],
        snapshot_family,
        client=baseline_client,
    )
    selected_result = generate_remote_responses(
        MODEL,
        [record],
        selected_family,
        client=selected_client,
    )

    baseline_messages = _called_messages(baseline_client)
    selected_messages = _called_messages(selected_client)
    baseline_client.create_chat_completion.assert_called_once()
    selected_client.create_chat_completion.assert_called_once()
    assert baseline_client.create_chat_completion.call_args.args[0] == MODEL
    assert selected_client.create_chat_completion.call_args.args[0] == MODEL
    assert selected_messages == baseline_messages
    assert _prompt_text(selected_messages) == COT_PROMPT[method].substitute(
        claim=record["claim"],
        context=record["context"],
        **captions,
    )
    for preserved_text in (record["claim"], record["context"], *captions.values()):
        assert preserved_text in _prompt_text(selected_messages)

    expected_paths = list(image_paths[:expected_image_count])
    assert len(_image_bytes(selected_messages)) == expected_image_count
    assert _image_bytes(selected_messages) == [
        path.read_bytes() for path in expected_paths
    ]
    assert baseline_result == selected_result
    assert selected_result[0]["parsed_prediction"] == parse_answer(raw_response)[
        "prediction"
    ]


@pytest.mark.parametrize(("method", "_captions", "_count"), REASONING_CASES)
def test_alternative_family_changes_only_text_block(
    image_paths, method, _captions, _count
):
    record = _record(method, image_paths)
    original_record = dict(record)

    baseline = prepare_remote_requests([record], COT_PROMPT)[0]
    alternative = prepare_remote_requests([record], _synthetic_family())[0]

    baseline_content = _content(baseline)
    alternative_content = _content(alternative)
    assert alternative_content[:-1] == baseline_content[:-1]
    assert alternative_content[-1]["type"] == baseline_content[-1]["type"] == "text"
    assert alternative_content[-1]["text"] == (
        "SYNTHETIC PROMPT FAMILY\n" + baseline_content[-1]["text"]
    )
    assert record == original_record


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_remote_cli_paths_use_the_once_resolved_prompt_family(
    tmp_path, image_paths, monkeypatch, dry_run
):
    alternative = _synthetic_family()
    monkeypatch.setitem(cli_module.PROMPT_DICT, "synthetic", alternative)

    sample = SimpleNamespace(
        sample_id="sample-1",
        record=_record("direct", image_paths),
    )
    adapter = SimpleNamespace(
        adapter_name="synthetic-adapter",
        load=Mock(return_value=[sample]),
    )
    import utils.dataset_adapters as dataset_adapters

    monkeypatch.setattr(
        dataset_adapters,
        "get_dataset_adapter",
        Mock(return_value=adapter),
    )

    prepared_families = []
    original_prepare = remote_api.prepare_remote_requests

    def recording_prepare(queries, prompt_family):
        prepared_families.append(prompt_family)
        return original_prepare(queries, prompt_family)

    monkeypatch.setattr(remote_api, "prepare_remote_requests", recording_prepare)

    client = Mock()
    client.create_chat_completion.return_value = (
        "Therefore, the final answer is: Answer: yes"
    )
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(cli_module, "_create_remote_client", client_factory)

    arguments = SimpleNamespace(
        dataset="SciVer",
        prompt="synthetic",
        method="cot",
        n=1,
        model=MODEL,
        dry_run=dry_run,
        live_api=not dry_run,
        output_dir=str(tmp_path / ("dry" if dry_run else "live")),
        data_path=str(tmp_path / "unused.json"),
        max_num=-1,
        resume=False,
        request_delay=0.0,
    )

    assert cli_module._run_remote(arguments) == 0

    assert prepared_families == [alternative]
    if dry_run:
        client_factory.assert_not_called()
    else:
        client_factory.assert_called_once_with()
        assert _prompt_text(_called_messages(client)).startswith(
            "SYNTHETIC PROMPT FAMILY\n"
        )
