import base64
import json
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from model_inference.remote_api import (
    InvalidRemoteInputError,
    LiveRequestDisabledError,
    generate_remote_responses,
    generate_response,
)
from model_inference.remote_config import RemoteAPIConfig
from utils.constant import COT_PROMPT


MODEL = "gemma-4-26B-A4B-it"
GROUND_TRUTH_MARKER = "GROUND_TRUTH_MUST_NOT_LEAK"
RAW_RESPONSE = "Reasoning. Therefore, the final answer is: Answer: yes"


def _write_image(path, color):
    Image.new("RGB", (2, 2), color=color).save(path, format="PNG")
    return path


@pytest.fixture
def evidence(tmp_path):
    first_image = _write_image(tmp_path / "first.png", "red")
    second_image = _write_image(tmp_path / "second.png", "green")
    paper_path = tmp_path / "paper.json"
    paper_path.write_text(
        json.dumps(
            {
                "sections": [
                    {
                        "section_id": "1.1",
                        "section_name": "Results",
                        "text": "First context paragraph.",
                    },
                    {
                        "section_id": "2",
                        "section_name": "Discussion",
                        "text": "Second context paragraph.",
                    },
                    {
                        "section_id": "3",
                        "section_name": "Excluded",
                        "text": "This paragraph was not selected.",
                    },
                ],
                "image_paths": [
                    {"caption": "First chart caption."},
                    {"caption": "Second chart caption."},
                ],
                "tables": [
                    {"capture": "First table caption."},
                    {"capture": "Second table caption."},
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "paper_path": str(paper_path),
        "first_image": first_image,
        "second_image": second_image,
        "context": (
            "Results:\nFirst context paragraph.\n"
            "Discussion:\nSecond context paragraph.\n"
        ),
    }


def _query(evidence, method):
    common = {
        "claim_type": method.title(),
        "claim": "The claim remains byte-for-byte unchanged.",
        "paper_path": evidence["paper_path"],
        "section": ["2.4", "1.1"],
        "label": GROUND_TRUTH_MARKER,
    }
    if method in {"direct", "analytical"}:
        return {
            **common,
            "type": "chart",
            "item": 0,
            "image_path": str(evidence["first_image"]),
        }
    return {
        **common,
        "item1_type": "chart",
        "item1": 0,
        "item1_path": str(evidence["first_image"]),
        "item2_type": "table",
        "item2": 1,
        "item2_path": str(evidence["second_image"]),
    }


def _request_messages(client):
    return client.create_chat_completion.call_args.args[1]


def _request_prompt(messages):
    return messages[0]["content"][-1]["text"]


def _request_image_bytes(messages):
    image_blocks = messages[0]["content"][:-1]
    return [
        base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
        for block in image_blocks
    ]


@pytest.mark.parametrize(
    ("method", "captions", "expected_image_count"),
    [
        ("direct", {"caption": "First chart caption."}, 1),
        ("analytical", {"caption": "First chart caption."}, 1),
        (
            "parallel",
            {
                "caption1": "First chart caption.",
                "caption2": "Second table caption.",
            },
            2,
        ),
        (
            "sequential",
            {
                "caption1": "First chart caption.",
                "caption2": "Second table caption.",
            },
            2,
        ),
    ],
)
def test_reasoning_method_uses_original_prompt_and_ordered_images(
    evidence, method, captions, expected_image_count
):
    query = _query(evidence, method)
    client = Mock()
    client.create_chat_completion.return_value = RAW_RESPONSE

    results = generate_remote_responses(
        MODEL,
        [query],
        COT_PROMPT,
        client=client,
    )

    client.create_chat_completion.assert_called_once()
    called_model, messages = client.create_chat_completion.call_args.args
    assert called_model == MODEL
    assert _request_prompt(messages) == COT_PROMPT[method].substitute(
        claim=query["claim"],
        context=evidence["context"],
        **captions,
    )

    image_bytes = _request_image_bytes(messages)
    assert len(image_bytes) == expected_image_count
    expected_paths = [evidence["first_image"]]
    if expected_image_count == 2:
        expected_paths.append(evidence["second_image"])
    assert image_bytes == [path.read_bytes() for path in expected_paths]
    assert GROUND_TRUTH_MARKER not in json.dumps(messages)
    assert results == [
        {
            "raw_response": RAW_RESPONSE,
            "parsed_prediction": "yes",
        }
    ]


def test_paper_path_keeps_legacy_prompt_when_inline_fields_are_also_present(
    evidence,
):
    query = _query(evidence, "direct")
    query["context"] = "INLINE_CONTEXT_MUST_NOT_REPLACE_PAPER_CONTEXT"
    query["caption"] = "INLINE_CAPTION_MUST_NOT_REPLACE_PAPER_CAPTION"
    client = Mock()
    client.create_chat_completion.return_value = RAW_RESPONSE

    generate_remote_responses(MODEL, [query], COT_PROMPT, client=client)

    prompt = _request_prompt(_request_messages(client))
    assert prompt == COT_PROMPT["direct"].substitute(
        claim=query["claim"],
        context=evidence["context"],
        caption="First chart caption.",
    )
    assert "INLINE_CONTEXT_MUST_NOT_REPLACE_PAPER_CONTEXT" not in prompt
    assert "INLINE_CAPTION_MUST_NOT_REPLACE_PAPER_CAPTION" not in prompt


def test_pipeline_entry_point_writes_raw_and_parsed_results(evidence, tmp_path):
    query = _query(evidence, "direct")
    client = Mock()
    client.create_chat_completion.return_value = (
        "Therefore, the final answer is: Answer: no"
    )
    output_path = tmp_path / "results.json"

    results = generate_response(
        model_name=MODEL,
        prompt=COT_PROMPT,
        queries=[query],
        output_path=str(output_path),
        client=client,
    )

    assert results == [
        {
            "raw_response": "Therefore, the final answer is: Answer: no",
            "parsed_prediction": "no",
        }
    ]
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written[0]["response"] == results[0]["raw_response"]
    assert written[0]["prediction"] == results[0]["parsed_prediction"]
    assert GROUND_TRUTH_MARKER not in json.dumps(_request_messages(client))


def test_injected_client_does_not_read_runtime_configuration(evidence):
    client = Mock()
    client.create_chat_completion.return_value = RAW_RESPONSE

    with patch(
        "model_inference.remote_api.validate_config_for_live_request"
    ) as load_config:
        generate_remote_responses(
            MODEL,
            [_query(evidence, "analytical")],
            client=client,
        )

    load_config.assert_not_called()
    client.create_chat_completion.assert_called_once()


def test_live_request_requires_explicit_opt_in_before_credentials_are_read(
    evidence,
):
    with patch(
        "model_inference.remote_api.validate_config_for_live_request"
    ) as load_config:
        with pytest.raises(LiveRequestDisabledError, match="disabled"):
            generate_remote_responses(
                MODEL,
                [_query(evidence, "direct")],
            )

    load_config.assert_not_called()


def test_explicit_live_opt_in_uses_mocked_client_boundary(evidence):
    config = RemoteAPIConfig(
        api_key="obviously-fake-key",
        api_url="https://invalid.example.test/chat/completions",
        timeout_seconds=12.5,
        max_retries=0,
    )
    client = Mock()
    client.create_chat_completion.return_value = RAW_RESPONSE

    with (
        patch(
            "model_inference.remote_api.validate_config_for_live_request",
            return_value=config,
        ) as load_config,
        patch(
            "model_inference.remote_api.RemoteChatCompletionsClient",
            return_value=client,
        ) as client_factory,
    ):
        results = generate_remote_responses(
            MODEL,
            [_query(evidence, "sequential")],
            allow_live_requests=True,
        )

    load_config.assert_called_once_with()
    client_factory.assert_called_once()
    client.create_chat_completion.assert_called_once()
    assert results[0]["raw_response"] == RAW_RESPONSE


def test_empty_batch_needs_no_client_or_runtime_configuration():
    with patch(
        "model_inference.remote_api.validate_config_for_live_request"
    ) as load_config:
        assert generate_remote_responses(MODEL, []) == []

    load_config.assert_not_called()


def test_invalid_query_schema_fails_before_client_call(evidence):
    client = Mock()
    query = _query(evidence, "direct")
    del query["image_path"]

    with pytest.raises(InvalidRemoteInputError, match="image_path"):
        generate_remote_responses(MODEL, [query], client=client)

    client.create_chat_completion.assert_not_called()


def test_empty_resolved_context_fails_before_client_call(evidence):
    client = Mock()
    query = _query(evidence, "direct")
    query["section"] = ["99"]

    with pytest.raises(InvalidRemoteInputError, match="non-empty context"):
        generate_remote_responses(MODEL, [query], client=client)

    client.create_chat_completion.assert_not_called()


def test_empty_caption_fails_before_client_call(evidence):
    paper_path = Path(evidence["paper_path"])
    paper = json.loads(paper_path.read_text(encoding="utf-8"))
    paper["image_paths"][0]["caption"] = "  "
    paper_path.write_text(json.dumps(paper), encoding="utf-8")
    client = Mock()

    with pytest.raises(InvalidRemoteInputError, match="non-empty text"):
        generate_remote_responses(MODEL, [_query(evidence, "direct")], client=client)

    client.create_chat_completion.assert_not_called()


def test_pipeline_entry_point_rejects_unsupported_sample_count(evidence, tmp_path):
    client = Mock()

    with pytest.raises(ValueError, match="n=1"):
        generate_response(
            MODEL,
            COT_PROMPT,
            [_query(evidence, "direct")],
            str(tmp_path / "unused.json"),
            n=2,
            client=client,
        )

    client.create_chat_completion.assert_not_called()
