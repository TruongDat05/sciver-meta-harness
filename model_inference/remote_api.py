"""Inference adapter for the provider-neutral remote API modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from string import Template
from typing import Any, Protocol, TypedDict

from model_inference.remote_client import (
    RemoteChatCompletionsClient,
    RetrySettings,
)
from model_inference.remote_config import (
    validate_config_for_live_request,
    validate_model_identifier,
)
from utils.answer_parser import Prediction, parse_answer
from utils.constant import COT_PROMPT
from utils.remote_input_processing import build_remote_messages


_SINGLE_IMAGE_METHODS = frozenset({"direct", "analytical"})
_MULTI_IMAGE_METHODS = frozenset({"parallel", "sequential"})
_REASONING_METHODS = _SINGLE_IMAGE_METHODS | _MULTI_IMAGE_METHODS


class LiveRequestDisabledError(RuntimeError):
    """Raised when creation of a live remote client was not opted into."""


class InvalidRemoteInputError(ValueError):
    """Raised when a query or paper does not match the SciVer input schema."""


class RemoteInferenceResult(TypedDict):
    """Raw model output and its normalized scientific-claim prediction."""

    raw_response: str
    parsed_prediction: Prediction


class RemoteClient(Protocol):
    """Small injectable boundary used by the inference adapter."""

    def create_chat_completion(
        self, model: str, messages: Sequence[Mapping[str, Any]]
    ) -> str: ...


def prepare_remote_requests(
    queries: Sequence[Mapping[str, Any]],
    prompt: Mapping[str, Template] = COT_PROMPT,
) -> list[list[dict[str, object]]]:
    """Build remote messages with the pipeline's prompts and evidence order."""

    requests = []
    for index, query in enumerate(queries):
        if not isinstance(query, Mapping):
            raise InvalidRemoteInputError(
                f"Query at index {index} must be a mapping."
            )

        method = _reasoning_method(query, index)
        if _uses_inline_evidence(query):
            context = _required_string(query, "context", index)
            user_prompt = _prepare_inline_prompt(prompt, method, query, context, index)
        else:
            paper = _read_paper(query, index)
            context = _prepare_context(paper, query, index)
            user_prompt = _prepare_prompt(prompt, method, paper, query, context, index)
        image_paths = _image_paths(query, method, index)
        requests.append(build_remote_messages(user_prompt, image_paths))
    return requests


def generate_remote_responses(
    model_name: str,
    queries: Sequence[Mapping[str, Any]],
    prompt: Mapping[str, Template] = COT_PROMPT,
    *,
    client: RemoteClient | None = None,
    allow_live_requests: bool = False,
) -> list[RemoteInferenceResult]:
    """Run remote inference without mutating the supplied query records.

    Supplying ``client`` is the offline test boundary and never reads runtime
    credentials. Without an injected client, callers must explicitly opt into
    live requests; configuration is loaded only after all local inputs have
    been successfully prepared.
    """

    model_identifier = validate_model_identifier(model_name)
    requests = prepare_remote_requests(queries, prompt)
    if not requests:
        return []

    resolved_client = client
    if resolved_client is None:
        if allow_live_requests is not True:
            raise LiveRequestDisabledError(
                "Live remote requests are disabled. Use an explicit "
                "provider-neutral opt-in flag to enable them."
            )
        config = validate_config_for_live_request()
        resolved_client = RemoteChatCompletionsClient(
            api_key=config.api_key,
            api_url=config.api_url,
            timeout=config.timeout_seconds,
            retry_settings=RetrySettings(max_retries=config.max_retries),
        )

    create_completion = getattr(
        resolved_client, "create_chat_completion", None
    )
    if not callable(create_completion):
        raise TypeError("client must provide a create_chat_completion method.")

    results: list[RemoteInferenceResult] = []
    for messages in requests:
        raw_response = create_completion(model_identifier, messages)
        parsed = parse_answer(raw_response)
        results.append(
            {
                "raw_response": raw_response,
                "parsed_prediction": parsed["prediction"],
            }
        )
    return results


def generate_response(
    model_name: str,
    prompt: Mapping[str, Template],
    queries: list[dict[str, Any]],
    output_path: str,
    n: int = 1,
    *,
    client: RemoteClient | None = None,
    allow_live_requests: bool = False,
) -> list[RemoteInferenceResult]:
    """Pipeline-compatible entry point that writes enriched query records."""

    if isinstance(n, bool) or not isinstance(n, int) or n != 1:
        raise ValueError("Remote inference currently requires n=1.")

    results = generate_remote_responses(
        model_name,
        queries,
        prompt,
        client=client,
        allow_live_requests=allow_live_requests,
    )
    for query, result in zip(queries, results):
        query["response"] = result["raw_response"]
        query["prediction"] = result["parsed_prediction"]

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(queries, output_file, indent=4, ensure_ascii=False)
    return results


def _reasoning_method(query: Mapping[str, Any], index: int) -> str:
    value = query.get("claim_type")
    if not isinstance(value, str) or not value.strip():
        raise InvalidRemoteInputError(
            f"Query at index {index} requires a non-empty claim_type."
        )
    method = value.lower()
    if method not in _REASONING_METHODS:
        supported = ", ".join(sorted(_REASONING_METHODS))
        raise InvalidRemoteInputError(
            f"Query at index {index} has unsupported claim_type; "
            f"expected one of: {supported}."
        )
    return method


def _read_paper(
    query: Mapping[str, Any], index: int
) -> Mapping[str, Any]:
    paper_path = _required_string(query, "paper_path", index)
    try:
        with open(paper_path, "r", encoding="utf-8") as paper_file:
            paper = json.load(paper_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidRemoteInputError(
            f"Unable to read valid paper JSON for query at index {index}."
        ) from exc
    if not isinstance(paper, Mapping):
        raise InvalidRemoteInputError(
            f"Paper JSON for query at index {index} must be an object."
        )
    return paper


def _prepare_context(
    paper: Mapping[str, Any], query: Mapping[str, Any], index: int
) -> str:
    section_ids = query.get("section")
    if (
        isinstance(section_ids, (str, bytes))
        or not isinstance(section_ids, Sequence)
        or not all(isinstance(section_id, str) for section_id in section_ids)
    ):
        raise InvalidRemoteInputError(
            f"Query at index {index} requires a sequence of section IDs."
        )

    top_sections: list[str] = []
    for section_id in section_ids:
        top_section = section_id.split(".")[0]
        if top_section not in top_sections:
            top_sections.append(top_section)

    sections = paper.get("sections")
    if isinstance(sections, (str, bytes)) or not isinstance(sections, Sequence):
        raise InvalidRemoteInputError(
            f"Paper JSON for query at index {index} requires sections."
        )

    context = ""
    has_context_text = False
    for section in sections:
        if not isinstance(section, Mapping):
            raise InvalidRemoteInputError(
                f"Paper section for query at index {index} must be an object."
            )
        section_id = section.get("section_id")
        section_name = section.get("section_name")
        section_text = section.get("text")
        if not all(
            isinstance(value, str)
            for value in (section_id, section_name, section_text)
        ):
            raise InvalidRemoteInputError(
                f"Paper section for query at index {index} is incomplete."
            )
        if section_id.split(".")[0] in top_sections:
            context += section_name + ":\n" + section_text + "\n"
            has_context_text = has_context_text or bool(section_text.strip())
    if not has_context_text:
        raise InvalidRemoteInputError(
            f"Query at index {index} requires non-empty context from its section IDs."
        )
    return context


def _prepare_prompt(
    prompts: Mapping[str, Template],
    method: str,
    paper: Mapping[str, Any],
    query: Mapping[str, Any],
    context: str,
    index: int,
) -> str:
    try:
        template = prompts[method]
    except KeyError as exc:
        raise InvalidRemoteInputError(
            f"Prompt mapping is missing the {method} reasoning method."
        ) from exc
    if not isinstance(template, Template):
        raise InvalidRemoteInputError(
            f"Prompt for the {method} reasoning method must be a Template."
        )

    claim = _required_string(query, "claim", index)
    if method in _SINGLE_IMAGE_METHODS:
        caption = _caption(
            paper,
            query.get("type"),
            query.get("item"),
            index,
        )
        return template.substitute(
            claim=claim,
            context=context,
            caption=caption,
        )

    caption1 = _caption(
        paper,
        query.get("item1_type"),
        query.get("item1"),
        index,
    )
    caption2 = _caption(
        paper,
        query.get("item2_type"),
        query.get("item2"),
        index,
    )
    return template.substitute(
        claim=claim,
        context=context,
        caption1=caption1,
        caption2=caption2,
    )


def _uses_inline_evidence(query: Mapping[str, Any]) -> bool:
    """Identify the additive flat normalized schema without changing legacy input."""

    return "paper_path" not in query


def _prompt_template(prompts: Mapping[str, Template], method: str) -> Template:
    try:
        template = prompts[method]
    except KeyError as exc:
        raise InvalidRemoteInputError(
            f"Prompt mapping is missing the {method} reasoning method."
        ) from exc
    if not isinstance(template, Template):
        raise InvalidRemoteInputError(
            f"Prompt for the {method} reasoning method must be a Template."
        )
    return template


def _prepare_inline_prompt(
    prompts: Mapping[str, Template],
    method: str,
    query: Mapping[str, Any],
    context: str,
    index: int,
) -> str:
    template = _prompt_template(prompts, method)
    values = {
        "claim": _required_string(query, "claim", index),
        "context": context,
    }
    if method in _SINGLE_IMAGE_METHODS:
        values["caption"] = _required_string(query, "caption", index)
    else:
        values["caption1"] = _required_string(query, "caption1", index)
        values["caption2"] = _required_string(query, "caption2", index)
    return template.substitute(**values)


def _caption(
    paper: Mapping[str, Any],
    item_type: Any,
    item_id: Any,
    index: int,
) -> str:
    if item_type not in {"chart", "table"}:
        raise InvalidRemoteInputError(
            f"Query at index {index} requires chart or table evidence."
        )
    collection_name = "image_paths" if item_type == "chart" else "tables"
    caption_name = "caption" if item_type == "chart" else "capture"
    collection = paper.get(collection_name)
    try:
        item = collection[item_id]
        caption = item[caption_name]
    except (KeyError, IndexError, TypeError) as exc:
        raise InvalidRemoteInputError(
            f"Paper evidence for query at index {index} is incomplete."
        ) from exc
    if not isinstance(caption, str) or not caption.strip():
        raise InvalidRemoteInputError(
            f"Paper caption for query at index {index} must be non-empty text."
        )
    return caption


def _image_paths(
    query: Mapping[str, Any], method: str, index: int
) -> tuple[str, ...]:
    if method in _SINGLE_IMAGE_METHODS:
        return (_required_string(query, "image_path", index),)
    return (
        _required_string(query, "item1_path", index),
        _required_string(query, "item2_path", index),
    )


def _required_string(
    mapping: Mapping[str, Any], name: str, index: int
) -> str:
    value = mapping.get(name)
    if not isinstance(value, str) or not value.strip():
        raise InvalidRemoteInputError(
            f"Query at index {index} requires a non-empty {name}."
        )
    return value


__all__ = [
    "InvalidRemoteInputError",
    "LiveRequestDisabledError",
    "RemoteInferenceResult",
    "generate_remote_responses",
    "generate_response",
    "prepare_remote_requests",
]
