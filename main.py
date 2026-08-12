"""SciVer command-line entry point.

The legacy model-based routing remains the default.  The remote path is
additive and requires both ``--provider remote`` and ``--live-api`` before a
client can be created or a request can be sent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from utils.constant import COT_PROMPT
from utils.cli_environment import load_cli_environment


PROMPT_DICT = {
    "cot": COT_PROMPT,
}
# Keep the historical public name for callers that import it.
prompt_dict = PROMPT_DICT

REMOTE_PROVIDER = "remote"
REMOTE_METHODS = tuple(PROMPT_DICT)
_SAMPLE_REASONING_METHODS = frozenset(
    {"direct", "analytical", "parallel", "sequential"}
)
_REPOSITORY_ROOT = Path(__file__).resolve().parent


def _select_legacy_generator(model_name: str) -> Callable[..., Any]:
    """Return the same generator selected by the original model-name routing."""

    if "gpt" in model_name:
        from model_inference.azure_gpt import generate_response

        return generate_response
    if "gemini" in model_name:
        from model_inference.openai_compatible import generate_response

        return generate_response

    model_list_path = _REPOSITORY_ROOT / "model_inference" / "vllm_model_list.json"
    with model_list_path.open("r", encoding="utf-8") as model_list_file:
        local_models = json.load(model_list_file)
    if model_name in local_models:
        from model_inference.vllm_inference import generate_response

        return generate_response
    raise ValueError(f"Invalid model name: {model_name}")


def main(
    model_name: str,
    prompt: Mapping[str, Any],
    queries: list,
    output_path: str,
    n: int = 1,
) -> None:
    """Run the historical inference path without changing its call contract."""

    generate_response = _select_legacy_generator(model_name)
    generate_response(
        model_name=model_name,
        prompt=prompt,
        queries=queries,
        output_path=output_path,
        n=n,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the additive CLI parser without importing inference clients."""

    from utils.dataset_adapters import SUPPORTED_DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=(REMOTE_PROVIDER,))
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument(
        "--data_path",
        "--dataset-path",
        "--dataset_path",
        dest="data_path",
        type=str,
        required=True,
    )
    parser.add_argument("--prompt", type=str, default="cot")
    parser.add_argument("--method", choices=REMOTE_METHODS, default="cot")
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=str,
        default="outputs",
    )
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--max_num", "--max-num", dest="max_num", type=int, default=5)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--request-delay", type=_non_negative_float, default=0.0)
    parser.add_argument("--live-api", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; exposed separately for focused CLI tests."""

    return build_argument_parser().parse_args(argv)


def _non_negative_float(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _legacy_cli(arguments: argparse.Namespace) -> int:
    from transformers.utils import logging

    logging.set_verbosity_error()
    try:
        prompt = PROMPT_DICT[arguments.prompt]
    except KeyError:
        print("Invalid prompt")
        return 1

    os.makedirs(arguments.output_dir, exist_ok=True)
    data_path_suffix = arguments.data_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(
        arguments.output_dir, f"{data_path_suffix}_{arguments.prompt}"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_name = arguments.model.split("/")[-1]
    output_path = os.path.join(output_dir, f"{output_name}.json")

    with open(arguments.data_path, "r", encoding="utf-8") as input_file:
        queries = json.load(input_file)
    if arguments.max_num > 0:
        queries = queries[: arguments.max_num]

    if os.path.exists(output_path) and arguments.overwrite:
        with open(output_path, "r", encoding="utf-8") as output_file:
            original_output = json.load(output_file)
        if len(original_output) >= len(queries):
            print(f"Output file {output_path} already exists. Skipping.")
            return 0
        print(f"Overwrite {output_path}")

    print(f"=========Running {arguments.model}=========\n")
    main(
        model_name=arguments.model,
        prompt=prompt,
        queries=queries,
        output_path=output_path,
        n=arguments.n,
    )
    return 0


def _remote_output_path(arguments: argparse.Namespace) -> Path:
    output_name = arguments.model.split("/")[-1]
    return (
        Path(arguments.output_dir)
        / f"{arguments.dataset}_{arguments.method}"
        / f"{output_name}.jsonl"
    )


def _sample_method(record: Mapping[str, Any], index: int) -> str:
    method = record.get("claim_type")
    if not isinstance(method, str) or method.lower() not in _SAMPLE_REASONING_METHODS:
        expected = ", ".join(sorted(_SAMPLE_REASONING_METHODS))
        raise ValueError(
            f"dataset record at index {index} requires claim_type in: {expected}"
        )
    return method.lower()


def _select_samples(samples: list[Any], method: str, max_num: int) -> list[Any]:
    if max_num < -1:
        raise ValueError("--max-num must be -1 or a non-negative integer")
    if max_num == -1:
        return samples
    return samples[:max_num]


def _create_remote_client() -> Any:
    """Validate live-only configuration and construct the HTTP client lazily."""

    from model_inference.remote_client import RemoteChatCompletionsClient, RetrySettings
    from model_inference.remote_config import validate_config_for_live_request

    config = validate_config_for_live_request()
    return RemoteChatCompletionsClient(
        api_key=config.api_key,
        api_url=config.api_url,
        timeout=config.timeout_seconds,
        retry_settings=RetrySettings(max_retries=config.max_retries),
    )


def _request_summary(messages: list[dict[str, object]]) -> tuple[int, int]:
    user_content = messages[-1].get("content")
    if not isinstance(user_content, list):
        raise ValueError("prepared request requires list-based user content")
    image_count = sum(
        1 for block in user_content if isinstance(block, Mapping) and block.get("type") == "image_url"
    )
    text_blocks = [
        block.get("text")
        for block in user_content
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    if len(text_blocks) != 1 or not isinstance(text_blocks[0], str):
        raise ValueError("prepared request requires exactly one text prompt")
    return image_count, len(text_blocks[0])


def _dry_run_remote(
    arguments: argparse.Namespace,
    adapter: Any,
    samples: list[Any],
    output_path: Path,
    prompt_family: Mapping[str, Any],
) -> int:
    from model_inference.remote_api import prepare_remote_requests

    for index, sample in enumerate(samples):
        method = _sample_method(sample.record, index)
        messages = prepare_remote_requests([sample.record], prompt_family)[0]
        image_count, prompt_length = _request_summary(messages)
        print(
            json.dumps(
                {
                    "dataset": arguments.dataset,
                    "adapter": adapter.adapter_name,
                    "model": arguments.model,
                    "reasoning_method": method,
                    "sample_id": sample.sample_id,
                    "image_count": image_count,
                    "prompt_length": prompt_length,
                    "output_location": str(output_path),
                },
                ensure_ascii=False,
            )
        )
    return 0


def _run_remote(arguments: argparse.Namespace) -> int:
    from model_inference.remote_config import validate_model_identifier
    from utils.dataset_adapters import get_dataset_adapter

    if arguments.dataset is None:
        raise ValueError("--dataset is required with --provider remote")
    try:
        prompt_family = PROMPT_DICT[arguments.prompt]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid prompt: {arguments.prompt!r}") from exc
    if arguments.n != 1:
        raise ValueError("remote inference requires --n 1")
    validate_model_identifier(arguments.model)

    # This gate intentionally precedes dataset loading, credentials, client
    # construction, and every possible request path.
    if not arguments.dry_run and arguments.live_api is not True:
        raise ValueError(
            "remote execution is disabled without both --provider remote and --live-api"
        )

    adapter = get_dataset_adapter(arguments.dataset)
    evidence_dir = (
        Path(arguments.output_dir)
        / "normalized_evidence"
        / arguments.dataset
    )
    samples = _select_samples(
        adapter.load(arguments.data_path, evidence_dir=evidence_dir),
        arguments.method,
        arguments.max_num,
    )
    output_path = _remote_output_path(arguments)

    if arguments.dry_run:
        return _dry_run_remote(
            arguments,
            adapter,
            samples,
            output_path,
            prompt_family,
        )

    from model_inference.remote_api import InvalidRemoteInputError, prepare_remote_requests
    from model_inference.remote_client import InvalidConfigurationError
    from model_inference.remote_config import RemoteConfigurationError
    from utils.answer_parser import parse_answer
    from utils.result_writer import (
        API_FAILURE,
        INVALID_INPUT,
        PARSE_FAILURE,
        SUCCESS,
        ResultWriter,
    )

    if output_path.exists() and output_path.stat().st_size and not arguments.resume:
        raise ValueError(
            f"result output already exists: {output_path}; use --resume to continue"
        )
    writer = ResultWriter(output_path)
    client = None
    run_id = f"{arguments.dataset}:{arguments.model}:{arguments.method}"
    sent_requests = 0
    failures = 0

    for index, sample in enumerate(samples):
        result_metadata = {}
        if "gold_label" in sample.record:
            result_metadata["gold_label"] = sample.record["gold_label"]
        try:
            method = _sample_method(sample.record, index)
        except ValueError as exc:
            method = arguments.method
            writer.write_result(
                run_id=run_id,
                sample_id=sample.sample_id,
                dataset=arguments.dataset,
                model=arguments.model,
                method=method,
                prompt_variant=arguments.prompt,
                prediction=None,
                parse_status="not_attempted",
                raw_response=None,
                request_status=INVALID_INPUT,
                error=exc,
                **result_metadata,
            )
            failures += 1
            continue

        identity = {
            "run_id": run_id,
            "sample_id": sample.sample_id,
            "dataset": arguments.dataset,
            "model": arguments.model,
            "method": method,
            "prompt_variant": arguments.prompt,
        }
        if arguments.resume and writer.is_successful(**identity):
            continue

        raw_response = None
        try:
            messages = prepare_remote_requests([sample.record], prompt_family)[0]
            if client is None:
                client = _create_remote_client()
            if sent_requests:
                time.sleep(arguments.request_delay)
            sent_requests += 1
            raw_response = client.create_chat_completion(arguments.model, messages)
            parsed = parse_answer(raw_response)
            if parsed["parse_status"] == "parsed":
                writer.write_result(
                    **identity,
                    prediction=parsed["prediction"],
                    parse_status="parsed",
                    raw_response=raw_response,
                    request_status=SUCCESS,
                    **result_metadata,
                )
            else:
                writer.write_result(
                    **identity,
                    prediction=parsed["prediction"],
                    parse_status="invalid",
                    raw_response=raw_response,
                    request_status=PARSE_FAILURE,
                    error_type="AnswerParseError",
                    error_message=parsed["parse_reason"],
                    **result_metadata,
                )
                failures += 1
        except (InvalidConfigurationError, RemoteConfigurationError):
            raise
        except Exception as exc:  # Persist one sample failure, then continue.
            request_status = (
                INVALID_INPUT if isinstance(exc, InvalidRemoteInputError) else API_FAILURE
            )
            writer.write_result(
                **identity,
                prediction=None,
                parse_status="not_attempted",
                raw_response=raw_response,
                request_status=request_status,
                error=exc,
                **result_metadata,
            )
            failures += 1
    return 1 if failures else 0


def cli(argv: Sequence[str] | None = None) -> int:
    """Run either the additive remote CLI or the unchanged legacy path."""

    arguments = parse_args(argv)
    if arguments.provider == REMOTE_PROVIDER:
        try:
            if arguments.live_api and not arguments.dry_run:
                load_cli_environment()
            return _run_remote(arguments)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if arguments.dry_run:
        print("error: --dry-run requires --provider remote", file=sys.stderr)
        return 2
    if arguments.live_api or arguments.resume or arguments.request_delay:
        print(
            "error: --live-api, --resume, and --request-delay require "
            "--provider remote",
            file=sys.stderr,
        )
        return 2
    if arguments.dataset is not None:
        print("error: --dataset requires --provider remote", file=sys.stderr)
        return 2
    return _legacy_cli(arguments)


if __name__ == "__main__":
    raise SystemExit(cli())
