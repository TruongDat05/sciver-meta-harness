import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

from PIL import Image
import pytest

import main as cli_module
from utils.result_writer import ResultWriter, SUCCESS, iter_result_records


MODEL = "Qwen2.5-VL-7B-Instruct"
RAW_RESPONSE = "Therefore, the final answer is: Answer: yes"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _set_fake_live_environment(monkeypatch):
    monkeypatch.setenv("API_KEY", "UNMISTAKABLY_FAKE_CLI_KEY")
    monkeypatch.setenv(
        "API_URL",
        "https://invalid.example.test/chat/completions",
    )


def _loaded_vllm_modules():
    return {
        module_name
        for module_name in sys.modules
        if module_name == "vllm"
        or module_name.startswith("vllm.")
        or module_name
        in {
            "model_inference.vllm_inference",
            "utils.vllm_input_preparation",
        }
    }


@pytest.fixture(autouse=True)
def block_http_requests():
    """Every CLI test mocks the HTTP request boundary."""

    with patch(
        "requests.sessions.Session.request",
        side_effect=AssertionError("CLI tests must not access the network"),
    ):
        yield


@pytest.fixture
def dataset_path(tmp_path):
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
                        "text": "Local evidence text.",
                    }
                ],
                "image_paths": [{"caption": "Local figure caption."}],
                "tables": [],
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "sample_id": f"sample-{index}",
            "claim_type": "direct",
            "claim": f"Claim {index}",
            "paper_path": str(paper_path),
            "section": ["1"],
            "type": "chart",
            "item": 0,
            "image_path": str(image_path),
            "label": "GROUND_TRUTH_MUST_NOT_LEAK",
        }
        for index in range(3)
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def remote_args(dataset_path, output_dir, *extra):
    return [
        "--provider",
        "remote",
        "--model",
        MODEL,
        "--dataset",
        "SciVer",
        "--dataset-path",
        str(dataset_path),
        "--method",
        "cot",
        "--output-dir",
        str(output_dir),
        *extra,
    ]


def fake_client():
    client = Mock()
    client.create_chat_completion.return_value = RAW_RESPONSE
    return client


def test_importing_main_in_a_fresh_process_does_not_import_vllm():
    probe = """
import sys
from unittest.mock import patch

with patch(
    "requests.sessions.Session.request",
    side_effect=AssertionError("import probe must not access HTTP"),
), patch(
    "socket.socket.connect",
    side_effect=AssertionError("import probe must not open sockets"),
):
    import main

loaded = [
    name
    for name in sys.modules
    if name == "vllm"
    or name.startswith("vllm.")
    or name
    in {
        "model_inference.vllm_inference",
        "utils.vllm_input_preparation",
    }
]
assert not loaded, loaded
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_argument_parsing_supports_hyphenated_and_legacy_aliases(
    dataset_path, tmp_path
):
    arguments = cli_module.parse_args(
        remote_args(
            dataset_path,
            tmp_path / "out",
            "--max-num",
            "1",
            "--dry-run",
            "--resume",
            "--request-delay",
            "0.25",
        )
    )

    assert arguments.provider == "remote"
    assert arguments.dataset == "SciVer"
    assert arguments.data_path == str(dataset_path)
    assert arguments.output_dir == str(tmp_path / "out")
    assert arguments.max_num == 1
    assert arguments.dry_run is True
    assert arguments.resume is True
    assert arguments.request_delay == 0.25

    legacy = cli_module.parse_args(
        [
            "--model",
            "legacy-model",
            "--data_path",
            str(dataset_path),
            "--output_dir",
            str(tmp_path),
            "--max_num",
            "2",
        ]
    )
    assert legacy.provider is None
    assert legacy.max_num == 2


def test_dry_run_needs_no_credentials_or_client_and_emits_safe_metadata(
    dataset_path, tmp_path, monkeypatch, capsys
):
    initially_loaded_vllm_modules = _loaded_vllm_modules()
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    client_factory = Mock(side_effect=AssertionError("client must not be created"))
    monkeypatch.setattr(cli_module, "_create_remote_client", client_factory)
    dotenv_loader = Mock(
        side_effect=AssertionError("dry-run must not load credentials")
    )
    monkeypatch.setattr(cli_module, "load_cli_environment", dotenv_loader)

    exit_code = cli_module.cli(
        remote_args(
            dataset_path, tmp_path / "out", "--max-num", "1", "--dry-run"
        )
    )

    assert exit_code == 0
    client_factory.assert_not_called()
    dotenv_loader.assert_not_called()
    output = capsys.readouterr()
    metadata = json.loads(output.out)
    assert set(metadata) == {
        "dataset",
        "adapter",
        "model",
        "reasoning_method",
        "sample_id",
        "image_count",
        "prompt_length",
        "output_location",
    }
    assert metadata["sample_id"] == "sample-0"
    assert metadata["image_count"] == 1
    assert "GROUND_TRUTH" not in output.out
    assert output.err == ""
    assert _loaded_vllm_modules() == initially_loaded_vllm_modules


def test_live_remote_cli_fails_clearly_when_credentials_are_missing(
    dataset_path,
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.setattr(cli_module, "load_cli_environment", Mock(return_value=False))

    exit_code = cli_module.cli(
        remote_args(dataset_path, tmp_path / "out", "--live-api")
    )

    assert exit_code == 2
    error = capsys.readouterr().err
    assert "Missing required remote API configuration" in error
    assert "API_URL and API_KEY" in error


@pytest.mark.parametrize(
    "bad_option",
    [
        ["--provider", "unknown"],
        ["--dataset", "UnknownDataset"],
        ["--method", "unknown"],
    ],
)
def test_unknown_provider_dataset_or_method_is_rejected(
    dataset_path, tmp_path, bad_option
):
    arguments = remote_args(dataset_path, tmp_path)
    option = bad_option[0]
    option_index = arguments.index(option)
    arguments[option_index : option_index + 2] = bad_option

    with pytest.raises(SystemExit):
        cli_module.parse_args(arguments)


def test_unknown_remote_model_is_rejected_before_client_creation(
    dataset_path, tmp_path, monkeypatch
):
    arguments = remote_args(dataset_path, tmp_path, "--dry-run")
    arguments[arguments.index(MODEL)] = "unknown-model"
    client_factory = Mock()
    monkeypatch.setattr(cli_module, "_create_remote_client", client_factory)

    assert cli_module.cli(arguments) == 2
    client_factory.assert_not_called()


@pytest.mark.parametrize(("max_num", "expected"), [("1", 1), ("-1", 3)])
def test_max_num_controls_remote_request_count(
    dataset_path, tmp_path, monkeypatch, max_num, expected
):
    _set_fake_live_environment(monkeypatch)
    client = fake_client()
    monkeypatch.setattr(cli_module, "_create_remote_client", Mock(return_value=client))

    exit_code = cli_module.cli(
        remote_args(
            dataset_path,
            tmp_path / max_num,
            "--max-num",
            max_num,
            "--live-api",
        )
    )

    assert exit_code == 0
    assert client.create_chat_completion.call_count == expected


def test_resume_uses_writer_and_skips_successful_samples(
    dataset_path, tmp_path, monkeypatch
):
    _set_fake_live_environment(monkeypatch)
    output_dir = tmp_path / "out"
    output_path = output_dir / "SciVer_cot" / f"{MODEL}.jsonl"
    writer = ResultWriter(output_path)
    writer.write_result(
        run_id=f"SciVer:{MODEL}:cot",
        sample_id="sample-0",
        dataset="SciVer",
        model=MODEL,
        method="direct",
        prediction="yes",
        parse_status="parsed",
        raw_response=RAW_RESPONSE,
        request_status=SUCCESS,
    )
    client = fake_client()
    monkeypatch.setattr(cli_module, "_create_remote_client", Mock(return_value=client))

    exit_code = cli_module.cli(
        remote_args(
            dataset_path,
            output_dir,
            "--max-num",
            "-1",
            "--resume",
            "--live-api",
        )
    )

    assert exit_code == 0
    assert client.create_chat_completion.call_count == 2
    assert [record["sample_id"] for record in iter_result_records(output_path)] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]


def test_existing_model_routing_remains_compatible(monkeypatch, tmp_path):
    generator = Mock()
    selector = Mock(return_value=generator)
    monkeypatch.setattr(cli_module, "_select_legacy_generator", selector)
    queries = [{"claim": "unchanged"}]

    cli_module.main(
        "legacy-model",
        {"prompt": "unchanged"},
        queries,
        str(tmp_path / "out.json"),
        2,
    )

    selector.assert_called_once_with("legacy-model")
    generator.assert_called_once_with(
        model_name="legacy-model",
        prompt={"prompt": "unchanged"},
        queries=queries,
        output_path=str(tmp_path / "out.json"),
        n=2,
    )


def test_remote_execution_without_live_api_stops_before_client_or_request(
    dataset_path, tmp_path, monkeypatch
):
    client_factory = Mock(side_effect=AssertionError("client must not be created"))
    monkeypatch.setattr(cli_module, "_create_remote_client", client_factory)

    exit_code = cli_module.cli(remote_args(dataset_path, tmp_path, "--max-num", "1"))

    assert exit_code == 2
    client_factory.assert_not_called()


def test_request_delay_is_only_applied_between_requests(
    dataset_path, tmp_path, monkeypatch
):
    _set_fake_live_environment(monkeypatch)
    client = fake_client()
    sleeps = Mock()
    monkeypatch.setattr(cli_module, "_create_remote_client", Mock(return_value=client))
    monkeypatch.setattr(cli_module.time, "sleep", sleeps)

    assert cli_module.cli(
        remote_args(
            dataset_path,
            tmp_path,
            "--max-num",
            "-1",
            "--request-delay",
            "0.5",
            "--live-api",
        )
    ) == 0

    assert sleeps.call_count == 2
    sleeps.assert_called_with(0.5)


def test_failed_sample_does_not_remove_previous_results(
    dataset_path, tmp_path, monkeypatch
):
    _set_fake_live_environment(monkeypatch)
    client = fake_client()
    client.create_chat_completion.side_effect = [
        RAW_RESPONSE,
        RuntimeError("safe failure"),
        RAW_RESPONSE,
    ]
    monkeypatch.setattr(cli_module, "_create_remote_client", Mock(return_value=client))

    assert cli_module.cli(
        remote_args(dataset_path, tmp_path, "--max-num", "-1", "--live-api")
    ) == 1

    output_path = tmp_path / "SciVer_cot" / f"{MODEL}.jsonl"
    records = list(iter_result_records(output_path))
    assert [record["sample_id"] for record in records] == [
        "sample-0",
        "sample-1",
        "sample-2",
    ]
    assert [record["request_status"] for record in records] == [
        "success",
        "api_failure",
        "success",
    ]
