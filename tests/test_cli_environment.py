from __future__ import annotations

import os
import subprocess

from utils.cli_environment import load_cli_environment
from utils.result_writer import API_FAILURE, ResultWriter


def test_dotenv_preserves_existing_environment_and_fills_missing_values(
    tmp_path,
    monkeypatch,
):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "API_KEY=UNMISTAKABLY_FAKE_FILE_KEY\n"
        "API_URL=https://invalid.example.test/from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("API_KEY", "UNMISTAKABLY_FAKE_EXISTING_KEY")
    monkeypatch.delenv("API_URL", raising=False)

    assert load_cli_environment(dotenv_path) is True
    assert os.environ["API_KEY"] == "UNMISTAKABLY_FAKE_EXISTING_KEY"
    assert (
        os.environ["API_URL"]
        == "https://invalid.example.test/from-file"
    )


def test_dotenv_secret_is_redacted_from_persisted_cli_failure(
    tmp_path,
    monkeypatch,
):
    secret = "UNMISTAKABLY_FAKE_DOTENV_SECRET"
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        f"API_KEY={secret}\n"
        "API_URL=https://invalid.example.test/redaction\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("API_URL", raising=False)
    load_cli_environment(dotenv_path)
    output = tmp_path / "results.jsonl"

    ResultWriter(output).write_result(
        run_id="offline-env-redaction",
        sample_id="sample-1",
        dataset="SciVer",
        model="test-model",
        method="direct",
        prediction=None,
        parse_status="not_attempted",
        raw_response=None,
        request_status=API_FAILURE,
        error=RuntimeError(secret),
    )

    serialized = output.read_text(encoding="utf-8")
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_git_ignores_local_dotenv_variants_but_not_example():
    for path in (".env", ".env.local", ".env.test"):
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", path],
            check=False,
        )
        assert ignored.returncode == 0
    example = subprocess.run(
        ["git", "check-ignore", "-q", ".env.example"],
        check=False,
    )

    assert example.returncode == 1
