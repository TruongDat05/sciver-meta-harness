"""Offline structural checks for the canonical thin server notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "sciver_meta_harness.ipynb"


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(notebook: dict[str, object]) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )


def test_notebook_has_all_required_operator_stages_in_order():
    source = _source(_notebook())
    headings = [
        "## 1. SETUP",
        "## 2. SETUP",
        "## 3. OFFLINE_SMOKE",
        "## 4. LIVE_SMOKE",
        "## 5. Smoke receipt validation",
        "## 6. FULL_SEARCH",
        "## 7. SEARCH status",
        "## 8. Freeze",
        "## 9. FINAL preflight",
        "## 10. FINAL",
        "## 11. Sanitized aggregate reporting",
    ]
    positions = [source.index(heading) for heading in headings]
    assert positions == sorted(positions)


def test_notebook_is_output_free_and_all_code_cells_compile():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("execution_count") is None
        assert cell.get("outputs", []) == []
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"notebook-cell-{index}")


def test_notebook_delegates_all_experiment_operations_to_server_interface():
    source = _source(_notebook())
    required_calls = {
        "validate_server_run_checkout",
        "prepare_run",
        "preflight_search_run",
        "run_meta_harness_smoke",
        "inspect_server_run_smoke_receipt",
        "start_or_resume_search_run",
        "inspect_server_run_status",
        "freeze_server_run_winner",
        "preflight_server_run_final",
        "start_or_resume_server_run_final",
        "inspect_server_run_final_status",
    }
    assert required_calls <= set(
        re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)
    )
    assert "from meta_harness.server_run import" in source
    forbidden = (
        "build_experiment_split",
        "evaluate_experiment_",
        "execute_solver_request",
        "Orchestrator",
        "execute_final_evaluation",
        "freeze_experiment_winner",
    )
    assert all(token not in source for token in forbidden)


def test_notebook_never_changes_checkout_or_installs_dependencies():
    source = _source(_notebook())
    forbidden_commands = (
        "git clone",
        "git fetch",
        "git pull",
        "git checkout",
        '["git", "clone"',
        '["git", "fetch"',
        '["git", "pull"',
        '["git", "checkout"',
    )
    assert all(command not in source for command in forbidden_commands)
    assert "subprocess.run" not in source
    assert '[sys.executable, "-m", "pip"' not in source
    assert "manually prepared repository checkout" in source
    assert "validate_server_run_checkout" in source
    assert "PINNED_COMMIT_SHA" in source
    assert "EXPECTED_REPOSITORY_ORIGIN" in source


def test_notebook_uses_exact_confirmation_strings_before_runtime_credentials():
    source = _source(_notebook())
    assert "LIVE_SMOKE_CONFIRMATION == 'RUN_LIVE_SMOKE'" in source
    assert "FULL_SEARCH_CONFIRMATION == 'RUN_FULL_SEARCH'" in source
    assert "FINAL_CONFIRMATION == 'RUN_FINAL_ONCE'" in source
    assert source.index("LIVE_SMOKE_CONFIRMATION == 'RUN_LIVE_SMOKE'") < source.index(
        "runtime_api_url, runtime_api_key = _runtime_api_values()"
    )
    assert "getpass('API_KEY (runtime only): ')" in source
    assert "raw_url = os.environ.get('API_URL')" in source
    assert "raw_key = os.environ.get('API_KEY')" in source
    assert "raw_key.strip()" in source
    assert "'\\r' in raw_key or '\\n' in raw_key" in source


def test_notebook_default_path_has_no_live_boolean_and_no_embedded_runtime_values():
    source = _source(_notebook())
    assert "authorize_smoke_execution=True" in source
    assert "authorize_search_execution=True" in source
    assert "authorize_final_execution=True" in source
    assert "RUN_SMOKE = True" not in source
    assert "RUN_FULL_SEARCH = True" not in source
    assert "RUN_FINAL = True" not in source
    assert "API_BASE_URL" not in source
    assert "data:image/" not in source.lower()
    assert not re.search(r"authorization\s*[:=]", source, flags=re.IGNORECASE)
    assert not re.search(r"bearer\s+[A-Za-z0-9._~+/=-]+", source, flags=re.IGNORECASE)
    assert not re.search(r"(?:sk|key)-[A-Za-z0-9]{16,}", source)
    assert "https://" not in source
    assert "http://" not in source
    assert "sample_id" not in source


def test_notebook_offline_smoke_precedes_any_credential_reader_call():
    source = _source(_notebook())
    offline_call = source.index("offline_smoke = preflight_search_run(")
    credential_call = source.index("runtime_api_url, runtime_api_key = _runtime_api_values()")
    assert offline_call < credential_call
    assert "solver_identity_sha256=" not in source[offline_call:credential_call]


def test_readme_references_the_only_canonical_notebook():
    tracked_notebooks = sorted(
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in (REPOSITORY_ROOT / "notebooks").glob("*.ipynb")
    )
    assert tracked_notebooks == ["notebooks/sciver_meta_harness.ipynb"]
    assert tracked_notebooks[0] in (REPOSITORY_ROOT / "README.md").read_text(
        encoding="utf-8"
    )
