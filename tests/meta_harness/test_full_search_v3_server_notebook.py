"""Offline structural checks for the canonical thin V3 server notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[2]
    / "notebooks"
    / "sciver_full_search_v3_server.ipynb"
)


def _notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source(notebook: dict[str, object]) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] in {"code", "markdown"}
    )


def test_notebook_has_all_required_operator_sections_in_order():
    source = _source(_notebook())
    headings = [
        "## 1. Non-secret operator settings",
        "## 2. Clone when absent, or verify the existing checkout",
        "## 3. Fetch and check out the exact pinned commit",
        "## 4. Install project dependencies into this Jupyter environment",
        "## 5. Verify server readiness",
        "## 6. Import the repository M6 interface and locate the source dataset",
        "## 7. Prepare or reuse the deterministic configured split",
        "## 8. Collect runtime API values without displaying them",
        "## 9. Offline SEARCH preflight",
        "## 10. Explicit live SEARCH guard",
        "## 11. Inspect SEARCH progress and terminal status",
        "## 12. Freeze the SEARCH winner only after terminal completion",
        "## 13. FINAL preflight",
        "## 14. Separate explicit paired FINAL guard",
        "## 15. Sanitized aggregate results and artifact locations",
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


def test_notebook_uses_only_the_m6_interface_for_experiment_operations():
    source = _source(_notebook())
    required_calls = {
        "prepare_full_search_v3_server_run",
        "preflight_full_search_v3_server_run",
        "start_or_resume_full_search_v3_server_run",
        "inspect_full_search_v3_server_status",
        "freeze_full_search_v3_server_winner",
        "preflight_full_search_v3_server_final",
        "start_or_resume_full_search_v3_server_final",
        "inspect_full_search_v3_server_final_status",
    }
    assert required_calls <= set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", source))
    assert "from meta_harness.full_search_v3_server import" in source
    forbidden = (
        "build_full_search_v3_split",
        "evaluate_full_search_v3_",
        "execute_solver_request",
        "FullSearchV3Orchestrator",
        "execute_full_search_v3_final",
        "freeze_full_search_v3_winner",
    )
    assert all(token not in source for token in forbidden)


def test_notebook_uses_pinned_checkout_runtime_credentials_and_separate_guards():
    source = _source(_notebook())
    for setting in (
        "GIT_REPOSITORY_URL",
        "PINNED_COMMIT_SHA",
        "WORKSPACE_DIRECTORY",
        "DATASET_PATH",
        "RUN_ID",
        "SOLVER_MODEL_ID",
        "CONFIG_PATH",
    ):
        assert f"{setting} =" in source
    assert "[\"git\", \"fetch\", \"--tags\", \"origin\", PINNED_COMMIT_SHA]" in source
    assert "[\"git\", \"checkout\", \"--detach\", PINNED_COMMIT_SHA]" in source
    assert "RUN_SEARCH = False" in source
    assert "if RUN_SEARCH:" in source
    assert "RUN_FINAL = False" in source
    assert "if RUN_FINAL and frozen_winner is not None:" in source
    assert "getpass(\"API_KEY (runtime only): \")" in source
    assert "api_url = os.environ.get(\"API_URL\") or input" in source
    assert "api_key = os.environ.get(\"API_KEY\") or getpass" in source
    assert "SOLVER_MODEL_ID does not match the repository's locked V3 solver configuration" in source
    assert "split_sizes_and_paper_groups" in source
    assert "paper_overlap_count" in source
    assert "\"checkpoints\"" in source


def test_notebook_has_no_embedded_secret_output_or_private_final_content():
    notebook = _notebook()
    source = _source(notebook)
    assert "data:image/" not in source.lower()
    assert not re.search(r"authorization\s*[:=]", source, flags=re.IGNORECASE)
    assert not re.search(r"bearer\s+[A-Za-z0-9._~+/=-]+", source, flags=re.IGNORECASE)
    assert "API_KEY =" not in source
    assert "API_URL =" not in source
    assert not re.search(r"(?:sk|key)-[A-Za-z0-9]{16,}", source)
    assert "FINAL\"][\"sample_ids\"]" not in source
    assert "sample_id" not in source
    assert "private_manifest_path" not in source.split("## 13. FINAL preflight")[0]


def test_notebook_has_no_embedded_urls():
    source = _source(_notebook())

    assert "http://" not in source
    assert "https://" not in source
