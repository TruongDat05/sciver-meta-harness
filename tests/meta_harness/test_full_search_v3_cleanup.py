"""Milestone 7 guards against reintroducing obsolete experiment surfaces."""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import meta_harness.config as config


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REMOVED_MODULES = (
    "meta_harness.baseline",
    "meta_harness.candidate_store",
    "meta_harness.evaluator",
    "meta_harness.experience_store",
    "meta_harness.finalize",
    "meta_harness.hard_search",
    "meta_harness.orchestrator",
    "meta_harness.proposer.codex_cli",
    "meta_harness.proposer.feedback",
    "meta_harness.reparse",
    "meta_harness.retry",
    "meta_harness.schemas",
    "meta_harness.split_manager",
    "meta_harness.staged_orchestrator",
    "utils.prompt_registry",
)
REMOVED_COMMANDS = (
    "prepare_meta_harness_data.py",
    "run_meta_harness.py",
    "finalize_meta_harness.py",
    "retry_meta_harness_failures.py",
    "reparse_meta_harness.py",
    "smoke_meta_harness_proposer.py",
    "export_meta_cot.py",
    "run_meta_cot_transfer.py",
)
CURRENT_MODULES = (
    "meta_harness.config",
    "meta_harness.prompt_family",
    "meta_harness.full_search_v3",
    "meta_harness.full_search_v3_preparation",
    "meta_harness.full_search_v3_solver",
    "meta_harness.full_search_v3_cache",
    "meta_harness.full_search_v3_retry",
    "meta_harness.full_search_v3_concurrency",
    "meta_harness.full_search_v3_evaluator",
    "meta_harness.full_search_v3_proposer",
    "meta_harness.full_search_v3_ranking",
    "meta_harness.full_search_v3_orchestrator",
    "meta_harness.full_search_v3_freeze",
    "meta_harness.full_search_v3_final",
    "meta_harness.full_search_v3_server",
)


def test_removed_legacy_modules_are_not_importable_or_exposed():
    assert all(importlib.util.find_spec(name) is None for name in REMOVED_MODULES)
    assert {
        "MetaHarnessConfig",
        "SearchProtocol",
        "SplitRatios",
        "load_meta_harness_config",
    }.isdisjoint(config.__all__)


def test_all_supported_full_search_v3_modules_import_offline():
    for module_name in CURRENT_MODULES:
        assert importlib.import_module(module_name).__name__ == module_name


def test_runtime_code_has_no_imports_from_deleted_modules():
    runtime_files = [
        *sorted((REPOSITORY_ROOT / "meta_harness").glob("**/*.py")),
        *sorted((REPOSITORY_ROOT / "scripts").glob("*.py")),
        *sorted((REPOSITORY_ROOT / "utils").glob("*.py")),
        REPOSITORY_ROOT / "main.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
    for module_name in REMOVED_MODULES:
        assert module_name not in source


def test_supported_documentation_has_no_removed_commands():
    supported_docs = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "docs" / "remote-benchmark.md",
        REPOSITORY_ROOT / "docs" / "sciver_full_search_v3_server_operations.md",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in supported_docs)
    for command in REMOVED_COMMANDS:
        assert f"python scripts/{command}" not in source


def test_canonical_server_notebook_is_the_only_tracked_notebook():
    notebooks = sorted(path.name for path in (REPOSITORY_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks == ["sciver_full_search_v3_server.ipynb"]
