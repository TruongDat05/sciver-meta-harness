import ast
import json
from pathlib import Path
import sys

import main as cli_module


NOTEBOOK_PATH = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "kaggle_remote_benchmark.ipynb"
)


def _notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells():
    return [
        "".join(cell["source"])
        for cell in _notebook()["cells"]
        if cell["cell_type"] == "code"
    ]


def _configuration():
    configuration_source = next(
        source for source in _code_cells() if "GITHUB_REPOSITORY =" in source
    )
    namespace = {}
    exec(compile(configuration_source, "notebook-configuration", "exec"), namespace)
    return namespace


def _benchmark_command_function():
    source = next(source for source in _code_cells() if "def benchmark_command" in source)
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "benchmark_command"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "sys": sys,
        "DATASET_NAME": "SciVer",
        "DATASET_FILE": Path("/tmp/normalized_records.json"),
        "MODEL_NAME": "Qwen2.5-VL-7B-Instruct",
        "METHOD": "cot",
        "PILOT_SAMPLE_COUNT": 7,
        "REQUEST_DELAY": 0.5,
        "RUN_OUTPUT_DIR": Path(
            "/kaggle/working/results/SciVer/Qwen2.5-VL-7B-Instruct/cot/test"
        ),
    }
    exec(compile(module, "notebook-command-builder", "exec"), namespace)
    return namespace["benchmark_command"]


def test_notebook_is_valid_json_and_every_code_cell_compiles():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    for index, source in enumerate(_code_cells()):
        compile(source, f"notebook-cell-{index}", "exec")


def test_notebook_has_no_saved_outputs_or_embedded_secret_values():
    notebook = _notebook()
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None

    source = "\n".join(_code_cells())
    assert "Authorization" not in source
    assert "data:image" not in source
    assert "<REFERENCE_NOTEBOOK_PATH>" not in source
    assert "RUN_ID" not in source


def test_configuration_defaults_to_credential_free_dry_run():
    configuration = _configuration()
    assert configuration["RUN_MODE"] == "dry_run"
    assert configuration["RUN_LIVE_API"] is False
    assert configuration["OUTPUT_ROOT"] == "/kaggle/working/results"
    required = {
        "GITHUB_REPOSITORY",
        "DATASET_NAME",
        "DATASET_PATH",
        "MODEL_NAME",
        "METHOD",
        "EXPERIMENT_ID",
        "RUN_MODE",
        "RUN_LIVE_API",
        "PILOT_SAMPLE_COUNT",
        "REQUEST_DELAY",
        "OUTPUT_ROOT",
    }
    assert required <= configuration.keys()
    assert configuration["GIT_REF"] or configuration["EXPECTED_GIT_SHA"]


def test_notebook_installs_only_lightweight_remote_dependencies_and_disables_gpu():
    install_cell = next(
        source
        for source in _code_cells()
        if '"pip", "install"' in source
    )
    install_tree = ast.parse(install_cell)
    install_call = next(
        node
        for node in ast.walk(install_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_checked"
        and isinstance(node.args[0], ast.List)
        and any(
            isinstance(item, ast.Constant) and item.value == "install"
            for item in node.args[0].elts
        )
    )
    installed_values = {
        item.value.casefold()
        for item in install_call.args[0].elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }

    assert not any(value.startswith("vllm") for value in installed_values)
    assert not any(value.startswith("torch") for value in installed_values)
    assert not any(value.startswith("transformers") for value in installed_values)
    assert '"requests>=2.31.0,<3"' in install_cell
    assert '"Pillow>=10,<12"' in install_cell
    assert '"pytest>=8,<10"' in install_cell
    assert 'offline_env["CUDA_VISIBLE_DEVICES"] = ""' in install_cell
    assert 'offline_env["HF_HUB_OFFLINE"] = "1"' in install_cell
    assert 'offline_env["TRANSFORMERS_OFFLINE"] = "1"' in install_cell


def test_secret_access_is_guarded_behind_non_default_live_mode():
    secret_cell = next(source for source in _code_cells() if "UserSecretsClient" in source)
    tree = ast.parse(secret_cell)
    guarded = next(node for node in tree.body if isinstance(node, ast.If))
    assert ast.unparse(guarded.test) == "RUN_MODE != 'dry_run'"
    guarded_source = ast.unparse(ast.Module(body=guarded.body, type_ignores=[]))
    assert "UserSecretsClient" in guarded_source
    assert "get_secret('API_KEY')" in guarded_source
    assert "get_secret('API_URL')" in guarded_source
    assert "parsed_api_url.scheme != 'https'" in guarded_source


def test_all_notebook_main_commands_match_the_current_cli():
    command_for = _benchmark_command_function()
    for mode in ("dry_run", "smoke", "pilot", "full"):
        command = command_for(mode)
        arguments = cli_module.parse_args(command[2:])
        assert arguments.provider == "remote"
        assert arguments.dataset == "SciVer"
        assert arguments.model == "Qwen2.5-VL-7B-Instruct"
        assert arguments.method == "cot"
        assert arguments.overwrite is False


def test_mode_commands_have_safe_counts_resume_and_one_output_directory():
    command_for = _benchmark_command_function()
    commands = {mode: command_for(mode) for mode in ("dry_run", "smoke", "pilot", "full")}

    def value(command, option):
        return command[command.index(option) + 1]

    assert value(commands["dry_run"], "--max-num") == "1"
    assert "--dry-run" in commands["dry_run"]
    assert "--live-api" not in commands["dry_run"]
    assert value(commands["smoke"], "--max-num") == "1"
    assert "--live-api" in commands["smoke"]
    assert "--resume" in commands["pilot"]
    assert value(commands["pilot"], "--max-num") == "7"
    assert "--resume" in commands["full"]
    assert value(commands["full"], "--max-num") == "-1"
    assert all("--overwrite" not in command for command in commands.values())
    live_output_dirs = {
        value(commands[mode], "--output-dir") for mode in ("smoke", "pilot", "full")
    }
    assert live_output_dirs == {
        "/kaggle/working/results/SciVer/Qwen2.5-VL-7B-Instruct/cot/test"
    }


def test_dataset_and_result_comparison_tables_are_present():
    source = "\n".join(_code_cells())
    for field in (
        "label distribution",
        "reasoning or claim-type distribution",
        "missing-context count",
        "missing-image count",
        "adapter readiness",
        "successful requests",
        "failed requests",
        "invalid inputs",
        "parsed predictions",
        "parse coverage",
        "accuracy",
        "output directory",
    ):
        assert field in source
    assert "scripts/evaluate_results.py" in source
    assert "run_manifest.json" in source
