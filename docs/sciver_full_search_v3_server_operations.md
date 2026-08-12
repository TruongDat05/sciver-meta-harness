# SciVer Full-Search v3 server operations

Use the canonical [server notebook](../notebooks/sciver_full_search_v3_server.ipynb) for normal operation. It calls the M6 interface and owns no split, request, evaluation, ranking, freeze, or FINAL logic. The optional `scripts/run_full_search_v3_server.py` command uses those same M6 APIs and the same durable run state when a terminal/background workflow is needed.

## Prepare a server session

Create an isolated Python environment, clone the repository, and work from a pinned commit. For example, use either of these environment choices:

```bash
conda create -n sciver-v3 python=3.11
conda activate sciver-v3
python -m pip install -r requirements.txt
```

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Start JupyterLab inside a persistent tmux session:

```bash
tmux new -s sciver-v3-jupyter
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

From a new SSH connection, reconnect with a local tunnel and then attach to the session if needed:

```bash
ssh -L 8888:127.0.0.1:8888 <server>
tmux attach -t sciver-v3-jupyter
```

Open `notebooks/sciver_full_search_v3_server.ipynb`, set its non-secret paths, repository URL, exact commit SHA, run ID, and dataset path, then run cells in order. Do not use a moving branch name in place of the pinned SHA.

## Runtime API values and dry-run

Enter `API_URL` and `API_KEY` only after opening the notebook. The notebook reads environment values when already present or prompts at runtime; the key uses `getpass`. Do not put either value in a notebook cell, shell script, command argument, run directory, or log.

Run the preparation and SEARCH preflight cells before authorizing live work. They validate exact split counts, paper disjointness, hashes, model/generation/parser identity, workload budgets, checkpoint locations, and compatible resume state. Preflight constructs neither a live solver nor a proposer client.

Keep `RUN_SMOKE = False`, `RUN_FULL_SEARCH = False`, and `RUN_FINAL = False` until each stage is separately authorized. SMOKE makes exactly one canonical P0 request from SEARCH, persists only a create-once compatible receipt under the run's isolated `smoke` directory, and never opens FINAL or production SEARCH state.

## SEARCH operation and monitoring

After explicit SMOKE authorization, set `RUN_SMOKE = True` and create or verify the receipt. Then obtain separate production authorization, set `RUN_FULL_SEARCH = True`, and rerun the guarded FULL_SEARCH cell. SEARCH fails closed if its SMOKE receipt is missing or incompatible. The same repository/workspace/run ID resumes only compatible durable SEARCH state. Inspect progress through the notebook's SEARCH status cell; it displays only aggregate state, metrics, hashes, and artifact locations.

For terminal operation, use the thin command wrapper. It never accepts credentials as arguments; any live request uses only inherited process `API_URL` and `API_KEY` values:

```bash
python scripts/run_full_search_v3_server.py search-preflight \
  --repository-root <repository> --run-id <run-id> \
  --search-safe-manifest <search-safe-manifest> \
  --search-records <search-records> --source-commit <pinned-sha>

python scripts/run_full_search_v3_server.py smoke \
  --repository-root <repository> --run-id <run-id> \
  --search-safe-manifest <search-safe-manifest> \
  --search-records <search-records> --source-commit <pinned-sha> \
  --live-smoke

python scripts/run_full_search_v3_server.py search \
  --repository-root <repository> --run-id <run-id> \
  --search-safe-manifest <search-safe-manifest> \
  --search-records <search-records> --source-commit <pinned-sha> \
  --live-search
```

Without `--live-smoke` or `--live-search`, the corresponding command fails closed before reading credentials or dispatching. Neither flag authorizes the other. JSON output is sanitized and may be redirected to a run-local operator log; do not redirect shell environment dumps or notebook output. Check a process lock before starting another runner:

```bash
python scripts/run_full_search_v3_server.py activity \
  --repository-root <repository> --run-id <run-id>
python scripts/run_full_search_v3_server.py search-status \
  --repository-root <repository> --run-id <run-id>
ps -ef | grep '[r]un_full_search_v3_server.py'
```

`activity` is advisory; the M4 lock is the authoritative race-safe gate. If it reports `held`, do not start a second SEARCH or FINAL process. If a process has exited, rerun the same guarded notebook or command with the same durable identity. Completed SEARCH cache entries and FINAL request hashes are reused; incompatible identity is rejected rather than resumed.

Monitor only sanitized command/notebook status and local storage:

```bash
df -h <workspace>
du -sh <workspace>/meta_harness/full_search_v3/<run-id>
```

## Freeze and FINAL

Freeze only after SEARCH reports `patience_stopped` or `max_stopped`. The M6 freeze call is create-once and validates terminal ranking before writing `meta_cot`. FINAL remains locked until that artifact exists.

Run FINAL preflight first. It exposes only safe identities and prompt hashes; it does not display private examples, IDs, labels, requests, or images. Then obtain separate explicit authorization and set `RUN_FINAL = True`, or use:

```bash
python scripts/run_full_search_v3_server.py final \
  --repository-root <repository> --run-id <run-id> \
  --dataset-path <dataset> --private-manifest <private-manifest> \
  --search-safe-manifest <search-safe-manifest> --live-final
```

`--live-final` is separate from `--live-search`; neither authorizes the other. FINAL status and completion receipts are sanitized and paired, and FINAL results cannot alter SEARCH state or the frozen winner.

## Interruptions and safe stopping

For a notebook kernel, SSH, or server interruption, do not delete or edit run artifacts. Restart the kernel/session, restore the same pinned checkout, workspace, run ID, and runtime environment, then rerun preparation and preflight before the appropriate guarded stage. Resume reuses the durable M4/M5 checkpoints and never intentionally redispatches completed work.

To stop an active foreground operation, send one interrupt (`Ctrl-C`) and wait for it to return. For a tmux operation, attach and interrupt there. Do not start a competing process, delete locks, use `kill -9`, or remove cache/checkpoint files. After it exits, inspect `activity`, status, and disk space, then resume with the same identity.
