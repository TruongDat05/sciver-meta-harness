# SciVer Meta-Harness server runbook

This is the authoritative operator runbook for the SciVer-only workflow. The notebook and wrapper compose repository APIs; they do not contain experiment logic. All status output is sanitized. Run from a pinned, clean checkout with dependencies installed and a private local dataset path.

## Setup before JupyterLab

Clone the repository and record its exact 40-character lowercase commit. The
notebook validates that commit, the canonical origin, and a clean worktree
including untracked non-ignored files; it fails closed on a mismatch and never
changes the checkout.

```bash
git clone https://github.com/TruongDat05/sciver-meta-harness.git
cd sciver-meta-harness
git rev-parse HEAD
git remote get-url origin
git status --short --untracked-files=all
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Create the environment and install dependencies before launching JupyterLab.
Open `notebooks/sciver_meta_harness.ipynb` from this checkout and
supply `PINNED_COMMIT_SHA`, `SCIVER_DATASET_PATH`, and `SCIVER_RUN_ID` through
the runtime environment or notebook prompts. Do not save deployment values in
the tracked notebook.

## Runtime values and URL contract

`API_URL` may come from the runtime environment or an ordinary notebook prompt.
`API_KEY` may come from the runtime environment or hidden `getpass` input.
Never place either value in a wrapper argument, notebook source or output,
artifact, receipt, log, or version-controlled file.

Meta-Harness treats `API_URL` as a base URL. It trims surrounding whitespace and redundant trailing slashes, then joins `/models` and `/chat/completions`; it never inserts `/v1`. URLs containing embedded credentials, query strings, or fragments are rejected. This differs from the [generic benchmark guide](remote-benchmark.md), whose `API_URL` is a complete Chat Completions URL.

## Supported wrapper operations

Set only local paths and safe identities; the wrapper reads runtime credentials internally during authorized live work.

```bash
ROOT='.'
RUN_ID='<run-id>'
SEARCH_MANIFEST="workspace/meta_harness/full_search_v3/${RUN_ID}/preparation/search/search_safe_manifest.json"
SEARCH_RECORDS="workspace/meta_harness/full_search_v3/${RUN_ID}/preparation/search/search_records.json"
PRIVATE_MANIFEST="workspace/meta_harness/full_search_v3/${RUN_ID}/preparation/private/private_split_manifest.json"
DATASET='/absolute/path/to/sciver.json'
COMMIT='<40-lowercase-hex-commit>'
```

```bash
# prepare
python scripts/run_meta_harness.py prepare --repository-root "$ROOT" --run-id "$RUN_ID" --dataset-path "$DATASET"

# offline SEARCH preflight
python scripts/run_meta_harness.py search-preflight --repository-root "$ROOT" --run-id "$RUN_ID" --search-safe-manifest "$SEARCH_MANIFEST" --search-records "$SEARCH_RECORDS" --source-commit "$COMMIT"

# isolated one-request smoke; explicit authorization
python scripts/run_meta_harness.py smoke --repository-root "$ROOT" --run-id "$RUN_ID" --search-safe-manifest "$SEARCH_MANIFEST" --search-records "$SEARCH_RECORDS" --source-commit "$COMMIT" --live-smoke

# SEARCH; separate explicit authorization
python scripts/run_meta_harness.py search --repository-root "$ROOT" --run-id "$RUN_ID" --search-safe-manifest "$SEARCH_MANIFEST" --search-records "$SEARCH_RECORDS" --source-commit "$COMMIT" --live-search

# inspection and terminal-winner freeze
python scripts/run_meta_harness.py activity --repository-root "$ROOT" --run-id "$RUN_ID"
python scripts/run_meta_harness.py search-status --repository-root "$ROOT" --run-id "$RUN_ID"
python scripts/run_meta_harness.py freeze --repository-root "$ROOT" --run-id "$RUN_ID"

# offline FINAL preflight, paired FINAL authorization, and status
python scripts/run_meta_harness.py final-preflight --repository-root "$ROOT" --run-id "$RUN_ID" --dataset-path "$DATASET" --private-manifest "$PRIVATE_MANIFEST" --search-safe-manifest "$SEARCH_MANIFEST" --solver-identity-sha256 '<64-lowercase-hex-sha256>'
python scripts/run_meta_harness.py final --repository-root "$ROOT" --run-id "$RUN_ID" --dataset-path "$DATASET" --private-manifest "$PRIVATE_MANIFEST" --search-safe-manifest "$SEARCH_MANIFEST" --live-final
python scripts/run_meta_harness.py final-status --repository-root "$ROOT" --run-id "$RUN_ID"
```

Without `--live-smoke`, `--live-search`, or `--live-final`, the matching live operation fails without dispatch. `search-preflight`, `activity`, `search-status`, `freeze`, `final-preflight`, and `final-status` do not read credentials or create a live client.

## Notebook operation

Open `notebooks/sciver_meta_harness.ipynb` only after manually preparing the checkout and environment. It validates the pinned commit, canonical origin, clean worktree, imports, and dataset path; it never changes the checkout or installs dependencies.

The notebook has three independent confirmations:

1. `RUN_LIVE_SMOKE` authorizes model-list validation and exactly one canonical P0 SEARCH-safe request.
2. `RUN_FULL_SEARCH` authorizes SEARCH.
3. `RUN_FINAL_ONCE` authorizes paired FINAL after a frozen winner and successful FINAL preflight.

Leaving any confirmation empty skips its live stage. Before a confirmed live
stage, the notebook reads `API_URL` and `API_KEY` at runtime only. Its
stage-local variables are cleared after the stage; existing environment
variables are not removed.

## Monitoring and safe resume

```text
workspace/meta_harness/full_search_v3/<run-id>/
├── preparation/
├── smoke/completion.json
├── search_cache/
├── orchestration_state.json
├── freeze/frozen_winner.json
└── final/
```

Use `activity`, `search-status`, and `final-status` rather than competing runners. To interrupt, send one `Ctrl-C`, wait for return, inspect state, and invoke the same stage with the same checkout, run ID, split, and compatible identity. Do not delete locks, caches, checkpoints, receipts, or immutable artifacts. Incompatible state fails closed.
