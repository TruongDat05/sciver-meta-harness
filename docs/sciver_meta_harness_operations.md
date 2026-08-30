# SciVer Meta-Harness operations runbook

This is the authoritative operator runbook for the SciVer-only
`sciver_full_search_v3` workflow. It documents how the deterministic data split
is produced and how to run the smoke, SEARCH, and paired FINAL stages. The root
launchers, the explicit wrapper, and the notebook compose supported repository
entry points; none of them reimplement experiment logic. All live stages are
explicit opt-in, and `API_URL`/`API_KEY` are runtime-only.

Run everything from a pinned, clean checkout with dependencies installed and a
private local dataset path.

## Fixed protocol and identity

- Protocol: `sciver_full_search_v3`.
- Split: exactly 2,000 SciVer samples with seed 42 → 1,000 paper-disjoint
  SEARCH records and 1,000 paper-disjoint FINAL records. Membership is
  deterministic and independent of labels, predictions, difficulty, and
  baseline errors. Preparation fails closed if exact counts or paper
  disjointness cannot be met.
- Solver: fixed `gemma-4-26B-A4B-it` with `temperature=0`, `top_p=1`, seed 42,
  `n=1`, non-streaming, `max_tokens=8192` (override with `SCIVER_SOLVER_MAX_TOKENS`).
- Proposer: `gpt-5.6-sol` with reasoning effort `high`.
- Evaluation: canonical P0 plus one valid prompt candidate per SEARCH iteration
  on the same complete 1,000-record SEARCH split; 38–50 iterations, patience 8,
  at most three proposal attempts per iteration. Ranking is SEARCH Macro-F1
  descending, Accuracy descending, prompt SHA-256 ascending, candidate ID
  ascending. The top-5 SEARCH candidates are frozen as additive `meta_cot`
  (P0 kept separate). FINAL evaluates frozen P0 and all 5 frozen candidates
  exactly once each on the identical 1,000 FINAL IDs.
- Workload ceiling: 51,000 SEARCH evaluations + 6,000 FINAL evaluations
  (57,000 logical solver evaluations total); transport retries excluded.

## Setup before running

Clone the repository, record its exact 40-character lowercase commit, and
confirm a clean worktree (the launchers and notebook validate this and fail
closed on a mismatch). Create the environment and install dependencies:

```bash
git clone https://github.com/TruongDat05/sciver-meta-harness.git
cd sciver-meta-harness
git rev-parse HEAD
git remote get-url origin
git status --short --untracked-files=all
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

### Configuration: `.env`

All runtime configuration is read from the repository `.env` file (the root
launchers and notebook load it). Provide these keys; never commit real values:

| Key | Purpose |
| --- | --- |
| `API_URL` | Base URL. The harness joins `/models` and `/chat/completions` itself; it never inserts `/v1`. No query strings or fragments. |
| `API_KEY` | Credential. Never printed, logged, or persisted to artifacts. |
| `PINNED_COMMIT_SHA` | Exact 40-char lowercase commit of the pinned clean checkout. |
| `SCIVER_DATASET_PATH` | Absolute path to the SciVer dataset JSON. |
| `SCIVER_RUN_ID` | Logical run identifier; artifacts land under this ID. |
| `SCIVER_SOLVER_MAX_TOKENS` | Optional solver `max_tokens` override (default 8192). |

A `.env` example is checked in with placeholder values. `API_KEY` and `API_URL`
may instead come from the process environment; existing environment values are
never overwritten.

## Data preparation and split

The split is created from **one** released SciVer dataset JSON. The input is
the default `data/sciver/testset.json`, or the absolute path in
`SCIVER_DATASET_PATH` / `--dataset-path`.

Preparation (`prepare_run`) is idempotent and fail-closed: it reads the source
dataset, hashes it, groups samples by verified paper identity, and picks the
deterministic paper-disjoint 1,000/1,000 SEARCH/FINAL split with seed 42. It
fails (rather than silently shrinking) if exact counts or paper disjointness
cannot be met, and it refuses to overwrite a different immutable artifact on
resume.

Artifacts under the run root:

```text
workspace/meta_harness/full_search_v3/<run-id>/
├── preparation/
│   ├── search/search_safe_manifest.json
│   ├── search/search_records.json        # immutable model-visible SEARCH order
│   └── private/private_split_manifest.json
├── smoke/completion.json
├── search_cache/
├── orchestration_state.json
├── freeze/frozen_winner.json
└── final/
```

SEARCH consumes only the SEARCH-safe artifacts; the private manifest and any
FINAL materialization stay outside the SEARCH/proposer boundary.

## Running stages

### Recommended: zero-config root launchers

`smoke.py`, `search.py`, and `final.py` resolve the repository root, the run
identity, the dataset, and the preparation layout internally. Defaults:
run-id `official_v3`; dataset `data/sciver/testset.json`; `source_commit`
resolved from Git HEAD.

Authorization is required per stage: pass the `--live-*` flag **or** set the
matching environment variable to exactly `1`
(`RUN_LIVE_SMOKE`, `RUN_FULL_SEARCH`, `RUN_FINAL_ONCE`). Without it the live
stage fails closed and nothing is dispatched.

```bash
# 1. SMOKE — auto-prepares the split, then one isolated canonical P0 request
python smoke.py --live-smoke
python smoke.py --live-smoke --run-id "<run-id>"          # override run
python smoke.py --live-smoke --dataset-path /abs/sciver.json

# 2. SEARCH — requires a completed SMOKE receipt; starts or resumes
python search.py --live-search
python search.py --live-search --run-id "<run-id>"

# 3. FINAL — requires terminal SEARCH; freezes top-5, then evaluates P0 + 5 frozen
python final.py --live-final
python final.py --live-final --run-id "<run-id>" --dataset-path /abs/sciver.json
```

`smoke.py` runs preparation automatically. `search.py` and `final.py` reject a
missing SEARCH/FINAL artifact with a message directing you to run the earlier
stage first.

### Explicit wrapper (`scripts/run_meta_harness.py`)

The same operations are available through the verbose, fully explicit wrapper.
Set non-secret local paths first:

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

Without `--live-smoke`, `--live-search`, or `--live-final`, the matching live
operation fails without dispatch. `search-preflight`, `activity`,
`search-status`, `freeze`, `final-preflight`, and `final-status` do not read
credentials or create a live client.

### Notebook operation

Open `notebooks/sciver_meta_harness.ipynb` after manually preparing the
checkout, `.env`, and environment. The notebook reads all runtime settings from
`.env` (`PINNED_COMMIT_SHA`, `SCIVER_DATASET_PATH`, `SCIVER_RUN_ID`, `API_URL`,
`API_KEY`), so no interactive typing is required. Live stages run automatically
when you execute their ordered cells:

1. Preparation and offline SEARCH preflight.
2. Live SMOKE (model-list validation + one canonical P0 request) and receipt
   validation.
3. SEARCH (auto-started or resumed) and status.
4. Freeze the terminal SEARCH top-5 candidates.
5. FINAL preflight, then the P0 + top-5 FINAL run and status.
6. A comparison of baseline (`cot`) vs the frozen `meta_cot` candidates (top-5)
   FINAL results.

The notebook never changes the checkout, installs dependencies, or persists
credentials. Run the root launchers instead if you prefer a terminal workflow.

## Monitoring and safe resume

Use `activity`, `search-status`, and `final-status` (or `search.py --live-search`
/ `final.py --live-final` which resume) rather than competing runners. To
interrupt, send one `Ctrl-C`, wait for return, inspect state, then invoke the
same stage with the same checkout, run ID, split, and compatible identity. Do
not delete locks, caches, checkpoints, receipts, or immutable artifacts.
Incompatible resume state fails closed.

## Safety

All live stages are explicit opt-in. Imports, preparation, preflight, tests,
and status inspection are offline. Tests mock all HTTP and proposer subprocess
boundaries. `API_URL` and `API_KEY` are runtime-only and must never be
committed, logged, passed as wrapper arguments, or written to artifacts.
