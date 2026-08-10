# SciVer Meta-Harness guide

## Scope

Meta-Harness is an additive, prompt-only optimization workflow for SciVer.
A candidate is data containing exactly four `string.Template` sources:
`direct`, `analytical`, `parallel`, and `sequential`. Candidates cannot change
the solver, claims, context, captions, evidence or image order, request shape,
labels, parser, metrics, or evaluation procedure.

The implemented lifecycle is:

```text
prepare paper split
  -> evaluate unchanged cot baseline on the reserved search papers
  -> build immutable 50-100 example hard-search manifest
  -> propose two prompt families from full search history
  -> static/schema checks
  -> 10-example search smoke
  -> complete the 80-example search evaluation
  -> repeat for the fixed search budget
  -> promote only search top-K to protected validation
  -> select the protected-validation winner
  -> freeze winner as meta_cot
  -> explicitly confirm one frozen-winner final-test execution
```

Final-test outcomes never select or modify a prompt. The proposer working
directory contains search experience only; protected-validation and final-test
records, labels, traces, metrics, and paths are absent. This implementation
does not fine-tune models, export weights, or download models. Archived v1/v2
runs keep their original validation-only behavior and remain read-only.

## Architecture

The implementation reuses production SciVer boundaries:

| Layer | Responsibility |
| --- | --- |
| `meta_harness/baseline.py` | Validate or atomically export `baseline_cot.json` from canonical `COT_PROMPT` text |
| `utils/dataset_adapters.py` | Load and validate released or normalized SciVer records |
| `meta_harness/split_manager.py` | Build and verify deterministic paper-disjoint splits |
| `meta_harness/hard_search.py` | Select deterministic baseline-error-heavy search examples with guard and diversity constraints |
| `meta_harness/experience_store.py` | Persist trusted search traces and a separate proposer-safe queryable view |
| `meta_harness/proposer/codex_cli.py` | Invoke the local read-only proposer through an injectable subprocess boundary |
| `meta_harness/schemas.py` | Validate candidate shape, placeholders, hashes, and forbidden content |
| `meta_harness/candidate_store.py` | Create immutable candidate files and an atomic status registry |
| `meta_harness/evaluator.py` | Reuse request preparation, solver interface, parser, writer, and metrics |
| `meta_harness/orchestrator.py` | Run baseline/search, budgets, frontier, early stopping, locking, and resume |
| `meta_harness/staged_orchestrator.py` | Run smoke/search/promotion/protected-validation stages with per-stage budgets |
| `meta_harness/finalize.py` | Select and freeze the winner, guard one-time final execution, and export manifests |
| `utils/prompt_registry.py` | Keep `cot` unchanged and register a verified frozen artifact as `meta_cot` |

`scripts/prepare_meta_harness_data.py`, `scripts/run_meta_harness.py`,
`scripts/finalize_meta_harness.py`, and
`scripts/run_meta_cot_transfer.py` are the command-line entry points. Importing
them does not create clients, launch subprocesses, or make requests.

## Installation and environment

Use the repository environment:

```bash
python -m pip install -r requirements.txt
python -m compileall -q .
pytest -q
```

Offline preparation, dry-run, freezing, artifact inspection, and tests need no
credentials. Create the ignored local environment file with:

```bash
cp .env.example .env
```

Fill `API_URL` with the compatible endpoint and `API_KEY` with its credential.
Live CLI commands load this file without overriding values already present in
the shell. Offline and dry-run commands do not load it. Do not place credential
values in JSON configuration, shell history, source files, candidate text, or
result metadata. `API_TIMEOUT_SECONDS` and `API_MAX_RETRIES` are optional and
retain their existing defaults when omitted.

The proposer subprocess receives a minimal environment with `API_URL` and
`API_KEY` removed. It uses local CLI authentication state, not solver
credentials. The proposer supports `gpt-5.6-sol` and configurable reasoning
effort; both values are part of immutable run identity.

The example search configuration is
`configs/meta_harness/sciver_gemma_codex.example.json`. The remote
OpenAI-compatible search supports `gemma-4-26B-A4B-it` and
`gemma-4-31B-it`. A model change always changes the config hash and requires a
new run ID.
`generation.temperature` and
`generation.max_tokens` must remain `null`, matching the existing request
interface. Changing the seed, split, model, evaluation procedure, or early
stopping configuration on resume is rejected.

## Dataset preparation and split rules

Run preparation before search. This is the only pre-finalization step that
reads the complete selected SciVer corpus:

```bash
python scripts/prepare_meta_harness_data.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --split-manifest workspace/meta_harness/data/staged-26b/split.json \
  --validation-output workspace/meta_harness/data/staged-26b/validation.json \
  --reserved-search-output workspace/meta_harness/data/staged-26b/reserved-search.json \
  --baseline-search-results /absolute/path/to/26b-search-baseline.results.jsonl \
  --hard-search-manifest workspace/meta_harness/data/staged-26b/hard-search.json \
  --search-output workspace/meta_harness/data/staged-26b/search.json
```

Add `--evidence-dir /absolute/path/to/evidence` only when the selected local
schema requires it.

Preparation loads records with the production adapter and groups claims by:

1. a verified `paper_id`, when present; otherwise
2. the full SHA-256 of the referenced local paper JSON bytes.

It never falls back to claim identity. Missing or unreadable paper data is an
error. The default deterministic group allocation uses seed `42` and target
ratios 20% search, 20% validation, and 60% final test. Whole-paper grouping can
make achieved sample ratios differ from the targets. The manifest verifies
that sample IDs and paper groups are pairwise disjoint.

After the reserved-search baseline has been evaluated offline or through an
explicitly gated run, preparation can atomically create:

- an immutable full split manifest; and
- a normalized file containing exactly protected-validation records;
- an immutable hard-search manifest; and
- a normalized file containing exactly the selected search records.

The hard-search builder selects 50-100 examples, prioritizes baseline errors,
keeps 20-30% baseline-correct guards, and balances labels, evidence modality,
reasoning methods, and papers. It refuses incomplete baseline results or any
record outside the already-reserved search paper groups.

Search and search dry-run reject a dataset file containing any search or
final-test sample. Finalization without confirmation reads only the split
manifest and run artifacts. Final-test data is loaded only when both
`--confirm-final-test` and `--live-api` are present.

## Verified commands

The command shapes below are covered by offline parser, fake-subprocess, and
fake-solver tests. Live commands were intentionally not executed during
implementation.

### Offline dry-run

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/staged-26b/split.json \
  --hard-search-manifest workspace/meta_harness/data/staged-26b/hard-search.json \
  --search-dataset-path workspace/meta_harness/data/staged-26b/search.json \
  --dataset-path workspace/meta_harness/data/staged-26b/validation.json \
  --run-id sciver-meta-26b-staged-pilot-v1 \
  --dry-run
```

Dry-run validates configuration, manifest hashes, and exact validation
membership; prints the planned upper bound; creates no run directory; and
calls no proposer, solver, network, or final-test path.

### Staged 26B pilot

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/staged-26b/split.json \
  --hard-search-manifest workspace/meta_harness/data/staged-26b/hard-search.json \
  --search-dataset-path workspace/meta_harness/data/staged-26b/search.json \
  --dataset-path workspace/meta_harness/data/staged-26b/validation.json \
  --run-id sciver-meta-26b-staged-pilot-v1 \
  --live-api
```

### New staged 31B run

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_31b_meta.example.json \
  --split-manifest workspace/meta_harness/data/staged-31b/split.json \
  --hard-search-manifest workspace/meta_harness/data/staged-31b/hard-search.json \
  --search-dataset-path workspace/meta_harness/data/staged-31b/search.json \
  --dataset-path workspace/meta_harness/data/staged-31b/validation.json \
  --run-id sciver-meta-31b-staged-v1 \
  --live-api
```

### Resume

Resume with the same configuration and no repeated completed work:

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/staged-26b/split.json \
  --hard-search-manifest workspace/meta_harness/data/staged-26b/hard-search.json \
  --search-dataset-path workspace/meta_harness/data/staged-26b/search.json \
  --dataset-path workspace/meta_harness/data/staged-26b/validation.json \
  --resume sciver-meta-26b-staged-pilot-v1 \
  --live-api
```

Staged resume accepts no identity changes. It skips completed candidates and
re-enters only a checkpointed API-failure stage. A model, split, parser,
generation, proposer, or protocol change requires a new run ID.

### Freeze the winner

This command selects by validation Macro-F1, then accuracy, observed tokens,
and candidate ID, writes the immutable artifact, and does not open final-test
data:

```bash
python scripts/finalize_meta_harness.py \
  --run-id sciver-meta-26b-staged-pilot-v1 \
  --split-manifest workspace/meta_harness/data/staged-26b/split.json
```

### Verify `cot` and `meta_cot` independently offline

The original `cot` family needs no artifact:

```bash
python main.py \
  --provider remote \
  --dataset SciVer \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --model gemma-4-26B-A4B-it \
  --prompt cot \
  --method cot \
  --max-num 1 \
  --dry-run
```

A fresh process registers `meta_cot` only from the verified frozen artifact:

```bash
python main.py \
  --provider remote \
  --dataset SciVer \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --model gemma-4-26B-A4B-it \
  --prompt meta_cot \
  --meta-cot-artifact workspace/meta_harness/runs/sciver-gemma-meta-main-v1/finalization/frozen_winner.json \
  --method cot \
  --max-num 1 \
  --dry-run
```

`--method cot` is retained for backward-compatible output naming. The result
field `prompt_variant` distinguishes `cot` from `meta_cot`; the result fields
`method` and `reasoning_method` remain one of the four reasoning methods.

### One-time Gemma final test

Do not run this command until the winner and repository revision have been
reviewed:

```bash
python scripts/finalize_meta_harness.py \
  --run-id sciver-meta-26b-staged-pilot-v1 \
  --split-manifest workspace/meta_harness/data/staged-26b/split.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --confirm-final-test \
  --live-api
```

For a staged run, only the exact frozen `meta_cot` winner executes once. Atomic
completion receipts make a retry return prior results instead of dispatching
again. Archived legacy runs retain their original paired `cot`/`meta_cot`
finalization contract.

### Archived cross-model transfer

Transfer is enabled only after the Gemma search-model completion receipt exists:

```bash
python scripts/run_meta_cot_transfer.py \
  --run-id sciver-gemma-meta-main-v1 \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --confirm-final-test \
  --live-api
```

The command evaluates unchanged `cot` and the same frozen `meta_cot` bytes once
on `Qwen2.5-VL-7B-Instruct`, `gemma-4-31B-it`, and `gemma-3-27b-it`.
Transfer scores cannot update selection or trigger new proposals.

## Ranking, budgets, and early stopping

Only candidates with complete request and parse coverage are rankable. Search
ranking uses Macro-F1, solver calls, tokens, latency, and candidate ID. After
the fixed search budget, only `promotion_top_k` candidates receive protected
validation; the winner is selected from those validation scores by Macro-F1,
accuracy, tokens, and candidate ID.

The v2 run state reports separate `search`, `protected_validation`, and
`final_test` budgets. Dry-run reports calls and estimated tokens for all three
stages without opening final data. `early_stopping_patience: null` disables
performance stopping. When enabled, it cannot stop before `min_iterations`.
Unresolved API failures checkpoint a `retry_pending` stage; resume retries only
those failed requests and never repeats completed samples or candidates.

## Output layout

```text
workspace/meta_harness/
├── data/
│   ├── sciver_split.json
│   └── sciver_validation.json
└── runs/<RUN_ID>/
    ├── .run.lock
    ├── run_state.json
    ├── candidate_registry.json
    ├── candidates/<CANDIDATE_ID>/candidate.json
    ├── proposer/iteration_####/attempt_#####.json
    ├── iterations/iteration_####/proposal.json
    ├── evaluations/<CANDIDATE_ID>/
    │   ├── validation.results.jsonl
    │   └── validation.metrics.json
    └── finalization/
        ├── frozen_winner.json
        ├── results_manifest.json
        └── executions/<MODEL>/
            ├── manifest.json
            ├── results.jsonl
            ├── metrics.json
            ├── results.csv
            ├── completion.json
            └── cot/
                ├── manifest.json
                ├── results.jsonl
                ├── metrics.json
                ├── results.csv
                └── completion.json
```

`run_state.json` includes the transition number, current iteration, immutable
configuration hash, consumed and configured budgets, candidate statuses,
scores, frontier, failures, early-stopping state, and sanitized proposer
metadata.

Candidate files contain ID, parent, search axis, hypothesis, the exact four
templates, expected trade-off, and prompt-source SHA-256. They are canonical,
create-once artifacts.

Evaluation JSONL records include:

- run, sample, dataset, model, prompt variant, candidate, and reasoning-method
  identity;
- local gold label, parsed prediction, parse and request status;
- sanitized raw response/error fields;
- latency, available usage, timestamp, and attempt count.

Metrics JSON contains prompt/candidate/config/split hashes, evaluator and
parser versions, sample IDs, Accuracy, per-class precision/recall/F1,
Macro-F1, coverage, confusion matrix, failure counts, and rankability.

The frozen winner contains the four templates, candidate and prompt hashes,
validation selection score, search solver configuration, split hashes, code
revision, and run ID. Completion receipts and manifests hash every JSON,
JSONL, and CSV result artifact.

## Safety and isolation

- Search accepts validation-only data and always requests split
  `validation`.
- The proposer receives parent prompts, aggregate validation scores, aggregate
  resource metrics, and sanitized failure counts only.
- Final-test IDs, records, labels, predictions, traces, and paths are never in
  proposer input.
- Gold-only fields are removed before request preparation.
- Image bytes are serialized only in the transient solver request; raw base64
  is never written to candidates, state, diagnostics, results, or manifests.
- `API_KEY`, `API_URL`, authorization values, endpoint strings, and long
  base64-like text are redacted from persisted failures.
- The local proposer uses an argument list with `shell=False`, an isolated
  temporary working directory, read-only sandboxing, a schema, timeout, and
  output-size limit.
- Every live solver path requires `--live-api`; final-test and transfer paths
  additionally require `--confirm-final-test`.
- Tests remove credentials, force model-library offline modes, and deny HTTP
  and socket connections suite-wide.
- No Meta-Harness path loads model weights.

## Reproducibility checklist

Before a main search:

- Confirm the branch and review a clean, intentional diff.
- Run `python -m compileall -q .`, `pytest -q`, and `git diff --check`.
- Preserve the baseline prompt fingerprint and pass baseline-equivalence
  tests.
- Record the configuration file, split hash, dataset hash, validation input
  hash, and run ID.
- Confirm validation input contains no search or final-test records.
- Review dry-run workload and choose fixed ceilings.
- Confirm local proposer authentication without exposing solver credentials.
- Keep `temperature` and `max_tokens` null.

Before finalization:

- Confirm the run is stopped and has a rankable validation candidate.
- Review candidate hashes, frontier, failure records, and budget ledger.
- Ensure the repository HEAD identifies the reviewed code. The frozen
  revision records HEAD, not uncommitted working-tree content.
- Freeze without `--confirm-final-test` first and archive
  `frozen_winner.json`.
- Verify `cot` and `meta_cot` dry-runs independently.
- Only then authorize the one-time Gemma final test.

Before transfer:

- Verify the Gemma search-model `completion.json` and all artifact hashes.
- Confirm every model manifest has the same prompt and split hashes.
- Do not edit the frozen artifact or prompt text.
- Treat transfer metrics as evaluation output, never search feedback.

## Troubleshooting

`dataset-path must contain exactly the fixed validation split`

: Run `prepare_meta_harness_data.py` and pass its
  `--validation-output`, not the full corpus, to search.

`run state already exists`

: Use `--resume RUN_ID`; do not reuse `--run-id`.

`resume configuration changed immutable fields`

: Restore the original configuration and split. A changed seed, model, sample
  membership, early-stopping policy, or evaluation procedure requires a new
  run ID.

`resource limits changed without explicit safe marking`

: Add one `--safe-resume-change` per increased limit. Decreases are rejected.

`another process is already writing this run`

: Let the active writer finish. Do not remove the lock while a process may
  still own it; the lock file itself is harmless after process exit.

`run has no rankable validation candidate`

: Inspect coverage and failure counts. Do not lower eligibility or use
  final-test results to repair selection.

`--prompt meta_cot requires --meta-cot-artifact`

: Pass the frozen winner from the intended run. A fresh process never guesses
  or silently selects a winner.

`the original-model final test must complete before transfer`

: Complete and verify the guarded Gemma execution first.

`completion artifact hash mismatch`

: Stop. Preserve the files for audit and do not rerun or overwrite the
  one-time result.

## Current limitations

- V1 search is SciVer-only and evaluates every valid candidate on the fixed
  validation split; it does not implement a separate search-screening stage.
- Candidate count is fixed at two per iteration.
- Generation controls remain omitted because the existing request interface
  does not send them.
- Token accounting depends on response usage metadata and is enforced between
  durable candidate transitions.
- Cost budgets and proposer-call budgets are not implemented.
- The frozen `meta_cot` family is registered from an explicit artifact path;
  it is not copied into source control automatically.
- Finalization evaluates only the frozen winner, not a new `cot` baseline.
- Final-test and transfer commands are live-only and were not executed during
  implementation or tests.
