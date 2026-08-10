# SciVer Meta-Harness terminal guide

This guide covers the implemented SciVer-only prompt search on branch
`feature/sciver-meta-harness-api`. It never requires a live request for setup,
split preparation, dry-run, offline tests, inspection, freezing, or export.
The implementation-to-test audit is recorded in the
[requirement trace](meta_harness_requirement_trace.md).

The commands below were checked against the repository entry-point `--help`
output. Codex commands were checked with `codex-cli 0.145.0`; installation and
ChatGPT login syntax also matches the current
[Codex CLI guide](https://developers.openai.com/codex/cli/) and
[authentication guide](https://developers.openai.com/codex/auth/).

## Cost and access labels

- **OFFLINE**: no Codex proposal, solver request, or final-test access.
- **CODEX QUOTA**: invokes `codex exec` and consumes the authenticated Codex
  account's allowance.
- **API CREDITS**: sends requests to the configured solver endpoint.
- **FINAL TEST — ONE TIME**: opens the held-out split and creates immutable
  completion receipts. Do not run while reviewing setup.

No command in this guide contains a real credential.

## Clone and check out the branch

**OFFLINE** after the initial clone:

```bash
git clone https://github.com/TruongDat05/SciVer.git
cd SciVer
git switch feature/sciver-meta-harness-api
git branch --show-current
git status --short
git merge-base --is-ancestor feature/unified-benchmark-api HEAD
```

The branch command must print `feature/sciver-meta-harness-api`; the ancestry
command must exit with status 0.

## Create the Python environment

Use Python 3.10, matching the repository quickstart:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
python -m pytest --version
```

Dependency installation uses package indexes but does not download model
weights. Meta-Harness uses only the remote solver path.

## Install and authenticate Codex CLI

Install or update Codex on Linux/macOS:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex --version
codex exec --help
codex login --help
```

Authenticate the proposer with the browser-based ChatGPT flow, then verify the
cached session:

```bash
codex login
codex login status
```

Authentication itself does not run a proposal. Search later invokes Codex with
an ephemeral session, an isolated temporary working directory,
`--sandbox read-only`, `--ignore-user-config`, `--ignore-rules`, structured
output, and a minimal environment. It explicitly selects `gpt-5.6-terra` with
`medium` reasoning. The solver's `API_KEY` and `API_URL` are removed from the
child environment.

## Configure the solver API

Copy the checked-in placeholder file:

```bash
cp .env.example .env
```

Edit `.env` locally so it contains the exact environment names used by the
code:

```dotenv
API_URL=<compatible-api-endpoint>
API_KEY=<unmistakably-fake-placeholder-replace-locally>
```

Do not commit `.env`. Shell values take precedence over the file. Optional
remote-client controls are `API_TIMEOUT_SECONDS` and `API_MAX_RETRIES`; they
are not required by Meta-Harness configuration.

Confirm only that the names are present, without printing their values:

```bash
python -c 'from pathlib import Path; names={line.split("=",1)[0] for line in Path(".env").read_text().splitlines() if "=" in line}; assert {"API_URL","API_KEY"} <= names'
```

## Prepare SciVer and the deterministic split

The input must be a released or normalized SciVer dataset accepted by the
production SciVer adapter. Every record must expose a verified `paper_id`, or
reference a readable local paper JSON whose complete bytes provide the paper
SHA-256 identity. Claim-level fallback is forbidden.

Review the frozen example configuration:

```bash
python -m json.tool configs/meta_harness/sciver_gemma_codex.example.json
```

It fixes:

- remote OpenAI-compatible model `gemma-4-26B-A4B-it`;
- Codex proposer model `gpt-5.6-terra` with `medium` reasoning;
- seed `42`;
- paper-group targets 20% search, 20% validation, 60% final test; and
- omitted generation overrides (`null` temperature and max tokens), preserving
  the existing request interface.

Prepare the immutable split manifest and validation-only search file:

```bash
python scripts/prepare_meta_harness_data.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --validation-output workspace/meta_harness/data/sciver_validation.json
```

If the selected released schema needs separate evidence, add:

```bash
python scripts/prepare_meta_harness_data.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --evidence-dir /absolute/path/to/SciVer/evidence \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --validation-output workspace/meta_harness/data/sciver_validation.json
```

Both commands are **OFFLINE**. Existing files are accepted only when their
canonical content is identical; different split or validation data is never
overwritten.

## Validate configuration and dry-run

The search dry-run is the implemented configuration validator. It verifies
the config hash, split hash, exact validation membership, model, frozen
generation settings, budgets, and planned request upper bound. It creates no
run directory.

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --run-id sciver-gemma-meta-dry-run \
  --dry-run \
  --max-iterations 2 \
  --max-candidates 4
```

This command is **OFFLINE** and does not load `.env`, construct an HTTP client,
or invoke Codex.

## Run the fake proposer/fake solver integration test

This is the complete offline workflow test, including search, resume, freeze,
one-time receipt behavior, and exact prompt transfer:

```bash
python -m pytest -q tests/meta_harness/test_full_pipeline_offline.py
```

Run the final-matrix and safety regressions too:

```bash
python -m pytest -q \
  tests/meta_harness/test_codex_cli.py \
  tests/meta_harness/test_orchestrator.py \
  tests/meta_harness/test_finalize.py
```

Both commands are **OFFLINE**. The suite-wide test guard denies socket and HTTP
access and uses fake subprocess/solver boundaries.

## Run a live solver smoke test

Review the dry-run output first.

**API CREDITS** — one validation example; no Codex proposal and no final-test
access:

```bash
python main.py \
  --provider remote \
  --dataset SciVer \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --model gemma-4-26B-A4B-it \
  --prompt cot \
  --method cot \
  --max-num 1 \
  --output-dir workspace/meta_harness/smoke \
  --live-api
```

Inspect the resulting JSONL before starting prompt search. Confirm that parsing
succeeds and no credential or image base64 appears in the file.

## Run a pilot search

**CODEX QUOTA + API CREDITS** — every candidate is evaluated on the fixed
validation split. The ceilings below are hard safeguards, not cost estimates:

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --run-id sciver-gemma-meta-pilot-v1 \
  --max-iterations 5 \
  --max-candidates 10 \
  --max-solver-calls 100000 \
  --max-tokens 20000000 \
  --max-wall-time 7200 \
  --max-consecutive-failures 3 \
  --patience 3 \
  --min-delta 0.005 \
  --min-iterations 5 \
  --live-api
```

Lower the resource ceilings after using the dry-run request count and live
smoke usage. Do not proceed if parse coverage is below 1.0 or unresolved API
failures remain.

## Run the main search

**CODEX QUOTA + API CREDITS**:

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --run-id sciver-gemma-meta-main-v1 \
  --max-iterations 15 \
  --max-candidates 30 \
  --max-solver-calls 100000 \
  --max-tokens 100000000 \
  --max-wall-time 86400 \
  --max-consecutive-failures 3 \
  --patience 3 \
  --min-delta 0.005 \
  --min-iterations 5 \
  --live-api
```

Search uses only `gemma-4-26B-A4B-it` and exactly the validation records in
`sciver_validation.json`. It never loads final-test samples.

## Resume an interrupted run

Use the same config, split, validation file, and frozen settings:

**CODEX QUOTA + API CREDITS** only for unfinished work:

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --resume sciver-gemma-meta-main-v1 \
  --live-api
```

To increase a hard ceiling, repeat its new value and explicitly approve only
that non-decreasing change:

```bash
python scripts/run_meta_harness.py \
  --config configs/meta_harness/sciver_gemma_codex.example.json \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path workspace/meta_harness/data/sciver_validation.json \
  --resume sciver-gemma-meta-main-v1 \
  --max-iterations 15 \
  --max-candidates 30 \
  --safe-resume-change max_iterations \
  --safe-resume-change max_candidates \
  --live-api
```

A frozen run cannot resume evolution, even with a resource extension.

## Inspect candidates, scores, and the Pareto frontier

All commands in this section are **OFFLINE**:

```bash
python -m json.tool \
  workspace/meta_harness/runs/sciver-gemma-meta-main-v1/run_state.json
```

```bash
find workspace/meta_harness/runs/sciver-gemma-meta-main-v1/candidates \
  -mindepth 2 -maxdepth 2 -name candidate.json -print | sort
```

```bash
python -c 'import json; p="workspace/meta_harness/runs/sciver-gemma-meta-main-v1/run_state.json"; s=json.load(open(p, encoding="utf-8")); print(json.dumps({"best_candidate_id":s["best_candidate_id"],"frontier":s["frontier"],"scores":s["scores"],"budgets":s["budgets"],"failures":s["failures"]}, indent=2, sort_keys=True))'
```

Rankable candidates require parse coverage 1.0 and zero unresolved API
failures. The winner is chosen across all completed candidates by validation
Macro-F1 descending, accuracy descending, observed tokens ascending, then
candidate ID ascending. The Pareto frontier separately tracks Macro-F1 versus
solver calls, tokens, and latency.

## Freeze the global winner

This command is **OFFLINE** and does not open the final-test dataset:

```bash
python scripts/finalize_meta_harness.py \
  --run-id sciver-gemma-meta-main-v1 \
  --split-manifest workspace/meta_harness/data/sciver_split.json
```

Review and archive:

```bash
python -m json.tool \
  workspace/meta_harness/runs/sciver-gemma-meta-main-v1/finalization/frozen_winner.json
```

Freezing records candidate, prompt, config, split, evaluator, parser, metric,
and code-revision identities. Later final results cannot change selection.

## Export and register `meta_cot`

Export the exact four frozen templates and provider-neutral metadata:

```bash
python scripts/export_meta_cot.py \
  --frozen-winner workspace/meta_harness/runs/sciver-gemma-meta-main-v1/finalization/frozen_winner.json \
  --output-dir prompts/optimized
```

The **OFFLINE** export creates immutable
`prompts/optimized/meta_cot.prompts.json` and
`prompts/optimized/meta_cot.metadata.json`, or verifies identical existing
files. It never overwrites `COT_PROMPT`.

Verify runtime registration without a request:

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

`--method cot` preserves historical output naming. `prompt_variant=meta_cot`
keeps the frozen prompt separate from the four per-sample reasoning methods.

## Run the one-time Gemma final evaluation

Stop and review the frozen winner, repository revision, manifest, and prior
final-test access before continuing.

**FINAL TEST — ONE TIME + API CREDITS**:

```bash
python scripts/finalize_meta_harness.py \
  --run-id sciver-gemma-meta-main-v1 \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --confirm-final-test \
  --live-api
```

This evaluates unchanged `cot` and frozen `meta_cot` once each on the frozen
Gemma identity. A retry verifies and returns both immutable completion receipts
without dispatching completed work.

## Transfer to the other three supported models

Run only after both Gemma search-model completion receipts exist.

**FINAL TEST — ONE TIME PER PROMPT/MODEL + API CREDITS**:

```bash
python scripts/run_meta_cot_transfer.py \
  --run-id sciver-gemma-meta-main-v1 \
  --split-manifest workspace/meta_harness/data/sciver_split.json \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --confirm-final-test \
  --live-api
```

The command evaluates unchanged `cot` and the exact same frozen `meta_cot`
bytes on:

- `Qwen2.5-VL-7B-Instruct`
- `gemma-4-31B-it`
- `gemma-3-27b-it`

It does not invoke the proposer, start another search, or adapt prompt text per
model.

## Artifact locations

```text
workspace/meta_harness/data/
  sciver_split.json                 immutable full split manifest
  sciver_validation.json            validation-only search input

workspace/meta_harness/runs/<RUN_ID>/
  run_state.json                    config, budgets, scores, frontier, resume
  candidate_registry.json           candidate hashes and mutable statuses
  candidates/<ID>/candidate.json    immutable candidate and four templates
  proposer/iteration_####/          sanitized proposer audit records
  iterations/iteration_####/        durable proposal checkpoints
  evaluations/<ID>/
    validation.results.jsonl        sanitized per-sample validation results
    validation.metrics.json         metrics, coverage, failures, identities
  finalization/
    frozen_winner.json              immutable global winner
    results_manifest.json           four-model/two-prompt completion summary
    executions/<MODEL>/
      results.jsonl                 meta_cot final results
      metrics.json
      manifest.json
      completion.json
      results.csv
      cot/
        results.jsonl               unchanged cot final results
        metrics.json
        manifest.json
        completion.json
        results.csv

prompts/optimized/
  meta_cot.prompts.json             exact exported four-template family
  meta_cot.metadata.json            provider-neutral frozen identities
```

Raw image base64, authorization headers, solver credentials, and complete
environments are not persisted. Per-sample result files necessarily contain
local labels for metrics, but labels never enter solver messages or proposer
context.

## Troubleshooting

### Codex authentication or proposer startup

```bash
codex --version
codex login status
codex exec --help
```

If login is absent, run `codex login`. Search proposer failures are classified
in `proposer/iteration_####/attempt_#####.json`. The proposer uses its cached
Codex login, not `API_KEY`.

### Solver API configuration

Missing configuration reports that both `API_URL` and `API_KEY` are required.
Check the names in `.env` without printing their values. Verify the endpoint
with the one-example smoke command before search. Authentication, permission,
rate-limit, timeout, server, network, malformed-JSON, and schema failures are
kept distinct by the existing remote client.

### Candidate parsing or schema failures

Inspect the sanitized proposer audit category and error. A candidate must have
exactly `direct`, `analytical`, `parallel`, and `sequential` templates with the
original placeholder sets and terminal yes/no contract. The Python wrapper
derives `source_sha256`; proposer output must not supply or guess a hash.

### Coverage failures

Inspect `validation.metrics.json`:

```bash
python -m json.tool \
  workspace/meta_harness/runs/sciver-gemma-meta-main-v1/evaluations/<CANDIDATE_ID>/validation.metrics.json
```

Coverage below 1.0, any unresolved API failure, or invalid input makes the
candidate unrankable. Do not change the parser or lower the gate during a run.

### Locking

`another process is already writing this run` means an active process holds
the advisory run lock. Let it finish. Lock files are durable markers; process
exit releases the operating-system lock, so an old file alone is harmless.
Do not delete a lock while another process may be active.

### Resume

`run state already exists` requires `--resume RUN_ID`.
`resume configuration changed immutable fields` requires restoring the
original config/split/evaluation settings or starting a new run.
Resource-limit increases require one matching `--safe-resume-change`; limits
cannot decrease. `finalized runs cannot resume` is permanent for that run ID.

### Finalization or one-time execution

`run has no rankable validation candidate` requires resolving validation
coverage/API failures without final-test feedback.
`final-test execution requires explicit confirmation` means the one-time flags
were omitted. Do not add them during setup.
`completion artifact hash mismatch` means a frozen result changed; stop,
preserve the artifacts, and investigate instead of overwriting or rerunning.

## Final offline verification

Before reporting or committing reviewed changes:

```bash
python -m compileall -q .
python -m pytest -q
git diff --check
git status --short
```

The default pytest configuration excludes tests marked `live_api`; the suite
also denies network access at the socket and HTTP boundaries.
