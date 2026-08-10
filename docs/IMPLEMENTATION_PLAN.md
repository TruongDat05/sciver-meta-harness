# SciVer Full-Search Meta-Harness Implementation Plan

## Authority and status

This is the canonical implementation plan for the new SciVer Meta-Harness
experiment. It is self-contained so that a fresh Codex session can continue
without access to the conversation that produced it.

- Plan approved: 2026-08-10
- Repository branch at approval: main
- Repository revision at approval:
  29c07e45f2a3ebeea23fea58b5e95c7e11f06f5a
- Target protocol: sciver_full_search_v1
- Current implementation status: Milestone 0 complete; Milestone 1 not started

The root AGENTS.md and this document are authoritative. Legacy source,
configuration, tests, documentation, and skill text describe an obsolete
experiment when they conflict with these documents.

Milestone 0 is documentation-only and must be completed before application
code, tests, configurations, scripts, or dependencies are changed.

## Experiment objective

Optimize the four SciVer prompt templates with Codex CLI while evaluating every
prompt through a self-hosted multimodal solver:

- Codex CLI proposes prompt candidates.
- Qwen/Qwen3.5-35B-A3B is served by vLLM through an OpenAI-compatible HTTP API.
- Trusted Python code performs inference orchestration, computes metrics,
  persists state, and selects the winner.

The Python Meta-Harness process does not load model weights. GPU count, tensor
parallelism, memory utilization, and other server-specific settings remain
external vLLM launch options.

## Fixed experiment protocol

### Data

The experiment has exactly two scored groups:

- SEARCH: exactly 1,000 SciVer samples.
- FINAL: exactly 1,000 SciVer samples.

Use deterministic seed 42. SEARCH and FINAL must be paper-disjoint. Split
membership must not use labels, model predictions, difficulty, or baseline
errors.

Paper identity is derived uniformly from canonical paper JSON SHA-256 unless a
complete, verified paper identifier is available for the entire corpus. Do not
mix identity sources within one split and never fall back to claim identity.

When complete paper groups cannot themselves total exactly 1,000 per side:

1. Deterministically assign whole papers exclusively to SEARCH or FINAL pools.
2. Require each pool to contain at least 1,000 eligible claims.
3. Deterministically select exactly 1,000 claims within each pool.
4. Leave all extra claims unused.

If the corpus contains fewer than 2,000 eligible claims, or no exclusive paper
partition can supply both pools, preparation fails with a feasibility report.
It must not duplicate claims, split a paper across groups, or silently change
the requested counts.

### Search

1. Evaluate the canonical initial prompt on all 1,000 SEARCH samples.
2. Persist per-sample predictions, parse results, usage, and aggregate metrics.
3. Build aggregate, opaque SEARCH feedback.
4. Ask Codex CLI for exactly one candidate prompt family.
5. Validate and persist the candidate before any solver request.
6. Evaluate the candidate on the same complete 1,000-sample SEARCH set.
7. Compute metrics and update the global best in Python.
8. Repeat until the iteration limit or patience condition.
9. Freeze the deterministic winner.
10. Evaluate only the frozen winner on all 1,000 FINAL samples through a
    separate explicit command.

The baseline is an eligible winner. With ten candidate iterations, the
worst-case SEARCH workload is 11,000 logical sample evaluations.

There is no hard-sample mining, reduced evaluation subset, smoke screening,
guard set, promotion stage, or protected validation. Representative failures
may help summarize feedback, but they never replace complete SEARCH scoring.

### Search schedule

- Candidates per iteration: 1
- Minimum completed candidate iterations: 3
- Maximum candidate iterations: 10
- Early-stopping patience: 3 completed non-improving candidates
- Bounded proposal attempts for an invalid or duplicate proposal: 3

Patience resets when Macro-F1 improves or, at equal Macro-F1, Accuracy
improves. A tie resolved only by a stable identifier does not reset patience.

### Metrics and ranking

Python selects the winner with this fixed ordering:

1. SEARCH Macro-F1 descending.
2. SEARCH Accuracy descending.
3. Prompt SHA-256 ascending.
4. Candidate ID ascending.

The same ordering governs global-best updates, final selection, reporting, and
resume reconciliation.

A successful HTTP response that cannot be parsed as yes or no is an abstaining
and incorrect prediction on the full 1,000-sample denominator. Metrics report
parse coverage and include an unparsed prediction column in the confusion
matrix. A candidate remains rankable after all 1,000 HTTP responses complete.

Unresolved transport or server failures leave an evaluation incomplete.
Authentication, permission, invalid-request, invalid-input, and context-limit
failures are fatal and actionable rather than model errors.

### Prompt candidate contract

Each candidate contains exactly these templates:

- direct
- analytical
- parallel
- sequential

Only template text may change. Preserve every existing placeholder, the answer
format, claim/context/caption inputs, image number and order, parser behavior,
label vocabulary, model, generation settings, and solver call behavior.

The canonical cot family is immutable. The frozen winner is exposed only as
the additive meta_cot prompt variant. Prompt variant and reasoning method
remain separate concepts.

### Final isolation

FINAL content or results cannot influence proposal, selection, patience, or
prompt modification.

- Search receives a search-safe manifest and SEARCH records only.
- The proposer receives no FINAL identifiers, paths, examples, labels,
  predictions, metrics, traces, or commitment.
- The FINAL dataset is not materialized during search.
- Freezing is offline and requires a terminal, fully scored SEARCH run.
- FINAL requires a separate CLI, the live-request flag, and the explicit final
  confirmation flag.
- Only the frozen winner runs on FINAL; the baseline and transfer models do
  not.
- FINAL metrics cannot modify the winner, frontier, history, or proposer
  input.
- A completion receipt prevents a second completed FINAL execution.
- Transport retries and resuming missing samples belong to the same single
  logical FINAL evaluation.

## Current design versus target design

| Area | Current repository | Approved target |
| --- | --- | --- |
| Solver | Gemma-focused search allowlist and configs | Qwen/Qwen3.5-35B-A3B through vLLM HTTP |
| Data groups | search, validation, final_test | SEARCH and FINAL only |
| Split sizing | 20/20/60 by number of paper groups | Exact 1,000/1,000 paper-exclusive pools |
| Search data | Baseline-error-heavy hard subset | Complete immutable 1,000-sample SEARCH |
| Candidate screening | 10-sample smoke and hard-search stages | No scoring subset or smoke gate |
| Promotion | Top-K to protected validation | None |
| Winner score | Protected-validation or legacy validation score | Complete SEARCH score |
| Proposer batch | Exactly two candidates | Exactly one candidate |
| Proposer feedback | Validation summaries or raw search traces | Aggregate/opaque SEARCH feedback only |
| Evaluator | Sequential request loop | Bounded concurrent scheduler |
| Generation controls | Temperature/max tokens omitted | Frozen deterministic 8K configuration |
| Resume | JSONL/state plus separate retry tool | Integrated per-sample resume and retry |
| Cache | Run-local completed-record skipping | Content-addressed cross-run SEARCH cache |
| Final | Baseline/winner pairs and model transfer | Winner only, separate explicit command |

## Current implementation map

### Reusable infrastructure

- meta_harness/baseline.py preserves and verifies the canonical prompt family.
- meta_harness/prompt_family.py validates and serializes prompt families.
- meta_harness/candidate_store.py provides immutable candidate persistence,
  hashing, registries, and locking.
- utils/answer_parser.py supplies the yes/no parser.
- evaluation/metrics.py supplies binary classification metrics.
- model_inference/remote_api.py prepares provider-neutral solver requests and
  strips inference from client construction.
- utils/remote_input_processing.py constructs ordered multimodal messages.
- utils/dataset_adapters.py contains the strict SciVer adapter and other
  existing dataset adapters.
- utils/result_writer.py already appends and fsyncs attempts and repairs a
  truncated final JSONL line.

### Components requiring redesign

- meta_harness/config.py hardcodes Gemma search models, three-way ratios, hard
  set sizes, guard fraction, smoke size, promotion count, and nullable
  generation settings.
- meta_harness/split_manager.py allocates counts of paper groups across three
  ratio-based splits rather than exact claim counts.
- scripts/prepare_meta_harness_data.py materializes validation data and invokes
  hard-search mining.
- meta_harness/evaluator.py evaluates sequentially and treats incomplete parse
  coverage as unrankable.
- model_inference/remote_client.py uses one mutable session and last_usage
  field and does not send the approved generation parameters.
- meta_harness/proposer/codex_cli.py requires two candidates, validation
  terminology, and an exploitation/exploration pair.
- meta_harness/proposer/feedback.py is tied to validation result names.
- meta_harness/orchestrator.py and meta_harness/staged_orchestrator.py implement
  competing protocols.
- meta_harness/finalize.py includes legacy validation selection, baseline/final
  pairs, and cross-model transfer.

### Obsolete protocol components

- meta_harness/hard_search.py
- meta_harness/staged_orchestrator.py
- meta_harness/experience_store.py
- meta_harness/retry.py
- meta_harness/reparse.py
- Retry, reparse, transfer, and staged-search scripts and tests
- Gemma-specific Meta-Harness configs
- Legacy hard-search/protected-validation documents

Dependency inspection found the hard-search module used by preparation, the
staged runner, its tests, and documentation. The staged orchestrator is the
only application caller of the raw experience store. Retry and reparse modules
are called by their dedicated scripts and tests. Remove these files only after
their entrypoints and test coverage have been replaced.

## Target architecture

### Preparation boundary

Implement paper_exclusive_exact_1000_v1:

1. Load the real SciVer corpus with the existing adapter.
2. Validate stable sample IDs, labels, local evidence, referenced paper JSON,
   and paper identity.
3. Emit counts, duplicates, unreadable inputs, paper-size distribution, and
   feasibility in inspection.json.
4. Require at least 2,000 eligible records.
5. Deterministically order paper groups from seed 42.
6. Use deterministic dynamic programming to choose a feasible SEARCH paper
   pool with total size between 1,000 and total minus 1,000. Prefer the
   feasible partition closest to balanced; resolve equal choices by seeded
   group order.
7. Assign all remaining papers to the FINAL pool.
8. Rank claims within each pool by SHA-256 of seed, group name, and stable
   sample ID; take the first 1,000.
9. Verify exact counts plus paper and sample disjointness.

Preparation writes:

- A private manifest containing SEARCH and FINAL membership.
- A search-safe manifest containing SEARCH membership, split/config hashes,
  FINAL count, and an opaque hash commitment to private FINAL membership.
- A materialized search.json with exactly the SEARCH records.

The search process is never passed the private manifest or full corpus path.

### Configuration

Replace legacy ratios and staged fields with a versioned configuration:

- dataset:
  - name: SciVer
  - seed: 42
  - search_samples: 1000
  - final_samples: 1000
  - split_algorithm: paper_exclusive_exact_1000_v1
- solver:
  - model: Qwen/Qwen3.5-35B-A3B
  - required immutable model_revision
  - required served_model identity
  - required vllm_version
  - required deployment_fingerprint
  - generation parameters
- execution:
  - max_in_flight: 8
  - request_timeout_seconds: 600
  - max_total_attempts_per_sample: 5
  - initial_backoff_seconds: 1
  - max_backoff_seconds: 30
- search:
  - min_iterations: 3
  - max_iterations: 10
  - patience: 3
  - proposal_attempts: 3
- proposer:
  - configurable Codex model and reasoning effort
  - initial defaults gpt-5.6-terra and medium
- cache and protocol schema versions

Generation is fixed initially to temperature 0, top_p 1, seed 42, n 1,
stream false, and max_tokens 8192.

API_URL is the only endpoint source. API_KEY is the only credential source.
Neither enters hashes or persisted artifacts.

Changing a semantic experiment identity starts a new run. This includes model
or revision, deployment, generation, split, prompt, parser, evaluator, scoring,
proposer model/effort, or search protocol. An exhausted attempt ceiling may be
raised only through an explicit, non-decreasing resume option.

### Solver client and scheduler

Retain the provider-neutral HTTP boundary and extend it additively:

- Send the frozen generation fields on every request.
- Return immutable completion metadata containing response text, safe usage,
  and HTTP attempt count.
- Preserve the existing string-returning method for non-Meta-Harness callers.
- Use a bounded thread pool with max_in_flight defaulting to 8.
- Give each worker a separate HTTP session/client.
- Retry only timeouts, network errors, HTTP 429, and HTTP 5xx.
- Do not retry parse failures or non-retryable HTTP 4xx responses.
- Use deterministic capped exponential backoff.

### SEARCH cache

Add meta_harness/cache.py. Cache keys include:

- cache and request serializer version;
- four-template prompt-family hash;
- rendered request digest;
- model, immutable revision, served identity, and deployment fingerprint;
- normalized generation settings;
- stable sample ID;
- claim, context, and caption hashes;
- ordered image-byte hashes.

Use atomic create-or-verify entries. Persist successful HTTP completion text,
safe usage, attempt count, and identity hashes. A successfully returned but
unparsable response is cacheable.

Do not cache transport/server failure, invalid input, credentials, endpoint,
Authorization data, labels, raw prompts, or image base64 data.

SEARCH can reuse the global cache. FINAL cannot read it and uses only run-local
attempt checkpoints.

### Persistent run state

Each run contains:

- An immutable run manifest with code revision, clean-worktree state, config,
  SEARCH data/manifest, model/deployment, generation, parser/evaluator,
  proposer, prompt schema, and cache identities.
- An exclusive run lock.
- Atomic mutable run_state.json.
- Immutable candidates and proposal checkpoints.
- Append-and-fsync sample attempts.
- Deterministic sample-ID-sorted completed result snapshots and metrics.
- Structured event logs.

State transitions cover baseline evaluation, proposal, candidate evaluation,
retry required, stopped, frozen, FINAL running, and FINAL complete.

On resume:

1. Verify all immutable identities.
2. Repair only a truncated last attempts line.
3. Reconcile state from durable proposals and sample attempts.
4. Schedule only missing or retryable samples.
5. Never reinvoke Codex for an already persisted proposal iteration.
6. Refuse a duplicate process through the run lock.

Logs contain stage, candidate, progress, cache counts, retry categories, and
timestamps. They exclude endpoint, credentials, prompts, responses, labels,
sample content, image data, and FINAL identifiers.

### Proposer

Keep the existing safe subprocess design:

- codex exec
- separate Codex authentication
- sanitized environment
- read-only sandbox
- bounded input/output and timeout
- fixed JSON output schema
- fake subprocess in tests
- local validation, hash derivation, and immutable audit artifacts

Change the proposer contract to one candidate and SEARCH terminology. Remove
the two-candidate portfolio and broad-rewrite similarity restriction. Retain
prompt-only validation, placeholder validation, duplicate detection,
hypothesis text, terminal answer-format checks, and safety scanning.

The proposer receives parent templates and only:

- aggregate SEARCH metrics;
- confusion matrix and false-positive/false-negative totals;
- per-reasoning-method and evidence-type aggregates;
- parse and infrastructure failure counts;
- deltas against baseline and parent;
- anonymous corrected, regressed, still-wrong, and unresolved overlap keys;
- prior candidate strategy history.

It receives no raw sample material. Its working directory is a generated safe
view containing only the approved schema, prompts, and feedback. The read-only
Codex sandbox prevents writes but is not treated as an adversarial filesystem
read jail; operational setup keeps private data outside the safe view.

### Freeze and FINAL

Freezing:

- Requires completed or early-stopped SEARCH with no incomplete candidate that
  could affect selection.
- Recomputes the deterministic winner from durable SEARCH results.
- Verifies prompt, candidate, config, split, parser, evaluator, and code hashes.
- Writes immutable finalization/frozen_winner.json.
- Makes no live request.

FINAL execution:

- Uses a separate script not imported by the search CLI.
- Requires both live and explicit final-confirmation flags.
- Loads the private manifest and original corpus only after verifying the
  frozen artifact and FINAL commitment.
- Evaluates the frozen winner on exactly the 1,000 FINAL IDs.
- Checkpoints each sample locally.
- Resumes only missing/retryable samples.
- Writes a completion receipt that blocks a second completed run.

## Artifact layout

    workspace/meta_harness/
    ├── data/<split_id>/
    │   ├── inspection.json
    │   ├── private_split_manifest.json
    │   ├── search_manifest.json
    │   └── search.json
    ├── cache/completions/v1/<prefix>/<cache-key>.json
    └── runs/<run_id>/
        ├── run_manifest.json
        ├── run_state.json
        ├── events.jsonl
        ├── .run.lock
        ├── candidates/<candidate-id>/candidate.json
        ├── proposer/iteration_<n>/
        │   ├── input-audit.json
        │   ├── codex-events.jsonl
        │   └── proposal.json
        ├── proposer_view/
        │   ├── history.json
        │   └── feedback.json
        ├── evaluations/search/<candidate-id>/
        │   ├── attempts.jsonl
        │   ├── results.jsonl
        │   ├── metrics.json
        │   └── status.json
        ├── feedback/iteration_<n>.json
        ├── finalization/frozen_winner.json
        ├── final/
        │   ├── manifest.json
        │   ├── attempts.jsonl
        │   ├── results.jsonl
        │   ├── metrics.json
        │   └── completion.json
        └── reports/
            ├── summary.json
            └── candidates.csv

The entire workspace remains excluded from version control.

## File-by-file decisions

### KEEP unchanged

- meta_harness/baseline.py
- meta_harness/prompt_family.py
- meta_harness/candidate_store.py
- utils/answer_parser.py
- Binary metric primitives in evaluation/metrics.py
- utils/dataset_adapters.py, unless real-corpus inspection proves a necessary
  schema correction
- model_inference/remote_api.py
- utils/remote_input_processing.py
- Existing non-Meta-Harness providers and main.py behavior
- .env.example and .gitignore

### MODIFY

- AGENTS.md: permanent full-search invariants.
- .agents/skills/meta-harness/SKILL.md: remove conflicting hard-search,
  promotion, and protected-validation instructions.
- meta_harness/config.py: versioned target config and target model.
- meta_harness/split_manager.py: exact two-group split and safe/private
  manifests.
- meta_harness/evaluator.py: concurrent full-SEARCH scoring, cache, abstention,
  and integrated resume.
- meta_harness/orchestrator.py: one full-search state machine.
- meta_harness/schemas.py and proposer/candidate_batch.schema.json: exactly one
  candidate.
- meta_harness/proposer/codex_cli.py: SEARCH objective, safe view, and bounded
  proposal retries.
- meta_harness/proposer/feedback.py: aggregate/opaque SEARCH history.
- meta_harness/finalize.py: SEARCH-winner freeze only.
- model_inference/remote_client.py: generation settings, immutable completion
  metadata, and worker-safe usage.
- model_inference/remote_config.py: target model support.
- utils/result_writer.py: request/cache identity and concurrent reconciliation.
- scripts/inspect_dataset.py: SciVer paper-group feasibility.
- scripts/prepare_meta_harness_data.py: exact split outputs; no hard mining.
- scripts/run_meta_harness.py: safe-manifest full search and resume.
- scripts/smoke_meta_harness_proposer.py: one-candidate SEARCH feedback.
- scripts/finalize_meta_harness.py: offline freeze only.
- scripts/export_meta_cot.py: SEARCH terminology and frozen winner.
- README.md and canonical Meta-Harness documentation.
- Dependency documentation and environment separation.

### REMOVE after dependency migration

- meta_harness/hard_search.py
- meta_harness/staged_orchestrator.py
- meta_harness/experience_store.py
- meta_harness/retry.py
- meta_harness/reparse.py
- scripts/retry_meta_harness_failures.py
- scripts/reparse_meta_harness.py
- scripts/run_meta_cot_transfer.py
- Tests dedicated to those obsolete protocols
- Gemma-specific Meta-Harness configurations
- Redundant legacy Meta-Harness design and protocol documents

Do not delete any item until imports, entrypoints, tests, and documentation have
been migrated and a final reference search shows no live dependency.

Old run artifacts remain untouched but cannot be resumed under
sciver_full_search_v1. No migration shim is planned.

### ADD

- docs/IMPLEMENTATION_PLAN.md
- meta_harness/cache.py
- configs/meta_harness/sciver_full_search.example.json
- requirements-meta-harness.txt
- requirements-remote-server.txt
- scripts/serve_remote_model.sh
- scripts/smoke_remote_model.py
- scripts/run_meta_harness_final.py
- docs/meta_harness/server_runbook.md
- Tests for cache, concurrency, complete SEARCH scoring, integrated resume, and
  FINAL isolation

The existing scripts/vllm_small.sh, scripts/vllm_large.sh, and scripts/api.sh
are legacy benchmark launchers, not model-server scripts. Leave them outside
the new workflow rather than repurposing them.

## Implementation milestones

### Milestone 0 — Persist approved design

Deliverables:

- Update root AGENTS.md with permanent experiment invariants.
- Create this canonical implementation plan.
- Make no application, test, config, script, or dependency change.

Exit criteria:

- Both documents are self-contained and mutually consistent.
- Legacy hard-search or protected-validation text is identified as obsolete,
  not approved behavior.
- No credential, endpoint, or sensitive payload is present.
- git diff --check passes.

### Milestone 1 — Protocol, configuration, and deterministic split

Deliverables:

- Align the Meta-Harness skill with the approved protocol before application
  changes.
- Replace the Meta-Harness configuration schema.
- Implement dataset inspection, feasibility reporting, exact splitting, and
  safe/private manifests.
- Rewrite split and configuration tests.

Exit criteria:

- Offline fixtures prove deterministic, input-order-independent exact
  1,000/1,000 membership.
- Paper and sample disjointness are verified.
- Insufficient and infeasible corpora fail clearly.
- Search-safe artifacts contain no FINAL identity.

### Milestone 2 — Solver boundary, cache, retries, and concurrency

Deliverables:

- Add explicit generation controls and immutable client completion metadata.
- Add the content-addressed SEARCH cache.
- Add bounded worker-local concurrency and retry classification.
- Extend durable sample attempt handling.

Exit criteria:

- Cache identity changes with every semantic request input.
- Image order remains significant.
- Concurrent completion order does not change final results or metrics.
- Interrupted runs skip completed samples.
- Tests make no live request.

### Milestone 3 — Full-SEARCH evaluator

Deliverables:

- Evaluate only the complete immutable SEARCH set.
- Add abstention/parse-failure metric semantics.
- Generate deterministic completed result snapshots and metrics.
- Remove arbitrary split and sample-subset selection from search evaluation.

Exit criteria:

- A synthetic 1,000-sample test proves baseline and every candidate receive the
  identical complete ID set.
- Parse and infrastructure failure behaviors match the protocol.
- No hard-search, smoke, promotion, or validation path is reachable.

### Milestone 4 — Codex proposer and search orchestration

Deliverables:

- Change proposer output to exactly one candidate.
- Replace raw experience traces with aggregate/opaque feedback.
- Consolidate orchestration into one baseline-plus-full-search state machine.
- Implement deterministic ranking and patience.

Exit criteria:

- Offline pipeline tests cover proposal validation, duplicates, resume,
  baseline winning, secondary ties, and early stopping.
- Codex is faked in tests and never chooses the winner.
- No proposer-visible artifact includes raw or FINAL data.

### Milestone 5 — Freeze and isolated FINAL

Deliverables:

- Simplify finalization to offline SEARCH-winner freezing.
- Add the separate winner-only FINAL CLI.
- Enforce two confirmation gates, commitment verification, local resume, and
  completion receipts.

Exit criteria:

- Sentinel tests prove FINAL never enters search or proposer boundaries.
- Only the winner is scheduled on exactly 1,000 FINAL samples.
- Completed FINAL work cannot be dispatched again.
- Final metrics cannot alter search state.

### Milestone 6 — Server workflow and operator interfaces

Deliverables:

- Add provider-neutral model-server launch and smoke scripts.
- Separate lightweight harness and model-server environment guidance.
- Add the complete server/tmux runbook.
- Update README and CLI help.

The runbook sequence is:

1. Clone the repository.
2. Create separate harness and model-server environments.
3. Install and record a vLLM version that supports the target model.
4. Authenticate Codex CLI independently.
5. Set API_URL and API_KEY.
6. Start the server with operator-selected GPU and parallelism flags.
7. Test text, one-image, and ordered two-image inference.
8. Test the Codex proposer with synthetic feedback.
9. Inspect and prepare the deterministic split.
10. Verify exact counts, disjointness, and commitment.
11. Dry-run the workload.
12. Run or resume SEARCH inside tmux.
13. Freeze the winner offline.
14. Explicitly run or resume FINAL.
15. Export and archive metrics and identity artifacts.

Exit criteria:

- All CLI dry runs are offline and do not create run state.
- Live smoke commands require an explicit live flag and are excluded from
  pytest.
- Application code contains no GPU-count or tensor-parallel assumption.

### Milestone 7 — Legacy removal and final verification

Deliverables:

- Confirm no live import or entrypoint uses obsolete Meta-Harness modules.
- Remove obsolete modules, scripts, configs, tests, and redundant docs.
- Run the complete offline suite and manual safety review.

Exit criteria:

    python -m compileall -q .
    pytest -q
    git diff --check

All commands pass, all HTTP and Codex boundaries are mocked in tests, and the
final diff contains no secret, provider advertising, base64 data, prompt drift,
image reordering, label leakage, FINAL leakage, hard-search behavior, implicit
model downloads, or unrelated changes.

## Test plan

### Unit tests

- Deterministic exact split and input-order independence.
- Paper exclusivity and duplicate sample detection.
- Canonical paper identity.
- Fewer-than-2,000 and infeasible-partition errors.
- Safe/private manifest hash and commitment verification.
- Configuration validation and semantic hash sensitivity.
- Exact outbound generation payload.
- Retryable versus fatal HTTP classification.
- Worker-local client/session behavior.
- Cache-key sensitivity to prompt, model/revision, generation, sample content,
  and image order.
- Atomic cache races, corruption detection, and sensitive-data exclusion.
- Parse failure as an abstention on the full denominator.
- Infrastructure failure as incomplete evaluation.
- Exactly-one-candidate schema and duplicate rejection.
- Aggregate-only proposer envelope and sanitized environment.
- Deterministic winner and patience behavior.
- Freeze identity checks, final gates, and completion receipt.

### Offline integration tests

- Fake baseline and candidates over 1,000 synthetic SEARCH records.
- Assert every prompt receives the same 1,000 sample IDs.
- Interrupt mid-candidate, resume, and assert no completed dispatch repeats.
- Rerun through cache and assert zero HTTP calls.
- Complete requests out of order and verify identical metrics and sorted
  results.
- Exercise retryable, exhausted, and fatal failures.
- Crash after proposal persistence and assert Codex is not reinvoked.
- Use sentinel FINAL IDs, labels, paths, and content and assert none appears in
  search calls, proposer input, logs, cache, or artifacts.
- Run FINAL partially, resume only missing samples, and then block a completed
  rerun.

The suite-wide test fixture continues to remove credentials, disable
accelerators/model downloads, block requests, and block socket connections.
Every Codex subprocess is faked.

### Dry-run tests

Dry runs must:

- Load and validate configuration and manifests.
- Report exact SEARCH and FINAL counts.
- Report at most 11,000 logical SEARCH evaluations and a separate 1,000 FINAL
  workload.
- Report concurrency, retries, cache behavior, generation settings, and
  identity hashes.
- Avoid reading credentials.
- Avoid constructing a live HTTP client.
- Avoid invoking Codex.
- Avoid writing run or cache state.

### Opt-in live smoke tests

Live smoke is an operator workflow, never part of pytest:

1. One text request.
2. One request with an explicitly supplied local image.
3. One ordered two-image request.
4. One Codex proposal from synthetic aggregate feedback.
5. Optionally, a few SEARCH records through a non-scoring smoke command.

Every solver smoke requires the live flag. Smoke tools cannot accept the
private FINAL manifest and cannot write evaluation metrics used for selection.

## Acceptance criteria

The redesign is accepted only when:

- SEARCH and FINAL each contain exactly 1,000 samples.
- No paper or sample crosses between SEARCH and FINAL.
- Split construction is deterministic from the approved seed and source.
- The baseline and every valid candidate are scored on all 1,000 SEARCH
  samples.
- No hard-search, subset-scoring, smoke-gating, promotion, or protected
  validation objective remains.
- Macro-F1, Accuracy, prompt hash, and candidate ID define the only winner
  ordering.
- Codex proposes prompt text but does not evaluate or choose.
- Python computes metrics and selects the winner.
- Qwen/vLLM inference remains behind the generic HTTP boundary.
- Prompt placeholders, canonical cot, image order, parser, labels, and solver
  settings remain frozen outside candidate control.
- Resume, checkpointing, retries, concurrency, and caching are covered by
  offline tests.
- FINAL is winner-only, separately gated, resumable, and unable to influence
  search.
- No test accesses the network or invokes Codex recursively.
- Existing non-Meta-Harness behavior remains compatible.
- Required verification commands and manual safety review pass.

## Preflight requirements

Before implementing data mappings:

- Obtain the intended SciVer corpus and local evidence directory.
- Inspect representative records and referenced paper JSON.
- Confirm at least 2,000 eligible unique records.
- Confirm paper-exclusive partition feasibility.
- Do not infer the real corpus size from synthetic tests.

Before a live experiment:

- Use a clean, immutable repository revision.
- Install the lightweight Meta-Harness environment.
- Install a separately managed vLLM server environment that supports the target
  model; do not assume the repository's legacy vllm 0.9.0.1 pin is compatible.
- Record the exact vLLM version, model revision, served-model identity, and
  deployment fingerprint.
- Authenticate Codex CLI in the server terminal.
- Set API_URL and API_KEY without persisting their values.
- Verify text and multimodal server inference.
- Verify the proposer with synthetic feedback.
- Verify split hashes, exact counts, paper/sample disjointness, and FINAL
  commitment.
- Review the dry-run workload and confirm disk, time, and server capacity.
- Run long work inside tmux with the documented resume command.

## Implementation status checklist

- [x] Milestone 0: persist the approved design
  - [x] Update root AGENTS.md
  - [x] Create docs/IMPLEMENTATION_PLAN.md
  - [x] Make no application/test/config/script/dependency changes
  - [x] Run git diff --check
- [ ] Milestone 1: protocol, configuration, and deterministic split
- [ ] Milestone 2: solver boundary, cache, retries, and concurrency
- [ ] Milestone 3: full-SEARCH evaluator
- [ ] Milestone 4: Codex proposer and search orchestration
- [ ] Milestone 5: freeze and isolated FINAL
- [ ] Milestone 6: server workflow and operator interfaces
- [ ] Milestone 7: legacy removal and final verification

Do not begin Milestone 1 without explicit user approval.
