---
name: meta-harness
description: >-
  Design, audit, implement, test, operate, and run the SciVer prompt-only
  Meta-Harness workflow. Use when a request involves the locked full-SEARCH
  protocol, proposer candidates, prompt search, SEARCH/FINAL isolation,
  workload accounting, checkpoints, resume, winner freezing, finalization, or
  related repository code and configuration. Analysis stays read-only,
  engineering may edit and test the repository offline, experiment operation
  may execute only explicitly authorized scope, and only an explicit
  next-candidate request enters the strict read-only JSON proposal contract.
---

# SciVer Meta-Harness

The authoritative SciVer protocol is `sciver_full_search_v3`. `AGENTS.md` and
`docs/IMPLEMENTATION_PLAN.md` control if this skill conflicts with them.
The earlier `sciver_full_search_v2` 750/1250 design was abandoned before any
official live experiment and is historical only.
This skill governs prompt-only optimization: a candidate contains exactly the four
prompt templates `direct`, `analytical`, `parallel`, and `sequential`; only
their text may change. Trusted Python, not the proposer, validates, evaluates,
scores, persists, ranks, selects, freezes, retries, and resumes work.

Use the user's requested outcome to choose one primary mode. Invoking
`$meta-harness` alone does not authorize execution and does not turn ordinary
repository work into a proposal iteration.

## Select one operating mode

1. **Analysis** — Explain, review, diagnose, compare, estimate, or design.
   Stay read-only and ground conclusions in repository or run evidence.
2. **Engineering** — Implement, fix, configure, document, or add tests. Edit
   only the authorized repository files and verify the smallest coherent change
   offline.
3. **Experiment operation** — Plan, dry-run, start, monitor, resume, freeze,
   or finalize a run. Execute only the scope explicitly authorized by the user.
4. **Proposal iteration** — Produce the one candidate prompt family for an
   active SEARCH iteration. Apply the strict read-only proposer contract below
   and return only schema-valid JSON.

For a mixed request, perform the authorized analysis or engineering first.
Enter Proposal iteration only when the final requested artifact is a candidate
and the active run supplies the required inputs and output schema.

## Locked protocol

### Dataset and split

- Protocol ID is exactly `sciver_full_search_v3`.
- Select exactly 2,000 SciVer samples with seed `42`: exactly 1000 SEARCH and
  exactly 1,000 FINAL samples.
- Group by verified paper identity; if that is unavailable, use the SHA-256 of
  the referenced paper JSON bytes. Complete paper groups must never be split
  across SEARCH and FINAL.
- Membership and allocation are deterministic and independent of labels, model
  predictions, difficulty, baseline correctness, baseline errors, and all
  other model-derived signals.
- Preparation must fail with a clear error if it cannot satisfy the exact
  counts while retaining paper disjointness. It must not silently shrink a
  split, fall back to claims, or change membership on resume.

### SEARCH and prompt schedule

- Evaluate canonical P0 (`cot`) on all 1000 immutable, manifest-ordered SEARCH
  records before candidate selection.
- Evaluate every valid candidate on exactly those same complete 1000 SEARCH
  records. There is no hard-search subset or objective, hard-sample mining,
  reduced scoring subset, smoke scoring gate, promotion, or protected
  validation.
- Sanitized aggregate metrics and representative opaque SEARCH error summaries
  may inform the proposer. They must not affect membership, replace full SEARCH
  scoring, or include labels, example IDs, trace-specific facts, or FINAL data.
- Codex proposes exactly one candidate prompt family per iteration. It may make
  at most three proposal attempts when output is invalid or duplicate; each
  rejected attempt is recorded before any solver call. Exhausted attempts are a
  durable proposal/infrastructure failure with a defined resume policy.
- Complete at least 38 and at most 50 candidate iterations. Stop only after
  eight consecutive completed iterations without a metric improvement, and
  never apply early stopping before iteration 38.

### Ranking and winner

P0 remains eligible to win. Trusted Python ranks every eligible
SEARCH-complete P0/candidate by this ascending key:

1. Negative SEARCH Macro-F1.
2. Negative SEARCH Accuracy.
3. Prompt SHA-256.
4. Candidate ID.

Only a Macro-F1 improvement, or an Accuracy improvement when Macro-F1 ties,
resets patience. A winner change caused solely by prompt SHA-256 or candidate
ID does not reset patience. The proposer must not self-score, rank, select, or
claim improvement.

### Frozen prompt boundary

Only the text of `direct`, `analytical`, `parallel`, and `sequential` is
searchable. The following remain frozen: canonical `cot`, all placeholders,
task inputs, claim/context/caption interface, image count and order, answer
format, parser, label vocabulary, evidence semantics, solver model, generation
parameters, and solver-call behavior. `meta_cot` is only an additive
registration of the frozen top-K candidates; it must never replace or route
through `cot`.

### FINAL isolation and execution

- FINAL records, IDs, labels, predictions, traces, metrics, paths, and
  availability signals are unavailable to the proposer and SEARCH
  orchestration. No raw FINAL data is exposed to SEARCH or the proposer.
- Freeze the offline top-5 SEARCH candidates (P0 kept separate) before FINAL is
  executable. FINAL is a separate, explicit execution stage; no prompt
  optimization is allowed after freeze.
- Evaluate frozen canonical P0 exactly once on all 1,000 immutable FINAL IDs,
  and evaluate each of the 5 frozen candidates exactly once on those identical
  IDs. FINAL results cannot alter prompts, SEARCH state, ranking, patience, or
  the frozen top-K selection.
- An interruption may resume missing work only under the same durable execution
  identity. Completed FINAL calls must not be redispatched.
- FINAL checkpoints and completion receipts are run-local and isolated from
  the global SEARCH completion cache.

### Workload accounting

Maximum logical SEARCH solver evaluations are:

```text
baseline:                 1000
50 candidate iterations: 50 * 1000 = 50,000
maximum SEARCH total:     51,000
```

FINAL logical solver evaluations are:

```text
P0:              1,000
5 frozen:        5 * 1,000 = 5,000
FINAL total:     6,000
```

The maximum full 50-iteration SEARCH plus FINAL workload is 57,000 logical
solver evaluations. Transport retries are excluded from logical counts.

Before an opt-in live execution, estimate solver calls, proposer calls, tokens,
and wall time from this contract. Fail closed when the required minimum
protocol or requested scope cannot fit explicit ceilings.

## Preserve experiment integrity

Apply these invariants in every mode:

- Keep only the locked SEARCH and FINAL stages. SEARCH selects the winner;
  FINAL is paired evaluation only and never optimization input.
- Keep finalized runs, evaluated candidates, manifests, hashes, state,
  checkpoints, and history append-only. Persist deterministic manifests, state
  transitions, retry state, cache identities, and completion receipts
  atomically.
- Require a new run identity when changing the solver, deployment identity,
  proposer model or effort, split, parser, generation settings, scoring,
  searchable surface, or protocol. Never resume across incompatible identity.
- Keep safe SEARCH caching, bounded retries, and concurrency controls that
  prevent duplicate requests. The cache key must include immutable protocol,
  config/split, prompt/candidate, solver/generation, parser, and sample
  identities; it must not share completed work across incompatible runs.
- Preserve evidence selection and order, image order, parser semantics, label
  vocabulary, retry semantics, and call behavior. Ground-truth labels are never
  model-visible input.
- Keep evaluation outside the proposer. The outer loop owns validation, solver
  calls, metrics, persistence, ranking, patience, freezing, and FINAL.

## Safety and external boundaries

- The fixed solver is gemma-4-26B-A4B-it through an OpenAI-compatible vLLM
  endpoint. Trusted Python owns all solver orchestration.
- Read the endpoint only from `API_URL` and credentials only from `API_KEY`.
  Never hardcode, persist, print, log, serialize, or otherwise expose either,
  an Authorization header, or image base64 data.
- Construct a live client and make a remote request only for explicit,
  provider-neutral opt-in execution. Importing a module, preparing data,
  dry-running, proposing, testing, or reading state must have no remote side
  effect. Never download model weights.
- Tests must remain offline. Mock every HTTP call at the request boundary and
  every Codex subprocess boundary; tests must never recursively invoke Codex
  or make a live solver, proposer, benchmark, or FINAL request.
- Never expose secrets, raw model payloads, gold labels, or FINAL material in
  artifacts, diagnostics, logs, feedback, or candidate output.

## Work from repository evidence

For Analysis, Engineering, and Experiment operation:

1. Read applicable `AGENTS.md` files and repository-local instructions.
2. Inspect relevant config snapshots, schemas, source, tests, immutable run
   metadata, and `git status` without altering unrelated user changes.
3. Use `rg` to trace the protocol from CLI/config through preparation,
   orchestration, evaluation, persistence, resume, freeze, and reporting.
4. Distinguish observed evidence from inference. Do not infer unseen runs or
   silently repair ambiguous state.
5. For multi-stage work, keep a short plan with one step in progress.
6. Implement the smallest coherent authorized change and run focused offline
   tests before the broadest practical offline suite.

Do not make live solver, proposer, benchmark, or FINAL calls unless the user
explicitly authorizes that execution.

## Engineering and operation guidance

### Validate candidates before scoring

Use a lightweight deterministic validation step before a SEARCH solver call.
It must validate the one-candidate schema, exactly four required template keys,
placeholders, label vocabulary, parser compatibility, and unchanged frozen
settings/call behavior. It completes with mocked services and rejects invalid
or duplicate candidates without spending solver calls.

### Operate safely

Before an authorized live execution, verify and report without revealing
secrets:

- a new or compatible resumable immutable run identity;
- exact protocol/count/seed/split identities and paper disjointness;
- frozen solver, deployment, generation, parser, and prompt-contract identity;
- SEARCH and paired FINAL workload ceilings, proposal-attempt limit,
  concurrency, retry policy, cache and checkpoint paths, and resume behavior;
- that FINAL is locked and inaccessible until the offline SEARCH freeze.

A dry run checks local files, exact split counts, paper disjointness,
configuration hashes, workload, cache/resume compatibility, and paired FINAL
call counts. It does not load credentials, construct a live client, invoke
Codex, create run state, or access FINAL records during SEARCH dry-run.

During SEARCH, record immutable identities before the first call, enforce
ceilings before each call, persist completed work before reporting progress,
and resume only compatible durable state. Classify remote failures as
infrastructure failures, never as model errors. Do not launch duplicate
processes or repeat completed samples, candidates, stages, or charged calls.

For FINAL, confirm the offline-frozen top-K selection and all experiment-defining
fields before the explicit command. Execute only the P0 + top-5 work, resume
only missing work, and report per-variant metrics, parse coverage, failure
rates, calls, tokens, and wall time without using the results to tune the
frozen selection.

Without explicit authorization, do not make live calls, launch an experiment or
FINAL evaluation, access credentials, push or publish, modify external
services, delete history, or rewrite immutable run artifacts.

## Strict proposal-iteration contract

Apply this section only in Proposal iteration.

### Required proposer inputs

Read the candidate output schema first, normally
`meta_harness/proposer/candidate_batch.schema.json`. Then use only files
mounted or explicitly enumerated as proposer-visible by the active SEARCH run:

- active protocol configuration, prompt contract, and iteration identity;
- canonical P0, permitted parent prompts, candidate lineage, and prompt diffs;
- sanitized aggregate SEARCH scores and representative opaque SEARCH failure
  summaries;
- approved normalized offline experience that has been checked for FINAL and
  label leakage.

The active configuration and schema are authoritative for exactly one candidate
family, template keys, prompt-length limits, required placeholders, label
vocabulary, parser-compatible output, fixed solver settings, and hash rules.
If a required input is missing, stop without fabricating state. If any supplied
file exposes FINAL content or paths, stop reading it and report contamination
to the outer loop.

### Remain read-only and isolated

Never:

- call a solver, evaluator, benchmark, or live service, or make a network
  request;
- invoke subagents, another Codex process, or another proposer;
- create, edit, or persist files, including temporary prototypes;
- update frontier, history, reports, run state, cache, checkpoints, selection,
  freeze, or finalization;
- evaluate, rank, select, or claim improvement for a candidate;
- change model parameters, parsing, metrics, data mapping, evidence selection
  or order, images, retries, token budgets, or solver-call counts;
- include example IDs, gold labels, memorized answers, trace-specific facts,
  secrets, FINAL material, or protected paths in a candidate.

The trusted outer loop owns all writes and evaluation.

### Produce one falsifiable candidate

- Return exactly one materially distinct candidate prompt family for the active
  iteration.
- Tie it to a recurring, general SEARCH failure pattern from permitted
  sanitized evidence.
- State a falsifiable hypothesis with a predicted measurable effect and a
  plausible trade-off; never state unmeasured improvement as fact.
- Reject a proposal whose only difference is stronger wording, verbosity,
  reordered prose, generic steps, or a request for more reasoning.

Possible mechanisms include claim decomposition, independent support and
contradiction tests, modality-aware reconciliation, missing-versus-conflicting
evidence handling, calibrated tie-breaking, and parser-compatible decision
contracts. These are illustrative, not a required reasoning sequence.

### Preserve the frozen prompt contract

For the candidate:

- include exactly `direct`, `analytical`, `parallel`, and `sequential`;
- change only their text and preserve every required placeholder exactly;
- preserve label vocabulary, parser-compatible output, claims, context,
  captions, images, and their order;
- preserve canonical `cot`, solver identity, generation settings, and solver
  call behavior;
- avoid dataset names, example-specific hints, external knowledge, tools, and
  unsupported citations;
- make every template a complete cold-start instruction.

Do not replace frozen baseline templates in repository state. Keep prompt
variant separate from reasoning method.

### Emit only the candidate

Validate the complete object against the active schema and all configured
limits. Use exact field names, types, nesting, IDs, parent references,
template keys, and hash semantics. Once inputs are present, return only the
schema-valid JSON object—no Markdown, commentary, status text, or code fences.

## Completion reports

For **Analysis**, lead with the finding and distinguish evidence from inference.

For **Engineering**, report changed files, implemented behavior, exact offline
test results, anything not run, and remaining risks.

For **Experiment operation**, report locked budgets and progress, checkpoints,
failures/retries, cache/resume status, winner-freeze status, and FINAL
isolation.

For **Proposal iteration**, emit only the candidate JSON.
