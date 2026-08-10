---
name: meta-harness
description: >-
  Design, audit, implement, test, operate, and run prompt-only Meta-Harness
  workflows across task domains. Use when a request involves Meta-Harness,
  proposer candidates, prompt or harness search, frontier/history or experience
  stores, hard search-set construction, search-protected-validation-final
  isolation, candidate promotion, experiment budgets, checkpoints, resume,
  finalization, or related repository code and configs. Select the operating
  mode from the requested deliverable: analysis stays read-only, engineering
  may edit and test the repository, experiment operation may execute only the
  authorized scope, and only an explicit next-candidate request enters the
  strict read-only JSON proposal contract.
---

# Meta-Harness

Treat Meta-Harness as an outer-loop search over a task-specific harness used by
a fixed solver. For prompt-only experiments, the searchable program is the
prompt contract; a separate evaluator executes and scores it.

Use the user's requested outcome to choose one primary mode. Invoking
`$meta-harness` alone does not authorize execution and does not turn ordinary
repository work into a proposal iteration.

## Select one operating mode

1. **Analysis** — Explain, review, diagnose, compare, estimate, or design. Stay
   read-only and ground conclusions in repository or run evidence.
2. **Engineering** — Implement, fix, refactor, configure, document, or add
   tests. Edit the repository and verify the smallest coherent change offline.
3. **Experiment operation** — Plan, dry-run, start, monitor, resume, promote,
   freeze, or finalize a run. Execute only the scope explicitly authorized by
   the user.
4. **Proposal iteration** — Produce the next configured candidate batch for an
   active outer loop. Apply the strict read-only proposer contract below and
   return only schema-valid JSON.

For a mixed request, perform the authorized analysis or engineering first.
Enter Proposal iteration only when the final requested artifact is a candidate
batch and the active run supplies the required inputs and output schema.

## Apply the primary design rule

Constrain objectives, outputs, experiment boundaries, and safety-relevant
behavior. Do not prescribe a rigid diagnosis procedure.

Within the proposer-visible workspace, allow the proposer to inspect and query
permitted prompt code, search scores, sanitized traces, candidate lineage, and
prior-run history in whatever order is useful. Long-horizon experience may
become more informative than the initial skill text. Keep the contract strict
about what must be produced and what must never be accessed or changed.

## Preserve experiment integrity

Apply these invariants in every mode:

- Keep search, protected validation, and final test as distinct stages.
- Expose only search-visible artifacts to the proposer. Never expose
  protected-validation or final-test examples, IDs, labels, predictions,
  traces, metrics, reports, or paths.
- Split grouped data, such as papers or users, before sampling examples so
  stages remain group-disjoint.
- Evaluate every candidate on search data first. Promote only the configured
  top candidates to protected validation.
- Freeze the selected winner before final evaluation. Evaluate final test once.
- Keep finalized runs, evaluated candidates, manifests, hashes, and history
  append-only. Resume from durable state without repeating completed work.
- Require a new run identity when changing the solver, proposer model or
  effort, split, parser, generation settings, scoring, searchable surface, or
  search protocol. Never resume across those changes.
- Keep prompt-only experiments prompt-only. Unless the user explicitly changes
  the experiment definition, preserve evidence selection and order, image
  order, parser semantics, label vocabulary, solver settings, metrics, retry
  semantics, and call behavior.
- Estimate solver calls, proposer calls, tokens, and wall time by stage before
  live execution. Fail closed when the configured minimum protocol cannot fit
  the supplied ceilings.
- Keep evaluation outside the proposer. The outer loop owns solver calls,
  scoring, promotion, persistence, and finalization.

## Work from repository evidence

For Analysis, Engineering, and Experiment operation:

1. Read applicable `AGENTS.md` files and repository-local instructions.
2. Inspect the current branch, `git status`, relevant config snapshots,
   schemas, source, tests, and immutable run metadata.
3. Use `rg` to locate contracts. Trace behavior from CLI/config entry points
   through planning, orchestration, evaluation, persistence, resume, and
   reporting.
4. Distinguish observed evidence from inference. Do not infer unseen runs or
   silently repair ambiguous state.
5. Preserve unrelated user changes, secrets, and immutable run artifacts.
6. For multi-stage work, keep a short plan with one step in progress.
7. Implement the smallest coherent change. Update validation, tests, examples,
   CLI help, and config checks when behavior changes.
8. Run focused offline tests first, then the broadest relevant offline suite
   that is practical.

Do not make live solver, proposer, benchmark, or final-test calls unless the
user explicitly authorizes that execution.

## Build a new domain or audit an existing harness

Use the following as readiness gates, not as a script for proposer reasoning.

### Write and refine the skill contract

Specify:

- optimization objectives and reporting metrics;
- searchable and frozen surfaces;
- proposer-visible inputs;
- forbidden data, actions, and mutations;
- required candidate artifacts and schema;
- validation, promotion, freeze, and finalization ownership.

Keep diagnosis flexible. Avoid forcing the proposer through a fixed sequence of
error analyses when it can query the permitted experience directly.

Before a costly run, use several short evolution runs—typically 3–5 iterations
each—to debug the skill, artifacts, isolation, and output contract. Treat this
as engineering calibration, not benchmark evidence. Prefer improving the skill
contract over increasing iteration count or population size when the loop
behaves unreliably.

### Establish a simple baseline and hard search set

- Implement a reproducible baseline harness first, such as a minimal zero-shot
  or few-shot prompt.
- Build search data from baseline errors or a diverse set of difficult
  instances. Do not use protected-validation or final-test results to curate it.
- Keep the search set fast and discriminative. Target roughly 50 full
  candidate evaluations per development run when practical; 50–100 examples
  often works for classification, while other domains should use an equivalent
  budget justified by measured runtime and variance.
- Verify label, modality, difficulty, and group diversity. Record the selection
  rule, seed, source hashes, and group-disjointness.
- If the baseline saturates search evaluation, redesign the search set before
  expanding the search budget.

Treat these counts as starting heuristics, never hidden constants.

### Make the experience store navigable

Write machine-readable artifacts, preferably JSON or JSONL, using a stable
hierarchy and consistent, regex-friendly names. Record at least:

- immutable run, iteration, candidate, and parent identities;
- prompt or harness source and content hashes;
- aggregate search scores and per-example sanitized traces;
- parse coverage, infrastructure failures, latency, tokens, and call counts;
- config and split hashes, timestamps, status, and retry/checkpoint state;
- promotion and freeze decisions with their governing rule.

Separate infrastructure, parse, and reasoning failures. Never copy protected
data into proposer-visible summaries.

When history becomes large, provide a small deterministic CLI that can:

- list the Pareto frontier;
- show top-k candidates and candidate details;
- compare or diff two candidates and their search results;
- filter by run, iteration, metric, status, or failure type.

Prefer stable JSON output plus a concise human-readable view. If useful offline
experience exists—such as prior rollouts, solved corpora, or paper-derived
ideas—normalize it into the same structure, retain provenance, and verify that
it contains no protected or final-test leakage before using it to warm-start
search.

### Validate cheaply before full evaluation

Create a lightweight, deterministic validation step that:

- imports the candidate module or loads the prompt artifact;
- instantiates the required harness object;
- calls every required public method on tiny fixtures;
- validates schema, placeholders, label vocabulary, and parser compatibility;
- confirms frozen settings and call behavior are unchanged;
- completes without network access when services are mocked.

Run this check before expensive scoring. Reject malformed candidates without
spending the full evaluation budget.

### Automate evaluation outside the proposer

Use a separate evaluator/orchestrator to:

- validate candidates;
- call the fixed solver;
- score search results;
- write code, scores, traces, costs, and status to the experience store;
- update the search frontier;
- promote only eligible top candidates;
- checkpoint atomically and resume idempotently.

Do not ask the proposer to run evaluation or decide that its own candidate
improved before results exist.

## Operate the experiment lifecycle

### Preflight

Before live execution, verify and report:

- new or resumable run identity and immutable source commit;
- config, baseline prompt, dataset, split, and hard-search hashes;
- solver model and generation settings;
- proposer model, effort, candidate count, and prompt contract;
- search, protected-validation, and final-test stage sizes;
- requested and effective candidate, call, token, time, and failure ceilings;
- checkpoint path, atomic persistence, retry policy, and resume behavior;
- lightweight candidate validation;
- final-test lock and proposer visibility boundary.

Stop when any unresolved item could materially change validity, cost, or run
identity. A dry-run must make requested/effective limits and stage budgets
visible without making live calls.

### Live search and protected validation

- Start only the authorized stages.
- Record source/config/split identities before the first call.
- Enforce ceilings before each additional call, not after overspending.
- Persist completed work before reporting progress.
- Promote according to the frozen rule, then freeze the selected winner and its
  hashes.
- Keep final test locked during search and protected validation.

### Monitor and resume

- Establish whether a live process exists before starting or resuming.
- Inspect logs, checkpoint state, and run artifacts without revealing secrets.
- Never launch a duplicate process for the same run.
- Resume only after the prior process has stopped, using compatible immutable
  identity and the recorded durable state.
- Do not repeat completed samples, candidates, stages, or charged calls.
- Classify API failures as infrastructure failures, not model errors.

### Finalize

- Confirm the winner and all experiment-defining fields are frozen.
- Run final test only when explicitly authorized and only once.
- Report baseline-versus-winner metrics, parse coverage, failure rates, calls,
  tokens, wall time, and any prespecified subgroup analysis.
- Do not tune or rewrite the winner after observing final-test results.

## Engineering and operation permissions

In Engineering mode, it is permitted to read relevant non-secret artifacts,
edit repository code/config/tests/docs/skills, add deterministic utilities, run
formatters and offline tests, inspect diffs, and run planners with mocked
services.

Without explicit authorization, do not:

- make live API, solver, proposer, or benchmark calls;
- launch a costly experiment or final evaluation;
- push, publish, or modify external services;
- access, print, or move credentials;
- delete, overwrite, or rewrite run history.

## Strict proposal-iteration contract

Apply this section only in Proposal iteration.

### Required proposer inputs

Read the candidate output schema first, normally
`meta_harness/proposer/candidate_batch.schema.json`. Then use only files mounted
or enumerated as proposer-visible by the active run, such as:

- active experiment config and prompt contract;
- current run manifest and iteration identity;
- baseline, frontier, and allowed parent candidates;
- candidate lineage, prior prompt source, diffs, and search summaries;
- sanitized search scores, traces, and failure summaries;
- optional normalized offline experience approved for proposer visibility.

Treat the active config and schema as authoritative for candidate count,
template keys, prompt-length limits, required placeholders, label vocabulary,
parser-compatible output, fixed solver settings, and source-hash rules.

Within this permitted workspace, freely query the evidence using `rg` or the
experience-store CLI. Do not impose a fixed diagnosis order.

If a required input is missing, identify it and stop without fabricating state.
If a supplied file exposes protected-validation or final-test content, stop
reading it and report contamination to the outer loop.

### Remain read-only and isolated

Never:

- call a solver, evaluator, benchmark, or live service;
- make network requests;
- invoke subagents, another Codex process, or another proposer;
- create, edit, or persist files, including temporary prototypes;
- update frontier, history, reports, run state, promotion, or finalization;
- evaluate, rank, promote, select, or claim improvement for candidates;
- change model parameters, parsing, metrics, data mapping, evidence selection
  or order, images, retries, token budgets, or solver-call counts;
- include example IDs, gold labels, memorized answers, trace-specific facts,
  secrets, or protected paths in a candidate.

The outer loop owns all writes and evaluation.

### Produce falsifiable, general candidates

- Return exactly the configured number of materially distinct candidates.
- When at least two candidates are required and the config defines no different
  portfolio, include one exploitation candidate grounded in positive search
  evidence and one exploration candidate using a different mechanism.
- Tie each candidate to a recurring, general search failure pattern.
- State a falsifiable hypothesis with a predicted measurable effect and a
  plausible trade-off. Never state unmeasured improvement as fact.
- Reject candidates whose only difference is stronger wording, verbosity,
  reordered prose, generic steps, or requests for more reasoning.
- Keep sibling candidates mechanistically different.

Possible mechanisms include claim decomposition, independent support and
contradiction tests, modality-aware reconciliation, missing-versus-conflicting
evidence handling, calibrated tie-breaking, and parser-compatible decision
contracts. These examples are illustrative, not a required diagnosis sequence.

### Preserve the frozen prompt contract

For every candidate:

- include exactly the template keys required by the active config;
- change only template text;
- preserve every required placeholder exactly;
- preserve label vocabulary and parser-compatible output;
- preserve claims, context, captions, images, and their order;
- preserve fixed solver settings and call behavior;
- avoid dataset names, example-specific hints, external knowledge, tools, and
  unsupported citations;
- make every template a complete cold-start instruction.

Do not replace frozen baseline templates in repository state. Keep prompt
variant separate from reasoning method.

### Emit only the candidate batch

Validate the complete object against the active schema and all configured
limits. Use exact field names, types, nesting, IDs, parent references, template
keys, and hash semantics.

Once all required inputs are present, return only the schema-valid JSON object.
Do not add Markdown, commentary, status text, or code fences.

## Completion reports

For **Analysis**, lead with the finding and distinguish evidence from inference.

For **Engineering**, report changed files, implemented behavior, exact offline
test results, anything not run, and remaining risks.

For **Experiment operation**, report stage budgets and progress, checkpoints,
failures/retries, and whether final-test isolation remains intact.

For **Proposal iteration**, emit only the candidate JSON.