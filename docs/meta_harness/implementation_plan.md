# Meta-Harness implementation plan

> Historical phase plan: Prompts 1–8 have now implemented the V1 runtime.
> Resolved behavior and verified commands are documented in
> [`docs/meta-harness.md`](../meta-harness.md). Any “future,” “unknown,” or
> “unresolved” wording below records the preimplementation decision state and
> is not the runtime contract.

## Planning constraints

This is a future-work plan. The current phase creates no Meta-Harness runtime
module and changes no existing runtime behavior.

Every implementation phase must:

- Start from a clean understanding of existing user changes.
- Remain additive and provider-neutral.
- Keep `cot`, legacy providers, CLI defaults, prompts, reasoning methods,
  evidence order, and parser behavior compatible.
- Use only `API_KEY` and `API_URL` for live remote configuration.
- Keep tests offline and mock every proposer and solver process/request
  boundary.
- Avoid model downloads.
- Never expose gold labels to model-visible or proposer-visible inputs.
- Never write credentials, authorization headers, or image base64 data.

The missing Meta-Harness reference repository must be made available and its
requested onboarding, example, skill, and contract-test files reviewed before
runtime design is considered final.

## 1. Prompt registry and baseline equivalence

### Work

- Introduce a provider-neutral prompt registry capable of returning an
  immutable mapping of the four reasoning methods to `string.Template`
  objects.
- Register the existing `COT_PROMPT` mapping as `cot` without copying or
  rewriting prompt text.
- Define exact placeholder validation for one- and two-evidence methods.
- Define canonical candidate serialization and prompt SHA-256 hashing.
- Route the remote path through its actually selected prompt mapping while
  retaining the existing default and output layout for `cot`.
- Keep prompt variant separate from record `claim_type` and reasoning method.

### Tests

- Snapshot exact `cot` template text and placeholder sets.
- Compare fully prepared legacy and remote `cot` prompts before and after the
  registry change for all four reasoning methods.
- Compare image bytes and ordering for one- and two-image samples.
- Reject missing, extra, renamed, or malformed placeholders before any client
  call.
- Prove `cot` remains the default and existing CLI invocations behave
  identically.
- Prove module import creates no client or request.

### Completion gate

The existing prompt fingerprint and prepared-request fixtures are unchanged,
and all old tests pass without updated expected prompt text.

## 2. Macro-F1, result identity and paper split

### Work

- Add `yes` and `no` precision, recall, and F1 plus binary Macro-F1 to the
  offline metrics layer.
- Freeze and document the zero-denominator convention.
- Add candidate-aware configuration identity while preserving the existing
  meaning of result `method`.
- Include experiment, prompt variant, candidate/prompt hash, split hash,
  solver model, generation settings, and parser/evaluator version in the
  resumable identity.
- Add a SciVer paper-group split builder using a verified paper ID if one is
  available, otherwise the full SHA-256 digest of referenced paper JSON bytes.
- Persist an immutable split manifest with target 20/20/60 ratios and
  explicit no-overlap checks.

### Tests

- Hand-check balanced, imbalanced, missing-class, all-error, and retry cases
  for per-class F1 and Macro-F1.
- Verify failures remain in coverage and all-labeled accuracy denominators.
- Verify results never resume across different candidates, prompt hashes,
  split hashes, models, or generation configurations.
- Verify old result files remain readable and existing resume behavior
  remains compatible.
- Verify claims from one paper cannot cross splits, including duplicate paper
  content referenced by different paths.
- Verify missing/unreadable papers fail rather than falling back to claim-level
  splitting.
- Verify split construction is local and never opens HTTP or sockets.

### Completion gate

Macro-F1 is independently validated, identity collisions are covered, and the
split manifest proves pairwise-disjoint paper groups and samples.

## 3. Candidate schema and immutable storage

### Work

- Finalize a versioned candidate schema containing exactly four templates,
  immutable ID, parent lineage, iteration, prompt hash, and sanitized proposer
  metadata.
- Implement strict schema and placeholder validation.
- Implement atomic create-once candidate storage and hash verification on
  load.
- Store the baseline as a candidate-compatible immutable artifact without
  changing `COT_PROMPT`.
- Decide content-derived versus assigned candidate IDs after reviewing the
  reference contracts.

### Tests

- Round-trip valid candidates without changing bytes.
- Reject missing/extra templates, invalid placeholders, unresolved
  substitutions, duplicate IDs, hash mismatches, and attempts to overwrite.
- Reject credential-, authorization-, endpoint-, label-, or image-bearing
  fields outside the schema.
- Exercise interrupted writes and concurrent create conflicts.
- Confirm validation has no solver, credential, or network dependency.

### Completion gate

Every evaluated prompt is recoverable by immutable ID and hash, and stored
candidate content cannot be silently replaced.

## 4. Evaluator with fake solver

### Work

- Build an evaluator around the production SciVer adapter, request
  preparation, image serialization, parser, result writer, and new metrics.
- Require an injected solver boundary by default.
- Provide a deterministic fake solver for offline tests, including controlled
  parse failures, API failures, retries, and interruptions.
- Add candidate, split, generation, and parser hashes to evaluation manifests.
- Implement eligibility checks for parse coverage `1.0` and zero unresolved
  API failures.

### Tests

- Run all four reasoning methods with sentinel prompt text.
- Assert exact claim, context, caption, and image order preservation.
- Assert gold labels and explanations never enter messages or serialized
  request payloads.
- Assert raw image data never enters result or diagnostic files.
- Assert fake successes, parse failures, invalid inputs, API failures, and
  retries produce correct durable records and metrics.
- Assert default evaluation cannot instantiate a live client and explicit live
  gating remains required.

### Completion gate

The complete evaluator workflow runs offline with the fake solver, produces
reproducible metrics, and preserves all existing SciVer input semantics.

## 5. Codex CLI proposer wrapper

### Work

- Review the installed Codex CLI and the available Meta-Harness reference
  contracts without invoking it during discovery.
- Define a versioned, provider-neutral proposer input envelope and strict
  candidate-only output schema.
- Implement subprocess execution without shell interpolation, with timeout,
  output-size bounds, sanitized diagnostics, and explicit exit classification.
- Restrict proposer inputs to the frozen domain specification, allowed
  aggregate feedback, frontier state, and budget.
- Exclude final-test information, per-example gold correctness, credentials,
  endpoint values, image data, and raw solver payloads.
- Make the process boundary injectable for tests.

### Tests

- Use a fake executable or callable; never invoke Codex recursively in tests.
- Accept exactly two valid candidates by default.
- Reject malformed JSON, extra fields, invalid templates, duplicates, partial
  output, timeout, excessive output, and nonzero exit.
- Verify candidate text cannot become shell syntax.
- Verify sanitized logs contain no forbidden material.
- Verify proposer failure produces no solver calls and is classified as
  infrastructure failure.

### Completion gate

The wrapper converts only valid machine-readable proposals into immutable
candidates and has no access path to final-test or secret-bearing artifacts.

## 6. Orchestrator, frontier, budget and resume

### Work

- Implement the smoke, pilot, and main state machine from the experiment
  protocol.
- Track two candidates per iteration by default.
- Maintain a global-best frontier across baseline and every completed
  iteration.
- Rank only eligible candidates by validation Macro-F1, then Accuracy, then
  the frozen tertiary rule.
- Implement minimum 5, target 10, maximum 15 main iterations.
- Implement early stopping after three completed non-improving iterations
  using the `0.005` threshold.
- Exclude infrastructure-failed iterations from early stopping.
- Enforce hard candidate, solver-call, proposer-call, time, token, and cost
  budgets once exact ceilings are decided.
- Persist every state transition atomically and reconstruct state on resume.

### Tests

- Cover global-best retention when later candidates regress.
- Cover qualifying, sub-threshold, and tied improvements.
- Cover the minimum-iteration gate and maximum-iteration stop.
- Cover proposer, solver, and storage failures without advancing early
  stopping.
- Cover interruption after every state transition and exact resume without
  duplicate proposal or evaluation.
- Cover budget exhaustion at each boundary.
- Prove final-test artifacts are neither read nor summarized by the
  orchestrator before freeze.

### Completion gate

An entirely fake smoke and pilot run can stop, resume, and reproduce the same
frontier and budget ledger from immutable artifacts.

## 7. Finalization, `meta_cot` export and model transfer

### Work

- Atomically freeze the global winner, configuration, split, parser, metrics,
  and repository revision.
- Lock proposal and frontier mutation after freeze.
- Permit one resumable final-test evaluation of the exact frozen identity.
- Export the winning four-template mapping additively as `meta_cot`.
- Preserve `cot` byte-for-byte and keep old CLI behavior unchanged.
- Evaluate the same frozen prompt without re-optimization on Qwen and all
  three supported Gemma models.
- Give each model transfer an isolated configuration and result identity.

### Tests

- Reject finalization of an ineligible, non-frontier, or hash-mismatched
  candidate.
- Reject any post-freeze candidate or split mutation.
- Prove final-test outcomes cannot call the proposer or update the winner.
- Prove `meta_cot` resolves to the exact winner text and `cot` remains
  unchanged.
- Prove model transfer changes only the model/configuration identity, not
  prompt text or split.
- Mock every solver request and verify no model weights are downloaded.

### Completion gate

The frozen artifact fully determines `meta_cot`, the final test is isolated,
and all four model reports share one prompt hash with no re-optimization.

## 8. Documentation and full offline verification

### Work

- Update user documentation for terminology, split creation, candidate
  artifacts, fake workflows, explicit live gating, resume, finalization, and
  transfer.
- Record all resolved decisions previously marked unknown.
- Document compatibility and migration behavior for old results and CLI
  invocations.
- Inspect the complete diff for prompt changes, image reordering,
  ground-truth leakage, secrets, provider identity, model downloads, and
  unrelated changes.

### Required verification

Run:

```text
python -m compileall -q .
pytest -q
git diff --check
```

Also:

- Confirm the suite-wide HTTP and socket denial fixture remains active.
- Inspect all tests that replace the request or subprocess boundary.
- Confirm no default or test path makes a live request.
- Confirm no import creates a remote client or downloads model weights.
- Compare the final `cot` prompt fingerprint and prepared-request snapshots to
  the baseline.
- Verify only intended files changed and review `AGENTS.md` for conflicts,
  vague instructions, or provider-specific language.

### Completion gate

All relevant tests pass offline, every definition-of-done requirement in
`AGENTS.md` is satisfied, and no live request, recursive proposer invocation,
commit, push, or merge is performed as part of verification.

## Decisions required before runtime implementation

1. Make the Meta-Harness reference repository available and reconcile its
   exact contracts with this plan.
2. Choose the deterministic paper-group allocation algorithm, seed, and
   stratification policy.
3. Freeze the per-class F1 zero-denominator convention.
4. Choose the validation promotion/pruning policy after search screening.
5. Choose the deterministic tertiary winner tie rule.
6. Verify endpoint support and select exact `temperature`, `max_tokens`, and
   `seed` behavior.
7. Set proposer, solver, token, time, cost, and retry budgets.
8. Decide how partially completed iterations count after infrastructure
   failure.
9. Define exact final-test interruption and resume semantics.
10. Decide whether Qwen transfer reuses an equivalent frozen final-test run.
