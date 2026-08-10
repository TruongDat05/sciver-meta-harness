# AGENTS.md

These rules apply to the entire repository. They are the permanent invariants
for the SciVer full-search Meta-Harness experiment. Treat the privacy,
security, experiment-isolation, and API-safety rules below as hard
requirements.

If legacy code, documentation, configuration, tests, or skill text conflicts
with this file, this file governs. Do not preserve the old experimental
protocol when doing so would violate these invariants.

## Project goal

Redesign the SciVer Meta-Harness as a prompt-only optimization system with
three independent roles:

- Codex CLI is the prompt proposer and optimization agent.
- Qwen/Qwen3.5-35B-A3B, served by vLLM through an OpenAI-compatible HTTP API,
  is the multimodal solver.
- Trusted Python code executes evaluations, computes metrics, manages state,
  and selects the winner.

The Meta-Harness Python process must never load model weights. GPU count,
tensor parallelism, memory settings, and other server-specific vLLM options
belong to the server launch configuration, not application logic.

Existing non-Meta-Harness providers and CLI behavior remain outside this
redesign and must not be changed incidentally. Compatibility with the old
Meta-Harness experiment protocol, configuration schema, or run directories is
not required.

## Privacy and provider neutrality

- Aside from the compatibility-protocol, proposer, framework, and model
  identifiers explicitly required by this file, do not identify or advertise
  a remote provider in source, documentation, file names, class names, CLI
  arguments, logs, tests, or artifacts.
- Use generic implementation names such as remote, remote_api, api_provider,
  and remote_model.
- Do not expose private dataset material, credentials, or sensitive request
  payloads through exceptions, diagnostics, logs, caches, or proposer input.

## Fixed experiment protocol

### Data groups

- The experiment has exactly two scored data groups: SEARCH and FINAL.
- SEARCH contains exactly 1,000 SciVer samples.
- FINAL contains exactly 1,000 SciVer samples.
- The split is deterministic with fixed seed 42.
- SEARCH and FINAL must be paper-disjoint.
- Derive paper identity from a verified paper identifier or a SHA-256 identity
  of the referenced paper JSON. Never split by claim.
- When complete paper groups cannot themselves total exactly 1,000 per side,
  assign papers exclusively to SEARCH or FINAL pools and deterministically
  sample exactly 1,000 claims within each pool. Extra claims remain unused.
- If the corpus has fewer than 2,000 eligible samples, or no paper-exclusive
  assignment can supply both exact counts, preparation must fail with a clear
  feasibility report. Never duplicate samples, split a paper across groups, or
  silently reduce either count.
- Do not use labels, predictions, difficulty, or model errors to choose split
  membership.

### Search evaluation

- Evaluate the canonical initial prompt on all 1,000 SEARCH samples.
- Score every valid prompt candidate on all 1,000 SEARCH samples.
- All candidates use the same immutable SEARCH membership.
- Representative or anonymous errors may inform proposer feedback, but they
  must never become an evaluation subset.
- Hard-sample mining and hard-search objectives are prohibited.
- Do not implement or use 50-, 80-, 100-, smoke-, guard-, screening-, or other
  reduced candidate-scoring subsets.
- Top-K promotion and protected-validation stages are prohibited.
- The baseline remains eligible to win.
- Candidate evaluation and winner selection belong to trusted Python code.
  Codex must not decide subjectively which prompt wins.

### Selection

- SEARCH Macro-F1 is the primary selection metric.
- SEARCH Accuracy is the secondary selection metric.
- Resolve any remaining tie deterministically by prompt SHA-256 ascending,
  then candidate ID ascending.
- Use the same frozen ordering for frontier updates, best-candidate selection,
  early stopping, freezing, reporting, and resume reconciliation.
- A tie resolved only by hash or candidate ID does not count as a metric
  improvement for patience.

### Search schedule

- Codex proposes exactly one candidate prompt family per iteration.
- Run at least 3 completed candidate iterations.
- Run at most 10 candidate iterations.
- Stop early after 3 consecutive completed candidates that do not improve
  Macro-F1 or, at equal Macro-F1, Accuracy.
- A duplicate or schema-invalid proposal is rejected before solver calls and
  may be regenerated through a bounded proposer retry.

### Final evaluation

- FINAL must remain completely isolated throughout prompt optimization.
- FINAL samples, IDs, paper identities, labels, predictions, metrics, traces,
  reports, and paths must never enter proposer input or search selection.
- FINAL results must never influence candidate generation, candidate
  selection, early stopping, prompt modification, or the search frontier.
- Freeze the deterministic SEARCH winner before any FINAL inference.
- Evaluate only the frozen winner on all 1,000 FINAL samples.
- FINAL evaluation requires a separate explicit command, an explicit live
  request flag, and an explicit final confirmation flag.
- FINAL is one logical evaluation. Transport retries and interruption recovery
  may resume missing work, but a completed FINAL run must not be dispatched
  again.
- Final metrics cannot mutate the winner or trigger another proposal.

## Prompt-only candidate contract

- A candidate contains exactly the direct, analytical, parallel, and
  sequential prompt templates.
- Only the text of those four templates may change.
- Preserve every template's placeholder interface exactly.
- Keep prompt variant, such as cot, meta_cot, or candidate ID, separate from
  reasoning method.
- Preserve the canonical cot prompt exactly. Expose the frozen winner only
  through the additive meta_cot variant.
- Preserve claims, context, captions, image count and order, evidence
  selection, parser behavior, label vocabulary, solver model, generation
  settings, and call behavior outside candidate control.
- Ground-truth labels must never enter prompts, message payloads,
  demonstrations, proposer-visible examples, or other model-visible input.

## Proposer boundary

- Codex CLI is independent from the solver API and uses its own terminal
  authentication.
- The proposer receives only approved SEARCH-derived information: parent
  prompts, aggregate metrics, confusion counts, aggregate error categories,
  parser/failure counts, candidate history, and opaque overlap keys.
- Do not expose raw claims, context, captions, images, model responses,
  predictions, labels, sample IDs, private paths, or FINAL commitments to the
  proposer.
- Run Codex through a bounded, audited subprocess with a sanitized
  environment, read-only sandbox, fixed output schema, timeout, and output
  limits.
- The Python outer loop validates and persists proposer output. Codex must not
  call the solver, run the benchmark, update state, select the winner, or
  access FINAL artifacts.
- Tests must never invoke Codex recursively; use a fake subprocess boundary.

## Solver and API safety

- Read credentials only from the API_KEY environment variable.
- Read the endpoint only from the API_URL environment variable.
- Never hardcode an API key or endpoint.
- Never log, print, serialize into diagnostics, cache, or expose the API key,
  an Authorization header, or image base64 data.
- Never use realistic-looking secrets in tests, fixtures, examples, or
  documentation. Use unmistakably fake placeholders.
- Importing a module must never make an API request or cause another remote
  side effect.
- Live requests are opt-in only and require an explicit provider-neutral CLI
  flag. The default path remains offline.
- Unit and integration tests must never access the network. Mock every HTTP
  call at the request boundary and block unexpected socket access.
- Never download model weights from application or test code.
- Preserve image ordering through dataset loading, message construction,
  serialization, caching, retry, and request dispatch.
- Never guess dataset field mappings. Inspect representative samples or a
  schema report first and return an actionable error when the schema is
  insufficient.

## Deterministic solver configuration

The initial approved solver configuration is:

- model: Qwen/Qwen3.5-35B-A3B
- temperature: 0
- top_p: 1
- seed: 42
- n: 1
- stream: false
- max_tokens: 8192

Record the immutable model revision, served-model identity, vLLM version, and
deployment fingerprint in each run. Changing the model, revision, deployment,
generation settings, split, parser, evaluator, prompt contract, scoring, or
search protocol requires a new run identity.

## Durability, caching, and resume

- Every run must support persistent state, per-sample checkpointing, bounded
  retries, interruption recovery, and deterministic resume.
- Persist each completed sample attempt before considering it complete.
- Resume must not repeat completed samples or completed proposer iterations.
- Use an exclusive run lock to prevent duplicate processes for one run.
- Cache SEARCH completions by at least prompt, model and revision, deployment,
  sample content, ordered image content, generation configuration, and cache
  schema version.
- Cache only successful HTTP completions, including successfully returned but
  unparsable text. Do not cache transport failures or invalid input.
- FINAL must not read the global SEARCH completion cache. It uses only
  run-local checkpoints for interruption recovery.
- Do not place credentials, endpoints, authorization data, gold labels, raw
  prompts, or image base64 data in cache metadata.
- Treat returned but unparsable responses as abstaining/incorrect predictions
  on the fixed 1,000-sample denominator. Treat unresolved infrastructure
  failures as incomplete evaluation, not model errors.
- Persist immutable run/config/split/prompt identities and reconcile mutable
  state from durable sample and proposal checkpoints.

## Development rules

- Milestone 0 in docs/IMPLEMENTATION_PLAN.md is the approved design baseline.
  Complete it before application-code changes.
- Follow the milestone order and keep the implementation status checklist
  current.
- Keep changes narrowly scoped and avoid unrelated cleanup.
- Reuse established repository patterns before adding abstractions.
- Add or update tests for every new behavior, error path, and safety gate.
- Return clear, actionable errors without secrets or sensitive payload data.
- Do not commit or push on behalf of the user.
- Do not modify unrelated files.
- Do not delete legacy code until its callers, tests, and documentation have
  been traced and migrated.

## Verification

Before declaring implementation complete, run:

    python -m compileall -q .
    pytest -q
    git diff --check

Confirm from the test setup and results that every HTTP request was mocked and
no test made a live request. Review the final diff for:

- secret material, endpoints, authorization headers, or realistic credentials;
- remote-provider advertising or provider-specific names outside required
  compatibility-protocol and model identifiers;
- image base64 data;
- accidental changes to the canonical prompts or placeholder interfaces;
- image reordering;
- ground-truth or FINAL leakage;
- hard-search, subset-scoring, promotion, or protected-validation behavior;
- implicit model downloads;
- unrelated changes.

## Definition of done

The redesign is complete only when:

- SEARCH and FINAL each contain exactly 1,000 paper-disjoint samples.
- The baseline and every candidate are scored on the same complete SEARCH set.
- No hard-search or protected-validation objective remains reachable.
- Python applies the approved deterministic ranking.
- FINAL isolation is enforced and tested.
- Resume, checkpointing, retries, caching, and concurrency are tested offline.
- Relevant tests and required verification commands pass.
- Existing non-Meta-Harness behavior remains compatible.
- No secret, sensitive payload, or unrelated change is present.

## Meta-Harness skill

Use the repository Meta-Harness skill for Meta-Harness analysis, engineering,
operation, and explicit proposal iterations. Select the operating mode that
matches the request.

The current skill contains legacy hard-search and protected-validation
guidance. Until that skill is updated in Milestone 1, ignore any skill guidance
that conflicts with this file or docs/IMPLEMENTATION_PLAN.md. An explicit
proposal iteration remains read-only and must emit only schema-valid candidate
JSON.
