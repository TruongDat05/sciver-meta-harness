# Meta-Harness experiment protocol

> Historical experiment draft: use
> [`docs/meta-harness.md`](../meta-harness.md) for implemented ranking,
> budgets, early stopping, isolation, finalization, and transfer behavior.

## Purpose

This protocol defines the initial SciVer V1 prompt-search experiment. It must
be frozen before the main run. Deviations require a new experiment identity
and cannot retroactively replace the original protocol.

## Fixed roles

- Proposer: Codex CLI with `gpt-5.6-terra` and `medium` reasoning.
- Search solver: remote OpenAI-compatible `gemma-4-26B-A4B-it`.
- Candidate unit: four prompt templates, one for each reasoning method.
- Mutable content: prompt text only.
- Primary selection metric: validation Macro-F1.
- Secondary selection metric: validation Accuracy.
- Dataset: SciVer only.

The parser, normalized labels, evidence selection, image order, solver model,
and frozen generation configuration are not candidate-controlled.

## Dataset qualification

Before splitting:

1. Inspect representative SciVer records or a generated schema report.
2. Load the complete selected corpus through the existing SciVer adapter.
3. Require a readable local paper JSON for every selected record.
4. Validate every record through production request preparation without
   creating a client or reading credentials.
5. Record sample, reasoning-method, and label distributions.
6. Fail on insufficient schema evidence instead of guessing mappings.

Generated SciVer records currently do not include `paper_id`. Unless a
verified stable identity is available, compute the paper group as the full
SHA-256 digest of the referenced paper JSON bytes.

## Leakage-safe split

Split paper groups, never individual claims.

Recommended group-level target ratios:

- Search: 20%
- Validation: 20%
- Final test: 60%

Required invariants:

- No paper digest appears in more than one split.
- No sample ID appears in more than one split.
- Every selected sample appears exactly once.
- Split membership is deterministic from the frozen algorithm and seed or
  ordering rule.
- The manifest records achieved sample and paper counts, because group sizes
  may prevent exact ratios.
- The final-test membership is inaccessible to the proposer.

The exact group allocation algorithm, seed, and any stratification by label or
reasoning method are unresolved. They must be chosen before splitting and
cannot be selected by inspecting final-test outcomes.

## Prompt baseline

The baseline candidate is the current `COT_PROMPT` mapping, registered as
`cot`. Before search:

- Record its canonical prompt hash.
- Verify exact placeholder signatures.
- Verify request equivalence for all four reasoning methods.
- Evaluate it with the same split, solver, parser, generation configuration,
  retry policy, and metric code used for candidates.

Baseline results are part of the frontier. A proposed candidate must beat the
global best under the same eligibility and ranking rules to become the winner.

## Candidate evaluation

Each proposed candidate is validated as an indivisible four-template object.
Invalid candidates consume proposer output but make no solver calls and do not
count as completed candidate evaluations.

For a valid candidate:

1. Run the declared search screening subset.
2. Resolve infrastructure failures according to the frozen retry policy.
3. Compute search metrics and eligibility diagnostics.
4. Run validation according to the frozen iteration policy.
5. Compute validation metrics from final attempts.
6. Update the global-best frontier if the candidate is eligible and ranks
   higher.

Whether every valid candidate reaches full validation or search screening may
prune candidates is unresolved. The choice must be fixed before the pilot and
must not use final-test information.

## Metrics

For actual class \(c\) and predicted class \(c\):

- Precision is `TP_c / (TP_c + FP_c)`.
- Recall is `TP_c / (TP_c + FN_c)`.
- F1 is the harmonic mean of precision and recall.
- Macro-F1 is `(F1_yes + F1_no) / 2`.
- Accuracy is correct predictions divided by all labeled examples.
- Parse coverage is successful parsed predictions divided by all selected
  examples.

Failures remain in the coverage and all-labeled accuracy denominators. They
must not be silently dropped to improve a score.

The zero-denominator convention for per-class F1 is unresolved and must be
frozen with unit tests before candidate evaluation.

## Winner eligibility and ranking

A candidate is eligible only when:

- Validation parse coverage is exactly `1.0`.
- Validation has zero unresolved API failures.

Eligible candidates rank by:

1. Higher validation Macro-F1.
2. Higher validation Accuracy.
3. A deterministic tertiary tie rule declared before the main run.

The winner is the global best across the baseline and all completed
iterations. It is never defined as the final candidate, the latest candidate,
or the winner of only the last iteration.

Invalid inputs are experiment-configuration failures, not eligible outcomes.
The exact policy for a candidate-independent invalid input discovered after
search begins is unresolved; the safe default is to halt and issue a new split
or experiment manifest rather than alter the sample set in place.

## Iteration defaults

### Smoke

- Iterations: 2
- Candidates per iteration: 2
- Examples: 10
- Purpose: exercise proposal, schema validation, fake or gated solver
  evaluation, metrics, frontier, persistence, and resume.

The 10 examples must come from the search split. Smoke outcomes do not select
the final winner unless a new protocol explicitly promotes the smoke run into
the main experiment.

### Pilot

- Iterations: 5
- Candidates per iteration: 2
- Screening size: 150 examples
- Purpose: validate cost estimates, failure rates, parse coverage, metric
  stability, and proposer contract before the main budget is opened.

The 150 examples must be a frozen subset of the search split. If the search
split has fewer than 150 examples, the behavior is currently unresolved and
must be recorded before the pilot.

### Main

- Minimum iterations: 5
- Target iterations: 10
- Maximum iterations: 15
- Candidates per iteration: 2 by default

The main run may stop after the minimum when the early-stopping condition is
met. It should continue toward the target while budget and stopping rules
permit, and must never exceed the maximum.

## Early stopping

Early stopping occurs after three completed iterations without a validation
Macro-F1 improvement of at least `0.005` over the previous global best.

Rules:

- Compare global-best validation Macro-F1 values, not iteration-local winners.
- An improvement smaller than `0.005` counts as no qualifying improvement.
- Reset the counter only on a qualifying improvement.
- Do not evaluate early stopping before five main iterations have completed.
- Infrastructure-failed iterations do not count as completed iterations and
  do not increment the counter.
- Invalid proposer output or an unavailable solver is an infrastructure
  failure unless the frozen policy classifies it otherwise.

How to count an iteration in which one of two candidates completes and the
other has an unresolved infrastructure failure is unresolved and must be
declared before the main run.

## Infrastructure failure handling

Infrastructure failures include proposer process failure, request transport
failure, exhausted retryable server failure, or persistence failure. They are
distinct from a model response that parses incorrectly.

The frozen policy must specify:

- Maximum proposer retries.
- Maximum solver retries and backoff.
- Whether retry attempts reuse the same semantic generation seed.
- When a failure becomes unresolved.
- Whether partial candidate evaluation resumes or restarts.
- How budget accounting treats failed attempts.

Infrastructure failures:

- Never receive an artificial metric score.
- Never make a candidate eligible.
- Never advance early stopping.
- Remain durably recorded for resume and audit.

## Generation reproducibility

Before the pilot, freeze and hash:

- Model identifier.
- `temperature`.
- `max_tokens`.
- `seed` or an explicit verified "unsupported" state.
- `n = 1`.
- Non-streaming behavior.
- Timeout and retry configuration.

The current remote client does not send `temperature`, `max_tokens`, or
`seed`; no values are selected by this document. A future live compatibility
check is required, explicitly opted into and kept outside tests.

If the solver is nondeterministic despite fixed supported controls, the number
of repetitions and aggregation rule are unknown and must be decided before
the main run, not after viewing validation or final-test results.

## Budget accounting

The orchestrator must record, at minimum:

- Proposed, valid, screened, and fully evaluated candidates.
- Solver calls attempted, succeeded, retried, and unresolved.
- Examples evaluated by split.
- Completed and infrastructure-failed iterations.
- Proposer calls and retries.
- Elapsed time if useful, without using it as identity.

Exact call, token, time, and cost ceilings are unresolved. The main experiment
must not start until hard ceilings and the behavior at each ceiling are added
to the frozen configuration.

## Freeze and final test

After search ends:

1. Select the global best using only validation metrics and eligibility.
2. Freeze the candidate ID, prompt hash and all four prompt texts.
3. Freeze the split manifest, solver configuration, parser, metrics, retry
   policy, and repository revision.
4. Write the winner manifest atomically.
5. Disable proposal and frontier mutation for that experiment.
6. Run the final-test split exactly once under the frozen configuration.

Final-test examples, predictions, labels, metrics, failures, or availability
signals must never reach the proposer. A final-test result cannot trigger
another candidate or change `meta_cot`.

The policy for a genuine infrastructure interruption during the one final-test
run must distinguish resume of the same immutable identity from a second
experiment. This exact recovery rule is unresolved.

## Post-freeze model transfer

After the winner and final-test protocol are frozen, evaluate the exact same
four prompt texts without re-optimization on:

- `Qwen2.5-VL-7B-Instruct`
- `gemma-4-31B-it`
- `gemma-4-26B-A4B-it`
- `gemma-3-27b-it`

Each model uses its own verified fixed generation configuration and result
identity. Model-specific prompt edits, candidate selection, or parser changes
are prohibited. Transfer results are analysis outcomes, not input to the
already frozen prompt search.

The `gemma-4-26B-A4B-it` report reuses the frozen search-model final-test
result. The other three models use transfer-labeled identities, avoiding a
duplicate search-model solver call.

## Reporting

Report for baseline, each candidate, winner, final test, and transfer runs:

- Candidate ID and prompt hash.
- Model and generation-configuration hash.
- Split-manifest hash and sample count.
- Per-class precision, recall, and F1.
- Macro-F1 and Accuracy.
- Parse coverage.
- Failure counts by type.
- Eligibility.
- Attempt and retry counts.

Never report credentials, endpoint values, authorization headers, image data,
ground-truth-bearing prompts, or remote provider identity.
