# Meta-Harness architecture

> Historical design document: use
> [`docs/meta-harness.md`](../meta-harness.md) for the implemented architecture,
> artifact layout, commands, and resolved runtime behavior.

## Design principles

The integration is additive around the existing SciVer pipeline:

- Keep `cot` and all legacy and remote behavior unchanged.
- Treat a prompt candidate as data, not executable code.
- Reuse the existing dataset adapter, request preparation, image
  serialization, answer parser, and result durability boundaries.
- Make prompt variant and reasoning method independent identity fields.
- Make every search decision reproducible from immutable local artifacts.
- Keep all tests offline with fake proposer and solver boundaries.
- Require the existing explicit live-request gate for any future solver call.

This document records the architecture proposed before implementation.

## Component view

```text
Verified SciVer input
        |
        v
Paper-group split builder -----> immutable split manifest
        |
        +-------- search -----------+
        |                           |
        |                    Candidate store <----- Codex CLI proposer wrapper
        |                           |                         ^
        |                           v                         |
        |                    Candidate validator              |
        |                           |                         |
        |                           v                         |
        |                    Solver evaluator ----------------+
        |                           |
        |                           v
        |                    Metrics and eligibility
        |                           |
        +-------- validation -------+
                                    |
                                    v
                          Global-best frontier
                                    |
                                    v
                        Frozen winner manifest
                           /                 \
                          v                   v
                 one final-test run     additive meta_cot
                                              |
                                              v
                                  post-freeze model transfer
```

Final-test artifacts are downstream of the frozen winner and have no path back
to the proposer, candidate store, or frontier.

## Prompt registry

The future registry has three logical kinds of entry:

- `cot`: the existing immutable four-template mapping.
- Candidate ID: an immutable four-template mapping loaded from candidate
  storage during search.
- `meta_cot`: a frozen four-template mapping exported from the winner.

The registry returns `Mapping[str, string.Template]`. It never returns one
string for the full candidate.

Registry validation must enforce:

- Exactly the four known reasoning-method keys.
- Exact placeholder sets for each key.
- Successful substitution with sentinel strings.
- Stable canonical serialization and SHA-256 fingerprinting.
- No mutation of the registered templates after construction.

Baseline-equivalence tests compare legacy `cot` substitutions and fully
prepared requests before and after registry integration. The existing
`COT_PROMPT` constant remains the source of truth for `cot`.

## Split builder

The split builder operates only on verified normalized SciVer records. It:

1. Resolves each record's local `paper_path`.
2. Hashes the referenced paper JSON bytes with SHA-256.
3. Groups every claim sharing a digest.
4. Assigns complete groups to search, validation, or final test.
5. Writes an immutable manifest containing dataset fingerprint, algorithm
   version, split seed or ordering rule, group digests, sample IDs, counts, and
   achieved ratios.
6. Verifies that group and sample intersections are empty.

The exact deterministic allocation algorithm and seed are unresolved. They
must be selected before the first experiment, written to the protocol, and
never changed after final-test isolation begins.

Split manifests are evaluator inputs. The proposer receives only the
information permitted by the domain specification and never receives the
final-test membership.

## Candidate schema and immutable store

Each candidate artifact should contain at least:

```json
{
  "schema_version": 1,
  "experiment_id": "immutable experiment identifier",
  "candidate_id": "content-derived or assigned immutable identifier",
  "parent_candidate_id": "nullable parent identifier",
  "iteration": 0,
  "candidate_index": 0,
  "templates": {
    "direct": "template text",
    "analytical": "template text",
    "parallel": "template text",
    "sequential": "template text"
  },
  "prompt_sha256": "full hexadecimal digest",
  "proposal_metadata": {
    "proposer": "codex_cli",
    "proposer_configuration_hash": "full hexadecimal digest"
  }
}
```

Timestamps may be recorded for audit but must not participate in the content
hash. Candidate IDs, canonicalization rules, and whether IDs are content
derived are unresolved pending reference-contract review.

Candidate files are create-once. Rewriting an existing candidate path or
reusing an ID for different text is an error. The store writes atomically and
checks the prompt hash on every load.

## Codex CLI proposer wrapper

The proposer wrapper is a subprocess boundary with:

- A fixed, versioned proposer instruction.
- A machine-readable input envelope containing only allowed context.
- A strict machine-readable output schema containing exactly two candidate
  objects by default.
- A timeout, output-size limit, and captured exit status.
- No shell interpolation of candidate text.
- Sanitized logs that exclude credentials, endpoint values, image data,
  model payloads, and final-test information.

Malformed, incomplete, duplicate, or contract-violating proposals are rejected
before candidate storage or solver evaluation. A proposer failure is an
infrastructure event, not a low-scoring candidate.

The wrapper must be injectable so tests use a fake proposer executable or
callable. This documentation phase must not invoke Codex recursively.

The exact Codex CLI invocation, output contract, and retry behavior are
unknown until the missing reference repository is available and the installed
CLI interface is inspected during implementation.

## Solver evaluator

The evaluator takes an immutable candidate, split manifest, solver
configuration, and result location. It:

1. Loads candidate templates through the validated prompt registry.
2. Loads only sample IDs assigned to the requested split.
3. Delegates evidence and message construction to the production SciVer
   request-preparation path.
4. Dispatches through an injected solver boundary.
5. Parses with the existing answer parser.
6. Appends a durable result for every attempt.
7. Produces metrics from final attempts only.

The evaluator does not allow a candidate to supply model parameters, message
roles, system prompts outside the frozen contract, evidence, parser rules, or
labels.

The fake solver is the default test boundary. It records message structure,
returns deterministic explicit answers or controlled failures, and makes no
network request. Tests must show that candidate text changes while claims,
context, captions, image order, and labels remain fixed.

## Generation configuration

Comparable solver calls require an immutable configuration including explicit
`temperature`, `max_tokens`, and `seed` support status. The current client
omits all three values; the architecture does not assume they are already
fixed.

Before main experiments:

- Verify which controls the compatibility endpoint accepts.
- Select and record exact values.
- Include values and "unsupported" states in configuration hashing.
- Ensure resume rejects a changed setting.
- Keep retry and timeout policy separate from semantic generation settings.

Any live compatibility check remains an explicit, provider-neutral, opt-in
operation outside the offline test suite.

## Result identity

Meta-Harness identity must distinguish:

- Experiment ID.
- Dataset fingerprint.
- Split name and split-manifest hash.
- Sample ID.
- Solver model.
- Prompt variant.
- Candidate ID and prompt hash when applicable.
- Reasoning method.
- Generation-configuration hash.
- Parser/evaluator version.

The current writer's identity is only
`run_id`, `sample_id`, `dataset`, `model`, and `method`. A backward-compatible
design may encode the new immutable configuration into a collision-resistant
`run_id` and add optional explicit fields, but it must not change the current
meaning of `method`. The exact migration is unresolved and requires tests with
old result files.

Resume rules:

- Skip only an exact successful identity.
- Retry exact failed identities according to the frozen failure policy.
- Never reuse a result across candidates, prompt hashes, splits, models, or
  generation settings.
- Reject duplicate attempt numbers and manifest mismatches.

## Metrics and eligibility

The metrics layer must add:

- `yes` precision, recall, and F1.
- `no` precision, recall, and F1.
- Macro-F1 as the arithmetic mean of both class F1 values.
- Accuracy.
- Parse coverage.
- Resolved and unresolved failure counts.

The class-F1 zero-denominator convention must be specified and tested before
search. The recommended convention is F1 `0.0` when a class has no true
positives and its precision or recall denominator is zero, but this remains a
decision until aligned with the unavailable reference contracts.

Eligibility is evaluated before ranking:

- Parse coverage must equal `1.0`.
- Unresolved API failures must equal zero.

Eligible candidates rank by validation Macro-F1, then validation Accuracy,
then a predeclared deterministic tertiary rule.

## Orchestrator and global-best frontier

The orchestrator is a local state machine:

```text
initialized
  -> baseline_evaluated
  -> proposing
  -> candidates_validated
  -> screening
  -> validation
  -> frontier_updated
  -> next_iteration | early_stopped | budget_exhausted
  -> frozen
  -> final_test_complete
  -> transfer_complete
```

Every transition is written atomically with input and output artifact hashes.
On resume, the orchestrator reconstructs state from immutable artifacts rather
than rerunning completed work or trusting a mutable counter.

The frontier stores the global best across all completed iterations. It never
assumes the latest candidate or iteration is best. Infrastructure-failed
iterations do not update the frontier and do not advance the early-stopping
counter.

## Finalization and transfer

Finalization writes a winner manifest that freezes:

- Candidate ID and prompt hash.
- All four prompt texts.
- Dataset and split manifest hashes.
- Solver and generation configuration.
- Parser and metric definitions.
- Selection metrics and eligibility evidence.
- Repository revision.

Only after this file is durably frozen may the final-test split be evaluated.
The final-test result cannot trigger another proposal or change the winner.

The exact winning mapping is then exported as `meta_cot`. The final matrix
evaluates the same frozen prompt, without re-optimization, on:

- `Qwen2.5-VL-7B-Instruct`
- `gemma-4-31B-it`
- `gemma-4-26B-A4B-it`
- `gemma-3-27b-it`

`gemma-4-26B-A4B-it` uses the guarded search-model final-test identity. Each of
the other three models uses a separate transfer identity while sharing the
winner prompt hash and split manifest.

## Safety and compatibility tests

Future tests must prove:

- Importing every new module has no remote side effect.
- Default evaluator and proposer tests cannot access HTTP or sockets.
- Live solver dispatch is impossible without the existing explicit flag.
- No prompt payload contains `gold_label` or other gold-only fields.
- No artifact or diagnostic contains credentials, authorization headers, or
  image base64 data.
- One- and two-image order is preserved end to end.
- `cot` prepared requests are baseline equivalent.
- Candidate IDs and prompt hashes isolate resume state.
- Final-test artifacts cannot be loaded by the proposer path.
- No code path downloads model weights.
