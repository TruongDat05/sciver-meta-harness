# Meta-Harness domain specification

> Historical design specification: use
> [`docs/meta-harness.md`](../meta-harness.md) for the implemented V1 contract,
> commands, and resolved decisions.

## Objective

V1 searches for a better SciVer prompt while keeping the solver, task,
evidence, parser, labels, and evaluation procedure fixed. Codex CLI is the
proposer, fixed to `gpt-5.6-terra` with `medium` reasoning.
`gemma-4-26B-A4B-it` is the fixed remote OpenAI-compatible search solver.

This is prompt optimization only. It is not fine-tuning, adapter training,
LoRA, weight modification, or model selection.

## Scope

V1 includes:

- The SciVer dataset only.
- Binary claim verification with the existing normalized `yes` and `no`
  labels.
- The existing four SciVer reasoning methods.
- Candidate proposal, immutable storage, offline-safe evaluation, validation
  selection, final freezing, additive export, and post-freeze model transfer.

V1 excludes:

- Prompt search on SciAtomicBench, MuSciClaims, or SciClaimEval.
- Changes to model weights or solver architecture.
- Changes to claims, context selection, captions, image content, image order,
  answer parsing, or gold-label mapping.
- Automatic dataset-field inference.
- Use of final-test examples, labels, predictions, metrics, or error traces by
  the proposer.
- Runtime implementation during this documentation phase.

## Terminology

The following terms are separate identity dimensions:

| Term | Allowed values or meaning |
| --- | --- |
| Prompt variant | Existing `cot`, frozen `meta_cot`, or an immutable candidate ID during search |
| Reasoning method | `direct`, `analytical`, `parallel`, or `sequential` |
| Candidate | One indivisible set of four prompt templates |
| Solver | The model that answers SciVer examples |
| Proposer | Codex CLI process that proposes candidate prompt text |
| Search split | Paper-disjoint examples used during iterative screening |
| Validation split | Paper-disjoint examples used to select the global winner |
| Final-test split | Paper-disjoint holdout opened only after freezing |

The existing result field `method` currently denotes reasoning method. Future
work must not silently repurpose it as prompt variant.

## Candidate contract

A candidate contains exactly four prompt templates:

- `direct`
- `analytical`
- `parallel`
- `sequential`

Only prompt text may differ from another candidate. Every other component is
fixed by the experiment configuration.

Each template must be a valid `string.Template` text with exactly the required
substitution interface:

| Reasoning method | Required placeholders |
| --- | --- |
| `direct` | `claim`, `context`, `caption` |
| `analytical` | `claim`, `context`, `caption` |
| `parallel` | `claim`, `context`, `caption1`, `caption2` |
| `sequential` | `claim`, `context`, `caption1`, `caption2` |

A candidate is invalid if it:

- Omits, renames, duplicates semantically, or adds a substitution placeholder.
- Contains an unresolved `$` substitution.
- Contains fewer or more than the four required templates.
- Alters request roles, message block structure, evidence paths, image order,
  dataset records, labels, parser behavior, model ID, or generation settings.
- Requests or embeds ground-truth labels, gold explanations, evaluation
  answers, credentials, endpoint details, or final-test information.
- Identifies a remote provider.

Literal dollar signs must use the `string.Template` escape form. Candidate
validation must substitute sentinel values into all four templates before any
solver call and reject the whole candidate on failure.

## Baseline and exported variant

`cot` is the immutable baseline and must continue to resolve to the existing
four `COT_PROMPT` templates without text, whitespace, placeholder, or routing
changes.

During search, candidates are addressed by immutable candidate IDs and prompt
hashes. A mutable alias such as "latest" is not a valid evaluation identity.

After selection and freeze, the winning four-template mapping is exposed
additively as `meta_cot`. Exporting `meta_cot` must not mutate, wrap, or
redirect `cot`. Baseline-equivalence tests must prove that all existing `cot`
requests are unchanged.

## Input and evidence contract

The evaluator consumes normalized SciVer samples produced by the existing
adapter. Each sample must include:

- A stable sample ID.
- A reasoning method in `claim_type`.
- Claim text.
- A readable local paper JSON reference and selected section IDs.
- One ordered evidence path for `direct` or `analytical`, or two ordered
  evidence paths for `parallel` or `sequential`.
- A gold label available only to the evaluator.

Request construction remains:

1. Derive context from the referenced paper and selected sections.
2. Resolve the method-specific caption or captions.
3. Substitute only the allowed prompt values.
4. Serialize image 1, optional image 2, then the text prompt.
5. Dispatch to the fixed solver.

The evaluator must assert that no gold-only field is present in messages or
serialized payloads. Candidate text and proposer feedback must never contain
sample gold labels or gold explanations.

## Paper grouping contract

No claim from the same paper may cross search, validation, or final-test
boundaries.

Generated normalized SciVer records currently lack `paper_id`. Unless a
verified stable paper identity is added after schema inspection, V1 will group
records by the full SHA-256 digest of the referenced paper JSON bytes. The
group key may be stored in split metadata but must not be inserted into model
messages.

Unreadable paper JSON, a missing reference, or insufficient schema evidence is
a hard input error. The implementation must not fall back to claim-level
splitting or guess a paper mapping.

## Solver contract

Search and validation use only `gemma-4-26B-A4B-it`. The candidate cannot
select or modify the solver.

The following must be fixed and recorded before comparable evaluation:

- Model identifier.
- Endpoint compatibility behavior.
- `temperature`.
- `max_tokens`.
- `seed`, if the endpoint supports it.
- Retry policy and request timeout.
- Parser version.
- Dataset and split fingerprints.
- Prompt candidate hash.

The current remote client omits `temperature`, `max_tokens`, and `seed`.
Therefore their values and support are unknown; no current run may be described
as fixed on those dimensions.

## Evaluation contract

Every evaluated example produces a durable final status:

- Success with a parsed `yes` or `no`.
- Parse failure.
- Invalid input.
- API or infrastructure failure.

Selection uses:

1. Validation Macro-F1 as the primary metric.
2. Validation Accuracy as the secondary metric.

An eligible winner must also have:

- Parse coverage exactly `1.0`.
- Zero unresolved API failures.

The winner is the best eligible candidate across every completed iteration,
not the last candidate and not merely the best candidate in the final
iteration. A deterministic tie rule must be frozen before the first main run;
the exact tertiary tie rule is currently unresolved.

## Proposer information boundary

The proposer may receive:

- This domain contract.
- The current candidate and immutable lineage metadata.
- Search and validation aggregate metrics allowed by the frozen protocol.
- Sanitized aggregate parse/error counts.
- Budget and iteration state.

The proposer must not receive:

- Final-test records or paper identifiers.
- Final-test predictions, metrics, logs, failures, or availability signals.
- Ground-truth labels or per-example correctness from any split.
- Credentials, authorization headers, endpoint values, or image data.
- Unredacted solver request or response payloads.

Whether the proposer may receive selected non-gold sample inputs or only
aggregate feedback must be resolved after the missing reference contracts are
available. Until then, the stricter aggregate-only boundary is the safe
default.

## Required outputs

A completed experiment produces:

- An immutable dataset/split manifest.
- An immutable configuration manifest.
- The baseline candidate and every proposed candidate with hashes and lineage.
- Durable per-example evaluation results with candidate-aware identity.
- Per-candidate search and validation summaries.
- A global-best frontier and decision log.
- A frozen winner manifest.
- An additive `meta_cot` export.
- One final-test report for the frozen winner.
- Post-freeze transfer reports for the other three supported models,
  with no prompt re-optimization.

All artifacts must remain provider-neutral and must not contain credentials,
authorization headers, image base64 data, or model-visible ground truth.
