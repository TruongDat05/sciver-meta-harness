# Meta-Harness baseline snapshot

## Scope and provenance

This snapshot records the repository state before any Meta-Harness runtime
implementation. It is an audit and planning artifact only.

- Audit date: 2026-07-27
- Branch: `feature/sciver-meta-harness-api`
- HEAD: `496d5cd0c863127fe78b65c738fd10f2c730fba7`
- Required base: `feature/unified-benchmark-api`
- Required base commit: `496d5cd0c863127fe78b65c738fd10f2c730fba7`
- Ancestry check: passed
- Initial worktree: clean
- Existing prompt fingerprint:
  `b2d1f354fac218a5a694b5c29fabb72ec2644945d709d9b31fccefd5d3cf1f7e`

The requested Meta-Harness reference directory,
`../meta-harness-reference`, was not present. A local search under the user
home directory also found no directory named `meta-harness-reference`.
Consequently, the requested reference files could not be read. Reference
behavior not stated explicitly in the task is unknown and must be checked
before implementation.

## Pre-edit verification

| Check | Result |
| --- | --- |
| `git branch --show-current` | `feature/sciver-meta-harness-api` |
| `git status --short` | clean |
| `git merge-base --is-ancestor feature/unified-benchmark-api HEAD` | passed |
| `python -m compileall -q .` | passed |
| `pytest -q` | not run: `pytest` is not installed in the current environment |
| `git diff --check` | passed |

The test result is therefore unknown, not passing or failing.

## Prompt definitions and selection

`utils/constant.py` defines `COT_PROMPT` as a mapping of four
`string.Template` objects. It is not one prompt string.

| Reasoning method | Required placeholders |
| --- | --- |
| `direct` | `claim`, `context`, `caption` |
| `analytical` | `claim`, `context`, `caption` |
| `parallel` | `claim`, `context`, `caption1`, `caption2` |
| `sequential` | `claim`, `context`, `caption1`, `caption2` |

`main.py` currently exposes only the prompt variant `cot` through
`PROMPT_DICT = {"cot": COT_PROMPT}`. The legacy path selects
`PROMPT_DICT[arguments.prompt]` and passes that mapping to its generator.

The remote CLI has overlapping names:

- `--prompt` is a free string with default `cot`; it is validated against
  `PROMPT_DICT`.
- `--method` has choices derived from `PROMPT_DICT`, so its only current value
  is also `cot`.
- The record's `claim_type`, not either CLI argument, chooses the reasoning
  method `direct`, `analytical`, `parallel`, or `sequential`.

The remote dry-run and live paths call
`prepare_remote_requests([sample.record], COT_PROMPT)` directly. Thus the
remote path accepts a prompt/method name but does not use the selected mapping
for request preparation. `_select_samples` also accepts the CLI method but
does not use it to filter or select samples. This naming and routing must be
made explicit before adding `meta_cot`.

The required terminology is:

- **Prompt variant:** `cot`, `meta_cot`, or an immutable candidate ID.
- **Reasoning method:** `direct`, `analytical`, `parallel`, or `sequential`.

These dimensions must not share a field or be substituted for each other.
Existing `cot` behavior and template text must remain byte-for-byte
equivalent.

## Remote request preparation

`model_inference/remote_api.py`:

- Selects a template from the prompt mapping using normalized
  `record["claim_type"]`.
- Supports legacy SciVer records containing `paper_path`, and flat normalized
  records containing inline context and captions.
- Reads selected paper sections in paper order after deriving the requested
  top-level section IDs.
- Resolves one caption for `direct` and `analytical`, and resolves `item1`
  before `item2` for `parallel` and `sequential`.
- Passes image paths in `(image_path,)` or `(item1_path, item2_path)` order.
- Does not read `gold_label`, `label`, rationales, or explanations while
  constructing a request.

`utils/remote_input_processing.py` serializes images in the input iterable's
order, then appends exactly one text block. The remote request content is
therefore image 1, optional image 2, then prompt text. Tests assert byte-level
image ordering and that gold-only fields are absent from serialized payloads.

`model_inference/remote_client.py` currently sends only:

- `model`
- `messages`
- `stream: false`

Although `utils/constant.py` contains generation constants, the remote client
does not send `temperature`, `max_tokens`, or `seed`. They are not fixed in the
current remote path. Exact supported generation controls and their eventual
values are unresolved until the compatibility endpoint is verified through
the separately gated live workflow.

## Dataset adapters and normalized records

`utils/dataset_adapters.py` provides strict adapters for SciVer,
SciAtomicBench, MuSciClaims, and SciClaimEval. The requested V1 Meta-Harness
scope is SciVer only; the other adapters remain compatibility surfaces and
must not change as a side effect.

The released SciVer adapter:

- Validates the referenced local paper JSON and selected evidence before
  producing a record.
- Produces a collision-safe sample ID based on dataset, split, source request
  ID, and row index.
- Preserves `paper_path`, section pointers, reasoning method, evidence
  pointers, and evidence path order.
- Copies the binary label only to `gold_label`.
- Drops gold explanations and unrelated source fields.

Generated normalized SciVer records do **not** contain `paper_id`. Already
normalized input is accepted without a guaranteed paper identity field.
Therefore a leakage-safe split cannot rely on `sample_id` or claim rows.

The proposed fallback group key is the full hexadecimal SHA-256 digest of the
bytes of the referenced paper JSON. All claims with the same digest must stay
in one split. Before implementation, representative SciVer samples or a schema
report must confirm that every selected record has a readable local
`paper_path`. If a verified, stable paper identity later becomes available,
the choice between that identity and the digest remains an explicit design
decision.

## Answer parsing

`utils/answer_parser.py` accepts explicit final-answer markers for binary
`yes` or `no`. It rejects empty, non-string, marker-free, and unresolved
conflicting responses as `invalid`. It does not treat bare occurrences of
`yes` or `no` in reasoning text as predictions.

For prompt search:

- Parse coverage is the fraction of final attempts with successful parsed
  answers.
- A candidate with any parse failure cannot satisfy parse coverage `1.0`.
- Parser behavior is part of the fixed evaluation environment; prompt
  optimization must not alter the parser to improve a candidate.

## Result identity and resume behavior

`utils/result_writer.py` currently identifies an attempt by:

1. `run_id`
2. `sample_id`
3. `dataset`
4. `model`
5. `method`

The stored `method` is the per-record reasoning method in `main.py`. The
current remote `run_id` is
`dataset:model:arguments.method`, where `arguments.method` currently means the
prompt-level value `cot`. Prompt variant and reasoning method are therefore
encoded inconsistently across identity fields.

Successful identities are skipped on resume. Failed outcomes remain
retryable, attempt counts advance, and evaluation uses the highest attempt
count. A malformed final JSONL fragment is repaired or ignored, while malformed
non-final records are rejected.

Current identity is insufficient for Meta-Harness candidates because it has no
explicit candidate ID, prompt hash, split identity, generation-settings hash,
or experiment identity. Future work must add those dimensions without
changing the meaning of existing fields or causing an old `cot` success to
skip a different prompt.

`utils/benchmark_workflow.py` separately creates an immutable manifest with a
dataset hash, prompt hash, model, method, experiment ID, request delay, and
configuration fingerprint. It does not currently include
`temperature`, `max_tokens`, or `seed`, and the main remote CLI does not use
that manifest for its result identity.

## Evaluation metrics

`evaluation/metrics.py`:

- Keeps exact run, dataset, model, and reasoning-method group boundaries.
- Evaluates only the latest attempt for each sample.
- Reports status counts, parse coverage, accuracy over all labeled samples,
  accuracy over parsed labeled samples, and a binary confusion matrix.
- Can combine reasoning methods for dataset-level accuracy.
- Provides an unweighted macro average of **dataset accuracy** across
  datasets.

It does not calculate per-class precision, recall, or F1, and it does not
calculate binary-class Macro-F1. The existing function named
`macro_average_accuracy` is not Macro-F1 and cannot be used as the primary
selection metric.

## CLI and offline safety

The remote path is additive and requires `--provider remote`. A non-dry live
run also requires `--live-api` before dataset loading, credential access,
client construction, or request dispatch. Module import has no remote side
effect.

The test suite has a suite-wide autouse fixture that:

- Removes `API_KEY` and `API_URL`.
- Forces accelerator/model-library offline modes.
- Rejects HTTP requests at the request boundary.
- Rejects socket connections.

Tests that exercise the live gate replace the request/client boundary with
fakes. Direct client tests inject a mock session. Inspection found no test
that intentionally performs a live request. Because `pytest` is unavailable,
this is a source audit rather than an executed test confirmation.

## Baseline risks to resolve in future work

1. Prompt variant and reasoning method are currently conflated in CLI and
   result naming.
2. The remote path bypasses its accepted prompt selection and always supplies
   `COT_PROMPT`.
3. Candidate-aware identity and resume protection do not exist.
4. SciVer normalized records have no guaranteed paper group ID.
5. Validation Macro-F1 cannot be computed by current metrics.
6. Remote generation controls are omitted, so solver determinism is unknown.
7. The external Meta-Harness contracts could not be checked because the
   reference repository is unavailable.

No runtime behavior was changed as part of this snapshot.
