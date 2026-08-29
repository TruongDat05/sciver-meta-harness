# Compatible-HTTP benchmark guide

This guide covers the generic benchmark CLI. It is independent from the
SciVer-only Meta-Harness and its [server runbook](sciver_meta_harness_operations.md);
it has its own dataset adapters, model matrix, output/checkpoint workflow, and
complete-URL contract.

SciVer includes an additive compatible-HTTP path for multimodal scientific
claim verification. Existing local inference paths remain available and
unchanged. Live access is opt-in: a request can be sent only when the command
uses both `--provider remote` and `--live-api`. Imports, tests, and dry-runs
remain offline, and this path never downloads model weights.

## Supported experiment matrix

The registered model identifiers are:

- `gemma-4-31B-it`
- `gemma-4-26B-A4B-it`
- `gemma-3-27b-it`

The registered datasets are:

- `SciVer`
- `SciAtomicBench`
- `MuSciClaims`
- `SciClaimEval`

Each dataset can be supplied in its released local schema or as an existing
normalized SciVer JSON file. Dedicated adapters validate exact fields and do
not infer alternatives. See [Dataset normalization](#dataset-normalization)
for the verified mappings.

## Architecture and safety boundary

```text
released local data or normalized JSON
        |
        v
strict dataset adapter and schema checks
        |
        v
original SciVer prompt + ordered local images
        |
        +---- --dry-run ----> safe request metadata only
        |
        +---- --live-api ---> remote Chat Completions request
                                  |
                                  v
                         durable JSONL checkpoint
                                  |
                                  v
                          offline evaluation
```

The CLI first validates the remote opt-in and registered model and dataset. A
dataset adapter then validates the released schema and creates the unified
record without retaining source rationales or gold-only annotations. Message
preparation reuses the original SciVer chain-of-thought prompt,
selects the record's `direct`, `analytical`, `parallel`, or `sequential`
reasoning template, and preserves image order. Single-evidence records send
`image_path`; two-evidence records send `item1_path` followed by `item2_path`.
The text prompt follows the image blocks.

The model-visible request contains the model identifier, ordered evidence, and
prompt only. Ground-truth labels and rationales are not added to the request.
If a normalized record has `gold_label`, it is copied directly to the local
result record for offline evaluation, never to the model message.

The HTTP client is constructed lazily, after explicit live opt-in and local
input preparation. It sends a non-streaming Chat Completions request and reads
the response text from `choices[0].message.content`. Credentials, request
headers, and image data are not written to diagnostics or result files.

## Dataset normalization

The adapters use the released schemas below. Missing inference fields,
unsupported provided labels, non-local evidence references, and unresolved
files are errors; similarly named fields are never substituted.

| Dataset | Verified raw fields | Binary label mapping | Evidence |
| --- | --- | --- | --- |
| SciVer | `request_id`, `claim`, `claim_type`, `paper_path`, `section`, evidence pointers, `label` | true/entailed → `yes`; false/refuted → `no` | Existing local chart/table image; context and caption are resolved from local paper JSON. |
| SciAtomicBench | `id`, `claim`, `label`, `table_caption`, `table_content` | `support` → `yes`; `refute` → `no` | Markdown table rendered locally as a readable PNG. |
| MuSciClaims | `claim_id`, `claim_text`, `label_2class`, `associated_figure_filepath`, `caption` | `SUPPORT` → `yes`; `NON_SUPPORT` → `no` | Released local figure. |
| SciClaimEval Task 1 | `claim_id`, `claim`, `evi_path`, `caption`, `context`; `label` on labeled splits | `Supported` → `yes`; `Refuted` → `no`; omitted for unlabeled test rows | Released Task 1 local PNG. |

Every generated ID is prefixed with its dataset name, namespaced by its split
or SciAtomic domain, and suffixed with the zero-based source row index so
repeated raw IDs cannot collide. All non-SciVer records receive
`claim_type="analytical"`. Schemas without context receive the explicit text
`No additional context is provided.`; no evidence or rationale is invented.
Non-SciVer records use the unified flat schema. SciVer records with
`paper_path` retain the established paper, section, and evidence pointers so the
original prompt-construction path remains authoritative. `gold_label` is kept
only on labeled records and is never model-visible.

The main CLI normalizes in memory. To inspect or reuse materialized records,
write them explicitly without making any network request:

```bash
python scripts/normalize_dataset.py \
  --dataset MuSciClaims \
  --dataset-path /absolute/path/to/test_set.jsonl \
  --output outputs/normalized/musciclaims.json
```

For SciAtomicBench, pass either one released domain JSON file or a directory.
A directory is accepted only when all four files are present: `fin.json`,
`mat.json`, `med.json`, and `ml.json`. Rendered evidence defaults to a sibling
`<output-stem>_evidence` directory; `--evidence-dir` can select another local
directory. Normalization never fetches remote images.

## Installation and configuration

Remote execution needs `requests` and Pillow but does not need local inference
packages or model weights:

```bash
python -m pip install "requests>=2.31.0,<3" "Pillow>=10,<12"
```

The two required environment variables are:

| Variable | Purpose |
| --- | --- |
| `API_KEY` | Credential for the remote OpenAI-compatible API. |
| `API_URL` | Complete remote Chat Completions URL. |

Set them only in the runtime environment. The placeholders below are
deliberately nonfunctional; replace them locally and never commit the values:

```bash
export API_KEY='<REMOTE_API_KEY>'
export API_URL='<REMOTE_CHAT_COMPLETIONS_URL>'
```

Do not pass either value as a CLI argument, paste it into a notebook cell, or
store it in a dataset, output file, archive, or run metadata. A dry-run does not
need either variable.

Two optional runtime controls are available: `API_TIMEOUT_SECONDS` is a
positive per-request timeout and defaults to `60`; `API_MAX_RETRIES` is a
non-negative retry count and defaults to `3`.

## Command workflow

The examples below run one dataset/model configuration. Set reusable,
non-secret shell variables first:

```bash
DATASET='SciVer'
DATASET_PATH='/absolute/path/to/released_or_normalized_data'
MODEL='gemma-4-26B-A4B-it'
REQUEST_DELAY='1.0'
```

The output path for each command is:

```text
<output-dir>/<dataset>_<method>/<model-basename>.jsonl
```

Here, `--method cot` selects the available prompt method and the output/run
namespace. Each valid JSONL record's `method` field is the sample's actual
reasoning type (`direct`, `analytical`, `parallel`, or `sequential`). A record
rejected before its reasoning type can be resolved uses `cot` in that field.

### 1. Offline dry-run

Run this before loading any credentials. It prepares one sample and prints only
the dataset, adapter, model, reasoning method, sample ID, image count, prompt
length, and prospective output location. It sends no request and writes no
result file.

```bash
python main.py \
  --provider remote \
  --dataset "$DATASET" \
  --dataset-path "$DATASET_PATH" \
  --model "$MODEL" \
  --method cot \
  --max-num 1 \
  --output-dir outputs/dry-run \
  --dry-run
```

### 2. One-sample live smoke test

After setting `API_KEY` and `API_URL`, send exactly one sample. Check the JSONL
record and confirm that the remote API accepts the multimodal image format
before increasing the sample count.

```bash
python main.py \
  --provider remote \
  --dataset "$DATASET" \
  --dataset-path "$DATASET_PATH" \
  --model "$MODEL" \
  --method cot \
  --max-num 1 \
  --output-dir outputs/smoke \
  --live-api
```

### 3. Twenty-sample pilot

Use a separate output directory so the smoke result is not mixed into the
pilot. The request delay spaces sample-level requests; retry backoff is handled
separately by the HTTP client.

```bash
python main.py \
  --provider remote \
  --dataset "$DATASET" \
  --dataset-path "$DATASET_PATH" \
  --model "$MODEL" \
  --method cot \
  --max-num 20 \
  --request-delay "$REQUEST_DELAY" \
  --output-dir outputs/pilot \
  --live-api
```

### 4. Full execution

`--max-num -1` selects every record. Start the first full run in its own output
directory:

```bash
python main.py \
  --provider remote \
  --dataset "$DATASET" \
  --dataset-path "$DATASET_PATH" \
  --model "$MODEL" \
  --method cot \
  --max-num -1 \
  --request-delay "$REQUEST_DELAY" \
  --output-dir outputs/full \
  --live-api
```

The CLI returns `1` if one or more samples fail after retries, but it persists
those failures and continues processing later samples. Invalid global
configuration returns `2` and stops the run.

### 5. Resume an interrupted or partially failed full run

Repeat the exact full command with `--resume`:

```bash
python main.py \
  --provider remote \
  --dataset "$DATASET" \
  --dataset-path "$DATASET_PATH" \
  --model "$MODEL" \
  --method cot \
  --max-num -1 \
  --request-delay "$REQUEST_DELAY" \
  --output-dir outputs/full \
  --live-api \
  --resume
```

Without `--resume`, the CLI refuses to overwrite a non-empty remote result.
Keep the dataset content, sample IDs, model, and method unchanged when resuming.

### 6. Evaluate results offline

Evaluation needs no credentials or network. Give it one or more JSONL files
that belong in the same report:

```bash
python scripts/evaluate_results.py \
  "outputs/full/${DATASET}_cot/${MODEL}.jsonl" \
  --summary-json outputs/evaluation/summary.json \
  --csv outputs/evaluation/summary.csv
```

The evaluator groups results by exact `run_id`, dataset, model, and reasoning
method. For repeated sample attempts, only the highest `attempt_count` is used.
The JSON report retains reasoning-method groups and adds per-dataset summaries
across methods. Each dataset summary contains SciVer Accuracy over all labeled
samples, parsed-only Accuracy, a yes/no confusion matrix, parse coverage, and
API/parse/invalid-input failure counts, including numerators and denominators.
`macro_average_accuracy` is the unweighted mean of dataset Accuracy values for
each model. Pass one full run for each of the four datasets in one invocation
to obtain the four-dataset macro average. Duplicate runs for the same model and
dataset are rejected because no ordering metadata exists to choose one safely.

Evaluate smoke, pilot, and full outputs separately. Combining those stages in
one command repeats sample identities under different run locations but the
same logical run identifier, which can create duplicate attempt numbers or
otherwise make the report ambiguous.

## Why a GPU is not required

Model inference happens behind the remote OpenAI-compatible API. The local
process performs only JSON parsing, prompt construction, PNG/JPEG validation
and encoding, HTTP I/O, result writing, and evaluation. No model weights are
loaded or downloaded for this workflow.

## Output record schema

Remote results use append-only JSON Lines: one JSON object per completed sample
attempt. The record fields are:

| Field | Meaning |
| --- | --- |
| `run_id` | Stable `<dataset>:<model>:<prompt-method>` identifier. |
| `sample_id` | JSON-serializable ID from the input, or the zero-based input index when absent. |
| `dataset` | Registered dataset name. |
| `model` | Registered model identifier. |
| `method` | Record reasoning type, or `cot` if input validation fails before the type is resolved. |
| `prediction` | Parsed `yes`/`no`, or `null` when unavailable. |
| `parse_status` | `parsed`, `invalid`, or `not_attempted`. |
| `raw_response` | Remote response text, or `null` if no response was available. |
| `request_status` | `success`, `api_failure`, `parse_failure`, or `invalid_input`. |
| `error_type` | Sanitized exception type, or `null`. |
| `error_message` | Sanitized actionable error text, or `null`. |
| `timestamp` | UTC ISO 8601 write time. |
| `attempt_count` | Per-identity benchmark attempt number, starting at `1`. |
| `gold_label` | Optional local label copied from normalized input for evaluation. |

`gold_label` is the only optional field in this table. The writer removes
credential material, request-header values, image data URIs, and long base64
content before serialization. Result files are created with owner-only
permissions where the operating system supports them.

## Retry and checkpoint behavior

There are two distinct recovery layers:

- **HTTP retry:** rate limits (`429`), selected server failures (`500`, `502`,
  `503`, `504`), timeouts, and network failures are retried. The default is
  three retries after the initial request. Backoff starts at one second,
  doubles, and is capped at 30 seconds. A valid `Retry-After` value is honored
  up to that cap. Permanent client errors are not retried.
- **Benchmark resume:** after every sample outcome, the JSONL line is appended,
  flushed, and synchronized to disk. On `--resume`, an exact identity that
  already has `request_status: "success"` is skipped. Failed or invalid
  outcomes remain retryable and receive a higher `attempt_count`.

If an interrupted write leaves a malformed final JSONL fragment, the result
writer removes only that final fragment before appending. Malformed records in
the middle of the file are rejected rather than silently ignored. HTTP retries
within a single CLI sample attempt do not increment `attempt_count`.

## Cost, rate limits, and process timeouts

- A dry-run makes no billable request. A smoke test bounds the first live stage
  to one sample; the pilot bounds the next stage to 20.
- A normal sample makes one Chat Completions request, but retryable failures can
  cause additional requests. A timeout may occur after remote processing has
  begun, so a retry can still add cost even when no response was received.
- Multimodal request cost depends on the remote service's pricing and on image,
  prompt, and response size. Use its usage reporting together with pilot
  latency and outcomes before estimating a full run.
- Set `--request-delay` to pace sample requests. It does not replace the
  client's handling of rate-limit responses. Reduce concurrency outside this
  CLI rather than starting competing processes against the same quota.
- `API_TIMEOUT_SECONDS` limits one HTTP attempt, not the whole experiment.
  HTTP retry delays, request pacing, dataset size, and process startup all add
  to wall-clock time.
- Processes can end independently of the HTTP timeout. Use `--resume`,
  keep stage outputs separate, and archive checkpoints frequently.

Do not run the entire four-dataset by four-model matrix in one process
session. A single combination can already approach a session's time or quota
limits; combining all 16 makes cost harder to bound, creates a burstier
rate-limit profile, increases the chance that a session timeout interrupts many
experiments, and makes failures and provenance harder to isolate. Use one
dataset/model configuration per session, finish dry-run/smoke/pilot validation,
then run and archive its full checkpoint before starting the next combination.

## Adding a model

Adding a model is a deliberate registry and validation change, not an arbitrary
CLI string:

1. Confirm that the remote OpenAI-compatible API exposes the exact identifier
   and accepts the repository's ordered `image_url` content blocks.
2. Add the identifier to `_MODEL_IDENTIFIERS` in
   `model_inference/remote_config.py`.
3. Update the exact registry expectation in `tests/test_remote_config.py` and
   add the model to the mocked dataset/model integration matrix in
   `tests/test_benchmark_pipeline.py`.
4. Run the offline test suite. Then perform the dry-run and one-sample live
   smoke test before a pilot.

Do not add model-specific credentials, endpoints, provider names, implicit
weight downloads, or automatic fallback behavior.

## Adding a dataset

Never guess a raw dataset mapping. First create a read-only schema report
outside the dataset directory:

```bash
python scripts/inspect_dataset.py \
  --dataset '<DATASET_NAME>' \
  --dataset-path /absolute/path/to/raw_dataset \
  --output /absolute/path/to/work/schema-report.json
```

Inspect representative records and explicitly define a strict adapter into a
local JSON top-level list. The flat single-evidence contract uses `sample_id`,
`claim_type`, `claim`, `context`, `caption`, `image_path`, and `gold_label`.
The established SciVer pointer contract remains supported: common fields are
`claim_type`, `claim`, `paper_path`, and `section`; direct and analytical
records require `type`, `item`, and `image_path`; parallel and sequential
records require `item1_type`, `item1`, `item1_path`, `item2_type`, `item2`, and
`item2_path`. Preserve `item1_path`/`item2_path` evidence order.

Each `paper_path` must reference local JSON whose `sections` entries contain
`section_id`, `section_name`, and `text`. Chart metadata uses `image_paths`
entries with `caption`; table metadata uses `tables` entries with `capture`.
Evidence files must be valid PNG or JPEG images. Do not place ground-truth
labels or rationales into claim, context, captions, or any other model-visible
field.

After the schema and normalization are verified:

1. Add the dataset name to `SUPPORTED_DATASETS` in
   `utils/dataset_adapters.py`. Introduce a dedicated adapter only if verified
   schema differences require one; do not infer mappings.
2. Add adapter tests for valid samples, missing/ambiguous fields, stable sample
   IDs, ground-truth isolation, and image ordering.
3. Add the dataset to the mocked end-to-end matrix in
   `tests/test_benchmark_pipeline.py`.
4. Run the required offline verification, followed by dry-run, one-sample
   smoke, and 20-sample pilot stages.

Keep dataset conversion offline and explicit. Remote image references must not
be fetched by inspection or normalization code.
