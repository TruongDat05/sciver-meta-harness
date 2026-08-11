# SciVer Full-Search Meta-Harness implementation plan

## 1. Authority and status

This is the canonical implementation plan for the SciVer Meta-Harness redesign.
Its locked experiment contract is `sciver_full_search_v3`. It supersedes the
older staged, hard-search, promotion, protected-validation, and transfer
descriptions for new SciVer prompt-search work. Historical documents and
runtime modules remain available as legacy evidence until a later authorized
milestone removes or archives them.

The earlier `sciver_full_search_v2` contract, with its 750/1250 split, was
abandoned before any official live experiment. It is historical only and is
not an active or resume-compatible experiment identity.

Milestone 0 is complete when this document and `AGENTS.md` persist the design.
It changes no runtime behavior. Subsequent work must follow the numbered
milestones below; no live experiment is authorized merely by this plan.

## 2. Experiment objective

Optimize only the text of the four existing SciVer prompt templates while
keeping the evaluation task and solver behavior fixed. The objective is to
select, using SEARCH data only, the best of canonical P0 (`cot`) and all valid
prompt candidates. The offline-frozen winner is P*. FINAL is a one-time,
paired evaluation of frozen P0 and frozen P* and is never optimization input.

The fixed solver is Qwen/Qwen3.5-35B-A3B served through a vLLM
OpenAI-compatible HTTP endpoint. Python owns all inference orchestration,
metrics, caching, retries, persistence, and selection. Codex CLI is a prompt
proposer only.

## 3. Fixed experiment protocol

### Dataset and split

- Protocol ID: `sciver_full_search_v3`.
- Select exactly 2,000 SciVer samples using seed `42`.
- Allocate exactly 1000 SEARCH samples and exactly 1,000 FINAL samples.
- SEARCH and FINAL must be paper-disjoint. Group by verified paper identity;
  otherwise use the SHA-256 of the referenced paper JSON bytes. Never split by
  claim.
- Deterministic selection and allocation must not use labels, predictions,
  difficulty, baseline errors, or any model-derived signal.
- Preparation must fail with an actionable error if the corpus cannot provide
  the exact counts while retaining paper exclusivity. It must never silently
  shrink a split, substitute a claim-level fallback, or change membership on
  resume.

### SEARCH and schedule

- Evaluate P0 on all 1000 SEARCH records before candidate selection.
- Every valid candidate is evaluated on exactly the same 1000 SEARCH records,
  in the manifest's immutable order.
- There is no hard subset, hard-search objective, smoke scoring gate,
  promotion, or protected-validation stage.
- Representative SEARCH errors may be sanitized and summarized for proposer
  feedback, but they never determine membership or replace full SEARCH scoring.
- Codex proposes exactly one candidate per iteration.
- Run at least 15 and at most 40 completed candidate iterations. Stop after
  eight consecutive completed iterations without a metric improvement; no
  early-stop decision is allowed before iteration 15.
- For each iteration, permit at most three proposal attempts for invalid or
  duplicate output. Rejected output makes no solver call. Exhaustion is a
  durably recorded infrastructure/proposal failure with defined resume policy.

### Ranking and patience

P0 remains eligible to win. Rank every SEARCH-complete, eligible P0/candidate
using this exact ascending sort key:

1. negative SEARCH Macro-F1;
2. negative SEARCH Accuracy;
3. prompt SHA-256;
4. candidate ID.

Only an improvement in Macro-F1 or, when Macro-F1 ties, Accuracy resets
patience. A winner change caused solely by prompt SHA-256 or candidate ID does
not reset patience. Python computes the metrics, hashes, ranking, and winner;
the proposer may not self-score or claim success.

### Frozen prompt boundary

Candidates contain exactly these template keys: `direct`, `analytical`,
`parallel`, and `sequential`. Only template text is mutable. The existing
placeholder interface, canonical `cot` text, answer format, parser, labels,
claim/context/caption construction, image number/order, model, and generation
semantics are immutable. `meta_cot` is an additive registration of the frozen
P* artifact; it must never alter or route through `cot`.

### FINAL isolation

FINAL records, IDs, labels, predictions, traces, metrics, paths, and
availability signals are unavailable to the proposer and SEARCH orchestration.
After offline SEARCH winner freeze, a separate explicit FINAL command may run.
It evaluates frozen P0 exactly once and frozen P* exactly once on the same
immutable 1,000 FINAL IDs. FINAL cannot alter prompts, ranking, patience,
SEARCH state, or the frozen winner. A restart after interruption resumes only
the same durable execution identity and never duplicates completed calls.

### Operational safety

Persist deterministic manifests, state transitions, retry state, and cache
identities atomically. Support safe SEARCH caching, bounded retries, and
concurrency without duplicate requests. Read the endpoint only from `API_URL`
and the credential only from `API_KEY`; never write either, an Authorization
header, or image base64 data. All tests remain offline and mock HTTP and Codex
subprocess boundaries.

### Workload accounting

Maximum SEARCH logical solver evaluations are 1,000 for P0 plus 40 × 1,000 =
40,000 for candidate iterations, for a maximum SEARCH total of 41,000. FINAL
is 1,000 for frozen P0 plus 1,000 for frozen P*, for a FINAL total of 2,000.
The maximum full 40-iteration run plus FINAL is therefore 43,000 logical
solver evaluations. Transport retries are excluded from logical counts.

## 4. Current-versus-target design

The current implementation is a legacy staged design: it partitions
search/validation/final roughly 20/20/60, builds a 50--100-item hard-search
set from baseline errors and labels, applies a smoke gate, proposes two
candidates per iteration, promotes top candidates to protected validation, and
uses validation for selection. Its default schedule is 10--15 iterations and
its ranking adds resource metrics. That behavior is incompatible with
`sciver_full_search_v3`.

The target has exactly two paper-disjoint splits with exact sample counts,
full-SEARCH scoring for P0 and every candidate, one candidate per iteration,
SEARCH-only selection, the 15--40/patience-8 schedule, and mandatory paired
FINAL after offline freeze. Existing legacy modules must not be silently
repurposed if their assumptions differ; replace their call paths deliberately
and preserve compatibility until Milestone 7.

## 5. Current implementation map

The following map is for planning only; no Milestone 0 code change is implied.

| Area | Current location | Target disposition |
| --- | --- | --- |
| Configuration | `meta_harness/config.py`, `configs/meta_harness/*.json` | Replace staged protocol fields with the fixed full-search contract. |
| Split preparation | `meta_harness/split_manager.py`, `scripts/prepare_meta_harness_data.py` | Implement exact two-way deterministic paper-disjoint allocation. |
| Hard-search selection | `meta_harness/hard_search.py` | Obsolete for the new protocol; retain untouched until legacy removal. |
| Evaluation | `meta_harness/evaluator.py` | Reuse production request/parser boundaries; add full-SEARCH identity/cache behavior. |
| Orchestration | `meta_harness/orchestrator.py`, `meta_harness/staged_orchestrator.py` | Add a dedicated full-search orchestrator; do not make staged behavior masquerade as target behavior. |
| Proposer | `meta_harness/proposer/codex_cli.py`, schema/feedback modules | Restrict to one prompt-text candidate and sanitized SEARCH-only feedback. |
| Freeze/FINAL | `meta_harness/finalize.py`, `scripts/finalize_meta_harness.py` | Freeze SEARCH winner then require paired P0/P* FINAL execution. |
| Legacy utilities | `hard_search.py`, `retry.py`, `reparse.py`, transfer scripts | Mark obsolete in documentation now; remove only in Milestone 7. |
| Tests | `tests/meta_harness/` | Replace staged assumptions with offline full-search coverage. |

## 6. Target architecture

1. **Protocol configuration** validates protocol ID, exact counts, seed,
   solver identity, generation semantics, schedule, ranking version, and
   proposal-attempt limit. These fields form immutable run identity.
2. **Preparation** loads normalized SciVer records, verifies paper identity,
   performs a deterministic label-independent selection/allocation, and writes
   one immutable split manifest plus separate SEARCH and FINAL materializations.
3. **Solver boundary** is an injected generic compatible HTTP client. It is
   constructed only for explicit live execution. It exposes request identity,
   bounded retry classification, and safe usage metadata without exposing
   credentials or image base64.
4. **Full-SEARCH evaluator** evaluates P0/candidates over all ordered SEARCH
   records. It reuses the existing request construction and parser unchanged,
   persists sanitized results, and computes trusted metrics.
5. **SEARCH cache** keys completed work by immutable protocol/config/split,
   prompt hash, candidate ID, solver/generation contract, parser version, and
   sample identity. It is append-only or create-once and never shares data
   across incompatible runs.
6. **Proposer boundary** gives Codex only the frozen template contract,
   permitted parent prompt(s), sanitized aggregate SEARCH metrics, and
   representative anonymized SEARCH error summaries. It requests one
   prompt-text candidate and validates it before any solver call.
7. **Search orchestrator** owns attempts, iteration state, retries, full
   evaluation, ranking, patience, atomic checkpoints, locks, and resume.
8. **Freeze/FINAL service** writes an immutable P* artifact from SEARCH only,
   then separately executes the paired P0/P* FINAL evaluation under a one-time
   completion receipt.

## 7. Artifact layout

Milestone 1 uses separate trusted-private and SEARCH-facing preparation
directories so future SEARCH code cannot load private FINAL membership through
the SEARCH-safe API:

```text
workspace/meta_harness/full_search/<PREPARATION_ID>/
├── private/
│   └── private_split_manifest.json  # trusted exact 1000/1000 membership
└── search/
    ├── search_safe_manifest.json    # SEARCH membership + opaque FINAL commitment
    └── search_records.json          # exactly 1,000 manifest-ordered SEARCH rows
```

Milestone 1 never materializes FINAL records. Later milestones may add run,
cache, checkpoint, and finalization layouts, but no private path may be mounted
or supplied to SEARCH/proposer processes. SEARCH-facing artifacts must omit
FINAL identities and records, credentials, Authorization headers, endpoint
values, proposer-visible gold labels, and image base64 data.

## 8. KEEP/MODIFY/REMOVE/ADD decisions

### KEEP

- Existing SciVer adapters, canonical prompts, request construction, parser,
  label normalization, image serialization/order guarantees, and generic live
  API safety gate.
- Offline test-wide HTTP/socket denial and fake subprocess/client patterns.
- Immutable candidate/prompt hashing and atomic persistence patterns where
  they meet the target contract.

### MODIFY

- Configuration, split manifest, CLI validation, ranking, feedback envelope,
  cache identity, finalization, and documentation to encode the locked
  protocol.
- Tests that encode hard-search, staged promotion, protected validation, two
  candidates, old iteration counts, or resource-based winner tie-breaks.

### REMOVE (Milestone 7 only)

- Legacy hard-search/staged protocol entry points, their configuration fields,
  obsolete tests, reparse/retry paths that only serve that protocol, and
  transfer behavior not in this experiment contract. Removal requires a
  compatibility audit and replacement coverage.

### ADD

- A protocol-specific exact-count splitter, full-SEARCH evaluator/orchestrator,
  one-candidate proposer contract, safe SEARCH cache, paired-FINAL execution
  receipt, dry-run preflight, operator documentation, and focused offline
  tests.

## 9. Milestones

### Milestone 0 — persist approved design

Update only `AGENTS.md` and this document. Do not alter runtime code, tests,
or legacy artifacts. Status: complete when the plan and permanent invariants
match this locked contract.

### Milestone 1 — protocol, configuration, deterministic split

Add `sciver_full_search_v3` configuration validation and exact deterministic
2,000/1000/1,000 paper-disjoint selection. Reject labels/predictions/difficulty
or baseline-error inputs to split membership and fail explicitly when exact
allocation is impossible. Persist immutable manifests and test resume identity.
Status: complete; the released 2,000-row source deterministically produces
1,000 SEARCH rows across 392 papers and 1,000 FINAL rows across 394 papers,
with split SHA-256
`8e5f28db7669026a4c419c972da4bb1caacf4ece0e2f1b8b7b8ab1dca204ec8c`.

### Milestone 2 — solver boundary, cache, retries, concurrency

Define an injected solver interface for Qwen/Qwen3.5-35B-A3B through the
generic compatible HTTP boundary. Add explicit live gating, safe cache keys,
bounded retry rules, request-level locking/concurrency, and redaction. No
module import may create a client or request.

### Milestone 3 — full-SEARCH evaluator

Implement P0/candidate evaluation over the complete ordered 1000 SEARCH IDs,
using frozen prompt/request/parser behavior. Compute Macro-F1 and Accuracy,
write sanitized durable results, and enforce full coverage/eligibility. Add
P0-equivalence and image-order regression tests.

### Milestone 4 — Codex proposer and search orchestration

Implement one-candidate iterations, up to three proposal attempts, full
SEARCH scoring, the exact ranking key, patience 8, minimum 15, maximum 40,
checkpoint/resume, and P0 eligibility. Restrict proposer input to prompt text
and permitted sanitized SEARCH feedback.

### Milestone 5 — freeze and isolated FINAL

Freeze the SEARCH winner offline and implement a separate explicit paired FINAL
run. Require exactly one P0 and one P* evaluation over the same 1,000 FINAL
IDs, enforce completion receipts/resume semantics, and prove FINAL cannot
change SEARCH state or winner.

### Milestone 6 — server workflow and operator interfaces

Add dry-run preflight, operator-facing configuration diagnostics, explicit live
smoke workflow, workload/budget estimates, safe resume commands, and final
artifact inspection. All commands remain provider-neutral and default offline.

### Milestone 7 — legacy removal and final verification

After migration coverage is complete, remove or archive obsolete staged,
hard-search, protected-validation, retry/reparse, and transfer-only paths.
Run the full offline suite and perform compatibility, privacy, prompt, image,
and final-isolation audits.

## 10. Unit test plan

- Configuration rejects any protocol ID, count, seed, schedule, ranking, or
  proposal-attempt deviation; accepts only the locked contract.
- Split tests prove deterministic exact 1000/1,000 allocation, paper
  disjointness, label/prediction/difficulty/baseline-error independence, and
  clear failure when exact allocation is impossible.
- Prompt tests prove exactly four templates, unchanged placeholders, `cot`,
  answer format, parser, labels, and image ordering.
- Evaluator tests prove P0 and every candidate receive all same ordered 1000
  IDs; no smoke, subset, promotion, or validation stage exists.
- Ranking tests cover all tie levels and confirm hash/ID-only changes do not
  reset patience; P0 can beat all candidates.
- Proposer tests require one prompt-text candidate, allow only three invalid/
  duplicate attempts, and prove rejection makes no solver call.
- Cache/retry/concurrency tests prove no duplicate completed calls, compatible
  resume, cache isolation, atomic state, and safe failure classification.
- FINAL tests prove paired identical IDs, once-per-variant receipts, freeze
  before access, and no backflow into SEARCH.
- Security tests verify credential/header/base64 redaction and default offline
  behavior at every boundary.

## 11. Offline integration test plan

Use a deterministic fake solver and fake Codex subprocess. Build a synthetic
paper-group corpus large enough for exact 2,000 selection, run P0 plus at least
the minimum 15 candidate iterations, interrupt at proposal/evaluation/cache/
freeze/final checkpoints, and resume. Assert identical state, rank, prompt
hashes, cache use, 1000 SEARCH coverage per valid prompt, and no FINAL access
before paired finalization. Mock every HTTP request and subprocess call; tests
must not invoke Codex recursively.

## 12. Dry-run test plan

Dry-run validates local files, exact split counts, paper disjointness,
configuration hashes, solver configuration shape, workload for 15--40
iterations, cache/resume compatibility, and paired FINAL call counts. It must
not load credentials, construct a live client, invoke Codex, create run state,
or access FINAL records during SEARCH dry-run. Tests assert all boundaries have
zero calls.

## 13. Opt-in live smoke plan

This is not a test and requires explicit operator authorization after all
offline checks pass. Preflight a new immutable run, validate `API_URL` and
`API_KEY` without printing values, and issue a minimal explicitly gated request
against a non-FINAL local record using frozen P0. Do not propose a candidate,
run a benchmark, create a hard subset, or open FINAL. Inspect sanitized output
and only then separately authorize a full SEARCH run.

## 14. Acceptance criteria

- The protocol is encoded as `sciver_full_search_v3` with exact counts,
  seed, paper disjointness, and deterministic label-independent selection.
- P0 and every valid candidate score the identical complete 1000 SEARCH IDs.
- One-candidate, 15--40, patience-8, three-attempt schedule and exact ranking
  are enforced, including P0 eligibility and no hash/ID patience reset.
- Only four prompt texts are candidate-controlled; all frozen interfaces are
  regression-tested.
- Codex is prompt proposer only; trusted Python controls inference, metrics,
  checkpoints, and selection.
- Freeze is offline and FINAL is an explicit isolated paired P0/P* execution
  over identical 1,000 IDs, with no influence on SEARCH or P*.
- Resume, retry, concurrency, cache, redaction, live gating, and no-network
  test requirements are verified offline.
- Legacy staged behavior is either kept clearly separate during migration or
  removed only in Milestone 7 with replacement coverage.

## 15. Server preflight requirements

Before any opt-in live run, verify locally and without exposing secrets:

- `API_URL` and `API_KEY` are present only in the process environment.
- The selected model is exactly Qwen/Qwen3.5-35B-A3B and the server presents
  the required compatible chat-completions interface.
- Generation semantics, timeout, retry policy, concurrency limit, request
  size/image support, and usage fields are frozen into the run identity.
- The immutable protocol/config/split manifests are new or resume-compatible;
  SEARCH has exactly 1000 records and FINAL exactly 1,000 paper-disjoint
  records.
- The minimum 15-iteration workload fits explicit operator ceilings before
  the first request; FINAL is not authorized by SEARCH preflight.
- The cache/run directory is writable, locked against concurrent writers, and
  contains no incompatible partial state.

Failure of any preflight condition aborts before inference. Endpoint values,
credentials, authorization material, raw request payloads, and image base64
must not appear in diagnostics.

## 16. Implementation status checklist

- [x] M0: approved contract persisted in `AGENTS.md` and this plan.
- [x] M1: protocol/configuration and exact deterministic split implemented.
- [ ] M2: solver boundary, cache, retries, and concurrency implemented.
- [ ] M3: full-SEARCH evaluator implemented.
- [ ] M4: one-candidate proposer and SEARCH orchestrator implemented.
- [ ] M5: offline freeze and isolated paired FINAL implemented.
- [ ] M6: server workflow and operator interfaces implemented.
- [ ] M7: legacy removal and final offline verification completed.
