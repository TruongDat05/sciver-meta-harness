# SciVer Meta-Harness specification

## Authority and scope

This is the normative specification for the SciVer-only `sciver_full_search_v3` prompt-only workflow. The repository is the reusable experiment engine; the server notebook is an operator layer that calls supported repository APIs and never reimplements splitting, evaluation, selection, retry, cache, or FINAL logic. This specification does not authorize live execution.

## Roles and ownership

| Role | Owner | Responsibility |
| --- | --- | --- |
| Engine | `meta_harness/{records,preparation,solver,search_cache,retry,request_executor,search_evaluator,prompt_proposer,ranking,search_orchestrator,winner_freeze,final_evaluation,server_run}.py` | Split preparation, request construction, solver dispatch, cache, retries, concurrency, metrics, ranking, checkpoint/resume, freeze, FINAL, and summaries. |
| Prompt proposer | Isolated proposer boundary | Supplies one candidate's template text only; cannot score itself or access FINAL. |
| Server composition | `meta_harness/server_run.py` | Checkout validation, offline preflight, explicit live gates, sanitized status, and composition of engine APIs. |
| Operator layer | Canonical notebook and wrapper | Supplies runtime paths and invokes supported operations without duplicating engine logic. |

## Locked protocol

- Protocol ID: `sciver_full_search_v3`.
- Select exactly 2,000 SciVer samples with seed 42: exactly 1,000 SEARCH and 1,000 FINAL records. The splits are paper-disjoint and selection is independent of labels, predictions, difficulty, baseline errors, and model-derived signals. Preparation fails if exact counts are impossible.
- The solver is `Qwen3.6-35B-A3B`, with `temperature=0`, `top_p=1`, seed 42, `n=1`, non-streaming output, and `max_tokens=8192`.
- Canonical P0 and each valid candidate use the same complete immutable ordered 1,000-record SEARCH split. A candidate has exactly `direct`, `analytical`, `parallel`, and `sequential`; only their text changes. Prompt interfaces, answer format, parser, labels, claims, context, captions, image count/order, model, and generation semantics remain fixed.
- Propose one candidate per iteration. Complete 15--40 candidate iterations, permitting at most three invalid or duplicate proposals per iteration. Rejected proposals dispatch no solver request. Stop after eight consecutive completed iterations without a metric improvement, never before iteration 15.
- Rank P0 and eligible candidates by SEARCH Macro-F1 descending, SEARCH Accuracy descending, prompt SHA-256 ascending, then candidate ID ascending. Only Macro-F1 improvement, or Accuracy improvement when Macro-F1 ties, resets patience; a hash/ID-only winner change does not.
- Maximum logical workload is 41,000 SEARCH evaluations (1,000 P0 plus 40,000 candidate) and 2,000 FINAL evaluations (1,000 each for P0 and P*): 43,000 total. Transport retries are excluded.

## Execution lifecycle

1. Validate the checkout and prepare or verify deterministic split artifacts.
2. Run offline SEARCH preflight without credentials, live client, proposer invocation, or HTTP.
3. Optionally run the explicitly authorized isolated smoke: model-list validation and one canonical P0 SEARCH-safe request.
4. Explicitly start or resume SEARCH, then inspect status until terminal.
5. Freeze the terminal SEARCH-only winner as immutable P* / `meta_cot`.
6. Run offline FINAL preflight, bound to frozen cross-stage identity.
7. Explicitly start or resume paired FINAL: frozen P0 once and frozen P* once on identical FINAL IDs.

FINAL artifacts, IDs, labels, predictions, traces, metrics, paths, and availability signals are unavailable to the proposer and SEARCH orchestration. FINAL never modifies prompts, rank, patience, SEARCH state, or the frozen winner.

## Immutable identity and artifacts

Manifests, hashes, state, receipts, cache entries, and frozen winner records bind protocol/configuration, split membership, prompt, parser, solver/generation contract, proposer identity, and source identity. Incompatible resume state fails closed. Persistence is atomic; cache, retries, and concurrency must not duplicate compatible completed work.

```text
workspace/meta_harness/full_search_v3/<run-id>/
├── preparation/
│   ├── private/
│   └── search/
├── smoke/completion.json
├── search_cache/
├── orchestration_state.json
├── freeze/frozen_winner.json
└── final/
```

SEARCH consumes SEARCH-safe artifacts only. Trusted preparation material and FINAL materialization remain outside the SEARCH/proposer boundary.

## Safety and offline boundaries

Imports have no remote side effects. Preparation, preflight, status, and tests are offline; tests mock HTTP and proposer subprocess boundaries. Live smoke, SEARCH, and FINAL each need separate authorization. Read the base endpoint only from `API_URL` and the credential only from `API_KEY`; never expose or persist either, an Authorization header, image base64, private sample data, labels, request/response content, or FINAL material. No model-weight download is part of this workflow.

## Implementation map

| Area | Canonical location |
| --- | --- |
| Locked configuration | `meta_harness/config.py` |
| Prompt contract and hashing | `meta_harness/prompt_family.py` |
| Records and preparation | `meta_harness/experiment.py`, `meta_harness/preparation.py` |
| Solver/cache/retry/concurrency | `meta_harness/solver.py` and adjacent modules |
| SEARCH/proposer/ranking/resume | evaluator, proposer, ranking, and orchestrator modules |
| Freeze and FINAL | `meta_harness/winner_freeze.py`, `meta_harness/final_evaluation.py` |
| Server and operators | `meta_harness/server_run.py`, wrapper, canonical notebook |

## Acceptance checks

- Validate exact counts, paper exclusivity, immutable ordering, locked configuration, request and parser identity, rank/patience behavior, freeze binding, FINAL isolation, and compatible resume.
- Confirm the wrapper supports `prepare`, `search-preflight`, `smoke --live-smoke`, `search --live-search`, `activity`, `search-status`, `freeze`, `final-preflight`, `final --live-final`, and `final-status`.
- Confirm the notebook has independent live confirmations for smoke, SEARCH, and FINAL.
- Run `python -m compileall -q .`, `pytest -q`, and `git diff --check` before accepting implementation changes.
