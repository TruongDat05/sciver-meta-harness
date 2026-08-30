# SciVer Meta-Harness

This repository contains the SciVer multimodal scientific-claim benchmark, existing local and compatible-HTTP inference paths, and the locked `sciver_full_search_v3` prompt-only experiment engine.

## Fixed contract

Meta-Harness selects exactly 2,000 SciVer samples with seed 42: 1,000 paper-disjoint SEARCH records and 1,000 paper-disjoint FINAL records. It evaluates canonical P0 and one valid prompt candidate per iteration on the same complete SEARCH split. Candidates may change only the text of the four existing templates: `direct`, `analytical`, `parallel`, and `sequential`.

SEARCH runs 38--50 completed candidate iterations, permits up to three invalid or duplicate proposal attempts per iteration, and stops after patience 8. Ranking is SEARCH Macro-F1 descending, SEARCH Accuracy descending, prompt SHA-256 ascending, then candidate ID ascending. The top-5 SEARCH candidates are frozen as additive `meta_cot` (P0 kept separate). A separately authorized FINAL stage evaluates frozen P0 and all 5 frozen candidates once each on the identical 1,000 FINAL IDs (6,000 calls); FINAL never changes SEARCH state, ranking, patience, or the frozen top-K selection.

The solver is fixed to `gemma-4-26B-A4B-it` with `temperature=0`, `top_p=1`, seed 42, `n=1`, non-streaming responses, and `max_tokens=8192`. Prompt optimization is driven by the `gpt-5.6-sol` proposer at `high` reasoning effort.

## Data split

Preparation reads a single released SciVer dataset JSON (default `data/sciver/testset.json`, or `SCIVER_DATASET_PATH`) and produces the deterministic, paper-disjoint 1,000 SEARCH / 1,000 FINAL split (seed 42) under `workspace/meta_harness/full_search_v3/<run-id>/preparation/`. See [Data preparation and split](docs/sciver_meta_harness_operations.md#data-preparation-and-split).

## Component map

- `meta_harness/config.py` locks and validates the identity.
- `meta_harness/{records,preparation,solver,search_cache,retry,request_executor,search_evaluator,prompt_proposer,ranking,search_orchestrator,winner_freeze,final_evaluation,server_run}.py` owns deterministic preparation, request construction, cache/retry/concurrency, SEARCH, freeze, and paired FINAL.
- `smoke.py`, `search.py`, and `final.py` are the zero-config root launchers; `scripts/run_meta_harness.py` is the explicit wrapper. Both delegate to the same `server_run` interface.
- `notebooks/sciver_meta_harness.ipynb` is the thin, ordered operator layer; it reads all settings from `.env` and does not implement experiment logic.
- `main.py`, `model_inference/`, `utils/`, and `evaluation/` retain the generic benchmark and local inference workflows.

## Operator lifecycle

1. Prepare a pinned, clean checkout and the environment before running.
2. Configure `.env` (`API_URL`, `API_KEY`, `PINNED_COMMIT_SHA`, `SCIVER_DATASET_PATH`, `SCIVER_RUN_ID`).
3. Run the one-request smoke and validate its receipt: `python smoke.py --live-smoke`.
4. Run SEARCH and monitor until terminal: `python search.py --live-search`.
5. Run the paired FINAL stage — this freezes the top-5 candidates, then evaluates frozen P0 and the 5 frozen candidates: `python final.py --live-final`.
6. Inspect sanitized status and artifacts under `workspace/meta_harness/full_search_v3/<run-id>/`.

```bash
git clone https://github.com/TruongDat05/sciver-meta-harness.git
cd sciver-meta-harness
python -m pip install -r requirements.txt
```

Then run the root launchers (`smoke.py`, `search.py`, `final.py`), the explicit
wrapper (`scripts/run_meta_harness.py`), or open
`notebooks/sciver_meta_harness.ipynb`.

The full command sequence, required confirmations, credentials boundary, and safe resume procedure are in the [Meta-Harness server runbook](docs/sciver_meta_harness_operations.md). The [generic benchmark guide](docs/remote-benchmark.md) is separate from and documents its own compatible-HTTP CLI, dataset adapters, and model matrix.

## Safety and verification

All live stages are explicit opt-in. `API_URL` and `API_KEY` are runtime-only; they must not be committed, logged, passed as wrapper arguments, or written to artifacts. Imports, preparation, preflight, tests, and status inspection are offline. Tests mock all HTTP and proposer subprocess boundaries.

```bash
python -m compileall -q .
pytest -q
git diff --check
```

## Citation

```text
@inproceedings{wang-etal-2025-sciver,
  title     = {SciVer: Evaluating Foundation Models for Multimodal Scientific Claim Verification},
  author    = {Wang, Chengye and Shen, Yifei and Kuang, Zexi and Cohan, Arman and Zhao, Yilun},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2025},
  pages     = {8562--8579}
}
```
