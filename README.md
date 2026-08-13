# SciVer Meta-Harness

This repository extends the SciVer multimodal scientific-claim benchmark with
provider-neutral remote inference and the locked `sciver_full_search_v3`
prompt-only experiment engine. Full-Search v3 is the primary server workflow;
the existing generic benchmark CLI, local inference paths, canonical prompts,
parsers, and evaluation utilities remain supported.

Full-Search v3 evaluates canonical P0 and one prompt-family candidate per
iteration on the same complete 1,000-record SEARCH split, freezes the
SEARCH-only winner as additive `meta_cot`, and then permits a separately
authorized paired P0/P* evaluation on the isolated 1,000-record FINAL split.
The solver is fixed to `Qwen3.6-35B-A3B`; the production generation
settings remain `temperature=0`, `top_p=1`, seed 42, `n=1`, non-streaming,
and `max_tokens=8192`.

## Repository architecture

- `meta_harness/full_search_v3*.py` owns configuration, deterministic
  preparation, the canonical request builder boundary, cache/retry/concurrency,
  full SEARCH, proposing, ranking/patience, freeze, and paired FINAL.
- `meta_harness/full_search_v3_server.py` composes those APIs for deployment,
  performs checkout and offline preflight validation, and owns the isolated
  model-list plus one-request live smoke boundary.
- `notebooks/sciver_full_search_v3_server.ipynb` is the only canonical notebook.
  It is a thin, ordered operator layer and contains no experiment logic.
- `scripts/run_full_search_v3_server.py` is the optional thin terminal wrapper.
- `model_inference/`, `utils/`, `evaluation/`, and `main.py` retain the
  supported generic benchmark and local inference paths.

## Server installation before JupyterLab

Prepare the repository and environment manually. The notebook never changes
the checkout and never installs dependencies.

```bash
git clone https://github.com/TruongDat05/sciver-meta-harness.git
cd sciver-meta-harness
git rev-parse HEAD
git remote get-url origin
git status --short --untracked-files=all
```

Record the exact 40-character lowercase commit SHA. The notebook requires that
SHA, the exact configured `origin`, and a clean worktree including untracked
non-ignored files. Ignored runtime artifacts under `workspace/meta_harness/`
are allowed. A mismatch fails closed without changing branches or files.

Create the environment before starting JupyterLab. Either:

```bash
conda create -n sciver-v3 python=3.11
conda activate sciver-v3
python -m pip install -r requirements.txt
```

or:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the private SciVer dataset outside version control and provide its
absolute path at notebook runtime. Do not copy samples, identifiers, labels,
responses, or images into documentation, logs, or notebook output.

Start JupyterLab in a persistent session:

```bash
tmux new -s sciver-v3-jupyter
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

From the operator machine, establish an SSH tunnel and reconnect if needed:

```bash
ssh -L 8888:127.0.0.1:8888 <server>
tmux attach -t sciver-v3-jupyter
```

Open `notebooks/sciver_full_search_v3_server.ipynb`. Supply
`PINNED_COMMIT_SHA`, `SCIVER_DATASET_PATH`, and `SCIVER_RUN_ID` through the
process environment or the notebook's runtime prompts. Do not edit and save
deployment values into the tracked notebook.

## Canonical notebook operation

Run the stages in order:

1. **SETUP** locates the existing repository, verifies the canonical origin,
   pinned commit, clean worktree, dependency metadata, imports, and dataset
   path.
2. **SETUP / preparation** creates or validates the exact deterministic seed-42
   1,000 SEARCH / 1,000 FINAL paper-disjoint artifacts through repository code.
3. **OFFLINE_SMOKE** validates the configuration, canonical P0 request and
   payload hash, parser identity, workload, paths, and compatible run state. It
   reads no credentials, creates no live client, invokes no proposer, and makes
   no HTTP request.
4. **LIVE_SMOKE** remains disabled unless the operator types exactly
   `RUN_LIVE_SMOKE`. Only then does the notebook read runtime `API_URL` and
   hidden `API_KEY`, perform `GET {API_URL}/models`, require the locked model,
   and perform exactly one canonical P0 SEARCH-safe
   `POST {API_URL}/chat/completions`.
5. **Smoke receipt validation** checks the sanitized create-once receipt. It
   contains only compatibility identities, safe hashes/counts, one logical
   call, and parse status.
6. **FULL_SEARCH** remains disabled unless the operator separately types
   `RUN_FULL_SEARCH`. It starts or resumes only compatible durable work.
7. **SEARCH status** may be rerun during operation. After terminal
   `patience_stopped` or `max_stopped`, **freeze** creates or verifies the
   immutable SEARCH-only winner.
8. **FINAL preflight** validates the frozen cross-stage identity offline.
9. **FINAL** remains disabled unless the operator separately types exactly
   `RUN_FINAL_ONCE`; it runs or resumes only the paired P0/P* FINAL work.
10. **Sanitized reporting** displays aggregate status and artifact paths only.

`API_URL` is the exact runtime base URL. The v3 transport trims surrounding
whitespace and redundant trailing slashes, then joins the separately identified
paths `/models` and `/chat/completions`; it never inserts `/v1`. URLs with
embedded credentials, query strings, or fragments are rejected. `API_KEY` is
trimmed at runtime, must be non-empty, and must contain no newline. Neither
value is stored in a notebook, command argument, run artifact, receipt, or log.

The existing generic remote client remains backward compatible: callers that
already pass a complete Chat Completions URL continue to use it as-is. Base-URL
joining is an additive v3 deployment mode.

## Monitoring, interruption, and resume

Use notebook status cells or the thin wrapper; do not start competing runners:

```bash
python scripts/run_full_search_v3_server.py activity \
  --repository-root . --run-id <run-id>
python scripts/run_full_search_v3_server.py search-status \
  --repository-root . --run-id <run-id>
python scripts/run_full_search_v3_server.py final-status \
  --repository-root . --run-id <run-id>
```

For a safe interruption, send one `Ctrl-C`, wait for the process to return,
inspect activity/status, and rerun the same stage with the same repository,
commit, run ID, split, deployment identity, and runtime configuration. Do not
delete locks, caches, checkpoints, receipts, or immutable artifacts. Compatible
completed SEARCH cache entries and FINAL request hashes are reused;
incompatible transport, solver, generation, parser, prompt, split, proposer, or
source identities fail closed.

Run-local artifacts are under:

```text
workspace/meta_harness/full_search_v3/<run-id>/
├── preparation/
├── smoke/completion.json
├── search_cache/
├── orchestration_state.json
├── freeze/frozen_winner.json
└── final/
```

See [the server operations guide](docs/sciver_full_search_v3_server_operations.md)
for terminal commands and detailed recovery guidance.

## Generic remote benchmarks

The additive provider-neutral benchmark CLI supports:

- Datasets: SciVer, SciAtomicBench, MuSciClaims, and SciClaimEval.
- Models: `Qwen2.5-VL-7B-Instruct`, `gemma-4-31B-it`,
  `gemma-4-26B-A4B-it`, and `gemma-3-27b-it`.
- Existing local providers and the explicit `--provider remote` path.

The generic remote CLI continues to support local dotenv loading for existing
workflows, but no credential example file is committed. Create a private,
ignored `.env` only if that CLI workflow needs it, using exactly `API_URL` and
`API_KEY`; unmistakably fake placeholders belong only in tests. Existing shell
values take precedence. Offline preparation, dry-run, evaluation, and tests do
not load or require credentials.

Example offline dry-run:

```bash
python main.py \
  --provider remote \
  --dataset SciVer \
  --dataset-path /absolute/path/to/SciVer/testset.json \
  --model Qwen2.5-VL-7B-Instruct \
  --method cot \
  --max-num 1 \
  --dry-run
```

Live generic benchmark access remains explicit through `--live-api`. See the
[remote benchmark guide](docs/remote-benchmark.md) for supported schema
adapters, normalization, checkpointing, evaluation, and CLI compatibility.

The original local inference and evaluation entry points remain available:

```bash
bash scripts/vllm_large.sh
python acc_evaluation.py
```

## Offline verification and security

All tests deny or mock HTTP, socket, solver, proposer, dataset, and subprocess
boundaries as appropriate. No test may invoke a live service or recursively
invoke Codex.

```bash
python -m compileall -q .
pytest -q
git diff --check
```

Never commit or expose API credentials, Authorization headers, runtime API
endpoints, request/response bodies, image base64, private sample identifiers,
labels, predictions, or FINAL material. Live stages are opt-in and independent;
module import, SETUP, OFFLINE_SMOKE, tests, and status inspection are offline.

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
