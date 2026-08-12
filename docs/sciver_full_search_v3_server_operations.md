# SciVer Full-Search v3 server operations

The canonical [server notebook](../notebooks/sciver_full_search_v3_server.ipynb)
is the normal operator interface. It delegates every scientific operation to
`meta_harness.full_search_v3_server`; the optional terminal wrapper calls the
same APIs and durable state.

## Prepare the server before JupyterLab

```bash
git clone https://github.com/TruongDat05/sciver-meta-harness.git
cd sciver-meta-harness
git rev-parse HEAD
git remote get-url origin
git status --short --untracked-files=all
```

Record the exact 40-character lowercase commit SHA. Create an environment and
install dependencies before launching JupyterLab:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
tmux new -s sciver-v3-jupyter
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Open `notebooks/sciver_full_search_v3_server.ipynb` from this checkout. The
notebook verifies the exact canonical origin URL, pinned HEAD, clean worktree
including untracked non-ignored files, `requirements.txt`, required imports,
and dataset path. It never changes the checkout or installs packages. Put the
private dataset outside version control and supply its absolute path at runtime.

## Offline and live gates

Run SETUP, preparation, and OFFLINE_SMOKE first. OFFLINE_SMOKE constructs and
hashes one canonical P0 SEARCH-safe request through the production request
builder and validates immutable configuration, split, parser, workload, and
resume metadata. It reads no credential, constructs no live client, invokes no
proposer, and writes no production SEARCH state.

At the notebook layer the exact independent confirmations are:

- LIVE_SMOKE: `RUN_LIVE_SMOKE`
- FULL_SEARCH: `RUN_FULL_SEARCH`
- FINAL: `RUN_FINAL_ONCE`

An empty value skips the stage. Any other non-empty value fails before reading
credentials or constructing a client. The notebook reads only runtime
`API_URL` and hidden `API_KEY`, trims surrounding whitespace, rejects an empty
value or key containing a newline, and never displays either value.

For v3, `API_URL` is the exact base URL. The transport uses precisely:

```text
GET  {API_URL}/models
POST {API_URL}/chat/completions
```

It strips redundant join slashes, never inserts `/v1`, and rejects embedded
credentials, query strings, and fragments. LIVE_SMOKE first requires the
locked `Qwen/Qwen3.5-35B-A3B` model in the sanitized model list, then sends
exactly one canonical P0 SEARCH-safe logical request and requires canonical
parsing. The receipt stores only source/config/split/request/parser/deployment/
transport identities, safe model-list hashes/counts, one logical call, and
parse status.

## Terminal operation

Credentials are never accepted as command arguments. Live commands read only
the process `API_URL` and `API_KEY` values.

```bash
python scripts/run_full_search_v3_server.py prepare \
  --repository-root . --run-id <run-id> \
  --dataset-path <dataset>

python scripts/run_full_search_v3_server.py search-preflight \
  --repository-root . --run-id <run-id> \
  --search-safe-manifest <search-safe-manifest> \
  --search-records <search-records> --source-commit <pinned-sha>

python scripts/run_full_search_v3_server.py smoke \
  --repository-root . --run-id <run-id> \
  --search-safe-manifest <search-safe-manifest> \
  --search-records <search-records> --source-commit <pinned-sha> \
  --live-smoke

python scripts/run_full_search_v3_server.py search \
  --repository-root . --run-id <run-id> \
  --search-safe-manifest <search-safe-manifest> \
  --search-records <search-records> --source-commit <pinned-sha> \
  --live-search
```

Without the matching live flag, a command fails before credential access or
dispatch. Smoke and SEARCH authorizations do not authorize one another.

Monitor only sanitized status and process locks:

```bash
python scripts/run_full_search_v3_server.py activity \
  --repository-root . --run-id <run-id>
python scripts/run_full_search_v3_server.py search-status \
  --repository-root . --run-id <run-id>
df -h <workspace>
du -sh workspace/meta_harness/full_search_v3/<run-id>
```

After SEARCH reports `patience_stopped` or `max_stopped`, freeze the immutable
winner and run the offline FINAL preflight:

```bash
python scripts/run_full_search_v3_server.py freeze \
  --repository-root . --run-id <run-id>
python scripts/run_full_search_v3_server.py final-preflight \
  --repository-root . --run-id <run-id> \
  --dataset-path <dataset> --private-manifest <private-manifest> \
  --search-safe-manifest <search-safe-manifest> \
  --solver-identity-sha256 <safe-deployment-identity>
```

After separate FINAL authorization:

```bash
python scripts/run_full_search_v3_server.py final \
  --repository-root . --run-id <run-id> \
  --dataset-path <dataset> --private-manifest <private-manifest> \
  --search-safe-manifest <search-safe-manifest> --live-final
```

## Safe interruption and resume

Send one `Ctrl-C` and wait. Do not delete locks, caches, checkpoints, smoke
receipts, frozen artifacts, or FINAL state, and do not start a competing
runner. Reopen the same pinned clean checkout, restore the same runtime values,
rerun checkout/preflight validation, inspect `activity`, and invoke the same
stage with the same run ID. Completed compatible SEARCH cache entries and FINAL
request hashes are reused. Any source, split, model, generation, parser,
proposer, deployment, transport-version, or path-identity change fails closed.

Artifacts are stored under
`workspace/meta_harness/full_search_v3/<run-id>/`. Status and receipts are
sanitized; they must never contain the runtime endpoint, credential,
Authorization header, request/response body, images, sample identifiers,
labels, predictions, or FINAL material.
