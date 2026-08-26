#!/usr/bin/env bash
# Terminal operator script for the Meta-Harness full-SEARCH v3 workflow.
#
# Mirrors notebooks/sciver_meta_harness.ipynb as ordered terminal commands.
# Reads API_URL / API_KEY from the repo .env at runtime only (never passed as
# wrapper arguments, never written to artifacts).
#
# Live stages are explicit opt-in, exactly like the notebook confirmations:
#   RUN_LIVE_SMOKE=1   authorizes the isolated one-request SMOKE
#   RUN_FULL_SEARCH=1  authorizes SEARCH
#   RUN_FINAL_ONCE=1   authorizes paired FINAL (needs a frozen winner)
# Leave them empty/0 to keep the stage offline (no dispatch).
#
# Prerequisites:
#   - python3 on PATH (repo deps installed)
#   - `codex` CLI on PATH for the SEARCH proposer (prompt_proposer.py)
#   - .env with real API_URL and API_KEY
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ensure the user-installed codex CLI on ~/.local/bin is discoverable
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) export PATH="$HOME/.local/bin:$PATH" ;;
esac

# --- load runtime credentials from .env (repo local, gitignored) ---
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo "error: .env not found" >&2
  exit 2
fi
: "${API_URL:?API_URL is required in .env}"
: "${API_KEY:?API_KEY is required in .env}"

# --- operator settings ---
RUN_ID="${SCIVER_RUN_ID:-run01}"
COMMIT="${PINNED_COMMIT_SHA:-$(git rev-parse HEAD)}"
DATASET="${SCIVER_DATASET_PATH:-$ROOT/data/sciver/testset.json}"
PREP_DIR="$ROOT/workspace/meta_harness/full_search_v3/$RUN_ID/preparation"
SEARCH_MANIFEST="$PREP_DIR/search/search_safe_manifest.json"
SEARCH_RECORDS="$PREP_DIR/search/search_records.json"
PRIVATE_MANIFEST="$PREP_DIR/private/private_split_manifest.json"

command -v codex >/dev/null 2>&1 || {
  echo "warning: 'codex' CLI not on PATH (required for the SEARCH proposer)." >&2
}

echo "root=$ROOT run_id=$RUN_ID commit=$COMMIT"
echo "dataset=$DATASET"

# 1. prepare (offline)
python3 scripts/run_meta_harness.py prepare \
  --repository-root "$ROOT" --run-id "$RUN_ID" --dataset-path "$DATASET"

# 2. offline SEARCH preflight (no dispatch)
python3 scripts/run_meta_harness.py search-preflight \
  --repository-root "$ROOT" --run-id "$RUN_ID" \
  --search-safe-manifest "$SEARCH_MANIFEST" --search-records "$SEARCH_RECORDS" \
  --source-commit "$COMMIT"

# 3. isolated one-request smoke (explicit authorization)
if [[ "${RUN_LIVE_SMOKE:-0}" == "1" ]]; then
  python3 scripts/run_meta_harness.py smoke \
    --repository-root "$ROOT" --run-id "$RUN_ID" \
    --search-safe-manifest "$SEARCH_MANIFEST" --search-records "$SEARCH_RECORDS" \
    --source-commit "$COMMIT" --live-smoke
fi

# 4. inspection
python3 scripts/run_meta_harness.py activity \
  --repository-root "$ROOT" --run-id "$RUN_ID"
python3 scripts/run_meta_harness.py search-status \
  --repository-root "$ROOT" --run-id "$RUN_ID"

# 5. SEARCH (separate explicit authorization)
if [[ "${RUN_FULL_SEARCH:-0}" == "1" ]]; then
  python3 scripts/run_meta_harness.py search \
    --repository-root "$ROOT" --run-id "$RUN_ID" \
    --search-safe-manifest "$SEARCH_MANIFEST" --search-records "$SEARCH_RECORDS" \
    --source-commit "$COMMIT" --live-search
fi

# 6. terminal-winner freeze (once SEARCH is patience/max stopped)
python3 scripts/run_meta_harness.py freeze \
  --repository-root "$ROOT" --run-id "$RUN_ID"

# 7. offline FINAL preflight
SOLVER_IDENTITY="$(SOLVER_URL="$API_URL" python3 - <<'PY'
import os, sys
sys.path.insert(0, ".")
from meta_harness.server_run import solver_identity_from_api_url
print(solver_identity_from_api_url(os.environ["SOLVER_URL"]))
PY
)"
python3 scripts/run_meta_harness.py final-preflight \
  --repository-root "$ROOT" --run-id "$RUN_ID" \
  --dataset-path "$DATASET" --private-manifest "$PRIVATE_MANIFEST" \
  --search-safe-manifest "$SEARCH_MANIFEST" \
  --solver-identity-sha256 "$SOLVER_IDENTITY"

# 8. paired FINAL (separate explicit authorization)
if [[ "${RUN_FINAL_ONCE:-0}" == "1" ]]; then
  python3 scripts/run_meta_harness.py final \
    --repository-root "$ROOT" --run-id "$RUN_ID" \
    --dataset-path "$DATASET" --private-manifest "$PRIVATE_MANIFEST" \
    --search-safe-manifest "$SEARCH_MANIFEST" --live-final
fi

# 9. FINAL status
python3 scripts/run_meta_harness.py final-status \
  --repository-root "$ROOT" --run-id "$RUN_ID"
