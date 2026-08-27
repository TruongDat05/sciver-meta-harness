#!/usr/bin/env python3
"""Live Meta-Harness full-SEARCH monitor.

Refreshes every N seconds and reports:
  - run status / P0 / winner / patience / stop reason
  - completed candidate iterations out of the maximum
  - the currently running evaluation (P0 or candidate id)
  - samples: processed, successful, parse-failed, infrastructure-failed
  - live throughput and estimated time remaining

Reads only the durable SEARCH artifacts under `workspace/meta_harness/`; it
never dispatches requests and never prints payloads or credentials.

Usage:
  python3 scripts/monitor_search.py [--repository-root DIR] [--run-id ID] [--interval N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _repo_default() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _run_dir(repo: Path, run_id: str) -> Path:
    return repo / "workspace" / "meta_harness" / "full_search_v3" / run_id


def _read_state(repo: Path, run_id: str) -> dict:
    return _load_json(_run_dir(repo, run_id) / "orchestration_state.json")


def _checkpoint_for(state: dict, repo: Path, run_id: str) -> tuple[str | None, dict]:
    """Return (label, checkpoint) for the evaluation currently in progress."""
    evaluations = _run_dir(repo, run_id) / "evaluations"
    for label, cid, ckpt in (
        ("P0 (cot)", "cot", evaluations / "cot.checkpoint.json"),
    ):
        if state.get("p0", {}).get("status") in ("pending", "evaluating", "incomplete"):
            return label, _load_json(ckpt)
        break
    for entry in state.get("iterations", []):
        if entry.get("status") in ("proposed", "evaluating", "incomplete"):
            cid = entry.get("candidate", {}).get("candidate_id")
            label = f"candidate {cid}"
            ckpt = evaluations / f"{cid}.checkpoint.json"
            return label, _load_json(ckpt)
    return None, {}


def _per_iteration_total(repo: Path, run_id: str) -> int:
    """Authoritative per-iteration sample count from the SEARCH record set."""
    records = _run_dir(repo, run_id) / "preparation" / "search" / "search_records.json"
    value = _load_json(records)
    if isinstance(value, list):
        return len(value)
    return 0


def _active_checkpoint_counts(checkpoint: dict) -> dict:
    """Summarize the in-flight checkpoint's completed samples."""
    samples = checkpoint.get("completed_samples") or []
    total = len(samples)
    infra = 0
    parse_fail = 0
    success = 0
    for s in samples:
        status = s.get("request_status")
        if status == "success":
            success += 1
        elif status in ("failed", "infrastructure_failure", "retry_exhausted"):
            infra += 1
        if s.get("parse_status") not in (None, "parsed"):
            parse_fail += 1
    return {
        "label": None,
        "processed": total,
        "success": success,
        "infra_fail": infra,
        "parse_fail": parse_fail,
        "total_records": None,
    }


def _completed_reports(state: dict):
    reports = []
    p0 = state.get("p0", {}).get("report")
    if p0:
        reports.append(("P0", p0))
    for entry in state.get("iterations", []):
        if entry.get("status") == "complete" and entry.get("report"):
            cid = entry.get("candidate", {}).get("candidate_id")
            reports.append((cid, entry["report"]))
    return reports


def _aggregate(repo: Path, run_id: str) -> dict:
    state = _read_state(repo, run_id)
    completed = [e for e in state.get("iterations", []) if e.get("status") == "complete"]
    max_iter = 40
    label, checkpoint = _checkpoint_for(state, repo, run_id)
    active = _active_checkpoint_counts(checkpoint)
    active["label"] = label
    per_iteration = _per_iteration_total(repo, run_id)
    if per_iteration:
        active["total_records"] = per_iteration

    sums = {"total_records": 0, "responses": 0, "parsed": 0, "abstentions": 0, "infra": 0}
    for _cid, r in _completed_reports(state):
        sums["total_records"] += r.get("total_records", 0)
        sums["responses"] += r.get("completed_solver_responses", 0)
        sums["parsed"] += r.get("parsed_predictions", 0)
        sums["abstentions"] += r.get("abstentions_or_parse_failures", 0)
        sums["infra"] += r.get("infrastructure_failures", 0)

    return {
        "status": state.get("status"),
        "stop_reason": state.get("stop_reason"),
        "p0_status": state.get("p0", {}).get("status"),
        "winner_id": state.get("winner_id"),
        "patience": state.get("patience"),
        "ranking": state.get("ranking"),
        "completed_iterations": len(completed),
        "max_iterations": max_iter,
        "active": active,
        "completed_sums": sums,
        "state_timestamp": _mtime(repo, run_id),
    }


def _mtime(repo: Path, run_id: str) -> float:
    p = _run_dir(repo, run_id) / "orchestration_state.json"
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _bar(done: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return "[n/a]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _render(data: dict, rate_per_s: float | None) -> str:
    active = data["active"]
    cs = data["completed_sums"]
    lines = []
    lines.append(f"run_status    : {data['status']}  (stop: {data['stop_reason']})")
    lines.append(f"P0            : {data['p0_status']}")
    done = data["completed_iterations"]
    lines.append(
        f"iterations    : {done}/{data['max_iterations']} {_bar(done, data['max_iterations'])}"
    )
    ref = active.get("processed") or 0
    lines.append(
        f"active eval   : {active['label'] or 'none'}  samples {ref}/{(active.get('total_records') or ref)}"
    )
    lines.append(
        f"  active now  : processed={active.get('processed', 0)} "
        f"success={active.get('success', 0)} infra_fail={active.get('infra_fail', 0)} "
        f"parse_fail={active.get('parse_fail', 0)}"
    )
    lines.append(
        f"completed evals: records={cs['total_records']} responses={cs['responses']} "
        f"parsed={cs['parsed']} abstentions/parse_fail={cs['abstentions']} infra_fail={cs['infra']}"
    )
    if data.get("winner_id"):
        lines.append(f"winner        : {data['winner_id']}")
    if data.get("patience"):
        p = data["patience"]
        lines.append(
            f"patience      : non_improving={p.get('consecutive_non_improving')} "
            f"best_f1={p.get('best_macro_f1')} best_acc={p.get('best_accuracy')}"
        )
    if rate_per_s is not None:
        lines.append(f"throughput    : {rate_per_s:.2f} samples/s")
    remaining = data["max_iterations"] - done
    if rate_per_s is not None and rate_per_s > 0 and active.get("total_records"):
        left = (remaining * active["total_records"] - active.get("processed", 0))
        if left > 0:
            lines.append(f"ETA           : ~{left / rate_per_s / 60:.1f} min")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=_repo_default())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    if args.run_id is None:
        dotenv = _repo_default() / ".env"
        value = None
        try:
            for line in dotenv.read_text(encoding="utf-8").splitlines():
                if line.startswith("SCIVER_RUN_ID="):
                    value = line.split("=", 1)[1].strip()
        except OSError:
            pass
        args.run_id = value or "run01"

    earlier = {"done": None, "processed": None, "t": None}
    try:
        while True:
            data = _aggregate(args.repository_root.resolve(), args.run_id)
            now = time.time()
            processed = (data["active"].get("processed") or 0) + (
                data["completed_sums"]["total_records"]
            )
            done = data["completed_iterations"]
            rate = None
            if earlier["t"] is not None and earlier["processed"] is not None:
                dt = now - earlier["t"]
                dp = processed - earlier["processed"]
                if dt > 0 and dp >= 0:
                    rate = dp / dt
            print("\033[2J\033[H", end="")  # clear screen (terminal)
            print(_render(data, rate))
            print(f"\n[refresh {time.strftime('%H:%M:%S')}] press Ctrl-C to stop")
            sys.stdout.flush()
            earlier = {"done": done, "processed": processed, "t": now}
            time.sleep(max(args.interval, 1.0))
    except KeyboardInterrupt:
        print("\nmonitor stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
