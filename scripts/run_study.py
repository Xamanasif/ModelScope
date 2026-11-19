"""Command-line study runner (same code path as the API job runner).

  python scripts/run_study.py development --actor "Your Name"
  python scripts/run_study.py final --study STUDY_ID --actor "Your Name" --confirm-final-holdout
  python scripts/run_study.py list

Training never starts automatically; the dashboard can also start these jobs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_policy  # noqa: E402
from app.registry import Registry  # noqa: E402
from app.study import current_identity, new_study, run_development_study, run_final_evaluation, stale_reasons  # noqa: E402


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    dev = sub.add_parser("development")
    dev.add_argument("--actor", required=True)
    fin = sub.add_parser("final")
    fin.add_argument("--study", required=True)
    fin.add_argument("--actor", required=True)
    fin.add_argument("--confirm-final-holdout", action="store_true")
    sub.add_parser("list")
    args = parser.parse_args()
    reg = Registry()
    policy = load_policy()
    if args.cmd == "list":
        ident = current_identity(policy)
        for s in reg.all("SELECT * FROM studies ORDER BY created_utc"):
            reasons = stale_reasons(s, ident)
            print(s["study_id"], s["status"], f"stale: {'; '.join(reasons)}" if reasons else "current")
        return
    if args.cmd == "development":
        ident = current_identity(policy)
        existing = reg.one("SELECT study_id FROM studies WHERE identity_key=? AND status IN "
                           "('development_complete','final_evaluated')", (ident["identity_key"],))
        if existing:
            log(f"Cached: {existing['study_id']} already has this dataset/policy/code identity; not retraining.")
            return
        study_id, _ = new_study(reg, args.actor, policy)
        log(f"study {study_id}")
        t = time.perf_counter()
        try:
            result = run_development_study(reg, study_id, policy, log)
        except Exception as exc:
            reg.execute("UPDATE studies SET status='failed', error=? WHERE study_id=?", (repr(exc), study_id))
            raise
        log(f"done in {time.perf_counter() - t:.1f}s: {json.dumps(result)}")
    if args.cmd == "final":
        result = run_final_evaluation(reg, args.study, args.actor, args.confirm_final_holdout, policy, log)
        log(json.dumps(result))


if __name__ == "__main__":
    main()
