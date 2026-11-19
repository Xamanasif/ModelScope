"""Trace ONE development (tuning-partition) customer through the whole pipeline.

  python scripts/trace_record.py --study STUDY_ID [--customer ID]

Prints raw values, each preprocessing stage, raw scores, calibrated
probabilities and the SHAP decomposition, all computed live from the trusted
registered artifacts. Demographic columns are not printed. Final-holdout
customers are refused, so the trace never touches sealed data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_policy  # noqa: E402
from app.data import ID_COLUMN, TARGET  # noqa: E402
from app.features import transformed_feature_map  # noqa: E402
from app.registry import Registry  # noqa: E402
from app.study import ROLES, load_prepared, load_trusted_model, partition  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--customer", type=int)
    args = ap.parse_args()
    reg = Registry()
    policy = load_policy()
    frame, parts, _, _ = load_prepared(policy)
    ex = reg.read_evidence(args.study, "explanations")
    cid = args.customer if args.customer is not None else ex["explained_ids"][0]
    if cid not in set(parts["tuning"]):
        raise SystemExit("Only tuning-partition (development) customers can be traced.")
    tune = partition(frame, parts, "tuning")
    row = tune[tune[ID_COLUMN] == cid]
    cols = ex["features"]
    print(f"Customer ID {cid} (tuning partition). Observed outcome DEFAULT_NEXT_MONTH = {int(row[TARGET].iloc[0])}")
    print("Raw predictor values (demographics not shown, not predictors by default):")
    print("  " + ", ".join(f"{c}={int(row[c].iloc[0])}" for c in cols))
    i = ex["explained_ids"].index(cid) if cid in ex["explained_ids"] else None
    for role in ROLES:
        model = load_trusted_model(reg, args.study, role)
        pre = model.base.named_steps["preprocess"]
        names, origins = transformed_feature_map(model.base, cols)
        z = pre.transform(row[cols])[0]
        print(f"\n[{role}] {ROLES[role]}: {len(names)} transformed columns")
        shown = [(n, v) for n, v in zip(names, z) if not (n.startswith("PAY_") and "_" in n[4:] and v == 0)]
        print("  transformed (non-zero one-hot and all numeric): " + ", ".join(f"{n}={v:.3f}" for n, v in shown[:24]))
        raw = float(model.raw_score(row[cols])[0])
        p = float(model.predict_proba(row[cols])[0])
        print(f"  raw score (log-odds) s = {raw:.4f}; uncalibrated p = {1 / (1 + np.exp(-raw)):.4f}")
        print(f"  calibration {model.method}: logit = {model.a:.4f} * s + {model.b:.4f} = {model.a * raw + model.b:.4f}"
              f" -> calibrated p = {p:.4f}")
        if i is not None:
            m = ex["models"][role]
            phi = np.array(m["local"]["phi"][i])
            top = np.argsort(-np.abs(phi))[:6]
            print(f"  SHAP (calibrated log-odds): base {m['base_value_calibrated_logit']:.4f} + sum {phi.sum():.4f}"
                  f" = {m['base_value_calibrated_logit'] + phi.sum():.4f}")
            print("  largest contributions: " + ", ".join(f"{cols[j]} {phi[j]:+.4f}" for j in top))
    case = reg.one("SELECT state FROM review_cases WHERE study_id=?", (args.study,))
    print(f"\nReview: this customer is part of the evidence for case state '{case['state']}'.")


if __name__ == "__main__":
    main()
