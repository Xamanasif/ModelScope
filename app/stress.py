"""Seeded stress suite on the TUNING partition (development evidence only).

Two different kinds of scenario:
- input_corruption: predictor values are damaged while each customer's
  observed outcome is held fixed. This is a robustness probe of the model's
  sensitivity to bad inputs; it is NOT a causal simulation of what customers
  would do in a recession.
- composition: rows are resampled (labels travel with their rows) to change
  the customer mix or the default prevalence. This approximates covariate and
  prevalence shift, but every row still comes from the same historical sample.

Both models are evaluated on identical perturbed data (same seed per scenario).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.data import BILL_AMT, PAY_AMT, PAY_STATUS
from app.drift import psi
from app.metrics import binary_metrics

SCENARIOS = [
    {"id": "missing_bill_10", "kind": "input_corruption",
     "description": "10% of BILL_AMT1-6 cells set to missing at random",
     "valid_range_note": "Missing values do not occur in the source; LR imputes train medians, HGB routes NaN natively."},
    {"id": "missing_pay_status_10", "kind": "input_corruption",
     "description": "10% of PAY_0..PAY_6 cells set to missing at random",
     "valid_range_note": "LR imputes the train most-frequent code; HGB routes NaN to the larger child (no NaN seen in training)."},
    {"id": "missing_amounts_30", "kind": "input_corruption",
     "description": "30% of all BILL_AMT and PAY_AMT cells set to missing at random",
     "valid_range_note": "Severe missingness probe."},
    {"id": "limit_in_thousands_10", "kind": "input_corruption",
     "description": "10% of rows report LIMIT_BAL divided by 1000 (unit error)",
     "valid_range_note": "Produces values below the training minimum (10,000); the data validator flags them."},
    {"id": "amount_noise_10pct", "kind": "input_corruption",
     "description": "BILL_AMT and PAY_AMT multiplied by lognormal noise (sigma 0.10), rounded",
     "valid_range_note": "Multiplicative noise keeps PAY_AMT >= 0 and preserves bill signs."},
    {"id": "pay0_reporting_plus1_20", "kind": "input_corruption",
     "description": "For 20% of rows with PAY_0 >= 1, PAY_0 increased by 1 (cap 9)",
     "valid_range_note": "Stays inside the documented delay-code range."},
    {"id": "composition_delinquent_x2", "kind": "composition",
     "description": "Rows with PAY_0 >= 1 sampled with weight 2 (n unchanged, with replacement)",
     "valid_range_note": "Changes P(X) and, as a consequence, prevalence."},
    {"id": "composition_young_x2", "kind": "composition",
     "description": "Rows with AGE < 30 sampled with weight 2",
     "valid_range_note": "Composition change on a demographic attribute (not a predictor by default)."},
    {"id": "prevalence_10pct", "kind": "composition",
     "description": "Resampled so 10% of customers default (class-conditional sampling)",
     "valid_range_note": "Pure prevalence shift: P(X|Y) preserved, P(Y) changed."},
    {"id": "prevalence_35pct", "kind": "composition",
     "description": "Resampled so 35% of customers default (class-conditional sampling)",
     "valid_range_note": "Pure prevalence shift."},
]


def _mask_cells(frame, columns, share, rng):
    out = frame.copy()
    for col in columns:
        out[col] = out[col].astype(float)
        out.loc[rng.random(len(out)) < share, col] = np.nan
    return out


def apply_scenario(frame: pd.DataFrame, y: np.ndarray, scenario_id: str, seed: int):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(frame)
    f = frame.reset_index(drop=True)
    if scenario_id == "missing_bill_10":
        return _mask_cells(f, BILL_AMT, 0.10, rng), y
    if scenario_id == "missing_pay_status_10":
        return _mask_cells(f, PAY_STATUS, 0.10, rng), y
    if scenario_id == "missing_amounts_30":
        return _mask_cells(f, BILL_AMT + PAY_AMT, 0.30, rng), y
    if scenario_id == "limit_in_thousands_10":
        out = f.copy()
        out["LIMIT_BAL"] = out["LIMIT_BAL"].astype(float)
        hit = rng.random(n) < 0.10
        out.loc[hit, "LIMIT_BAL"] = out.loc[hit, "LIMIT_BAL"] / 1000.0
        return out, y
    if scenario_id == "amount_noise_10pct":
        out = f.copy()
        for col in BILL_AMT + PAY_AMT:
            out[col] = np.rint(out[col].to_numpy(dtype=float) * rng.lognormal(0.0, 0.10, n))
        return out, y
    if scenario_id == "pay0_reporting_plus1_20":
        out = f.copy()
        hit = (out["PAY_0"] >= 1).to_numpy() & (rng.random(n) < 0.20)
        out.loc[hit, "PAY_0"] = np.minimum(out.loc[hit, "PAY_0"] + 1, 9)
        return out, y
    if scenario_id in ("composition_delinquent_x2", "composition_young_x2"):
        mask = (f["PAY_0"] >= 1) if scenario_id == "composition_delinquent_x2" else (f["AGE"] < 30)
        w = np.where(mask.to_numpy(), 2.0, 1.0)
        idx = rng.choice(n, size=n, replace=True, p=w / w.sum())
        return f.iloc[idx].reset_index(drop=True), y[idx]
    if scenario_id in ("prevalence_10pct", "prevalence_35pct"):
        target = 0.10 if scenario_id == "prevalence_10pct" else 0.35
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        n_pos = int(round(target * n))
        idx = np.concatenate([rng.choice(pos, n_pos, replace=True), rng.choice(neg, n - n_pos, replace=True)])
        rng.shuffle(idx)
        return f.iloc[idx].reset_index(drop=True), y[idx]
    raise ValueError(f"Unknown scenario {scenario_id}")


def run_stress_suite(models: dict, frame: pd.DataFrame, y, feature_cols, train_reference: pd.DataFrame,
                     policy, threshold) -> dict:
    seed0 = policy["stress"]["seed"]
    clean = {name: binary_metrics(y, m.predict_proba(frame[feature_cols]), threshold) for name, m in models.items()}
    clean_scores = {name: m.predict_proba(frame[feature_cols]) for name, m in models.items()}
    train_min_limit = float(train_reference["LIMIT_BAL"].min())
    results = []
    for k, sc in enumerate(SCENARIOS):
        seed = seed0 + k
        fs, ys = apply_scenario(frame, y, sc["id"], seed)
        row = {**sc, "seed": seed, "n": int(len(ys)), "prevalence": float(np.mean(ys)),
               "rows_below_train_min_limit": int((fs["LIMIT_BAL"] < train_min_limit).sum()),
               "missing_cells": int(fs[feature_cols].isna().sum().sum()), "models": {}}
        for name, m in models.items():
            p = m.predict_proba(fs[feature_cols])
            met = binary_metrics(ys, p, threshold)
            row["models"][name] = {
                "roc_auc": met["roc_auc"], "log_loss": met["log_loss"], "brier": met["brier"],
                "average_precision": met["average_precision"], "mean_predicted": met["mean_predicted"],
                "recall": met["recall"], "specificity": met["specificity"],
                "delta_roc_auc": met["roc_auc"] - clean[name]["roc_auc"],
                "delta_log_loss": met["log_loss"] - clean[name]["log_loss"],
                "score_psi_vs_clean": psi(pd.Series(clean_scores[name]), pd.Series(p)),
            }
        results.append(row)
    return {
        "partition": policy["stress"]["partition"],
        "clean": {k: {m: v[m] for m in ("roc_auc", "log_loss", "brier", "average_precision",
                                         "mean_predicted", "prevalence")} for k, v in clean.items()},
        "scenarios": results,
        "caveat": ("Input-corruption scenarios hold outcomes fixed: robustness probes, not causal "
                   "simulations of economic stress. Composition scenarios resample historical rows."),
    }
