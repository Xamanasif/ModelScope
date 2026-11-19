"""Segment (slice) analysis with support counts, uncertainty and caution labels.

Descriptive audit only. Slices are pre-specified in the policy; credit-limit
band edges are fitted on the TRAIN partition (quartiles), never on the
partition being evaluated. Demographic slices are not a legal fairness
determination and are not corrected for multiple comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from app.bootstrap import bootstrap_single, fast_auc
from app.data import PAY_STATUS, SEX_LABELS, education_group, marriage_group


def fit_band_edges(train_frame: pd.DataFrame) -> dict:
    q = np.quantile(train_frame["LIMIT_BAL"], [0.25, 0.5, 0.75])
    return {"LIMIT_BAL_quartile_edges": [float(v) for v in q]}


def slice_labels(frame: pd.DataFrame, edges: dict, age_bands: list) -> dict[str, pd.Series]:
    e = edges["LIMIT_BAL_quartile_edges"]
    limit = frame["LIMIT_BAL"]
    limit_band = pd.Series(np.select(
        [limit <= e[0], limit <= e[1], limit <= e[2]],
        [f"Q1 <= {e[0]:,.0f}", f"Q2 ({e[0]:,.0f}, {e[1]:,.0f}]", f"Q3 ({e[1]:,.0f}, {e[2]:,.0f}]"],
        default=f"Q4 > {e[2]:,.0f}"), index=frame.index)
    pay0 = frame["PAY_0"]
    recent = pd.Series(np.select([pay0 <= 0, pay0 == 1], ["PAY_0 <= 0 (no delay code)", "PAY_0 = 1"],
                                 default="PAY_0 >= 2"), index=frame.index)
    delayed_months = (frame[PAY_STATUS] >= 1).sum(axis=1)
    history = pd.Series(np.select([delayed_months == 0, delayed_months <= 2],
                                  ["0 delayed months", "1-2 delayed months"], default="3+ delayed months"),
                        index=frame.index)
    age = frame["AGE"]
    age_band = pd.Series("unbanded", index=frame.index)
    for lo, hi in age_bands:
        age_band[(age >= lo) & (age <= hi)] = f"{lo}-{hi}"
    return {
        "credit_limit_band": limit_band,
        "recent_repayment_status": recent,
        "six_month_delay_count": history,
        "SEX": frame["SEX"].map(SEX_LABELS).fillna("undocumented_code"),
        "EDUCATION": education_group(frame["EDUCATION"]),
        "MARRIAGE": marriage_group(frame["MARRIAGE"]),
        "AGE_BAND": age_band,
    }


DEMOGRAPHIC_SLICES = {"SEX", "EDUCATION", "MARRIAGE", "AGE_BAND"}


def _model_slice(y, p, threshold, overall_auc, min_events, seed, n_boot):
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    out = {"mean_predicted": float(p.mean()), "observed_minus_predicted": float(y.mean() - p.mean())}
    pred = p >= threshold
    out["recall"] = float(pred[y == 1].mean()) if n_pos else None
    out["false_positive_rate"] = float(pred[y == 0].mean()) if n_neg else None
    out["flag_rate"] = float(pred.mean())
    if n_pos == 0 or n_neg == 0:
        out.update(roc_auc=None, roc_auc_ci=[None, None],
                   roc_auc_unavailable_reason="single class in slice: AUC undefined")
    else:
        out["roc_auc"] = float(roc_auc_score(y, p))
        ci, failed = bootstrap_single(y, p, fast_auc, n_boot, seed)
        out["roc_auc_ci"] = ci
        out["roc_auc_bootstrap_failed"] = failed
        out["roc_auc_unavailable_reason"] = None
        out["auc_ci_below_overall"] = bool(ci[1] is not None and ci[1] < overall_auc)
        if min(n_pos, n_neg) < min_events:
            out["roc_auc_caution"] = f"fewer than {min_events} events in a class; AUC unstable"
    gap_ci, _ = bootstrap_single(y, p, lambda yy, pp: float(yy.mean() - pp.mean()), n_boot, seed + 1)
    out["observed_minus_predicted_ci"] = gap_ci
    return out


def slice_report(frame, y, probs: dict, edges, policy, threshold, overall_auc: dict):
    cfg = policy["slices"]
    labels = slice_labels(frame, edges, cfg["age_bands"])
    y = np.asarray(y).astype(int)
    rows = []
    for family, series in labels.items():
        for group in sorted(series.unique()):
            mask = (series == group).to_numpy()
            n = int(mask.sum())
            ys = y[mask]
            warnings = []
            if n < cfg["minimum_support"]:
                warnings.append(f"support {n} < {cfg['minimum_support']}")
            if ys.sum() < cfg["minimum_events"]:
                warnings.append(f"defaults {int(ys.sum())} < {cfg['minimum_events']}")
            row = {"slice": family, "group": str(group), "demographic": family in DEMOGRAPHIC_SLICES,
                   "n": n, "n_default": int(ys.sum()),
                   "prevalence": float(ys.mean()) if n else None, "warnings": warnings, "models": {}}
            if n == 0:
                row["unavailable_reason"] = "empty slice"
            else:
                # Deterministic per-slice seed (Python's hash() is salted per process).
                seed = int.from_bytes(f"{family}|{group}".encode(), "little") % (2 ** 31)
                for name, p in probs.items():
                    row["models"][name] = _model_slice(ys, np.asarray(p)[mask], threshold, overall_auc[name],
                                                       cfg["minimum_events"], seed, cfg["bootstrap_resamples"])
            rows.append(row)
    return {"edges": edges, "rows": rows, "threshold": threshold,
            "note": cfg["language"],
            "uncertainty": f"Percentile bootstrap within slice, {cfg['bootstrap_resamples']} resamples."}
