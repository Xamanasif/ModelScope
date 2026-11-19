"""Distribution-distance diagnostics and SIMULATED monitoring batches.

- Covariate shift: P(X) changes. Detectable without labels (PSI, KS).
- Prevalence (label) shift: P(Y) changes. Needs labels to confirm; unlabeled
  data only shows indirect signs (e.g. the mean predicted probability moves).
- Concept drift: P(Y | X) changes. CANNOT be identified from unlabeled inputs;
  it requires labelled outcomes. Batch S6 below demonstrates this: inputs are
  unchanged (PSI near 0) while outcomes and performance change.

PSI thresholds are configurable review triggers (policy.drift), not regulatory
limits. Every batch produced here is SIMULATED from the tuning partition and is
not a live monitoring feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4


def _bins(reference: pd.Series, n_bins: int):
    ref = reference.dropna()
    if ref.nunique() <= 12:
        return "categorical", sorted(ref.unique().tolist())
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return "quantile", edges.tolist()


def _shares(values: pd.Series, kind, spec):
    n = len(values)
    missing = values.isna().to_numpy()
    v = values.to_numpy(dtype=float)
    if kind == "categorical":
        counts = [np.sum(v == c) for c in spec]
        other = n - missing.sum() - sum(counts)
        counts = counts + [other]
    else:
        counts = np.histogram(v[~missing], bins=np.asarray(spec))[0].tolist()
    counts = counts + [missing.sum()]
    return np.asarray(counts, dtype=float) / max(n, 1)


def psi(reference: pd.Series, current: pd.Series, n_bins=10) -> float:
    """Population Stability Index: sum (c - r) * ln(c / r) over bins.

    Bins come from the reference only (quantiles, or categories for <= 12
    distinct values) plus an explicit missing-value bin and an 'unseen
    category' bin; empty shares are floored at 1e-4 to keep the log finite.
    """
    kind, spec = _bins(reference, n_bins)
    r = np.clip(_shares(reference, kind, spec), EPS, None)
    c = np.clip(_shares(current, kind, spec), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, columns, policy) -> list[dict]:
    cfg = policy["drift"]
    rows = []
    for col in columns:
        value = psi(reference[col], current[col], cfg["psi_bins"])
        ref, cur = reference[col].dropna(), current[col].dropna()
        ks = ks_2samp(ref, cur).statistic if len(cur) and len(ref) else None
        status = ("review" if value >= cfg["psi_review_trigger"] else
                  "watch" if value >= cfg["psi_watch"] else "stable")
        rows.append({"feature": col, "psi": value, "ks_statistic": None if ks is None else float(ks),
                     "missing_share": float(current[col].isna().mean()), "status": status})
    return sorted(rows, key=lambda r: -r["psi"])


def simulated_batches(frame: pd.DataFrame, y: np.ndarray, seed: int, size: int = 1500) -> list[dict]:
    """Six SIMULATED batches drawn from the tuning partition.

    S1-S5: oversample customers with a recent delay code (PAY_0 >= 1) by an
    increasing factor -> covariate shift + accompanying prevalence shift.
    S6: inputs drawn like S1 but 40% of defaults among PAY_0 >= 2 relabelled
    as non-default -> a simulated change in P(Y|X) (concept drift).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    delinquent = (frame["PAY_0"] >= 1).to_numpy()
    factors = [1.0, 1.25, 1.5, 2.0, 3.0]
    batches = []
    for i, factor in enumerate(factors, start=1):
        w = np.where(delinquent, factor, 1.0)
        idx = rng.choice(len(frame), size=size, replace=True, p=w / w.sum())
        batches.append({"batch": f"S{i}", "label": "SIMULATED", "kind": "covariate_and_prevalence_shift",
                        "description": f"Customers with PAY_0 >= 1 oversampled x{factor}",
                        "frame": frame.iloc[idx].reset_index(drop=True), "y": y[idx].copy()})
    idx = rng.choice(len(frame), size=size, replace=True)
    y6 = y[idx].copy()
    target = np.where((frame["PAY_0"].to_numpy()[idx] >= 2) & (y6 == 1))[0]
    flip = rng.choice(target, size=int(0.4 * len(target)), replace=False) if len(target) else []
    y6[flip] = 0
    batches.append({"batch": "S6", "label": "SIMULATED", "kind": "concept_drift",
                    "description": "Inputs sampled as in S1; 40% of defaults with PAY_0 >= 2 relabelled "
                                   "non-default (P(Y|X) changed, P(X) unchanged)",
                    "frame": frame.iloc[idx].reset_index(drop=True), "y": y6})
    return batches
