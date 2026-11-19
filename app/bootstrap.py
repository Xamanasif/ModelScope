"""Paired nonparametric bootstrap for model comparisons.

"Paired" = each resample draws one set of customer indices and evaluates BOTH
models on exactly those customers, so the difference's interval reflects the
correlation between the two models' errors. Intervals are percentile intervals
conditional on the fitted models and this evaluation sample; they do not
include uncertainty from re-training on a different training sample.

Resamples where a metric is undefined (e.g. only one class drawn) are counted
in `n_failed` and excluded; they are never silently treated as zero.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def fast_auc(y, p) -> float:
    """Exact ROC-AUC via the Mann-Whitney U statistic with mid-ranks for ties.

    AUC = (sum of positive-class ranks - n1 (n1 + 1) / 2) / (n1 * n0).
    Identical to sklearn.metrics.roc_auc_score (tested) but without sklearn's
    per-call input validation, which dominated bootstrap run time.
    """
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("AUC undefined for a single class")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_average_precision(y, p) -> float:
    """Step-wise average precision AP = sum_k (R_k - R_{k-1}) * P_k, evaluated at
    each distinct score threshold (ties grouped), matching
    sklearn.metrics.average_precision_score (tested)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n1 = int(y.sum())
    if n1 == 0:
        raise ValueError("AP undefined without positives")
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    last_of_group = np.r_[np.where(np.diff(ps) != 0)[0], len(ps) - 1]
    tp = np.cumsum(ys)[last_of_group]
    fp = (last_of_group + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_log_loss(y, p, eps=1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


METRICS = {
    "roc_auc": (fast_auc, "higher_is_better"),
    "average_precision": (fast_average_precision, "higher_is_better"),
    "log_loss": (fast_log_loss, "lower_is_better"),
    "brier": (lambda y, p: float(np.mean((np.asarray(p, dtype=float) - np.asarray(y)) ** 2)), "lower_is_better"),
}


def _percentile_ci(values, confidence):
    if len(values) == 0:
        return [None, None]
    alpha = (1 - confidence) / 2
    return [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))]


def paired_bootstrap(y, p_baseline, p_challenger, n_resamples=1000, seed=2026, confidence=0.95,
                     metrics=None):
    y = np.asarray(y).astype(int)
    pa = np.asarray(p_baseline, dtype=float)
    pb = np.asarray(p_challenger, dtype=float)
    if not (len(y) == len(pa) == len(pb)) or len(y) == 0:
        raise ValueError("Paired bootstrap needs equal-length non-empty inputs")
    names = metrics or list(METRICS)
    rng = np.random.default_rng(seed)
    draws = {m: {"baseline": [], "challenger": [], "difference": []} for m in names}
    failed = 0
    n = len(y)
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            failed += 1
            continue
        for m in names:
            fn = METRICS[m][0]
            a, b = fn(yb, pa[idx]), fn(yb, pb[idx])
            draws[m]["baseline"].append(a)
            draws[m]["challenger"].append(b)
            draws[m]["difference"].append(b - a)
    out = {}
    for m in names:
        fn, direction = METRICS[m]
        point_a, point_b = (fn(y, pa), fn(y, pb)) if y.min() != y.max() else (None, None)
        diffs = np.asarray(draws[m]["difference"])
        out[m] = {
            "direction": direction,
            "baseline": point_a, "challenger": point_b,
            "difference_challenger_minus_baseline": (point_b - point_a) if point_a is not None else None,
            "baseline_ci": _percentile_ci(draws[m]["baseline"], confidence),
            "challenger_ci": _percentile_ci(draws[m]["challenger"], confidence),
            "difference_ci": _percentile_ci(diffs, confidence),
            "share_resamples_challenger_better": (
                float(np.mean(diffs > 0) if direction == "higher_is_better" else np.mean(diffs < 0))
                if len(diffs) else None),
        }
    return {"n": n, "n_resamples": n_resamples, "n_failed": failed, "n_used": n_resamples - failed,
            "seed": seed, "confidence": confidence, "method": "paired percentile bootstrap over customers",
            "metrics": out}


def bootstrap_single(y, p, fn, n_resamples=200, seed=0, confidence=0.95):
    """CI for one statistic of one model (used for slices). Returns (ci, n_failed)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    vals, failed = [], 0
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            failed += 1
            continue
        vals.append(fn(y[idx], p[idx]))
    return _percentile_ci(vals, confidence), failed
