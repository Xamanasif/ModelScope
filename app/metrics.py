"""Metrics for binary labels; the positive class is explicitly label 1 (default).

Three different questions:
- Discrimination/ranking: ROC-AUC, average precision (AP; step-wise sum
  sum_k (R_k - R_{k-1}) P_k as in sklearn, NOT trapezoidal PR-AUC).
- Probability quality: log loss and Brier score (proper scoring rules),
  calibration curve, calibration-in-the-large, calibration slope, ECE.
- Decisions at a threshold: confusion matrix, precision, recall, specificity.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             log_loss, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def _check(y_true, probabilities):
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Labels and probabilities must be non-empty equal-length vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary 0/1")
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes required; a completed slice report must show unavailable")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1")
    return y.astype(int), p


def threshold_metrics(y, p, threshold):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "flag_rate": float(pred.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def binary_metrics(y_true, probabilities, threshold=.5):
    y, p = _check(y_true, probabilities)
    tm = threshold_metrics(y, p, threshold)
    return {
        "n": len(y), "n_positive": int(y.sum()), "prevalence": float(y.mean()), "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted": float(p.mean()),
        "precision": tm["precision"], "recall": tm["recall"], "specificity": tm["specificity"],
        "confusion_matrix": tm["confusion_matrix"],
    }


def wilson_interval(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return None, None
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def calibration_table(y_true, probabilities, n_bins=10):
    """Equal-count (quantile) bins: observed default rate vs mean prediction.

    Wilson intervals describe binomial noise of the observed rate within a bin
    only; they ignore uncertainty in the bin edges and in the model.
    """
    y, p = _check(y_true, probabilities)
    order = np.argsort(p, kind="mergesort")
    rows = []
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        k, n = int(y[chunk].sum()), len(chunk)
        lo, hi = wilson_interval(k, n)
        rows.append({"n": n, "mean_predicted": float(p[chunk].mean()), "observed_rate": k / n,
                     "observed_ci_low": lo, "observed_ci_high": hi,
                     "p_min": float(p[chunk].min()), "p_max": float(p[chunk].max())})
    return rows


def expected_calibration_error(y_true, probabilities, n_bins=10):
    """Count-weighted mean |observed - predicted| over quantile bins.

    A binning-dependent diagnostic, NOT a proper scoring rule: a model can have
    ECE = 0 while ranking customers no better than chance (predict prevalence).
    """
    rows = calibration_table(y_true, probabilities, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["observed_rate"] - r["mean_predicted"]) for r in rows) / n)


def calibration_slope_intercept(y_true, probabilities):
    """Logistic recalibration y ~ alpha + beta * logit(p).

    beta < 1 suggests over-confident (too extreme) predictions; beta > 1
    under-confident. Intercept here is from the joint fit, not
    calibration-in-the-large; we report mean(y) - mean(p) separately.
    """
    y, p = _check(y_true, probabilities)
    z = logit(np.clip(p, 1e-12, 1 - 1e-12)).reshape(-1, 1)
    fit = LogisticRegression(C=np.inf, max_iter=1000).fit(z, y)
    return {"slope": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0]),
            "calibration_in_the_large": float(y.mean() - p.mean())}


def threshold_grid(y_true, probabilities, thresholds):
    y, p = _check(y_true, probabilities)
    return [threshold_metrics(y, p, t) for t in thresholds]


def curve_points(y_true, probabilities, max_points=200):
    """Down-sampled ROC and precision-recall curves for plotting."""
    y, p = _check(y_true, probabilities)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)

    def thin(a, b):
        idx = np.unique(np.linspace(0, len(a) - 1, min(max_points, len(a))).astype(int))
        return [[round(float(a[i]), 5), round(float(b[i]), 5)] for i in idx]
    return {"roc": thin(fpr, tpr), "pr": thin(recall, precision)}


def full_evaluation(y_true, probabilities, threshold, grid):
    return {
        "metrics": binary_metrics(y_true, probabilities, threshold),
        "calibration": {
            "table": calibration_table(y_true, probabilities),
            "ece_quantile_10": expected_calibration_error(y_true, probabilities),
            **calibration_slope_intercept(y_true, probabilities),
        },
        "threshold_grid": threshold_grid(y_true, probabilities, grid),
        "curves": curve_points(y_true, probabilities),
    }


def bayes_cost_threshold(cost_fp: float, cost_fn: float) -> float:
    """Threshold minimising expected cost IF probabilities are calibrated:
    flag when p * c_fn > (1 - p) * c_fp  <=>  p > c_fp / (c_fp + c_fn)."""
    if cost_fp < 0 or cost_fn < 0 or cost_fp + cost_fn == 0:
        raise ValueError("Costs must be non-negative and not both zero")
    return cost_fp / (cost_fp + cost_fn)
