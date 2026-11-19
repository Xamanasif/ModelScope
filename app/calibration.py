"""Probability calibration fitted on the CALIBRATION partition only.

Sigmoid (Platt) calibration: p = 1 / (1 + exp(-(a * s + b))), where s is the
base model's raw decision score (log-odds for both logistic regression and
HistGradientBoosting). a and b are found by minimising log loss on the
calibration partition. Because the calibrated log-odds are an affine function
of the raw score, additive explanations of s can be mapped exactly to the
calibrated log-odds scale by multiplying by a (see app.explain).

This is plain maximum-likelihood Platt scaling without Platt's (1999) target
smoothing; that choice is documented in docs/DECISIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import log_loss


def fit_platt(scores, y) -> tuple[float, float]:
    s = np.asarray(scores, dtype=float)
    t = np.asarray(y, dtype=float)
    if len(np.unique(t)) != 2:
        raise ValueError("Calibration partition needs both classes")

    def nll(theta):
        z = theta[0] * s + theta[1]
        # log(1 + exp(z)) - t*z, numerically stable
        value = np.sum(np.logaddexp(0, z) - t * z)
        p = expit(z)
        grad = np.array([np.sum((p - t) * s), np.sum(p - t)])
        return value, grad

    result = minimize(nll, x0=np.array([1.0, 0.0]), jac=True, method="BFGS")
    if not result.success and not np.isfinite(result.fun):
        raise RuntimeError(f"Platt fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


@dataclass
class CalibratedModel:
    """A fitted base pipeline plus a frozen calibration map.

    method == "none" means a=1, b=0: probabilities equal the base model's own.
    """
    base: object
    method: str
    a: float = 1.0
    b: float = 0.0

    def raw_score(self, X) -> np.ndarray:
        return np.asarray(self.base.decision_function(X), dtype=float)

    def calibrated_logit(self, X) -> np.ndarray:
        return self.a * self.raw_score(X) + self.b

    def predict_proba(self, X) -> np.ndarray:
        return expit(self.calibrated_logit(X))


def calibration_decision(base, X_cal, y_cal, X_tune, y_tune) -> dict:
    """Fit sigmoid on calibration rows; choose none vs sigmoid on tuning rows.

    The decision uses only development partitions. The calibration-partition
    log losses are reported but flagged as in-sample for the calibrator.
    """
    a, b = fit_platt(base.decision_function(X_cal), y_cal)
    sigmoid = CalibratedModel(base, "sigmoid", a, b)
    none = CalibratedModel(base, "none")
    p_none_tune = none.predict_proba(X_tune)
    p_sig_tune = sigmoid.predict_proba(X_tune)
    ll_none = log_loss(y_tune, p_none_tune, labels=[0, 1])
    ll_sig = log_loss(y_tune, p_sig_tune, labels=[0, 1])
    chosen = sigmoid if ll_sig <= ll_none else none
    return {
        "model": chosen,
        "record": {
            "method": chosen.method, "a": chosen.a, "b": chosen.b,
            "sigmoid_params": {"a": a, "b": b},
            "tuning_log_loss": {"none": ll_none, "sigmoid": ll_sig},
            "calibration_partition_log_loss_in_sample": {
                "none": log_loss(y_cal, none.predict_proba(X_cal), labels=[0, 1]),
                "sigmoid": log_loss(y_cal, sigmoid.predict_proba(X_cal), labels=[0, 1]),
                "warning": "In-sample for the calibrator; not an unbiased estimate of calibration quality.",
            },
            "rule": "sigmoid unless it increases tuning-partition log loss",
        },
    }
