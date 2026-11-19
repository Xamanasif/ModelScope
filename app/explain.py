"""SHAP explanations on a bounded development sample.

Algorithms (tied to installed shap; see docs/DECISIONS.md D-009):
- Logistic regression: shap.LinearExplainer with an Independent masker built
  from `n_background` TRAIN rows (interventional; for a linear model in the
  transformed space phi_j = w_j * (z_j - mean_background(z_j))).
- HistGradientBoosting: shap.TreeExplainer, feature_perturbation=
  "interventional", with the SAME train background, so both models'
  explanations answer the same question ("versus a typical training
  customer"). With shap 0.51.0 + scikit-learn 1.9.1 we observed that this
  explainer violates additivity on NON-integer continuous inputs (errors of
  several log-odds), while it is additive to ~1e-7 on integer-valued inputs
  such as UCI 350 (tests/test_modelling.py::test_hgb_interventional_condition).
  We therefore never trust it blindly: additivity is measured on every
  explained row and a violation is reported (additivity_ok = False) and
  flagged by the review checks rather than hidden.

Output space: SHAP values explain the base model's RAW score (log-odds). With
sigmoid calibration (calibrated logit = a * raw + b) they are mapped exactly to
the calibrated log-odds scale by multiplying by a; the base value becomes
a * E[raw] + b. They are NOT additive in probability space.

One-hot columns are summed back to their original column (additivity makes
this exact). SHAP is not causal and not a fairness certificate; correlated
features (e.g. BILL_AMT1..6) can share or swap credit arbitrarily.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from app.features import build_model, transformed_feature_map


def _raw_shap(base_pipeline, family, X_explain, X_background):
    pre = base_pipeline.named_steps["preprocess"]
    est = base_pipeline.named_steps["model"]
    Ze = pre.transform(X_explain)
    if family == "logistic_regression":
        Zb = pre.transform(X_background)
        explainer = shap.LinearExplainer(est, shap.maskers.Independent(Zb, max_samples=len(Zb)))
        phi = np.asarray(explainer.shap_values(Ze))
        base_value = float(np.ravel(explainer.expected_value)[0])
        algorithm = "LinearExplainer (Independent masker, interventional)"
    elif family == "hist_gradient_boosting":
        Zb = pre.transform(X_background)
        explainer = shap.TreeExplainer(est, data=Zb, feature_perturbation="interventional")
        # shap's own check raises; we measure and report additivity ourselves.
        phi = np.asarray(explainer.shap_values(Ze, check_additivity=False))
        base_value = float(np.ravel(explainer.expected_value)[0])
        algorithm = "TreeExplainer (interventional, train background)"
    else:
        raise ValueError(family)
    return phi, base_value, algorithm


def aggregate(phi, origins, columns):
    out = np.zeros((phi.shape[0], len(columns)))
    index = {c: i for i, c in enumerate(columns)}
    for j, origin in enumerate(origins):
        out[:, index[origin]] += phi[:, j]
    return out


def explain_model(cal_model, family, X_explain: pd.DataFrame, X_background: pd.DataFrame,
                  columns: list[str], tolerance: float) -> dict:
    base = cal_model.base
    phi_t, base_value, algorithm = _raw_shap(base, family, X_explain, X_background)
    _, origins = transformed_feature_map(base, columns)
    phi = aggregate(phi_t, origins, columns)
    raw = cal_model.raw_score(X_explain)
    additivity_error = float(np.max(np.abs(phi.sum(axis=1) + base_value - raw)))
    a, b = cal_model.a, cal_model.b
    phi_cal = a * phi
    base_cal = a * base_value + b
    cal_error = float(np.max(np.abs(phi_cal.sum(axis=1) + base_cal - cal_model.calibrated_logit(X_explain))))
    mean_abs = np.abs(phi_cal).mean(axis=0)
    order = np.argsort(-mean_abs)
    direction = []
    for j, col in enumerate(columns):
        x = X_explain[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & (np.std(x) > 0)
        direction.append(float(spearmanr(x[ok], phi_cal[ok, j]).statistic) if ok.sum() > 2 and np.std(phi_cal[ok, j]) > 0 else None)
    return {
        "algorithm": algorithm,
        "output_space": "calibrated log-odds (raw-score SHAP multiplied by calibration slope a)",
        "calibration_map": {"a": a, "b": b, "method": cal_model.method},
        "base_value_raw": base_value,
        "base_value_calibrated_logit": base_cal,
        "additivity_max_abs_error_raw": additivity_error,
        "additivity_max_abs_error_calibrated": cal_error,
        "additivity_tolerance": tolerance,
        "additivity_ok": bool(additivity_error <= tolerance and cal_error <= tolerance),
        "global": [{"feature": columns[j], "mean_abs_shap": float(mean_abs[j]),
                    "value_shap_spearman": direction[j], "rank": int(r + 1)}
                   for r, j in enumerate(order)],
        "local_phi": phi_cal,  # removed before JSON serialisation by the caller
    }


def ranking_stability(family, params, include_demographics, seed, X_train, y_train, X_explain,
                      X_background, columns, reference_global, n_refits, tolerance):
    """Refit on bootstrap resamples of TRAIN and compare global SHAP rankings.

    Uses the uncalibrated raw-score space (calibration multiplies all values by
    the same a > 0 and cannot change a ranking).
    """
    from app.calibration import CalibratedModel
    rng = np.random.default_rng(seed)
    ref_rank = {g["feature"]: g["rank"] for g in reference_global}
    ref_top5 = {g["feature"] for g in reference_global[:5]}
    rows = []
    for i in range(n_refits):
        idx = rng.integers(0, len(X_train), len(X_train))
        model = build_model(family, params, include_demographics, seed)
        model.fit(X_train.iloc[idx], y_train.iloc[idx])
        res = explain_model(CalibratedModel(model, "none"), family, X_explain, X_background, columns, tolerance)
        rank = {g["feature"]: g["rank"] for g in res["global"]}
        rho = spearmanr([ref_rank[c] for c in columns], [rank[c] for c in columns]).statistic
        rows.append({"refit": i + 1, "spearman_rank_correlation": float(rho),
                     "top5_overlap": len(ref_top5 & {g["feature"] for g in res["global"][:5]}),
                     "top5": [g["feature"] for g in res["global"][:5]],
                     "additivity_ok": res["additivity_ok"]})
    return {"method": f"{n_refits} refits on bootstrap resamples of the train partition, selected hyperparameters",
            "refits": rows}
