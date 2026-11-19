"""Leakage, training-only preprocessing, calibration separation, metrics,
bootstrap, stress, drift and SHAP. SYNTHETIC fixtures only."""
import numpy as np
import pandas as pd
import pytest
import shap
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from app.bootstrap import fast_auc, fast_average_precision, paired_bootstrap
from app.calibration import CalibratedModel, calibration_decision, fit_platt
from app.data import ID_COLUMN, TARGET, to_semantic
from app.drift import psi, simulated_batches
from app.explain import explain_model
from app.features import build_model, feature_columns, signed_log1p
from app.metrics import (bayes_cost_threshold, binary_metrics, calibration_slope_intercept,
                         expected_calibration_error, threshold_metrics, wilson_interval)
from app.models import candidate_configs
from app.stress import SCENARIOS, apply_scenario
from tests.fixtures import synthetic_uci_frame

COLS = feature_columns(False)


@pytest.fixture(scope="module")
def data():
    f = to_semantic(synthetic_uci_frame(2400, seed=1))
    return f.iloc[:1400].reset_index(drop=True), f.iloc[1400:1900].reset_index(drop=True), \
        f.iloc[1900:].reset_index(drop=True)


@pytest.fixture(scope="module")
def models(data):
    tr, _, _ = data
    lr = build_model("logistic_regression", {"C": 1.0}, False, 0).fit(tr[COLS], tr[TARGET])
    hgb = build_model("hist_gradient_boosting", {"max_iter": 40}, False, 0).fit(tr[COLS], tr[TARGET])
    return {"logistic_regression": lr, "hist_gradient_boosting": hgb}


# ---------------- leakage ----------------------------------------------------
def test_feature_list_excludes_target_and_id():
    for demo in (False, True):
        cols = feature_columns(demo)
        assert TARGET not in cols and ID_COLUMN not in cols
    assert not set(feature_columns(False)) & {"SEX", "EDUCATION", "MARRIAGE", "AGE"}


@pytest.mark.parametrize("family,params", [("logistic_regression", {"C": 1.0}),
                                           ("hist_gradient_boosting", {"max_iter": 40})])
def test_target_cannot_leak_into_features(family, params):
    """Labels are shuffled so that features carry NO information. The frame
    passed to fit/predict still contains the target and an exact copy of it
    under a feature-like name; if either entered the model, held-out AUC would
    be ~1. The test fails if target information reaches the predictors."""
    f = to_semantic(synthetic_uci_frame(2000, seed=5))
    rng = np.random.default_rng(0)
    f[TARGET] = rng.permutation(f[TARGET].to_numpy())
    f["LEAK_COPY_OF_TARGET"] = f[TARGET]
    tr, te = f.iloc[:1500], f.iloc[1500:]
    model = build_model(family, params, False, 0).fit(tr.drop(columns=[]), tr[TARGET])
    auc = roc_auc_score(te[TARGET], model.predict_proba(te)[:, 1])
    assert auc < 0.62, f"held-out AUC {auc:.3f} on shuffled labels suggests target leakage"
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    assert not any(TARGET in n or "LEAK" in n or n == ID_COLUMN for n in names)


def test_preprocessing_statistics_come_from_training_rows_only(data):
    tr, tu, _ = data
    model = build_model("logistic_regression", {"C": 1.0}, False, 0).fit(tr[COLS], tr[TARGET])
    amounts = model.named_steps["preprocess"].named_transformers_["amounts"]
    expected_mean = signed_log1p(tr[["LIMIT_BAL"]].to_numpy()).mean()
    assert amounts.named_steps["scale"].mean_[0] == pytest.approx(expected_mean)
    assert amounts.named_steps["impute"].statistics_[0] == pytest.approx(tr["LIMIT_BAL"].median())
    before = amounts.named_steps["scale"].mean_.copy()
    model.predict_proba(tu[COLS])  # predicting must not refit anything
    assert np.array_equal(before, amounts.named_steps["scale"].mean_)


def test_candidate_configs_bounded_and_reproducible():
    spec = {"search": "random", "search_seed": 7, "n_configurations": 4,
            "untuned": {"a": 1}, "search_space": {"a": [1, 2, 3], "b": [0, 1]}}
    a, b = candidate_configs(spec), candidate_configs(spec)
    assert a == b and len(a) <= 5 and a[0] == {"a": 1}


# ---------------- calibration ------------------------------------------------
def test_platt_recovers_known_map():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 * s - 1.0)))).astype(int)
    a, b = fit_platt(s, y)
    assert a == pytest.approx(0.5, abs=0.05) and b == pytest.approx(-1.0, abs=0.06)


def test_calibration_fit_uses_only_calibration_rows(data, models):
    tr, tu, ca = data
    base = models["hist_gradient_boosting"]
    rec = calibration_decision(base, ca[COLS], ca[TARGET], tu[COLS], tu[TARGET])["record"]
    a, b = fit_platt(base.decision_function(ca[COLS]), ca[TARGET])
    assert rec["sigmoid_params"] == {"a": a, "b": b}
    # Changing tuning labels cannot change the fitted map (only the none/sigmoid choice).
    flipped = 1 - tu[TARGET]
    rec2 = calibration_decision(base, ca[COLS], ca[TARGET], tu[COLS], flipped)["record"]
    assert rec2["sigmoid_params"] == rec["sigmoid_params"]
    # Changing calibration labels does change it.
    rec3 = calibration_decision(base, ca[COLS], 1 - ca[TARGET], tu[COLS], tu[TARGET])["record"]
    assert rec3["sigmoid_params"] != rec["sigmoid_params"]
    assert "In-sample" in rec["calibration_partition_log_loss_in_sample"]["warning"]


def test_identity_calibration_equals_base_probabilities(data, models):
    _, tu, _ = data
    for m in models.values():
        np.testing.assert_allclose(CalibratedModel(m, "none").predict_proba(tu[COLS]),
                                   m.predict_proba(tu[COLS])[:, 1], atol=1e-12)


# ---------------- metrics ----------------------------------------------------
def test_metric_edge_cases():
    m = binary_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
    assert m["log_loss"] == pytest.approx(-np.mean(np.log([.9, .8, .8, .9])))
    with pytest.raises(ValueError):
        binary_metrics([0, 1, 2], [.1, .2, .3])
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [.1, .2], threshold=1.5)
    # Threshold above every score: no positives predicted; precision defined as 0, not an error.
    t = threshold_metrics(np.array([0, 1]), np.array([.1, .2]), 0.9)
    assert t["precision"] == 0 and t["recall"] == 0 and t["specificity"] == 1


def test_fast_metrics_match_sklearn_with_ties():
    rng = np.random.default_rng(3)
    for digits in (0, 1, 3):
        y = rng.integers(0, 2, 1500)
        p = np.round(rng.random(1500), digits)
        assert fast_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
        assert fast_average_precision(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_calibration_diagnostics_behave():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 50000)
    y = (rng.random(50000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.01
    cs = calibration_slope_intercept(y, p)
    assert cs["slope"] == pytest.approx(1, abs=0.05)
    over = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))  # over-confident transform
    assert calibration_slope_intercept(y, over)["slope"] < 0.5
    # ECE is not a proper score: constant prevalence prediction is "calibrated" but useless.
    const = np.full_like(p, y.mean())
    assert expected_calibration_error(y, const) < 0.01
    assert roc_auc_score(y, const) == 0.5


def test_wilson_and_bayes_threshold():
    lo, hi = wilson_interval(0, 10)
    assert lo == pytest.approx(0, abs=1e-12) and 0 < hi < 0.35
    assert bayes_cost_threshold(1, 4) == pytest.approx(0.2)
    with pytest.raises(ValueError):
        bayes_cost_threshold(0, 0)


# ---------------- bootstrap ----------------------------------------------------
def test_paired_bootstrap_deterministic_and_paired():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    pa = np.clip(y * 0.3 + rng.random(400) * 0.6, 0, 1)
    r1 = paired_bootstrap(y, pa, pa, n_resamples=50, seed=9)
    r2 = paired_bootstrap(y, pa, pa, n_resamples=50, seed=9)
    assert r1 == r2
    # Identical models: every paired difference is exactly zero.
    assert r1["metrics"]["roc_auc"]["difference_ci"] == [0.0, 0.0]


def test_one_class_resamples_are_counted_not_zeroed():
    y = np.array([0, 0, 0, 1])
    r = paired_bootstrap(y, [.1, .2, .3, .9], [.2, .1, .3, .8], n_resamples=200, seed=0)
    assert r["n_failed"] > 0 and r["n_used"] + r["n_failed"] == 200
    assert all(v is not None for v in r["metrics"]["roc_auc"]["difference_ci"])


# ---------------- stress / drift ---------------------------------------------
def test_stress_reproducible_and_labels_fixed_for_corruption(data):
    _, tu, _ = data
    y = tu[TARGET].to_numpy()
    for sc in SCENARIOS:
        a, ya = apply_scenario(tu, y, sc["id"], 5)
        b, yb = apply_scenario(tu, y, sc["id"], 5)
        pd.testing.assert_frame_equal(a, b)
        assert np.array_equal(ya, yb)
        if sc["kind"] == "input_corruption":
            assert np.array_equal(ya, y), sc["id"]
            assert (a["PAY_AMT1"].dropna() >= 0).all()
    c, _ = apply_scenario(tu, y, "missing_bill_10", 6)
    a, _ = apply_scenario(tu, y, "missing_bill_10", 5)
    assert not a["BILL_AMT1"].isna().equals(c["BILL_AMT1"].isna())
    _, yp = apply_scenario(tu, y, "prevalence_35pct", 1)
    assert yp.mean() == pytest.approx(0.35, abs=0.002)


def test_models_score_missing_values(data, models):
    _, tu, _ = data
    m, _ = apply_scenario(tu, tu[TARGET].to_numpy(), "missing_amounts_30", 1)
    for model in models.values():
        assert np.isfinite(model.predict_proba(m[COLS])).all()


def test_psi_properties_and_concept_drift_invisible_to_psi(data):
    tr, tu, _ = data
    assert psi(tr["LIMIT_BAL"], tr["LIMIT_BAL"]) == pytest.approx(0, abs=1e-12)
    shifted = tr["LIMIT_BAL"] * 3
    assert psi(tr["LIMIT_BAL"], shifted) > 0.25
    batches = simulated_batches(tu, tu[TARGET].to_numpy(), seed=1, size=600)
    assert all(b["label"] == "SIMULATED" for b in batches)
    s1, s5, s6 = batches[0], batches[4], batches[5]
    assert psi(tu["PAY_0"], s5["frame"]["PAY_0"]) > psi(tu["PAY_0"], s1["frame"]["PAY_0"])
    assert s6["kind"] == "concept_drift" and s6["y"].mean() < s1["y"].mean() + 0.05


# ---------------- explanations -----------------------------------------------
@pytest.mark.parametrize("family", ["logistic_regression", "hist_gradient_boosting"])
def test_shap_additivity_in_calibrated_logit_space(data, models, family):
    tr, tu, ca = data
    base = models[family]
    a, b = fit_platt(base.decision_function(ca[COLS]), ca[TARGET])
    res = explain_model(CalibratedModel(base, "sigmoid", a, b), family, tu[COLS].iloc[:60], tr[COLS].iloc[:50],
                        COLS, 1e-6)
    assert res["additivity_ok"], res["additivity_max_abs_error_calibrated"]
    assert {g["feature"] for g in res["global"]} == set(COLS)


def test_demographic_onehot_explanations_aggregate_to_original_columns(data):
    tr, tu, _ = data
    cols = feature_columns(True)
    m = build_model("logistic_regression", {"C": 1.0}, True, 0).fit(tr[cols], tr[TARGET])
    res = explain_model(CalibratedModel(m, "none"), "logistic_regression", tu[cols].iloc[:30], tr[cols].iloc[:30],
                        cols, 1e-6)
    assert res["additivity_ok"] and {g["feature"] for g in res["global"]} == set(cols)


@pytest.mark.skipif(shap.__version__ != "0.51.0" or sklearn.__version__ != "1.9.1",
                    reason="documents a behaviour observed with shap 0.51.0 + scikit-learn 1.9.1 only")
def test_hgb_interventional_condition():
    """Documents decision D-009: interventional TreeSHAP on HGB is additive for
    integer-valued inputs (like UCI 350) but not for non-integer continuous
    inputs with these library versions. If this test starts failing, the
    library behaviour changed and the decision should be revisited."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X[:, 0] + X[:, 1] ** 2 + rng.normal(size=2000) > 1).astype(int)

    def error(XX):
        m = HistGradientBoostingClassifier(max_iter=100, random_state=0, early_stopping=False).fit(XX, y)
        e = shap.TreeExplainer(m, data=XX[:100], feature_perturbation="interventional")
        phi = e.shap_values(XX[:300], check_additivity=False)
        return np.max(np.abs(phi.sum(1) + e.expected_value - m.decision_function(XX[:300])))
    assert error(np.round(X * 100)) < 1e-6   # integer-valued: additive
    assert error(X) > 1e-3                   # continuous non-integer: violated
