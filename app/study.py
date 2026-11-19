"""End-to-end study orchestration enforcing the pre-registered protocol.

run_development_study  (never reads final-partition labels or features)
  1. verify dataset hash and data quality; build and verify split manifest
  2. hyperparameter search by CV inside TRAIN; refit on TRAIN
  3. fit sigmoid on CALIBRATION; choose none/sigmoid on TUNING
  4. evaluate calibrated models on TUNING; paired bootstrap
  5. apply the frozen champion rule; write selection.json (frozen)
  6. repeated CV on TRAIN+TUNING, slices, stress, drift, SHAP (all development)
run_final_evaluation  (one-shot audit after selection is frozen)
  - requires explicit confirmation, logs access, verifies artifact hashes,
    evaluates the frozen models once and stores the result as an audit.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from app import review
from app.bootstrap import paired_bootstrap
from app.calibration import CalibratedModel, calibration_decision
from app.config import (ARTIFACT_DIR, DATA_DIR, DATASET_FILE, MANIFEST_FILE, canonical_json, code_identity,
                        load_policy, policy_hash, sha256_bytes, sha256_file)
from app.data import ID_COLUMN, TARGET, read_dataset, validate_dataset
from app.drift import feature_drift, simulated_batches, psi
from app.explain import explain_model, ranking_stability
from app.features import (HGB_PREPROCESSING_VERSION, LR_PREPROCESSING_VERSION, build_model,
                          feature_columns)
from app.metrics import binary_metrics, full_evaluation
from app.models import search_and_fit
from app.registry import IntegrityError, Registry, now
from app.slices import fit_band_edges, slice_report
from app.splitting import split_customers, split_manifest

ROLES = {"baseline": "logistic_regression", "challenger": "hist_gradient_boosting"}
PREPROCESSING = {"logistic_regression": LR_PREPROCESSING_VERSION, "hist_gradient_boosting": HGB_PREPROCESSING_VERSION}


class StudyError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


def current_identity(policy: dict | None = None) -> dict:
    policy = policy or load_policy()
    data_path = DATA_DIR / DATASET_FILE
    dataset_sha = sha256_file(data_path) if data_path.exists() else None
    code = code_identity()
    p_hash = policy_hash(policy)
    key = sha256_bytes(canonical_json([dataset_sha, p_hash, code["source_sha256"]])) if dataset_sha else None
    return {"dataset_sha256": dataset_sha, "policy_sha256": p_hash, "code": code, "identity_key": key}


def stale_reasons(study: dict, identity: dict) -> list[str]:
    reasons = []
    if study["dataset_sha256"] != identity["dataset_sha256"]:
        reasons.append("dataset file changed")
    if study["policy_sha256"] != identity["policy_sha256"]:
        reasons.append("validation policy changed")
    if study["code_source_sha256"] != identity["code"]["source_sha256"]:
        reasons.append("backend code changed")
    return reasons


def load_prepared(policy: dict):
    """Load, verify and split the dataset. Returns (frame, parts, manifest, quality)."""
    path = DATA_DIR / DATASET_FILE
    if not path.exists():
        raise StudyError(f"No dataset at {path}. Run scripts/download_data.py (see README).", 409)
    dataset_sha = sha256_file(path)
    expected = policy["dataset"].get("expected_sha256")
    if expected and dataset_sha != expected:
        raise StudyError(f"Dataset SHA-256 {dataset_sha[:12]}… differs from the pre-registered "
                         f"{expected[:12]}…: a different file is a different study.", 409)
    frame = read_dataset(path)
    quality = validate_dataset(frame)
    if not quality["ok"]:
        raise StudyError("Data-quality validation failed; training refused.", 422)
    sp = policy["split"]
    parts = split_customers(frame, TARGET, ID_COLUMN, seed=sp["seed"], fractions=sp["fractions"])
    manifest = split_manifest(parts, sp["seed"], sp["fractions"], dataset_sha)
    if sp.get("expected_manifest_sha256") and manifest["manifest_sha256"] != sp["expected_manifest_sha256"]:
        raise StudyError("Split manifest hash differs from the pre-registered value; artifacts are incompatible.", 409)
    return frame, parts, manifest, quality


def partition(frame, parts, name):
    return frame.set_index(ID_COLUMN).loc[parts[name]].reset_index()


def _xy(df, cols):
    return df[cols], df[TARGET].astype(int)


def _save_model(directory: Path, role: str, model: CalibratedModel) -> tuple[str, str]:
    path = directory / "models" / f"{role}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path), sha256_file(path)


def load_trusted_model(reg: Registry, study_id: str, role: str) -> CalibratedModel:
    """Load ONLY locally generated artifacts registered for this study, after
    verifying their SHA-256 and that they live under ARTIFACT_DIR. joblib/pickle
    executes code on load, so uploaded model files are never accepted."""
    row = reg.one("SELECT * FROM models WHERE study_id=? AND role=?", (study_id, role))
    if row is None:
        raise StudyError(f"No {role} model registered for {study_id}", 404)
    path = Path(row["artifact_path"]).resolve()
    if ARTIFACT_DIR.resolve() not in path.parents:
        raise IntegrityError("Model artifact outside the trusted artifact directory")
    if sha256_file(path) != row["artifact_sha256"]:
        raise IntegrityError(f"Model artifact {role} changed after registration; refusing to load")
    return joblib.load(path)


def new_study(reg: Registry, actor: str, policy: dict) -> tuple[str, Path]:
    ident = current_identity(policy)
    if ident["dataset_sha256"] is None:
        raise StudyError("No dataset registered; run scripts/download_data.py first.", 409)
    study_id = f"study-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{ident['identity_key'][:8]}"
    directory = ARTIFACT_DIR / "studies" / study_id
    directory.mkdir(parents=True, exist_ok=False)
    reg.execute("INSERT INTO studies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (study_id, ident["identity_key"], ident["dataset_sha256"], ident["policy_sha256"],
                 ident["code"]["source_sha256"], ident["code"]["git_revision"], None, "running", now(),
                 None, None, actor, str(directory)))
    review.create_case(reg, study_id)
    return study_id, directory


def run_development_study(reg: Registry, study_id: str, policy: dict, progress=lambda m: None) -> dict:
    study = reg.one("SELECT * FROM studies WHERE study_id=?", (study_id,))
    directory = Path(study["directory"])
    t_start = time.perf_counter()
    progress("loading and validating data")
    frame, parts, manifest, quality = load_prepared(policy)
    if manifest["dataset_sha256"] != study["dataset_sha256"]:
        raise StudyError("Dataset changed between study creation and execution", 409)
    reg.write_evidence(study_id, directory, "data_quality", quality, "data")
    reg.write_evidence(study_id, directory, "split_manifest", manifest, "split")
    reg.execute("UPDATE studies SET split_manifest_sha256=? WHERE study_id=?", (manifest["manifest_sha256"], study_id))
    source_manifest = None
    if (DATA_DIR / MANIFEST_FILE).exists():
        source_manifest = json.loads((DATA_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))

    include_demo = policy["features"]["demographic_predictors"]
    cols = feature_columns(include_demo)
    seed = policy["models"]["model_seed"]
    train, tune, cal = (partition(frame, parts, k) for k in ("train", "tuning", "calibration"))
    X_tr, y_tr = _xy(train, cols)
    X_tu, y_tu = _xy(tune, cols)
    X_ca, y_ca = _xy(cal, cols)
    threshold = policy["threshold"]["primary"]
    grid = policy["threshold"]["reported_grid"]

    fitted, calibrated, cal_records = {}, {}, {}
    for role, family in ROLES.items():
        fitted[role] = search_and_fit(family, policy["models"][role], X_tr, y_tr, policy["models"]["cv_folds"],
                                      seed, include_demo, progress)
        progress(f"{role}: calibration")
        decision = calibration_decision(fitted[role]["model"], X_ca, y_ca, X_tu, y_tu)
        calibrated[role] = decision["model"]
        cal_records[role] = decision["record"]

    progress("development evaluation on tuning partition")
    evaluation = {}
    probs_tu = {}
    for role, model in calibrated.items():
        p_cal = model.predict_proba(X_tu)
        p_unc = CalibratedModel(model.base, "none").predict_proba(X_tu)
        probs_tu[role] = p_cal
        untuned = build_model(ROLES[role], fitted[role]["untuned_params"], include_demo, seed).fit(X_tr, y_tr)
        evaluation[role] = {
            "calibrated": full_evaluation(y_tu, p_cal, threshold, grid),
            "uncalibrated_metrics": binary_metrics(y_tu, p_unc, threshold),
            "untuned_uncalibrated_metrics": binary_metrics(y_tu, untuned.predict_proba(X_tu)[:, 1], threshold),
        }
    unc = policy["uncertainty"]
    progress("paired bootstrap on tuning partition")
    boot = paired_bootstrap(y_tu, probs_tu["baseline"], probs_tu["challenger"], unc["n_resamples"], unc["seed"],
                            unc["confidence"])

    # ---- frozen champion rule (policy.champion_selection) -----------------
    ll_ci_upper = boot["metrics"]["log_loss"]["difference_ci"][1]
    champion = "challenger" if (ll_ci_upper is not None and ll_ci_upper < 0) else "baseline"
    artifacts = {}
    for role, model in calibrated.items():
        path, digest = _save_model(directory, role, model)
        artifacts[role] = digest
        reg.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (study_id, role, ROLES[role], canonical_json(fitted[role]["selected_params"]).decode(),
                     PREPROCESSING[ROLES[role]], canonical_json(cal_records[role]).decode(), path, digest,
                     fitted[role]["search_seconds"], fitted[role]["refit_seconds"]))
    selection = {
        "frozen_utc": now(),
        "rule": policy["champion_selection"]["rule"],
        "log_loss_difference_ci": boot["metrics"]["log_loss"]["difference_ci"],
        "champion": champion,
        "champion_family": ROLES[champion],
        "threshold": threshold,
        "calibration": {r: {"method": c.method, "a": c.a, "b": c.b} for r, c in calibrated.items()},
        "artifact_sha256": artifacts,
        "policy_sha256": study["policy_sha256"],
        "split_manifest_sha256": manifest["manifest_sha256"],
        "note": "Frozen before any final-partition access. Final results cannot change this record.",
    }
    selection_sha = reg.write_evidence(study_id, directory, "selection", selection, "selection")

    progress("repeated development cross-validation")
    repeated = repeated_development_cv(policy, fitted, pd.concat([X_tr, X_tu]), pd.concat([y_tr, y_tu]), include_demo, seed)

    development = {
        "study_id": study_id,
        "partition_sizes": manifest["counts"],
        "prevalence": {k: float(frame.set_index(ID_COLUMN).loc[v, TARGET].mean()) for k, v in parts.items()},
        "features": cols,
        "demographic_predictors": include_demo,
        "models": {role: {
            "family": ROLES[role], "preprocessing_version": PREPROCESSING[ROLES[role]],
            "untuned_params": fitted[role]["untuned_params"], "selected_params": fitted[role]["selected_params"],
            "search_results": fitted[role]["search_results"], "search_seconds": fitted[role]["search_seconds"],
            "refit_seconds": fitted[role]["refit_seconds"], "n_configurations": fitted[role]["n_configurations"],
            "calibration": cal_records[role]} for role in ROLES},
        "evaluation": evaluation,
        "paired_bootstrap_tuning": boot,
        "repeated_cv": repeated,
        "selection_sha256": selection_sha,
        "champion": champion,
        "source_manifest": source_manifest,
    }
    reg.write_evidence(study_id, directory, "development", development, "development")

    progress("slice analysis (tuning)")
    edges = fit_band_edges(train)
    overall = {r: evaluation[r]["calibrated"]["metrics"]["roc_auc"] for r in ROLES}
    reg.write_evidence(study_id, directory, "slices_tuning",
                       {"partition": "tuning", **slice_report(tune, y_tu, probs_tu, edges, policy, threshold, overall)},
                       "slices")

    progress("stress suite (tuning)")
    from app.stress import run_stress_suite
    reg.write_evidence(study_id, directory, "stress",
                       run_stress_suite(calibrated, tune, y_tu.to_numpy(), cols, train, policy, threshold), "stress")

    progress("drift diagnostics (simulated batches)")
    reg.write_evidence(study_id, directory, "drift", drift_evidence(calibrated, train, tune, y_tu, cols, policy),
                       "drift")

    progress("SHAP explanations")
    reg.write_evidence(study_id, directory, "explanations",
                       explanation_evidence(calibrated, fitted, train, tune, cols, policy, include_demo, seed),
                       "explanations")

    reg.execute("UPDATE studies SET status='development_complete', completed_utc=? WHERE study_id=?",
                (now(), study_id))
    case = reg.one("SELECT case_id FROM review_cases WHERE study_id=?", (study_id,))
    review.system_transition(reg, case["case_id"], "evaluated",
                             {"champion": champion, "elapsed_seconds": round(time.perf_counter() - t_start, 1)},
                             candidate_role=champion)
    progress("complete")
    return {"study_id": study_id, "champion": champion}


def repeated_development_cv(policy, fitted, X, y, include_demo, seed):
    cfg = policy["uncertainty"]["repeated_development_cv"]
    rskf = RepeatedStratifiedKFold(n_splits=cfg["folds"], n_repeats=cfg["repeats"], random_state=cfg["seed"])
    rows = []
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    for k, (tr, va) in enumerate(rskf.split(X, y)):
        row = {"fold": k + 1}
        for role, family in ROLES.items():
            m = build_model(family, fitted[role]["selected_params"], include_demo, seed).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            row[role] = {"roc_auc": float(roc_auc_score(y.iloc[va], p)),
                         "log_loss": float(log_loss(y.iloc[va], p, labels=[0, 1]))}
        row["auc_difference"] = row["challenger"]["roc_auc"] - row["baseline"]["roc_auc"]
        row["log_loss_difference"] = row["challenger"]["log_loss"] - row["baseline"]["log_loss"]
        rows.append(row)

    def summary(values):
        v = np.asarray(values)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}
    return {
        "scheme": f"{cfg['folds']}-fold x {cfg['repeats']} repeats on train+tuning (uncalibrated, selected params)",
        "caveat": "Fold results overlap in training data, so their spread understates variance; descriptive only.",
        "folds": rows,
        "summary": {
            "baseline_auc": summary([r["baseline"]["roc_auc"] for r in rows]),
            "challenger_auc": summary([r["challenger"]["roc_auc"] for r in rows]),
            "auc_difference": summary([r["auc_difference"] for r in rows]),
            "log_loss_difference": summary([r["log_loss_difference"] for r in rows]),
            "folds_challenger_better_log_loss": int(sum(r["log_loss_difference"] < 0 for r in rows)),
        },
    }


def drift_evidence(models, train, tune, y_tu, cols, policy):
    ref_scores = {r: pd.Series(m.predict_proba(train[cols])) for r, m in models.items()}
    sanity = feature_drift(train, tune, cols, policy)
    batches = []
    for b in simulated_batches(tune, y_tu.to_numpy(), policy["stress"]["seed"] + 100):
        fd = feature_drift(train, b["frame"], cols, policy)
        row = {k: b[k] for k in ("batch", "label", "kind", "description")}
        row["n"] = int(len(b["y"]))
        row["max_feature_psi"] = fd[0]
        row["features_review"] = [f["feature"] for f in fd if f["status"] == "review"]
        row["features_watch"] = [f["feature"] for f in fd if f["status"] == "watch"]
        row["models"] = {}
        for r, m in models.items():
            p = m.predict_proba(b["frame"][cols])
            entry = {"score_psi": psi(ref_scores[r], pd.Series(p)), "mean_predicted": float(p.mean())}
            # Labelled view (available here only because the batch is simulated from labelled data).
            met = binary_metrics(b["y"], p, policy["threshold"]["primary"])
            entry.update(observed_rate=met["prevalence"], roc_auc=met["roc_auc"], log_loss=met["log_loss"])
            row["models"][r] = entry
        batches.append(row)
    return {
        "label": "SIMULATED",
        "reference": "train partition",
        "sanity_train_vs_tuning": sanity,
        "batches": batches,
        "thresholds": {k: policy["drift"][k] for k in ("psi_watch", "psi_review_trigger")},
        "interpretation": ("PSI detects covariate shift without labels. Prevalence shift is confirmed only "
                           "with labels. Concept drift (S6) leaves PSI near zero and is visible only in "
                           "labelled performance. None of these batches is a live monitoring feed."),
    }


def explanation_evidence(calibrated, fitted, train, tune, cols, policy, include_demo, seed):
    cfg = policy["explanations"]
    rng = np.random.default_rng(cfg["seed"])
    ex_idx = np.sort(rng.choice(len(tune), size=min(cfg["n_explain"], len(tune)), replace=False))
    bg_idx = np.sort(rng.choice(len(train), size=min(cfg["n_background"], len(train)), replace=False))
    X_ex = tune.iloc[ex_idx][cols]
    X_bg = train.iloc[bg_idx][cols]
    out = {"partition": cfg["explain_partition"], "n_explain": int(len(ex_idx)),
           "explained_ids": tune.iloc[ex_idx][ID_COLUMN].astype(int).tolist(),
           "background": {"partition": "train", "n": int(len(bg_idx)),
                          "used_by": "both models (LinearExplainer masker; interventional TreeExplainer data)"},
           "features": cols, "models": {}, "stability": {},
           "limitations": [
               "SHAP attributes the model's output, not real-world causes.",
               "Correlated features (BILL_AMT1..6, PAY_0..PAY_6) can share or swap attribution.",
               "Values depend on the background sample (200 train rows); a different background gives different values.",
               "Explanations are not a fairness certificate.",
               "Additive in calibrated log-odds, not in probability."]}
    for role, model in calibrated.items():
        res = explain_model(model, ROLES[role], X_ex, X_bg, cols, cfg["additivity_tolerance"])
        phi = res.pop("local_phi")
        res["local"] = {"phi": np.round(phi, 6).tolist(),
                        "calibrated_probability": model.predict_proba(X_ex).tolist(),
                        "raw_score": model.raw_score(X_ex).tolist()}
        out["models"][role] = res
        out["stability"][role] = ranking_stability(ROLES[role], fitted[role]["selected_params"], include_demo,
                                                   seed, train[cols], train[TARGET].astype(int), X_ex, X_bg, cols,
                                                   res["global"], cfg["stability_refits"], cfg["additivity_tolerance"])
    out["feature_values"] = X_ex.to_numpy().tolist()
    return out


def run_final_evaluation(reg: Registry, study_id: str, actor: str, confirm: bool, policy: dict,
                         progress=lambda m: None) -> dict:
    study = reg.one("SELECT * FROM studies WHERE study_id=?", (study_id,))
    if study is None:
        raise StudyError("Unknown study", 404)
    selection_row = reg.one("SELECT sha256 FROM evidence WHERE study_id=? AND name='selection'", (study_id,))
    if not confirm:
        reg.log_holdout(study_id, actor or "unknown", "final_evaluation", None, "refused: not confirmed")
        raise StudyError("Final-holdout access requires explicit confirmation", 400)
    if reg.one("SELECT 1 AS x FROM evidence WHERE study_id=? AND name='final'", (study_id,)):
        reg.log_holdout(study_id, actor, "final_evaluation", selection_row["sha256"], "refused: already evaluated")
        raise StudyError("Final holdout already evaluated for this study (one-shot audit)", 409)
    if study["status"] != "development_complete" or selection_row is None:
        reg.log_holdout(study_id, actor, "final_evaluation", None, "refused: selection not frozen")
        raise StudyError("Final evaluation requires a completed development study with a frozen selection", 409)
    reasons = stale_reasons(study, current_identity(policy))
    if reasons:
        reg.log_holdout(study_id, actor, "final_evaluation", selection_row["sha256"], "refused: stale study")
        raise StudyError(f"Study is stale ({'; '.join(reasons)}); start a new study instead", 409)
    selection = reg.read_evidence(study_id, "selection")  # verifies hash
    models = {role: load_trusted_model(reg, study_id, role) for role in ROLES}
    for role, model in models.items():
        row = reg.one("SELECT artifact_sha256 FROM models WHERE study_id=? AND role=?", (study_id, role))
        if row["artifact_sha256"] != selection["artifact_sha256"][role]:
            raise IntegrityError("Model artifact differs from the frozen selection record")
    reg.log_holdout(study_id, actor, "final_evaluation", selection_row["sha256"], "granted")
    progress("evaluating frozen models on final holdout")
    frame, parts, manifest, _ = load_prepared(policy)
    if manifest["manifest_sha256"] != study["split_manifest_sha256"]:
        raise IntegrityError("Split manifest differs from the study's manifest")
    development = reg.read_evidence(study_id, "development")
    cols = development["features"]
    final = partition(frame, parts, "final")
    X_f, y_f = _xy(final, cols)
    threshold = selection["threshold"]
    probs = {role: m.predict_proba(X_f) for role, m in models.items()}
    unc = policy["uncertainty"]
    evaluation = {role: full_evaluation(y_f, p, threshold, policy["threshold"]["reported_grid"])
                  for role, p in probs.items()}
    boot = paired_bootstrap(y_f, probs["baseline"], probs["challenger"], unc["n_resamples"], unc["seed"],
                            unc["confidence"])
    result = {
        "study_id": study_id, "accessed_by": actor, "accessed_utc": now(),
        "selection_sha256": selection_row["sha256"], "champion": selection["champion"],
        "n": int(len(y_f)), "prevalence": float(y_f.mean()),
        "evaluation": evaluation, "paired_bootstrap_final": boot,
        "development_vs_final": {role: {
            m: {"tuning": development["evaluation"][role]["calibrated"]["metrics"][m],
                "final": evaluation[role]["metrics"][m]}
            for m in ("roc_auc", "average_precision", "log_loss", "brier")} for role in ROLES},
        "audit_note": ("One-shot audit after selection was frozen. Any weakness found here is recorded as a "
                       "finding; changing the models in response requires a new study with fresh unseen data "
                       "or must be labelled exploratory."),
    }
    reg.write_evidence(study_id, Path(study["directory"]), "final", result, "final")
    progress("slices on final holdout")
    train = partition(frame, parts, "train")
    overall = {r: evaluation[r]["metrics"]["roc_auc"] for r in ROLES}
    reg.write_evidence(study_id, Path(study["directory"]), "slices_final",
                       {"partition": "final", **slice_report(final, y_f, probs, fit_band_edges(train), policy,
                                                             threshold, overall)}, "slices")
    reg.execute("UPDATE studies SET status='final_evaluated' WHERE study_id=?", (study_id,))
    case = reg.one("SELECT case_id FROM review_cases WHERE study_id=?", (study_id,))
    review.refresh_evidence(reg, case["case_id"], "final-holdout audit added")
    progress("complete")
    return {"study_id": study_id, "champion": selection["champion"]}
