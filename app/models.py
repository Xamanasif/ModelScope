"""Bounded, reproducible hyperparameter search inside the TRAIN partition only.

Every candidate configuration (including the untuned default) is scored by
mean log loss over stratified K-fold CV on the train partition. Tuning,
calibration and final partitions are never touched here. The best
configuration is then refit on the whole train partition.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from app.features import build_model


def candidate_configs(spec: dict) -> list[dict]:
    space = spec["search_space"]
    keys = sorted(space)
    grid = [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]
    if spec["search"] == "grid":
        chosen = grid
    elif spec["search"] == "random":
        rng = np.random.default_rng(spec["search_seed"])
        n = min(spec["n_configurations"], len(grid))
        chosen = [grid[i] for i in sorted(rng.choice(len(grid), size=n, replace=False))]
    else:
        raise ValueError(f"Unknown search type {spec['search']}")
    untuned = dict(spec["untuned"])
    configs = [untuned] + [c for c in chosen if c != untuned]
    return configs


def cross_validate_config(family, params, X, y, folds, seed, include_demographics):
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    losses, aucs, fit_seconds = [], [], 0.0
    for train_idx, valid_idx in cv.split(X, y):
        model = build_model(family, params, include_demographics, seed)
        start = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds += time.perf_counter() - start
        p = model.predict_proba(X.iloc[valid_idx])[:, 1]
        losses.append(log_loss(y.iloc[valid_idx], p, labels=[0, 1]))
        aucs.append(roc_auc_score(y.iloc[valid_idx], p))
    return {
        "params": params,
        "cv_log_loss_mean": float(np.mean(losses)), "cv_log_loss_sd": float(np.std(losses, ddof=1)),
        "cv_roc_auc_mean": float(np.mean(aucs)), "cv_roc_auc_sd": float(np.std(aucs, ddof=1)),
        "fit_seconds_total": round(fit_seconds, 3),
    }


def search_and_fit(family: str, spec: dict, X_train, y_train, folds: int, seed: int,
                   include_demographics: bool, progress=None) -> dict:
    configs = candidate_configs(spec)
    results = []
    start = time.perf_counter()
    for i, params in enumerate(configs):
        if progress:
            progress(f"{family}: CV config {i + 1}/{len(configs)}")
        row = cross_validate_config(family, params, X_train, y_train, folds, seed, include_demographics)
        row["is_untuned_default"] = (i == 0)
        results.append(row)
    search_seconds = time.perf_counter() - start
    best = min(results, key=lambda r: (r["cv_log_loss_mean"], configs.index(r["params"])))
    model = build_model(family, best["params"], include_demographics, seed)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    refit_seconds = time.perf_counter() - t0
    return {
        "model": model,
        "family": family,
        "selected_params": best["params"],
        "untuned_params": configs[0],
        "search_results": results,
        "search_seconds": round(search_seconds, 2),
        "refit_seconds": round(refit_seconds, 3),
        "n_configurations": len(configs),
    }
