"""SYNTHETIC test fixtures. Not credit observations and not model evidence.

The generator mimics the UCI 350 *schema and code ranges* (including the
undocumented codes -2/0, EDUCATION 0/5/6, MARRIAGE 0) so the pipeline can be
exercised offline. Relationships between variables are invented.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def synthetic_uci_frame(n: int = 1600, seed: int = 0, source_headers: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice(np.arange(10_000, 500_001, 10_000), n)
    pay = np.clip(rng.choice([-2, -1, 0, 0, 0, 1, 2, 3], size=(n, 6)), -2, 8)
    bill = np.rint(rng.gamma(2.0, limit[:, None] / 6, size=(n, 6)) - rng.integers(0, 2000, (n, 6)))
    pay_amt = np.rint(rng.gamma(1.0, 3000, size=(n, 6)))
    logit = -1.3 + 1.6 * (pay[:, 0] >= 1) + 0.8 * (pay[:, 1] >= 2) - 0.000003 * limit + rng.normal(0, 0.3, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    frame = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "X1": limit,
        "X2": rng.choice([1, 2], n),
        "X3": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[.01, .35, .45, .15, .02, .01, .01]),
        "X4": rng.choice([0, 1, 2, 3], n, p=[.01, .45, .5, .04]),
        "X5": rng.integers(21, 75, n),
        **{f"X{6 + i}": pay[:, i] for i in range(6)},
        **{f"X{12 + i}": bill[:, i].astype(int) for i in range(6)},
        **{f"X{18 + i}": pay_amt[:, i].astype(int) for i in range(6)},
        "Y": y,
    })
    if not source_headers:
        from app.data import SOURCE_TO_SEMANTIC
        frame = frame.rename(columns=SOURCE_TO_SEMANTIC)
    return frame


def small_policy() -> dict:
    """Real policy with budgets shrunk for speed and the pre-registered official
    dataset/split hashes removed (the synthetic file has different hashes)."""
    policy = json.loads((ROOT / "configs" / "validation_policy.json").read_text(encoding="utf-8"))
    p = copy.deepcopy(policy)
    p["status"] = "TEST_ONLY_small_budget"
    p["dataset"]["expected_sha256"] = None
    p["split"]["expected_manifest_sha256"] = None
    p["models"]["baseline"]["search_space"]["C"] = [0.1, 1.0]
    p["models"]["challenger"]["search_space"] = {"learning_rate": [0.1], "max_iter": [30, 60],
                                                 "max_leaf_nodes": [15], "min_samples_leaf": [20],
                                                 "l2_regularization": [0.0]}
    p["models"]["challenger"]["untuned"]["max_iter"] = 30
    p["models"]["challenger"]["n_configurations"] = 2
    p["models"]["cv_folds"] = 3
    p["uncertainty"]["n_resamples"] = 60
    p["uncertainty"]["repeated_development_cv"] = {"folds": 3, "repeats": 1, "seed": 11}
    p["slices"]["bootstrap_resamples"] = 30
    p["explanations"].update(n_explain=40, n_background=30, stability_refits=2)
    return p
