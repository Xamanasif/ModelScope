"""Feature policy and preprocessing pipelines.

Every statistic a preprocessing step learns (imputation medians, scaling
means/standard deviations, one-hot categories) is learned inside an sklearn
Pipeline, so `pipeline.fit(X_train, y_train)` is the ONLY place those
statistics are estimated. Calibration, tuning and final rows only ever pass
through `transform`/`predict`.

Preprocessing versions are part of a model's identity; bump them when the
transformation logic changes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from app.data import BILL_AMT, DEMOGRAPHIC, ID_COLUMN, PAY_AMT, PAY_STATUS, TARGET

LR_PREPROCESSING_VERSION = "lr-pre-v1: median impute; signed log1p + standardise amounts; PAY_* codes clipped to [-2,3] then one-hot"
HGB_PREPROCESSING_VERSION = "hgb-pre-v1: raw numeric passthrough (NaN handled natively); demographic codes one-hot when enabled"
PAY_CODES = [-2, -1, 0, 1, 2, 3]  # 3 means "3 or more months' delay code"
CATEGORICAL_DEMOGRAPHICS = ["SEX", "EDUCATION", "MARRIAGE"]


def feature_columns(include_demographics: bool) -> list[str]:
    cols = ["LIMIT_BAL", *PAY_STATUS, *BILL_AMT, *PAY_AMT]
    if include_demographics:
        cols = cols + DEMOGRAPHIC
    forbidden = {ID_COLUMN, TARGET}
    if forbidden & set(cols):  # defensive; would indicate a coding error
        raise RuntimeError("Identity/target in feature list")
    return cols


# Module-level functions so fitted pipelines can be serialised with joblib.
def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def clip_pay_codes(x):
    return np.clip(np.rint(np.asarray(x, dtype=float)), -2, 3)


def group_demographic_codes(x):
    """EDUCATION {0,5,6} -> 0 ('undocumented_code'); other columns unchanged.
    Applied column-wise to [SEX, EDUCATION, MARRIAGE] in that order."""
    x = np.array(x, dtype=float, copy=True)
    edu = x[:, 1]
    edu[np.isin(edu, [0, 5, 6])] = 0
    return x


def _demographic_onehot() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("group", FunctionTransformer(group_demographic_codes, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


def build_logistic(C: float, include_demographics: bool, seed: int) -> Pipeline:
    amounts = ["LIMIT_BAL", *BILL_AMT, *PAY_AMT]
    transformers = [
        ("amounts", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("signed_log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), amounts),
        ("pay_status", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("clip", FunctionTransformer(clip_pay_codes, feature_names_out="one-to-one")),
            ("onehot", OneHotEncoder(categories=[PAY_CODES] * len(PAY_STATUS),
                                     handle_unknown="ignore", sparse_output=False)),
        ]), PAY_STATUS),
    ]
    if include_demographics:
        transformers += [
            ("age", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["AGE"]),
            ("demographics", _demographic_onehot(), CATEGORICAL_DEMOGRAPHICS),
        ]
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, verbose_feature_names_out=False)),
        ("model", LogisticRegression(C=C, max_iter=5000, random_state=seed)),
    ])


def build_hgb(params: dict, include_demographics: bool, seed: int) -> Pipeline:
    numeric = ["LIMIT_BAL", *PAY_STATUS, *BILL_AMT, *PAY_AMT] + (["AGE"] if include_demographics else [])
    transformers = [("numeric", "passthrough", numeric)]
    if include_demographics:
        transformers.append(("demographics", _demographic_onehot(), CATEGORICAL_DEMOGRAPHICS))
    return Pipeline([
        ("preprocess", ColumnTransformer(transformers, verbose_feature_names_out=False)),
        ("model", HistGradientBoostingClassifier(early_stopping=False, random_state=seed, **params)),
    ])


def build_model(family: str, params: dict, include_demographics: bool, seed: int) -> Pipeline:
    if family == "logistic_regression":
        return build_logistic(params["C"], include_demographics, seed)
    if family == "hist_gradient_boosting":
        return build_hgb(params, include_demographics, seed)
    raise ValueError(f"Unknown model family {family}")


def transformed_feature_map(pipeline: Pipeline, input_columns: list[str]) -> tuple[list[str], list[str]]:
    """Return (transformed feature names, original column each one came from).

    Used to aggregate one-hot SHAP values back to the original column, which
    is valid because SHAP values are additive.
    """
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    by_length = sorted(input_columns, key=len, reverse=True)
    origins = []
    for name in names:
        match = next((c for c in by_length if name == c or name.startswith(c + "_")), None)
        if match is None:
            raise RuntimeError(f"Cannot map transformed feature {name}")
        origins.append(match)
    return names, origins


def as_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.loc[:, columns]
