"""Dataset schema, loading and data-quality validation for UCI dataset 350.

Source facts (UCI API metadata and `variable_info`, accessed 2026-11-18):
- ID column `ID`; target `Y` = "default payment next month",
  documented as "default payment (Yes = 1, No = 0)" -> positive class is 1.
- X1..X23 are documented as LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
  PAY_0, PAY_2..PAY_6, BILL_AMT1..6, PAY_AMT1..6 (in that order).
- Repayment status scale is documented as -1 = pay duly, 1..8 = delay of
  that many months, 9 = nine months and above. Codes -2 and 0 occur in the
  data but are NOT defined by the source; we keep them as distinct codes and
  never invent a meaning for them.
- EDUCATION documented 1..4; codes 0, 5, 6 occur but are undocumented.
- MARRIAGE documented 1..3; code 0 occurs but is undocumented.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "ID"
TARGET = "DEFAULT_NEXT_MONTH"

# Official positional names -> documented semantic names (from UCI metadata
# `variables[].description`). scripts/download_data.py re-checks this mapping
# against the live metadata and refuses to continue if it disagrees.
SOURCE_TO_SEMANTIC = {
    "ID": "ID", "X1": "LIMIT_BAL", "X2": "SEX", "X3": "EDUCATION", "X4": "MARRIAGE",
    "X5": "AGE", "X6": "PAY_0", "X7": "PAY_2", "X8": "PAY_3", "X9": "PAY_4",
    "X10": "PAY_5", "X11": "PAY_6",
    **{f"X{12 + i}": f"BILL_AMT{1 + i}" for i in range(6)},
    **{f"X{18 + i}": f"PAY_AMT{1 + i}" for i in range(6)},
    "Y": TARGET,
}
PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]
DEMOGRAPHIC = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]
EXPECTED_COLUMNS = list(SOURCE_TO_SEMANTIC.values())

# Documented codes vs codes observed but undocumented. "Invalid" means outside
# any plausible code range, which is a hard failure.
DOCUMENTED = {
    "SEX": {1, 2},
    "EDUCATION": {1, 2, 3, 4},
    "MARRIAGE": {1, 2, 3},
    **{c: {-1, 1, 2, 3, 4, 5, 6, 7, 8, 9} for c in PAY_STATUS},
}
UNDOCUMENTED_TOLERATED = {
    "EDUCATION": {0, 5, 6},
    "MARRIAGE": {0},
    **{c: {-2, 0} for c in PAY_STATUS},
}
EDUCATION_LABELS = {1: "graduate_school", 2: "university", 3: "high_school", 4: "others"}
MARRIAGE_LABELS = {1: "married", 2: "single", 3: "others"}
SEX_LABELS = {1: "male", 2: "female"}
AGE_RANGE = (18, 100)


def validate_frame(frame: pd.DataFrame, target: str, id_column: str) -> None:
    """Generic hard checks (kept from the starter; used by the splitter)."""
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names")
    missing = {target, id_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError("Customer IDs must be unique and non-missing")
    if frame[target].isna().any() or not frame[target].isin([0, 1]).all():
        raise ValueError("Target must contain only 0/1 with no missing values")
    if frame[target].nunique() != 2:
        raise ValueError("Dataset must contain both target classes")
    if len(frame.columns) < 3:
        raise ValueError("No predictor columns")


def feature_target(frame: pd.DataFrame, target: str, id_column: str):
    validate_frame(frame, target, id_column)
    # The label and the customer identity are never predictors.
    return frame.drop(columns=[target, id_column]).copy(), frame[target].astype(int).copy()


def to_semantic(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename official X1..X23/Y headers to documented names.

    Accepts files already using semantic names. Mixed or unknown headers are
    left untouched so that validation reports them as unexpected columns.
    """
    source_without_target = set(SOURCE_TO_SEMANTIC) - {"Y"}
    if source_without_target.issubset(frame.columns):  # Y optional (unlabelled batches)
        return frame.rename(columns=SOURCE_TO_SEMANTIC)
    return frame


def read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return to_semantic(frame)


@dataclass
class Check:
    name: str
    status: str  # pass | info | warn | fail
    detail: str
    count: int | None = None
    values: dict = field(default_factory=dict)


def _int_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    values = series.dropna().to_numpy(dtype=float)
    return bool(np.all(np.isfinite(values)) and np.all(np.equal(np.mod(values, 1), 0)))


def validate_dataset(frame: pd.DataFrame, require_target: bool = True) -> dict:
    """Return a structured data-quality report. `ok` is False on any hard failure.

    require_target=False is for unlabelled monitoring batches: the target column
    may then be absent (performance cannot be computed for such a batch).

    Hard failures (training refused): schema mismatch, non-numeric/non-integer
    columns, missing or duplicate IDs, bad target, missing predictor values in
    the source file, out-of-range codes/amounts. Undocumented-but-tolerated codes
    produce warnings with counts, never silent recoding.
    """
    checks: list[Check] = []
    cols = list(frame.columns)
    labelled = TARGET in cols
    expected = EXPECTED_COLUMNS if (require_target or labelled) else [c for c in EXPECTED_COLUMNS if c != TARGET]
    unexpected = sorted(set(cols) - set(expected))
    missing = sorted(set(expected) - set(cols))
    dup_cols = sorted({c for c in cols if cols.count(c) > 1})
    if unexpected or missing or dup_cols:
        checks.append(Check("schema", "fail",
                            f"missing={missing} unexpected={unexpected} duplicated={dup_cols}"))
        return _report(frame, checks)
    checks.append(Check("schema", "pass", f"{len(expected)} expected columns present, none unexpected"))
    checks.append(Check("row_count", "pass" if len(frame) > 0 else "fail", f"{len(frame)} rows", len(frame)))

    bad_types = [c for c in expected if not _int_like(frame[c])]
    checks.append(Check("numeric_integer_types", "fail" if bad_types else "pass",
                        f"non-integer or non-numeric columns: {bad_types}" if bad_types
                        else "all columns numeric and integer-valued"))
    if bad_types:
        return _report(frame, checks)

    n_missing = frame.isna().sum()
    total_missing = int(n_missing.sum())
    checks.append(Check("missing_values", "fail" if total_missing else "pass",
                        "missing cells per column: " + str({k: int(v) for k, v in n_missing.items() if v})
                        if total_missing else "no missing cells", total_missing))

    ids = frame[ID_COLUMN]
    id_bad = int(ids.isna().sum() + ids.duplicated().sum())
    checks.append(Check("unique_customer_ids", "fail" if id_bad else "pass",
                        f"{id_bad} missing/duplicate IDs" if id_bad else f"{ids.nunique()} unique IDs", id_bad))

    if not labelled:
        checks.append(Check("target_binary", "info", "no target column: unlabelled batch; performance unavailable"))
        return _finish_domains(frame, checks)
    y = frame[TARGET]
    bad_y = int((~y.isin([0, 1])).sum())
    classes = sorted(y.dropna().unique().tolist())
    y_status = "fail" if bad_y or len(set(classes) & {0, 1}) < 2 else "pass"
    checks.append(Check("target_binary", y_status,
                        f"positive class = 1 (source: 'default payment (Yes = 1, No = 0)'); "
                        f"invalid values={bad_y}; prevalence={float((y == 1).mean()):.4f}",
                        bad_y, {"prevalence": float((y == 1).mean()), "classes": classes}))
    return _finish_domains(frame, checks)


def _finish_domains(frame: pd.DataFrame, checks: list) -> dict:
    for col, documented in DOCUMENTED.items():
        counts = frame[col].value_counts().to_dict()
        tolerated = UNDOCUMENTED_TOLERATED.get(col, set())
        undoc = {int(k): int(v) for k, v in counts.items() if k not in documented and k in tolerated}
        invalid = {int(k): int(v) for k, v in counts.items() if k not in documented and k not in tolerated}
        if invalid:
            checks.append(Check(f"domain_{col}", "fail", f"values outside documented/tolerated codes: {invalid}",
                                sum(invalid.values()), invalid))
        elif undoc:
            checks.append(Check(f"domain_{col}", "warn",
                                f"undocumented codes present (kept as distinct codes, not reinterpreted): {undoc}",
                                sum(undoc.values()), undoc))
        else:
            checks.append(Check(f"domain_{col}", "pass", "all values documented"))

    age = frame["AGE"]
    bad_age = int(((age < AGE_RANGE[0]) | (age > AGE_RANGE[1])).sum())
    checks.append(Check("domain_AGE", "fail" if bad_age else "pass",
                        f"{bad_age} values outside {AGE_RANGE}; observed {int(age.min())}-{int(age.max())}", bad_age))
    bad_limit = int((frame["LIMIT_BAL"] <= 0).sum())
    checks.append(Check("domain_LIMIT_BAL", "fail" if bad_limit else "pass",
                        f"{bad_limit} non-positive credit limits", bad_limit))
    neg_pay = int((frame[PAY_AMT] < 0).sum().sum())
    checks.append(Check("domain_PAY_AMT", "fail" if neg_pay else "pass",
                        f"{neg_pay} negative payment amounts", neg_pay))
    neg_bill = int((frame[BILL_AMT] < 0).sum().sum())
    checks.append(Check("domain_BILL_AMT_negative", "info",
                        f"{neg_bill} negative bill-statement cells; plausible credit balances, kept as-is",
                        neg_bill))
    dup_rows = int(frame.drop(columns=[ID_COLUMN]).duplicated().sum())
    checks.append(Check("duplicate_feature_rows", "info",
                        f"{dup_rows} rows identical to another row except ID (including target). "
                        "IDs are distinct so they are treated as distinct customers; cannot be "
                        "confirmed from the source.", dup_rows))
    return _report(frame, checks)


def _report(frame: pd.DataFrame, checks: list[Check]) -> dict:
    return {
        "ok": not any(c.status == "fail" for c in checks),
        "rows": int(len(frame)),
        "columns": [str(c) for c in frame.columns],
        "checks": [asdict(c) for c in checks],
        "summary": {s: sum(c.status == s for c in checks) for s in ("pass", "info", "warn", "fail")},
    }


def education_group(values: pd.Series) -> pd.Series:
    return values.map(EDUCATION_LABELS).fillna("undocumented_code")


def marriage_group(values: pd.Series) -> pd.Series:
    return values.map(MARRIAGE_LABELS).fillna("undocumented_code")
