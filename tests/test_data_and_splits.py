"""Data validation, provenance-independent schema checks and split isolation.
All frames are SYNTHETIC fixtures."""
import numpy as np
import pandas as pd
import pytest

from app.data import ID_COLUMN, TARGET, feature_target, to_semantic, validate_dataset
from app.splitting import split_customers, split_manifest
from tests.fixtures import synthetic_uci_frame


def semantic(n=1600, seed=0):
    return to_semantic(synthetic_uci_frame(n, seed))


def status(report, name):
    return next(c for c in report["checks"] if c["name"] == name)["status"]


def test_valid_fixture_passes_with_undocumented_codes_as_warnings():
    r = validate_dataset(semantic())
    assert r["ok"]
    assert status(r, "domain_PAY_0") == "warn"      # -2 and 0 are undocumented, not invalid
    assert status(r, "domain_EDUCATION") == "warn"  # 0/5/6
    assert status(r, "target_binary") == "pass"


def test_unexpected_and_missing_columns_fail():
    f = semantic()
    f["EXTRA"] = 1
    assert not validate_dataset(f)["ok"]
    assert not validate_dataset(semantic().drop(columns=["PAY_AMT6"]))["ok"]


@pytest.mark.parametrize("mutate", [
    lambda f: f.assign(**{TARGET: f[TARGET].replace({1: 2})}),       # bad target code
    lambda f: f.assign(**{ID_COLUMN: np.r_[f[ID_COLUMN].iloc[:-1], 1]}),  # duplicate customer
    lambda f: f.assign(PAY_0=f["PAY_0"].where(f.index != 0, -5)),     # invalid repayment code
    lambda f: f.assign(PAY_AMT1=f["PAY_AMT1"].where(f.index != 0, -10)),  # negative payment
    lambda f: f.assign(AGE=f["AGE"].where(f.index != 0, 7)),          # implausible age
    lambda f: f.assign(LIMIT_BAL=f["LIMIT_BAL"].astype(float).where(f.index != 0, np.nan)),  # missing
    lambda f: f.assign(BILL_AMT1=f["BILL_AMT1"].astype(str)),          # non-numeric
    lambda f: f.assign(LIMIT_BAL=f["LIMIT_BAL"] + 0.5),                # non-integer
])
def test_hard_failures(mutate):
    assert not validate_dataset(mutate(semantic()))["ok"]


def test_single_class_target_fails():
    f = semantic()
    f[TARGET] = 0
    assert not validate_dataset(f)["ok"]


def test_unlabelled_batch_allowed_only_when_requested():
    f = semantic().drop(columns=[TARGET])
    assert not validate_dataset(f)["ok"]
    assert validate_dataset(f, require_target=False)["ok"]


def test_id_and_target_never_predictors():
    X, y = feature_target(semantic(), TARGET, ID_COLUMN)
    assert TARGET not in X.columns and ID_COLUMN not in X.columns
    assert len(y) == len(X)


def test_split_disjoint_complete_reproducible_and_seed_sensitive():
    f = semantic(2000)
    a = split_customers(f, TARGET, ID_COLUMN, seed=42)
    b = split_customers(f, TARGET, ID_COLUMN, seed=42)
    c = split_customers(f, TARGET, ID_COLUMN, seed=43)
    assert a == b
    assert a["final"] != c["final"]
    flat = [i for v in a.values() for i in v]
    assert len(flat) == len(set(flat)) == len(f)
    for name, ids in a.items():
        assert len(ids) == len(set(ids)), f"duplicate within {name}"
    assert {k: len(v) for k, v in a.items()} == {"train": 1200, "tuning": 300, "calibration": 200, "final": 300}


def test_split_stratification_and_non_divisible_size():
    f = semantic(403, seed=3)
    parts = split_customers(f, TARGET, ID_COLUMN)
    assert sum(len(v) for v in parts.values()) == 403
    y = f.set_index(ID_COLUMN)[TARGET]
    overall = y.mean()
    for ids in parts.values():
        assert abs(y.loc[ids].mean() - overall) < 0.06


def test_manifest_hash_changes_with_seed_fraction_or_data():
    f = semantic(1000)
    fr = {"train": .6, "tuning": .15, "calibration": .1, "final": .15}
    base = split_manifest(split_customers(f, TARGET, ID_COLUMN, 42, fr), 42, fr, "abc")["manifest_sha256"]
    assert base == split_manifest(split_customers(f, TARGET, ID_COLUMN, 42, fr), 42, fr, "abc")["manifest_sha256"]
    assert base != split_manifest(split_customers(f, TARGET, ID_COLUMN, 7, fr), 7, fr, "abc")["manifest_sha256"]
    fr2 = {"train": .5, "tuning": .2, "calibration": .15, "final": .15}
    assert base != split_manifest(split_customers(f, TARGET, ID_COLUMN, 42, fr2), 42, fr2, "abc")["manifest_sha256"]
    assert base != split_manifest(split_customers(f, TARGET, ID_COLUMN, 42, fr), 42, fr, "xyz")["manifest_sha256"]


def test_bad_fractions_rejected():
    with pytest.raises(ValueError):
        split_customers(semantic(), TARGET, ID_COLUMN, fractions={"train": .7, "tuning": .15, "calibration": .1, "final": .15})


def test_source_header_mapping():
    raw = synthetic_uci_frame(50)
    assert list(to_semantic(raw).columns)[1:6] == ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    mixed = raw.rename(columns={"X1": "LIMIT"})
    assert not validate_dataset(to_semantic(mixed))["ok"]
