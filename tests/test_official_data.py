"""Checks against the real official file, if the user has downloaded it.
Skipped (not failed) when data/raw/uci_credit_default.csv is absent."""
import json
from pathlib import Path

import pytest

from app.config import sha256_file
from app.data import ID_COLUMN, TARGET, read_dataset, validate_dataset
from app.splitting import split_customers, split_manifest

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "raw" / "uci_credit_default.csv"
pytestmark = [pytest.mark.official_data,
              pytest.mark.skipif(not REAL.exists(), reason="official dataset not downloaded")]


def test_official_file_matches_preregistered_hash_and_split():
    policy = json.loads((ROOT / "configs" / "validation_policy.json").read_text(encoding="utf-8"))
    digest = sha256_file(REAL)
    assert digest == policy["dataset"]["expected_sha256"]
    frame = read_dataset(REAL)
    report = validate_dataset(frame)
    assert report["ok"] and report["rows"] == 30000
    sp = policy["split"]
    parts = split_customers(frame, TARGET, ID_COLUMN, sp["seed"], sp["fractions"])
    m = split_manifest(parts, sp["seed"], sp["fractions"], digest)
    assert m["counts"] == sp["expected_counts"]
    assert m["manifest_sha256"] == sp["expected_manifest_sha256"]
