"""Four disjoint customer partitions; NOT a time-based (out-of-time) experiment.

Rows of UCI 350 are a cross-sectional sample: row order is not chronological,
so no "later cohort" exists to hold out. We therefore split by unique customer
ID with stratification on the target, and say explicitly that this measures
in-distribution generalisation only.
"""
from __future__ import annotations

from sklearn.model_selection import train_test_split

from app.config import canonical_json, sha256_bytes
from app.data import validate_frame

PARTITIONS = ("train", "tuning", "calibration", "final")
DEFAULT_FRACTIONS = {"train": 0.60, "tuning": 0.15, "calibration": 0.10, "final": 0.15}


def split_customers(frame, target, id_column, seed=42, fractions=None):
    fractions = fractions or DEFAULT_FRACTIONS
    if set(fractions) != set(PARTITIONS) or abs(sum(fractions.values()) - 1) > 1e-9:
        raise ValueError(f"Fractions must cover {PARTITIONS} and sum to 1")
    validate_frame(frame, target, id_column)
    ids = frame[id_column].to_numpy()
    labels = frame.set_index(id_column)[target]

    def split(values, size):
        return train_test_split(values, test_size=size, random_state=seed,
                                stratify=labels.loc[values].to_numpy())
    try:
        # Allocate integer counts once: floating ratios can ceil 40.00000000001
        # to 41. Python round uses ties-to-even; the remainder goes to training.
        n = len(ids)
        n_final = round(n * fractions["final"])
        n_calibration = round(n * fractions["calibration"])
        n_tuning = round(n * fractions["tuning"])
        if min(n_final, n_calibration, n_tuning) < 2:
            raise ValueError("Partitions too small")
        development, final = split(ids, n_final)
        train_tuning, calibration = split(development, n_calibration)
        train, tuning = split(train_tuning, n_tuning)
    except ValueError as exc:
        raise ValueError("Insufficient class support for four stratified partitions") from exc
    parts = {"train": train.tolist(), "tuning": tuning.tolist(),
             "calibration": calibration.tolist(), "final": final.tolist()}
    assert_disjoint_complete(parts, ids)
    return parts


def assert_disjoint_complete(parts: dict, ids) -> None:
    names = list(parts)
    for name in names:
        if len(set(parts[name])) != len(parts[name]):
            raise RuntimeError(f"Customer repeated within partition {name}")
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(parts[name]) & set(parts[other]):
                raise RuntimeError("Customer leakage across partitions")
    if set().union(*(set(v) for v in parts.values())) != set(ids.tolist() if hasattr(ids, "tolist") else ids):
        raise RuntimeError("Split lost or added customer IDs")


def split_manifest(parts: dict, seed: int, fractions: dict, dataset_sha256: str) -> dict:
    """Canonical manifest: sorted IDs per partition and a hash over all of it.

    The hash identifies exactly which customers sit in which partition. Any
    change of seed, fractions or source file produces a different hash, which
    makes previously built artifacts incompatible (see app.study).
    """
    body = {
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "fractions": fractions,
        "ids": {k: sorted(int(i) for i in v) for k, v in parts.items()},
    }
    return {**body, "counts": {k: len(v) for k, v in parts.items()},
            "manifest_sha256": sha256_bytes(canonical_json(body))}
