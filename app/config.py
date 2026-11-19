"""Paths, the pre-registered validation policy and version identities.

Everything that defines *which experiment* produced an artifact is reduced to a
hash here: the dataset file (SHA-256), the policy JSON (canonical SHA-256) and
the code (git revision when available plus a hash of the app/*.py sources).
A study is only reusable while all three identities still match.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MODELSCOPE_DATA_DIR", ROOT / "data" / "raw"))
ARTIFACT_DIR = Path(os.environ.get("MODELSCOPE_ARTIFACT_DIR", ROOT / "artifacts"))
POLICY_PATH = Path(os.environ.get("MODELSCOPE_POLICY", ROOT / "configs" / "validation_policy.json"))
DB_PATH = Path(os.environ.get("MODELSCOPE_DB", ARTIFACT_DIR / "modelscope.db"))
DATASET_FILE = "uci_credit_default.csv"
MANIFEST_FILE = "source_manifest.json"


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, no whitespace) used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_policy(path: Path | None = None) -> dict:
    with open(path or POLICY_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def policy_hash(policy: dict) -> str:
    return sha256_bytes(canonical_json(policy))


# Modules whose code can change a numeric result. API, job runner, registry,
# review workflow and report rendering are excluded: editing them does not
# change what a model predicts, so it must not invalidate evidence.
RESULT_MODULES = ("bootstrap", "calibration", "config", "data", "drift", "explain", "features",
                  "metrics", "models", "slices", "splitting", "stress", "study")


def code_identity() -> dict:
    """Code version: hash of result-affecting app modules plus git revision.

    Conservative within RESULT_MODULES: even a comment edit there invalidates
    cached studies (line endings are normalised so Windows/Unix checkouts agree).
    """
    h = hashlib.sha256()
    for path in sorted(ROOT / "app" / f"{m}.py" for m in RESULT_MODULES):
        h.update(path.name.encode())
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
    revision = None
    dirty = None
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        if out.returncode == 0:
            revision = out.stdout.strip()
            status = subprocess.run(["git", "status", "--porcelain", "app"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5)
            dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"source_sha256": h.hexdigest(), "git_revision": revision, "git_app_dirty": dirty}
