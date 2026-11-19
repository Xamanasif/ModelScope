"""SQLite registry: studies, evidence index, models, jobs, holdout log and review.

Single source of truth rule: numeric results live in JSON evidence files under
artifacts/studies/<study_id>/. The database stores *references* to them with
their SHA-256, so a changed file is detected as tampered/stale on load rather
than silently disagreeing with a second copy in the database.

Append-only tables (review_events, finding_events, holdout_access_log) have
SQLite triggers that abort UPDATE/DELETE, and review events are hash-chained.
This is NOT tamper-proof: anyone with write access to the .db file can drop
the triggers or rewrite it. It protects against accidental edits and makes
casual tampering detectable via the hash chain.

PostgreSQL: the SQL below is mostly portable; a port would need a driver,
'%s' placeholders and SERIAL/IDENTITY columns. Not implemented.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.config import DB_PATH, canonical_json, sha256_bytes

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  sha256 TEXT PRIMARY KEY, path TEXT NOT NULL, rows INTEGER, manifest_json TEXT,
  registered_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS studies (
  study_id TEXT PRIMARY KEY, identity_key TEXT NOT NULL, dataset_sha256 TEXT NOT NULL,
  policy_sha256 TEXT NOT NULL, code_source_sha256 TEXT NOT NULL, git_revision TEXT,
  split_manifest_sha256 TEXT, status TEXT NOT NULL, created_utc TEXT NOT NULL,
  completed_utc TEXT, error TEXT, created_by TEXT NOT NULL, directory TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS evidence (
  study_id TEXT NOT NULL REFERENCES studies(study_id), name TEXT NOT NULL,
  path TEXT NOT NULL, sha256 TEXT NOT NULL, kind TEXT NOT NULL, created_utc TEXT NOT NULL,
  PRIMARY KEY (study_id, name));
CREATE TABLE IF NOT EXISTS models (
  study_id TEXT NOT NULL REFERENCES studies(study_id), role TEXT NOT NULL,
  family TEXT NOT NULL, params_json TEXT NOT NULL, preprocessing_version TEXT NOT NULL,
  calibration_json TEXT NOT NULL, artifact_path TEXT NOT NULL, artifact_sha256 TEXT NOT NULL,
  search_seconds REAL, refit_seconds REAL, PRIMARY KEY (study_id, role));
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, study_id TEXT, identity_key TEXT,
  status TEXT NOT NULL, progress TEXT, error TEXT, requested_by TEXT NOT NULL,
  created_utc TEXT NOT NULL, finished_utc TEXT);
CREATE TABLE IF NOT EXISTS holdout_access_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT NOT NULL, at_utc TEXT NOT NULL,
  actor TEXT NOT NULL, purpose TEXT NOT NULL, selection_sha256 TEXT, outcome TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_cases (
  case_id TEXT PRIMARY KEY, study_id TEXT NOT NULL UNIQUE REFERENCES studies(study_id),
  candidate_role TEXT, state TEXT NOT NULL, version INTEGER NOT NULL,
  evidence_sha256 TEXT, created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, at_utc TEXT NOT NULL,
  actor TEXT NOT NULL, actor_role TEXT NOT NULL, identity_verified INTEGER NOT NULL DEFAULT 0,
  event_type TEXT NOT NULL, from_state TEXT, to_state TEXT, payload_json TEXT NOT NULL,
  evidence_sha256 TEXT, request_id TEXT UNIQUE, prev_hash TEXT, event_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS findings (
  finding_id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL, category TEXT NOT NULL,
  severity TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL,
  evidence_ref TEXT NOT NULL, owner TEXT NOT NULL, status TEXT NOT NULL,
  reviewer_comment TEXT, created_by TEXT NOT NULL, created_utc TEXT NOT NULL,
  updated_utc TEXT NOT NULL, source_check TEXT);
CREATE TABLE IF NOT EXISTS finding_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, finding_id INTEGER NOT NULL, at_utc TEXT NOT NULL,
  actor TEXT NOT NULL, event_type TEXT NOT NULL, payload_json TEXT NOT NULL);
"""
APPEND_ONLY = ("review_events", "finding_events", "holdout_access_log")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or DB_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self.connect() as db:
            db.executescript(SCHEMA)
            for table in APPEND_ONLY:
                for op in ("UPDATE", "DELETE"):
                    db.execute(f"""CREATE TRIGGER IF NOT EXISTS {table}_no_{op.lower()}
                        BEFORE {op} ON {table} BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only'); END;""")

    @contextmanager
    def connect(self):
        with self._lock:
            db = sqlite3.connect(self.path, timeout=30)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

    # ---- generic helpers -------------------------------------------------
    def one(self, sql, args=()):
        with self.connect() as db:
            row = db.execute(sql, args).fetchone()
            return dict(row) if row else None

    def all(self, sql, args=()):
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, args).fetchall()]

    def execute(self, sql, args=()):
        with self.connect() as db:
            return db.execute(sql, args).lastrowid

    # ---- evidence ----------------------------------------------------------
    def write_evidence(self, study_id: str, directory: Path, name: str, obj, kind: str) -> str:
        data = json.dumps(obj, indent=1, default=_json_default, allow_nan=False).encode("utf-8")
        path = directory / f"{name}.json"
        path.write_bytes(data)
        digest = sha256_bytes(data)
        self.execute("INSERT OR REPLACE INTO evidence VALUES (?,?,?,?,?,?)",
                     (study_id, name, str(path), digest, kind, now()))
        return digest

    def read_evidence(self, study_id: str, name: str):
        row = self.one("SELECT * FROM evidence WHERE study_id=? AND name=?", (study_id, name))
        if not row:
            return None
        data = Path(row["path"]).read_bytes()
        if sha256_bytes(data) != row["sha256"]:
            raise IntegrityError(f"Evidence {name} for {study_id} changed after registration")
        return json.loads(data)

    def evidence_bundle_hash(self, study_id: str) -> str:
        ev = self.all("SELECT name, sha256 FROM evidence WHERE study_id=? ORDER BY name", (study_id,))
        models = self.all("SELECT role, artifact_sha256 FROM models WHERE study_id=? ORDER BY role", (study_id,))
        return sha256_bytes(canonical_json({"evidence": ev, "models": models}))

    # ---- holdout log -------------------------------------------------------
    def log_holdout(self, study_id, actor, purpose, selection_sha, outcome):
        self.execute("INSERT INTO holdout_access_log (study_id, at_utc, actor, purpose, selection_sha256, outcome)"
                     " VALUES (?,?,?,?,?,?)", (study_id, now(), actor, purpose, selection_sha, outcome))


class IntegrityError(RuntimeError):
    pass


def _json_default(obj):
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")
