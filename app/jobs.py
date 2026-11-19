"""Bounded background jobs: one worker thread, one job at a time.

- Submitting while another job is queued/running returns 409 (prevents
  accidental concurrent duplicate training).
- A development study whose dataset/policy/code identity matches an existing
  completed study is NOT retrained; the existing study is returned (cache).
- Jobs left 'running' by a crashed/restarted server are marked 'interrupted'
  at start-up; their study is marked failed. Failures keep the error text.
"""
from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from app.config import load_policy
from app.registry import Registry, now
from app.study import StudyError, current_identity, new_study, run_development_study, run_final_evaluation


class JobConflict(Exception):
    pass


class JobRunner:
    def __init__(self, registry: Registry):
        self.reg = registry
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="modelscope-job")
        self.lock = threading.Lock()
        for job in self.reg.all("SELECT * FROM jobs WHERE status IN ('queued','running')"):
            self.reg.execute("UPDATE jobs SET status='interrupted', error=?, finished_utc=? WHERE job_id=?",
                             ("server stopped while job was active", now(), job["job_id"]))
            if job["study_id"]:
                self.reg.execute("UPDATE studies SET status='failed', error='interrupted' "
                                 "WHERE study_id=? AND status='running'", (job["study_id"],))

    def _active(self):
        return self.reg.one("SELECT * FROM jobs WHERE status IN ('queued','running') LIMIT 1")

    def _progress(self, job_id):
        def report(message):
            self.reg.execute("UPDATE jobs SET progress=? WHERE job_id=?", (message, job_id))
        return report

    def submit_development(self, actor: str) -> dict:
        policy = load_policy()
        with self.lock:
            active = self._active()
            if active:
                raise JobConflict(f"Job {active['job_id']} ({active['kind']}) is already {active['status']}")
            ident = current_identity(policy)
            if ident["identity_key"] is None:
                raise StudyError("No dataset available; run scripts/download_data.py first", 409)
            cached = self.reg.one("SELECT study_id FROM studies WHERE identity_key=? AND status IN "
                                  "('development_complete','final_evaluated') ORDER BY created_utc DESC",
                                  (ident["identity_key"],))
            if cached:
                return {"cached": True, "study_id": cached["study_id"],
                        "message": "A completed study with identical dataset, policy and code exists; not retraining."}
            study_id, _ = new_study(self.reg, actor, policy)
            job_id = f"job-{uuid.uuid4().hex[:12]}"
            self.reg.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (job_id, "development_study", study_id, ident["identity_key"], "queued", None, None,
                              actor, now(), None))
        self.pool.submit(self._run, job_id, study_id,
                         lambda progress: run_development_study(self.reg, study_id, policy, progress))
        return {"cached": False, "job_id": job_id, "study_id": study_id}

    def submit_final(self, study_id: str, actor: str, confirm: bool) -> dict:
        policy = load_policy()
        with self.lock:
            active = self._active()
            if active:
                raise JobConflict(f"Job {active['job_id']} ({active['kind']}) is already {active['status']}")
            job_id = f"job-{uuid.uuid4().hex[:12]}"
            self.reg.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (job_id, "final_evaluation", study_id, None, "queued", None, None, actor, now(), None))
        self.pool.submit(self._run, job_id, None,
                         lambda progress: run_final_evaluation(self.reg, study_id, actor, confirm, policy, progress))
        return {"job_id": job_id, "study_id": study_id}

    def _run(self, job_id, study_id, fn):
        self.reg.execute("UPDATE jobs SET status='running' WHERE job_id=?", (job_id,))
        try:
            fn(self._progress(job_id))
            self.reg.execute("UPDATE jobs SET status='succeeded', finished_utc=? WHERE job_id=?", (now(), job_id))
        except Exception as exc:  # recorded and surfaced to the UI, never swallowed
            detail = f"{type(exc).__name__}: {exc}"
            self.reg.execute("UPDATE jobs SET status='failed', error=?, finished_utc=? WHERE job_id=?",
                             (detail + "\n" + traceback.format_exc(limit=5), now(), job_id))
            if study_id:
                self.reg.execute("UPDATE studies SET status='failed', error=? WHERE study_id=?", (detail, study_id))

    def wait_idle(self, timeout=None):
        """Test helper: block until submitted work has finished."""
        self.pool.submit(lambda: None).result(timeout=timeout)
