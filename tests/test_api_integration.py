"""End-to-end API workflow on a SYNTHETIC fixture (no network, temp dirs).

Covers: no training on page load, job single-flight, identity cache, failed
inputs, one-shot final holdout with access log, review controls over HTTP,
stale-artifact rejection and report download. Tests in this module share one
study and therefore run in file order.
"""
import json
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import DATA_DIR, DATASET_FILE, POLICY_PATH
from tests.fixtures import synthetic_uci_frame

STATE = {}


@pytest.fixture(scope="module")
def client():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    synthetic_uci_frame(1600, seed=0).to_csv(DATA_DIR / DATASET_FILE, index=False)
    from app.main import app, jobs
    with TestClient(app) as c:
        c.jobs = jobs
        yield c


def wait(client, job_id, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError("job timed out")


def test_invalid_requests_rejected(client):
    assert client.post("/api/jobs/development-study", json={"actor": "dev"}).status_code == 422
    assert client.post("/api/jobs/development-study", json={"actor": "dev", "confirm": False}).status_code == 400
    assert client.post("/api/jobs/development-study", json={"actor": "", "confirm": True}).status_code == 422
    assert client.get("/api/studies/nope").status_code == 404
    assert client.get("/api/jobs/nope").status_code == 404


def test_upload_validation_reports_before_success(client):
    good = synthetic_uci_frame(200, seed=9).to_csv(index=False).encode()
    r = client.post("/api/data/validate-upload", content=good, headers={"content-type": "text/csv"}).json()
    assert r["validated"] and r["accepted"] and r["stored"] is False
    bad = synthetic_uci_frame(200, seed=9).assign(Y=2).to_csv(index=False).encode()
    r = client.post("/api/data/validate-upload", content=bad).json()
    assert r["validated"] and not r["accepted"]
    assert client.post("/api/data/validate-upload", content=b"").status_code == 400
    assert client.post("/api/data/validate-upload", content=b"\x00\xff\xfe garbage").status_code in (200, 422)


def test_development_job_single_flight_and_cache(client):
    r = client.post("/api/jobs/development-study", json={"actor": "dev", "confirm": True})
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["cached"] is False
    second = client.post("/api/jobs/development-study", json={"actor": "dev", "confirm": True})
    assert second.status_code == 409  # concurrent duplicate prevented
    job = wait(client, body["job_id"])
    assert job["status"] == "succeeded", job["error"]
    STATE["study"] = body["study_id"]
    again = client.post("/api/jobs/development-study", json={"actor": "dev", "confirm": True}).json()
    assert again["cached"] is True and again["study_id"] == STATE["study"]


def test_development_evidence_is_complete_and_final_untouched(client):
    sid = STATE["study"]
    s = client.get(f"/api/studies/{sid}").json()
    names = {e["name"] for e in s["evidence"]}
    assert {"development", "selection", "slices_tuning", "stress", "drift", "explanations"} <= names
    assert "final" not in names and s["holdout_access_log"] == []
    dev = client.get(f"/api/studies/{sid}/evidence/development").json()
    assert set(dev["models"]) == {"baseline", "challenger"}
    assert dev["partition_sizes"] == {"train": 960, "tuning": 240, "calibration": 160, "final": 240}
    ex = client.get(f"/api/studies/{sid}/evidence/explanations").json()
    assert all(m["additivity_ok"] for m in ex["models"].values())
    cid = ex["explained_ids"][0]
    loc = client.get(f"/api/studies/{sid}/explanations/local/{cid}").json()
    for m in loc["models"].values():
        assert abs(m["calibrated_logit"] - (m["base_value_calibrated_logit"]
                                            + sum(c["shap"] for c in m["contributions"]))) < 1e-4
    assert client.get(f"/api/studies/{sid}/explanations/local/999999").status_code == 404
    assert client.get(f"/api/studies/{sid}/evidence/not_a_thing").status_code == 404
    assert client.get(f"/api/studies/{sid}/threshold-scenario",
                      params={"cost_fp": 1, "cost_fn": 5, "partition_name": "final"}).status_code == 409
    sc = client.get(f"/api/studies/{sid}/threshold-scenario", params={"cost_fp": 1, "cost_fn": 5}).json()
    assert sc["threshold"] == pytest.approx(1 / 6) and "USER-SUPPLIED" in sc["label"]


def test_review_cannot_accept_before_final(client):
    sid = STATE["study"]
    rv = client.get(f"/api/studies/{sid}/review").json()
    assert rv["case"]["state"] == "evaluated"
    req = {"action": "start_review", "actor": "Reviewer", "rationale": "Starting the documented review now",
           "confirm": True, "expected_version": rv["case"]["version"],
           "expected_evidence_sha256": rv["case"]["evidence_sha256"], "request_id": uuid.uuid4().hex}
    assert client.post(f"/api/studies/{sid}/review/transition", json=req).status_code == 200
    dup = client.post(f"/api/studies/{sid}/review/transition", json=req)
    assert dup.status_code == 409
    rv = client.get(f"/api/studies/{sid}/review").json()
    acc = {**req, "action": "accept", "expected_version": rv["case"]["version"], "request_id": uuid.uuid4().hex}
    r = client.post(f"/api/studies/{sid}/review/transition", json=acc)
    assert r.status_code == 409 and "final-holdout" in r.json()["detail"]
    STATE["stale_version"] = rv["case"]["version"]
    STATE["stale_evidence"] = rv["case"]["evidence_sha256"]


def test_final_holdout_requires_confirmation_and_is_one_shot(client):
    sid = STATE["study"]
    bad = client.post(f"/api/studies/{sid}/final-evaluation",
                      json={"actor": "dev", "confirm_final_holdout_access": True, "acknowledgement": "ok"})
    assert bad.status_code == 400
    r = client.post(f"/api/studies/{sid}/final-evaluation",
                    json={"actor": "dev", "confirm_final_holdout_access": False,
                          "acknowledgement": "selection is frozen"})
    assert wait(client, r.json()["job_id"])["status"] == "failed"
    r = client.post(f"/api/studies/{sid}/final-evaluation",
                    json={"actor": "dev", "confirm_final_holdout_access": True,
                          "acknowledgement": "selection is frozen"})
    assert wait(client, r.json()["job_id"])["status"] == "succeeded"
    r = client.post(f"/api/studies/{sid}/final-evaluation",
                    json={"actor": "dev", "confirm_final_holdout_access": True,
                          "acknowledgement": "selection is frozen"})
    job = wait(client, r.json()["job_id"])
    assert job["status"] == "failed" and "already evaluated" in job["error"]
    log = client.get(f"/api/studies/{sid}").json()["holdout_access_log"]
    assert [e["outcome"] for e in log] == ["refused: not confirmed", "granted", "refused: already evaluated"]
    fin = client.get(f"/api/studies/{sid}/evidence/final").json()
    sel = client.get(f"/api/studies/{sid}/evidence/selection").json()
    assert fin["champion"] == sel["champion"]  # final cannot change the frozen selection


def test_stale_decision_after_evidence_change_and_accept(client):
    sid = STATE["study"]
    rv = client.get(f"/api/studies/{sid}/review").json()
    assert rv["case"]["evidence_sha256"] != STATE["stale_evidence"]
    base = {"action": "accept", "actor": "Reviewer", "rationale": "Accepting for research with limitations",
            "confirm": True}
    stale = {**base, "expected_version": STATE["stale_version"], "expected_evidence_sha256": STATE["stale_evidence"],
             "request_id": uuid.uuid4().hex}
    assert client.post(f"/api/studies/{sid}/review/transition", json=stale).status_code == 409
    f = client.post(f"/api/studies/{sid}/findings", json={
        "actor": "Reviewer", "category": "calibration", "severity": "medium", "title": "Synthetic fixture finding",
        "description": "Test finding", "evidence_ref": "development.json", "owner": "dev"})
    assert f.status_code == 201
    assert client.post(f"/api/studies/{sid}/findings", json={
        "actor": "Reviewer", "category": "bogus", "severity": "medium", "title": "x", "description": "x",
        "evidence_ref": "x", "owner": "x"}).status_code == 400
    rv = client.get(f"/api/studies/{sid}/review").json()
    ok = {**base, "expected_version": rv["case"]["version"], "expected_evidence_sha256": rv["case"]["evidence_sha256"],
          "request_id": uuid.uuid4().hex}
    r = client.post(f"/api/studies/{sid}/review/transition", json=ok)
    assert r.status_code == 200 and r.json()["state"] == "accepted_for_research"
    assert client.get(f"/api/studies/{sid}/review").json()["event_chain"]["ok"]


def test_reports_downloadable(client):
    sid = STATE["study"]
    md = client.get(f"/api/studies/{sid}/report.md")
    assert md.status_code == 200 and "Final holdout audit (evaluated once)" in md.text
    assert "not tamper-proof" in md.text
    js = client.get(f"/api/studies/{sid}/results.json").json()
    assert js["evidence"]["final"]["n"] == 240 and "split_manifest" not in js["evidence"]


def test_monitoring_batch_upload(client):
    sid = STATE["study"]
    frame = synthetic_uci_frame(300, seed=4).drop(columns=["Y"])
    r = client.post(f"/api/studies/{sid}/monitoring-batch", content=frame.to_csv(index=False).encode()).json()
    assert r["accepted"] and not r["labelled"] and "performance" not in r["models"]["baseline"]
    lab = synthetic_uci_frame(300, seed=4)
    r = client.post(f"/api/studies/{sid}/monitoring-batch", content=lab.to_csv(index=False).encode()).json()
    assert "performance" in r["models"]["challenger"]
    # A credit limit reported in thousands passes schema validation but is out of range.
    lab["X1"] = lab["X1"] // 1000
    r = client.post(f"/api/studies/{sid}/monitoring-batch", content=lab.to_csv(index=False).encode()).json()
    assert r["accepted"] and r["out_of_training_range"]["LIMIT_BAL"]["below_train_min"] == 300


def test_tampered_model_artifact_is_refused(client):
    sid = STATE["study"]
    s = client.get(f"/api/studies/{sid}").json()
    from app.main import registry
    path = Path(registry.one("SELECT artifact_path FROM models WHERE study_id=? AND role='baseline'", (sid,))
                ["artifact_path"])
    original = path.read_bytes()
    try:
        path.write_bytes(original + b"tamper")
        r = client.get(f"/api/studies/{sid}/threshold-scenario", params={"cost_fp": 1, "cost_fn": 5})
        assert r.status_code == 500 and "Integrity" in r.json()["detail"]
    finally:
        path.write_bytes(original)
    assert s["models"]


def test_policy_change_makes_study_stale_and_new_identity(client):
    sid = STATE["study"]
    original = POLICY_PATH.read_text(encoding="utf-8")
    try:
        p = json.loads(original)
        p["split"]["seed"] = 43
        POLICY_PATH.write_text(json.dumps(p), encoding="utf-8")
        s = client.get(f"/api/studies/{sid}").json()
        assert "validation policy changed" in s["stale_reasons"]
        # A stale study cannot be (re)used: a new development job would create a new study.
        r = client.post("/api/jobs/development-study", json={"actor": "dev", "confirm": True}).json()
        assert r["cached"] is False and r["study_id"] != sid
        client.jobs.wait_idle(timeout=300)
    finally:
        POLICY_PATH.write_text(original, encoding="utf-8")
