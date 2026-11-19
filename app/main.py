"""ModelScope API (FastAPI). Binds to 127.0.0.1 by default (see run.ps1).

Opening the dashboard never trains anything: training and the final-holdout
audit run only through explicit POST requests that create background jobs.
Uploads are parsed as CSV text only; no uploaded model or code is ever loaded.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import review
from app.config import DATA_DIR, DATASET_FILE, MANIFEST_FILE, ROOT, load_policy, policy_hash
from app.data import ID_COLUMN, TARGET, to_semantic, validate_dataset
from app.drift import feature_drift, psi
from app.jobs import JobConflict, JobRunner
from app.metrics import bayes_cost_threshold, binary_metrics, threshold_metrics
from app.registry import IntegrityError, Registry
from app.report import build_report
from app.study import (ROLES, StudyError, current_identity, load_prepared, load_trusted_model, partition,
                       stale_reasons)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
EVIDENCE_NAMES = ("data_quality", "split_manifest", "development", "selection", "slices_tuning", "stress",
                  "drift", "explanations", "final", "slices_final")

app = FastAPI(title="ModelScope", version="1.0.0",
              description="Credit-risk model validation research prototype (local). Not production software.")
registry = Registry()
jobs = JobRunner(registry)


@app.exception_handler(StudyError)
@app.exception_handler(review.ReviewError)
async def domain_error(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=getattr(exc, "status_code", 400))


@app.exception_handler(JobConflict)
async def job_conflict(_, exc):
    return JSONResponse({"detail": str(exc)}, status_code=409)


@app.exception_handler(IntegrityError)
async def integrity_error(_, exc):
    return JSONResponse({"detail": f"Integrity check failed: {exc}"}, status_code=500)


# ---------------------------------------------------------------- helpers
def _study(study_id: str) -> dict:
    s = registry.one("SELECT * FROM studies WHERE study_id=?", (study_id,))
    if s is None:
        raise HTTPException(404, "Unknown study")
    return s


def _evidence(study_id: str, name: str, required=True):
    data = registry.read_evidence(study_id, name)
    if data is None and required:
        raise HTTPException(404, f"No {name} evidence for this study")
    return data


def _case(study_id: str):
    return registry.one("SELECT * FROM review_cases WHERE study_id=?", (study_id,))


def _study_summary(s: dict, identity: dict) -> dict:
    case = _case(s["study_id"])
    sel = registry.read_evidence(s["study_id"], "selection") if s["status"] != "running" else None
    return {**s, "stale_reasons": stale_reasons(s, identity),
            "review_state": case["state"] if case else None,
            "champion": sel["champion"] if sel else None,
            "final_evaluated": registry.one("SELECT 1 AS x FROM evidence WHERE study_id=? AND name='final'",
                                            (s["study_id"],)) is not None}


# ---------------------------------------------------------------- status
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/api/status")
def status():
    policy = load_policy()
    identity = current_identity(policy)
    studies = registry.all("SELECT * FROM studies ORDER BY created_utc DESC")
    manifest_path = DATA_DIR / MANIFEST_FILE
    return {
        "dataset_present": (DATA_DIR / DATASET_FILE).exists(),
        "source_manifest": json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None,
        "identity": identity,
        "policy_version": policy.get("policy_version"),
        "policy_status": policy.get("status"),
        "studies": [_study_summary(s, identity) for s in studies],
        "active_job": registry.one("SELECT * FROM jobs WHERE status IN ('queued','running') LIMIT 1"),
        "scope": "Local research prototype; no production, regulatory or bank approval is implied.",
    }


@app.get("/api/policy")
def policy_view():
    policy = load_policy()
    return {"policy": policy, "sha256": policy_hash(policy)}


# ---------------------------------------------------------------- data
@app.get("/api/data/quality")
def data_quality():
    policy = load_policy()
    try:
        frame, parts, manifest, quality = load_prepared(policy)
    except StudyError as exc:
        return {"available": False, "reason": str(exc)}
    y = frame.set_index(ID_COLUMN)[TARGET]
    ids = {k: set(v) for k, v in parts.items()}
    names = list(ids)
    overlaps = {f"{a}&{b}": len(ids[a] & ids[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    return {
        "available": True,
        "quality": quality,
        "split": {"counts": manifest["counts"], "manifest_sha256": manifest["manifest_sha256"],
                  "expected_manifest_sha256": policy["split"].get("expected_manifest_sha256"),
                  "seed": manifest["seed"], "fractions": manifest["fractions"],
                  "prevalence": {k: float(y.loc[v].mean()) for k, v in parts.items()},
                  "pairwise_overlap": overlaps,
                  "union_equals_dataset": set().union(*ids.values()) == set(frame[ID_COLUMN])},
        "source_manifest": json.loads((DATA_DIR / MANIFEST_FILE).read_text(encoding="utf-8"))
        if (DATA_DIR / MANIFEST_FILE).exists() else None,
        "code_tables": {
            "PAY_status": "documented: -1 pay duly, 1..8 months delay, 9 nine+ months; -2 and 0 observed but undocumented",
            "EDUCATION": "documented 1 graduate school, 2 university, 3 high school, 4 others; 0/5/6 undocumented",
            "MARRIAGE": "documented 1 married, 2 single, 3 others; 0 undocumented",
            "SEX": "1 male, 2 female (audit slice only by default)",
        },
        "temporal_note": policy["split"]["temporal_note"],
    }


async def _read_csv_body(request: Request) -> pd.DataFrame:
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty upload; send CSV text as the request body")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    try:
        return to_semantic(pd.read_csv(io.BytesIO(body)))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(422, f"Not a readable CSV file: {exc}") from exc


@app.post("/api/data/validate-upload")
async def validate_upload(request: Request):
    """Validate a CSV against the UCI 350 schema. Nothing is stored or trained."""
    frame = await _read_csv_body(request)
    report = validate_dataset(frame)
    return {"validated": True, "accepted": report["ok"], "stored": False, "report": report}


# ---------------------------------------------------------------- jobs
class DevJobRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=80)
    confirm: bool


class FinalJobRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=80)
    confirm_final_holdout_access: bool
    acknowledgement: str = Field(min_length=1, max_length=400)


@app.post("/api/jobs/development-study", status_code=202)
def start_development(req: DevJobRequest):
    if not req.confirm:
        raise HTTPException(400, "Training requires confirm=true")
    return jobs.submit_development(req.actor.strip())


@app.post("/api/studies/{study_id}/final-evaluation", status_code=202)
def start_final(study_id: str, req: FinalJobRequest):
    _study(study_id)
    if "frozen" not in req.acknowledgement.lower():
        raise HTTPException(400, "Acknowledgement must confirm that the selection is frozen")
    return jobs.submit_final(study_id, req.actor.strip(), req.confirm_final_holdout_access)


@app.get("/api/jobs")
def list_jobs():
    return registry.all("SELECT * FROM jobs ORDER BY created_utc DESC LIMIT 50")


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = registry.one("SELECT * FROM jobs WHERE job_id=?", (job_id,))
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


# ---------------------------------------------------------------- studies
@app.get("/api/studies")
def list_studies():
    identity = current_identity()
    return [_study_summary(s, identity) for s in registry.all("SELECT * FROM studies ORDER BY created_utc DESC")]


@app.get("/api/studies/{study_id}")
def get_study(study_id: str):
    s = _study(study_id)
    summary = _study_summary(s, current_identity())
    summary["evidence"] = registry.all("SELECT name, sha256, kind, created_utc FROM evidence WHERE study_id=?",
                                       (study_id,))
    summary["models"] = registry.all("SELECT role, family, params_json, preprocessing_version, calibration_json,"
                                     " artifact_sha256, search_seconds, refit_seconds FROM models WHERE study_id=?",
                                     (study_id,))
    summary["holdout_access_log"] = registry.all("SELECT * FROM holdout_access_log WHERE study_id=? ORDER BY id",
                                                 (study_id,))
    return summary


@app.get("/api/studies/{study_id}/evidence/{name}")
def get_evidence(study_id: str, name: str):
    if name not in EVIDENCE_NAMES:
        raise HTTPException(404, "Unknown evidence type")
    _study(study_id)
    data = _evidence(study_id, name)
    if name == "explanations":  # local matrices are served by the local endpoint
        data = {k: v for k, v in data.items() if k != "feature_values"}
        data["models"] = {r: {k: v for k, v in m.items() if k != "local"} for r, m in data["models"].items()}
    return data


@app.get("/api/studies/{study_id}/explanations/local/{customer_id}")
def local_explanation(study_id: str, customer_id: int):
    ex = _evidence(study_id, "explanations")
    if customer_id not in ex["explained_ids"]:
        raise HTTPException(404, "Customer is not in the bounded development explanation sample")
    i = ex["explained_ids"].index(customer_id)
    out = {"customer_id": customer_id, "partition": ex["partition"], "features": ex["features"],
           "values": ex["feature_values"][i], "models": {}}
    for role, m in ex["models"].items():
        phi = m["local"]["phi"][i]
        out["models"][role] = {
            "base_value_calibrated_logit": m["base_value_calibrated_logit"],
            "contributions": sorted(({"feature": f, "value": v, "shap": s}
                                     for f, v, s in zip(ex["features"], ex["feature_values"][i], phi)),
                                    key=lambda r: -abs(r["shap"])),
            "calibrated_logit": m["base_value_calibrated_logit"] + float(np.sum(phi)),
            "calibrated_probability": m["local"]["calibrated_probability"][i],
            "output_space": m["output_space"]}
    return out


@app.get("/api/studies/{study_id}/threshold-scenario")
def threshold_scenario(study_id: str, cost_fp: float, cost_fn: float, partition_name: str = "tuning"):
    """User-supplied cost scenario. Costs are hypothetical inputs, not observed losses."""
    if partition_name not in ("tuning", "final"):
        raise HTTPException(400, "partition_name must be tuning or final")
    try:
        t = bayes_cost_threshold(cost_fp, cost_fn)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if partition_name == "final" and not registry.read_evidence(study_id, "final"):
        raise HTTPException(409, "Final holdout not yet evaluated")
    policy = load_policy()
    frame, parts, _, _ = load_prepared(policy)
    df = partition(frame, parts, partition_name)
    cols = _evidence(study_id, "development")["features"]
    out = {"label": "USER-SUPPLIED COST SCENARIO (not observed banking losses)", "cost_fp": cost_fp,
           "cost_fn": cost_fn, "threshold": t, "partition": partition_name, "models": {}}
    for role in ROLES:
        p = load_trusted_model(registry, study_id, role).predict_proba(df[cols])
        tm = threshold_metrics(df[TARGET].to_numpy(), p, t)
        cm = tm["confusion_matrix"]
        out["models"][role] = {**tm, "scenario_cost_per_customer": (cm["fp"] * cost_fp + cm["fn"] * cost_fn) / len(p)}
    return out


@app.post("/api/studies/{study_id}/monitoring-batch")
async def monitoring_batch(study_id: str, request: Request):
    """Compare a user-supplied batch (labels optional) with the train reference.
    Result is returned, not stored, so it cannot alter the review evidence bundle."""
    frame = await _read_csv_body(request)
    report = validate_dataset(frame, require_target=False)
    if not report["ok"]:
        return {"validated": True, "accepted": False, "report": report}
    policy = load_policy()
    full, parts, _, _ = load_prepared(policy)
    train = partition(full, parts, "train")
    cols = _evidence(study_id, "development")["features"]
    out = {"validated": True, "accepted": True, "label": "USER-SUPPLIED BATCH (not persisted)",
           "report": report, "n": len(frame), "labelled": TARGET in frame.columns,
           "feature_drift": feature_drift(train, frame, cols, policy), "models": {},
           "out_of_training_range": {
               c: {"below_train_min": int((frame[c] < train[c].min()).sum()),
                   "above_train_max": int((frame[c] > train[c].max()).sum())}
               for c in cols if ((frame[c] < train[c].min()) | (frame[c] > train[c].max())).any()},
           "range_note": "Values outside the train partition's observed range (e.g. a credit limit "
                         "reported in thousands). Schema validation alone does not catch these."}
    for role in ROLES:
        model = load_trusted_model(registry, study_id, role)
        p, ref = model.predict_proba(frame[cols]), model.predict_proba(train[cols])
        entry = {"score_psi": psi(pd.Series(ref), pd.Series(p)), "mean_predicted": float(p.mean())}
        if out["labelled"] and frame[TARGET].nunique() == 2:
            entry["performance"] = binary_metrics(frame[TARGET], p, policy["threshold"]["primary"])
        elif out["labelled"]:
            entry["performance_unavailable"] = "batch labels contain a single class"
        out["models"][role] = entry
    return out


# ---------------------------------------------------------------- review
class TransitionRequest(BaseModel):
    action: str
    actor: str = Field(min_length=1, max_length=80)
    rationale: str = Field(max_length=4000)
    confirm: bool
    expected_version: int
    expected_evidence_sha256: str
    request_id: str = Field(min_length=8, max_length=80)


class FindingRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=80)
    category: str
    severity: str
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    evidence_ref: str = Field(min_length=1, max_length=300)
    owner: str = Field(min_length=1, max_length=80)
    source_check: str | None = None


class FindingUpdate(BaseModel):
    actor: str = Field(min_length=1, max_length=80)
    status: str | None = None
    comment: str | None = Field(default=None, max_length=4000)
    remediation: str | None = Field(default=None, max_length=4000)


def _checks(study_id: str):
    s = _study(study_id)
    ev = {n: registry.read_evidence(study_id, n) for n in ("development", "final", "slices_tuning", "stress",
                                                           "explanations")}
    return review.deterministic_checks(load_policy(), ev["development"], ev["final"], ev["slices_tuning"],
                                       ev["stress"], ev["explanations"], stale_reasons(s, current_identity()))


@app.get("/api/studies/{study_id}/review")
def get_review(study_id: str):
    s = _study(study_id)
    case = _case(study_id)
    if case and s["status"] != "running":
        current = registry.evidence_bundle_hash(study_id)
        case["evidence_current"] = current == case["evidence_sha256"]
    findings = registry.all("SELECT * FROM findings WHERE case_id=? ORDER BY finding_id", (case["case_id"],))
    for f in findings:
        f["history"] = registry.all("SELECT * FROM finding_events WHERE finding_id=? ORDER BY id", (f["finding_id"],))
    return {
        "case": case,
        "stale_reasons": stale_reasons(s, current_identity()),
        "final_evaluated": registry.read_evidence(study_id, "final") is not None,
        "checks": _checks(study_id) if s["status"] not in ("running", "failed") else [],
        "findings": findings,
        "events": registry.all("SELECT * FROM review_events WHERE case_id=? ORDER BY event_id", (case["case_id"],)),
        "event_chain": review.verify_event_chain(registry, case["case_id"]),
        "allowed_actions": [a for a, (frm, _) in review.TRANSITIONS.items() if frm == case["state"]],
        "severity_rubric": review.SEVERITIES, "categories": review.CATEGORIES,
        "finding_statuses": review.FINDING_STATUSES,
        "identity_note": "Reviewer names are entered locally and are not authenticated.",
    }


@app.post("/api/studies/{study_id}/review/transition")
def review_transition(study_id: str, req: TransitionRequest):
    s = _study(study_id)
    case = _case(study_id)
    return review.transition(registry, case["case_id"], req.action, req.actor, req.rationale, req.confirm,
                             req.expected_version, req.expected_evidence_sha256, req.request_id,
                             study_is_stale=bool(stale_reasons(s, current_identity())),
                             final_complete=registry.read_evidence(study_id, "final") is not None)


@app.post("/api/studies/{study_id}/findings", status_code=201)
def create_finding(study_id: str, req: FindingRequest):
    _study(study_id)
    case = _case(study_id)
    return review.add_finding(registry, case["case_id"], req.actor, req.category, req.severity, req.title,
                              req.description, req.evidence_ref, req.owner, req.source_check)


@app.patch("/api/findings/{finding_id}")
def patch_finding(finding_id: int, req: FindingUpdate):
    return review.update_finding(registry, finding_id, req.actor, req.status, req.comment, req.remediation)


# ---------------------------------------------------------------- reports
@app.get("/api/studies/{study_id}/report.md", response_class=PlainTextResponse)
def report_md(study_id: str):
    s = _study(study_id)
    case = _case(study_id)
    evidence = {n: registry.read_evidence(study_id, n) for n in EVIDENCE_NAMES}
    findings = registry.all("SELECT * FROM findings WHERE case_id=? ORDER BY finding_id", (case["case_id"],))
    events = registry.all("SELECT * FROM review_events WHERE case_id=? ORDER BY event_id", (case["case_id"],))
    text = build_report(s, stale_reasons(s, current_identity()), evidence, case, findings, events,
                        _checks(study_id), review.verify_event_chain(registry, case["case_id"]))
    return PlainTextResponse(text, media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{study_id}-report.md"'})


@app.get("/api/studies/{study_id}/results.json")
def results_json(study_id: str):
    s = _study(study_id)
    evidence = {}
    for n in EVIDENCE_NAMES:
        data = registry.read_evidence(study_id, n)
        if data is not None and n == "explanations":
            data = {k: v for k, v in data.items() if k != "feature_values"}
            data["models"] = {r: {k: v for k, v in m.items() if k != "local"} for r, m in data["models"].items()}
        if n != "split_manifest":
            evidence[n] = data
    return JSONResponse({"study": _study_summary(s, current_identity()), "evidence": evidence,
                         "evidence_sha256": registry.all("SELECT name, sha256 FROM evidence WHERE study_id=?",
                                                         (study_id,))},
                        headers={"Content-Disposition": f'attachment; filename="{study_id}-results.json"'})


# ---------------------------------------------------------------- frontend
DIST = ROOT / "frontend" / "dist"
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa(path: str):
    if path.startswith("api/"):
        raise HTTPException(404, "Unknown API route")
    index = DIST / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse("<!doctype html><title>ModelScope</title><h1>ModelScope API running</h1>"
                        "<p>The dashboard has not been built. Run <code>npm install</code> and "
                        "<code>npm run build</code> in <code>frontend/</code>. API docs: <a href='/docs'>/docs</a>.</p>")
