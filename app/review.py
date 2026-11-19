"""Human review workflow for one study's champion candidate.

State machine (acceptance means "accepted for this local research prototype",
never bank, regulatory or production approval):

    draft --(system: development evidence complete)--> evaluated
    evaluated --(reviewer: start_review)--> in_review
    in_review --(reviewer: accept)--> accepted_for_research      [terminal]
    in_review --(reviewer: request_changes)--> changes_requested
    in_review --(reviewer: reject)--> rejected                   [terminal]
    changes_requested --(reviewer: reopen)--> in_review

Controls on every human transition:
- explicit `confirm: true`, a written rationale and a locally entered actor
  name (NOT verified authentication; stored with identity_verified = 0);
- optimistic concurrency: `expected_version` must equal the case version,
  otherwise the request is stale (someone/something changed the case);
- `expected_evidence_sha256` must equal the current evidence bundle hash,
  otherwise the decision refers to evidence that has since changed;
- `request_id` is unique: re-sending the same submission is rejected as a
  duplicate instead of being applied twice;
- accept additionally requires a completed final-holdout audit, a non-stale
  study and no open critical/high findings.
Deterministic checks only *flag* concerns; nothing here accepts automatically.
"""
from __future__ import annotations

import json
import sqlite3

from app.config import canonical_json, sha256_bytes
from app.registry import Registry, now

STATES = ("draft", "evaluated", "in_review", "accepted_for_research", "changes_requested", "rejected")
TRANSITIONS = {
    "start_review": ("evaluated", "in_review"),
    "accept": ("in_review", "accepted_for_research"),
    "request_changes": ("in_review", "changes_requested"),
    "reject": ("in_review", "rejected"),
    "reopen": ("changes_requested", "in_review"),
}
SEVERITIES = {
    "critical": "Invalidates the evidence (e.g. leakage, wrong target, corrupted data). Blocks acceptance.",
    "high": "Material weakness for the intended research use. Blocks acceptance while open.",
    "medium": "Limitation that must be documented or remediated in a later study.",
    "low": "Minor issue; document.",
    "info": "Observation for the record.",
}
CATEGORIES = ("data_quality", "leakage", "discrimination", "calibration", "stability_robustness",
              "segment_performance", "explainability", "documentation", "process_control", "other")
FINDING_STATUSES = ("open", "in_remediation", "resolved", "risk_accepted", "closed_no_action")
OPEN_STATUSES = ("open", "in_remediation")


class ReviewError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _event_hash(prev_hash, body):
    return sha256_bytes((prev_hash or "").encode() + canonical_json(body))


def append_event(reg: Registry, db, case_id, actor, role, event_type, from_state, to_state, payload,
                 evidence_sha, request_id=None):
    prev = db.execute("SELECT event_hash FROM review_events WHERE case_id=? ORDER BY event_id DESC LIMIT 1",
                      (case_id,)).fetchone()
    prev_hash = prev["event_hash"] if prev else None
    at = now()
    body = {"case_id": case_id, "at": at, "actor": actor, "role": role, "type": event_type,
            "from": from_state, "to": to_state, "payload": payload, "evidence": evidence_sha,
            "request_id": request_id}
    try:
        db.execute("INSERT INTO review_events (case_id, at_utc, actor, actor_role, identity_verified, event_type,"
                   " from_state, to_state, payload_json, evidence_sha256, request_id, prev_hash, event_hash)"
                   " VALUES (?,?,?,?,0,?,?,?,?,?,?,?,?)",
                   (case_id, at, actor, role, event_type, from_state, to_state, json.dumps(payload),
                    evidence_sha, request_id, prev_hash, _event_hash(prev_hash, body)))
    except sqlite3.IntegrityError as exc:
        if "request_id" in str(exc):
            raise ReviewError("Duplicate submission: this request_id was already recorded", 409) from exc
        raise


def create_case(reg: Registry, study_id: str) -> str:
    case_id = f"case-{study_id}"
    with reg.connect() as db:
        db.execute("INSERT INTO review_cases VALUES (?,?,?,?,?,?,?,?)",
                   (case_id, study_id, None, "draft", 1, None, now(), now()))
        append_event(reg, db, case_id, "system", "system", "case_created", None, "draft", {}, None)
    return case_id


def system_transition(reg: Registry, case_id: str, to_state: str, payload: dict, candidate_role=None):
    """Only the draft -> evaluated step is performed by the system."""
    with reg.connect() as db:
        case = db.execute("SELECT * FROM review_cases WHERE case_id=?", (case_id,)).fetchone()
        if not (case["state"] == "draft" and to_state == "evaluated"):
            raise ReviewError(f"System may not move {case['state']} -> {to_state}", 409)
        study_id = case["study_id"]
    evidence = reg.evidence_bundle_hash(study_id)
    with reg.connect() as db:
        db.execute("UPDATE review_cases SET state=?, version=version+1, evidence_sha256=?, candidate_role=?,"
                   " updated_utc=? WHERE case_id=?", (to_state, evidence, candidate_role, now(), case_id))
        append_event(reg, db, case_id, "system", "system", "evidence_complete", "draft", to_state, payload, evidence)


def refresh_evidence(reg: Registry, case_id: str, reason: str):
    """Record that the evidence bundle changed (e.g. final audit added)."""
    case = reg.one("SELECT * FROM review_cases WHERE case_id=?", (case_id,))
    evidence = reg.evidence_bundle_hash(case["study_id"])
    with reg.connect() as db:
        db.execute("UPDATE review_cases SET evidence_sha256=?, version=version+1, updated_utc=? WHERE case_id=?",
                   (evidence, now(), case_id))
        append_event(reg, db, case_id, "system", "system", "evidence_changed", case["state"], case["state"],
                     {"reason": reason, "previous_evidence": case["evidence_sha256"]}, evidence)


def transition(reg: Registry, case_id: str, action: str, actor: str, rationale: str, confirm: bool,
               expected_version: int, expected_evidence_sha256: str, request_id: str,
               study_is_stale: bool, final_complete: bool) -> dict:
    if action not in TRANSITIONS:
        raise ReviewError(f"Unknown action {action}; allowed {sorted(TRANSITIONS)}")
    if not confirm:
        raise ReviewError("Explicit confirmation required (confirm=true)")
    if not actor or not actor.strip():
        raise ReviewError("Reviewer name required")
    if len((rationale or "").strip()) < 20:
        raise ReviewError("Rationale of at least 20 characters required")
    if not request_id:
        raise ReviewError("request_id required for duplicate protection")
    from_state, to_state = TRANSITIONS[action]
    with reg.connect() as db:
        dup = db.execute("SELECT 1 FROM review_events WHERE request_id=?", (request_id,)).fetchone()
        if dup:
            raise ReviewError("Duplicate submission: this request_id was already recorded", 409)
        case = db.execute("SELECT * FROM review_cases WHERE case_id=?", (case_id,)).fetchone()
        if case is None:
            raise ReviewError("Unknown review case", 404)
        if case["version"] != expected_version:
            raise ReviewError(f"Stale decision: case is at version {case['version']}, "
                              f"request was based on {expected_version}. Reload and re-check.", 409)
        current_evidence = reg.evidence_bundle_hash(case["study_id"])
        if expected_evidence_sha256 != current_evidence or case["evidence_sha256"] != current_evidence:
            raise ReviewError("Stale decision: the evidence bundle changed since you loaded it", 409)
        if case["state"] != from_state:
            raise ReviewError(f"Cannot {action} from state {case['state']}", 409)
        if action == "accept":
            if study_is_stale:
                raise ReviewError("Cannot accept: study evidence is stale (dataset/policy/code changed)", 409)
            if not final_complete:
                raise ReviewError("Cannot accept before the final-holdout audit has been run", 409)
            blocking = db.execute(
                "SELECT COUNT(*) AS n FROM findings WHERE case_id=? AND severity IN ('critical','high') "
                "AND status IN ('open','in_remediation')", (case_id,)).fetchone()["n"]
            if blocking:
                raise ReviewError(f"Cannot accept: {blocking} open critical/high finding(s)", 409)
        db.execute("UPDATE review_cases SET state=?, version=version+1, updated_utc=? WHERE case_id=?",
                   (to_state, now(), case_id))
        append_event(reg, db, case_id, actor.strip(), "reviewer", action, from_state, to_state,
                     {"rationale": rationale.strip(),
                      "scope": "local research prototype decision; not production or regulatory approval"},
                     current_evidence, request_id)
    return reg.one("SELECT * FROM review_cases WHERE case_id=?", (case_id,))


def add_finding(reg: Registry, case_id, actor, category, severity, title, description, evidence_ref, owner,
                source_check=None) -> dict:
    if category not in CATEGORIES:
        raise ReviewError(f"category must be one of {CATEGORIES}")
    if severity not in SEVERITIES:
        raise ReviewError(f"severity must be one of {list(SEVERITIES)}")
    for name, value in (("actor", actor), ("title", title), ("description", description),
                        ("evidence_ref", evidence_ref), ("owner", owner)):
        if not value or not str(value).strip():
            raise ReviewError(f"{name} is required")
    case = reg.one("SELECT * FROM review_cases WHERE case_id=?", (case_id,))
    if case is None:
        raise ReviewError("Unknown review case", 404)
    if case["state"] in ("accepted_for_research", "rejected"):
        raise ReviewError("Case is closed; record findings in a new study", 409)
    with reg.connect() as db:
        if source_check:
            dup = db.execute("SELECT finding_id FROM findings WHERE case_id=? AND source_check=?",
                             (case_id, source_check)).fetchone()
            if dup:
                raise ReviewError(f"A finding for check {source_check} already exists (#{dup['finding_id']})", 409)
        cur = db.execute("INSERT INTO findings (case_id, category, severity, title, description, evidence_ref,"
                         " owner, status, reviewer_comment, created_by, created_utc, updated_utc, source_check)"
                         " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                         (case_id, category, severity, title.strip(), description.strip(), evidence_ref.strip(),
                          owner.strip(), "open", None, actor.strip(), now(), now(), source_check))
        fid = cur.lastrowid
        db.execute("INSERT INTO finding_events (finding_id, at_utc, actor, event_type, payload_json) VALUES (?,?,?,?,?)",
                   (fid, now(), actor.strip(), "created", json.dumps({"severity": severity, "category": category})))
        append_event(reg, db, case_id, actor.strip(), "reviewer", "finding_added", case["state"], case["state"],
                     {"finding_id": fid, "severity": severity, "title": title.strip()}, case["evidence_sha256"])
    return reg.one("SELECT * FROM findings WHERE finding_id=?", (fid,))


def update_finding(reg: Registry, finding_id, actor, status=None, comment=None, remediation=None) -> dict:
    if not actor or not actor.strip():
        raise ReviewError("actor is required")
    if status is None and not comment and not remediation:
        raise ReviewError("Nothing to update")
    if status is not None and status not in FINDING_STATUSES:
        raise ReviewError(f"status must be one of {FINDING_STATUSES}")
    f = reg.one("SELECT * FROM findings WHERE finding_id=?", (finding_id,))
    if f is None:
        raise ReviewError("Unknown finding", 404)
    if status in ("resolved", "risk_accepted", "closed_no_action") and not (comment or remediation):
        raise ReviewError("Closing a finding requires a comment or remediation note")
    with reg.connect() as db:
        db.execute("UPDATE findings SET status=COALESCE(?, status), reviewer_comment=COALESCE(?, reviewer_comment),"
                   " updated_utc=? WHERE finding_id=?", (status, comment, now(), finding_id))
        db.execute("INSERT INTO finding_events (finding_id, at_utc, actor, event_type, payload_json) VALUES (?,?,?,?,?)",
                   (finding_id, now(), actor.strip(), "updated",
                    json.dumps({"from_status": f["status"], "to_status": status or f["status"],
                                "comment": comment, "remediation": remediation})))
    return reg.one("SELECT * FROM findings WHERE finding_id=?", (finding_id,))


def verify_event_chain(reg: Registry, case_id: str) -> dict:
    events = reg.all("SELECT * FROM review_events WHERE case_id=? ORDER BY event_id", (case_id,))
    prev = None
    for e in events:
        body = {"case_id": e["case_id"], "at": e["at_utc"], "actor": e["actor"], "role": e["actor_role"],
                "type": e["event_type"], "from": e["from_state"], "to": e["to_state"],
                "payload": json.loads(e["payload_json"]), "evidence": e["evidence_sha256"],
                "request_id": e["request_id"]}
        if e["prev_hash"] != prev or e["event_hash"] != _event_hash(prev, body):
            return {"ok": False, "broken_at_event": e["event_id"], "n_events": len(events)}
        prev = e["event_hash"]
    return {"ok": True, "n_events": len(events)}


def deterministic_checks(policy: dict, development: dict | None, final: dict | None, slices: dict | None,
                         stress: dict | None, explanations: dict | None, stale_reasons: list) -> list[dict]:
    """Configured checks that FLAG concerns for a human. They never decide."""
    cfg = policy["review_checks"]
    checks = []

    def add(check_id, status, message, evidence_ref, category, suggested_severity):
        checks.append({"check_id": check_id, "status": status, "message": message, "evidence_ref": evidence_ref,
                       "category": category, "suggested_severity": suggested_severity})

    add("study_identity_current", "flag" if stale_reasons else "ok",
        "Stale: " + "; ".join(stale_reasons) if stale_reasons else "Dataset, policy and code identity match",
        "studies table", "process_control", "high")
    if development:
        for role, ev in development["evaluation"].items():
            cal = ev["calibrated"]["calibration"]
            citl = cal["calibration_in_the_large"]
            add(f"citl_{role}", "flag" if abs(citl) > cfg["calibration_in_the_large_abs_max"] else "ok",
                f"{role}: observed - mean predicted on tuning = {citl:+.4f} (limit ±{cfg['calibration_in_the_large_abs_max']})",
                f"development.json#/evaluation/{role}/calibrated/calibration", "calibration", "medium")
            add(f"ece_{role}", "flag" if cal["ece_quantile_10"] > cfg["ece_max"] else "ok",
                f"{role}: ECE (10 quantile bins) on tuning = {cal['ece_quantile_10']:.4f} (limit {cfg['ece_max']})",
                f"development.json#/evaluation/{role}/calibrated/calibration", "calibration", "medium")
    if stress:
        for role in stress["clean"]:
            worst = min(stress["scenarios"], key=lambda s: s["models"][role]["delta_roc_auc"])
            drop = -worst["models"][role]["delta_roc_auc"]
            add(f"stress_{role}", "flag" if drop > cfg["stress_auc_drop_max"] else "ok",
                f"{role}: largest AUC drop {drop:.4f} under '{worst['id']}' (limit {cfg['stress_auc_drop_max']})",
                f"stress.json#/scenarios/{worst['id']}", "stability_robustness", "medium")
    if slices:
        low = [f"{r['slice']}={r['group']}" for r in slices["rows"] if r["warnings"]]
        add("slice_support", "flag" if low else "ok",
            f"{len(low)} slice(s) below support/event minimums: {', '.join(low[:4])}{' …' if len(low) > 4 else ''}" if low
            else "All slices meet support minimums", "slices_tuning.json", "segment_performance", "low")
        below = [f"{r['slice']}={r['group']} ({m})" for r in slices["rows"] for m, v in r["models"].items()
                 if v.get("auc_ci_below_overall")]
        add("slice_auc_below_overall", "flag" if below else "ok",
            f"{len(below)} slice/model pair(s) with AUC CI wholly below the overall AUC (expected for "
            f"slices that restrict the range of PAY_0): {', '.join(below[:4])}{' …' if len(below) > 4 else ''}" if below
            else "No slice AUC interval lies wholly below the overall AUC", "slices_tuning.json",
            "segment_performance", "medium")
    if explanations:
        for role, ex in explanations["models"].items():
            add(f"shap_additivity_{role}", "ok" if ex["additivity_ok"] else "flag",
                f"{role}: max additivity error {ex['additivity_max_abs_error_calibrated']:.2e}",
                f"explanations.json#/models/{role}", "explainability", "high")
    if final and development:
        for role in final["evaluation"]:
            dev_auc = development["evaluation"][role]["calibrated"]["metrics"]["roc_auc"]
            fin_auc = final["evaluation"][role]["metrics"]["roc_auc"]
            add(f"final_vs_dev_{role}", "flag" if dev_auc - fin_auc > cfg["final_vs_development_auc_drop_max"] else "ok",
                f"{role}: AUC tuning {dev_auc:.4f} vs final {fin_auc:.4f}",
                f"final.json#/evaluation/{role}", "discrimination", "medium")
    else:
        add("final_audit", "pending", "Final-holdout audit not yet run", "final.json", "process_control", "info")
    return checks
