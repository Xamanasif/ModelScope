"""Registry integrity and review state machine (isolated temporary database)."""
import sqlite3
import uuid

import pytest

from app import review
from app.registry import IntegrityError, Registry


@pytest.fixture()
def setup(tmp_path):
    reg = Registry(tmp_path / "r.db")
    reg.execute("INSERT INTO studies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("s1", "k", "d", "p", "c", None, "m", "development_complete", "t", None, None, "dev", str(tmp_path)))
    reg.write_evidence("s1", tmp_path, "development", {"x": 1}, "development")
    case = review.create_case(reg, "s1")
    review.system_transition(reg, case, "evaluated", {}, candidate_role="baseline")
    return reg, case, tmp_path


def act(reg, case, action, final=True, stale=False, **over):
    c = reg.one("SELECT * FROM review_cases WHERE case_id=?", (case,))
    args = dict(actor="Reviewer A", rationale="Reasoned decision text over twenty chars", confirm=True,
                expected_version=c["version"], expected_evidence_sha256=reg.evidence_bundle_hash("s1"),
                request_id=uuid.uuid4().hex, study_is_stale=stale, final_complete=final)
    args.update(over)
    return review.transition(reg, case, action, **args)


def test_happy_path_and_terminal_state(setup):
    reg, case, _ = setup
    assert act(reg, case, "start_review")["state"] == "in_review"
    assert act(reg, case, "accept")["state"] == "accepted_for_research"
    with pytest.raises(review.ReviewError):
        act(reg, case, "reopen")
    assert review.verify_event_chain(reg, case)["ok"]


def test_invalid_transitions_rejected(setup):
    reg, case, _ = setup
    with pytest.raises(review.ReviewError, match="Cannot accept from"):
        act(reg, case, "accept")
    with pytest.raises(review.ReviewError, match="Unknown action"):
        act(reg, case, "approve_for_production")


def test_changes_requested_then_reopen(setup):
    reg, case, _ = setup
    act(reg, case, "start_review")
    assert act(reg, case, "request_changes")["state"] == "changes_requested"
    assert act(reg, case, "reopen")["state"] == "in_review"
    assert act(reg, case, "reject")["state"] == "rejected"


def test_confirmation_rationale_actor_required(setup):
    reg, case, _ = setup
    for over in ({"confirm": False}, {"rationale": "too short"}, {"actor": "  "}):
        with pytest.raises(review.ReviewError):
            act(reg, case, "start_review", **over)


def test_duplicate_submission_rejected(setup):
    reg, case, _ = setup
    act(reg, case, "start_review", request_id="same-request-1")
    with pytest.raises(review.ReviewError, match="Duplicate") as e:
        act(reg, case, "request_changes", request_id="same-request-1")
    assert e.value.status_code == 409


def test_stale_version_rejected(setup):
    reg, case, _ = setup
    c = reg.one("SELECT * FROM review_cases WHERE case_id=?", (case,))
    act(reg, case, "start_review")
    with pytest.raises(review.ReviewError, match="Stale decision"):
        act(reg, case, "accept", expected_version=c["version"])


def test_stale_evidence_rejected(setup):
    reg, case, tmp = setup
    act(reg, case, "start_review")
    old_hash = reg.evidence_bundle_hash("s1")
    reg.write_evidence("s1", tmp, "final", {"y": 2}, "final")
    with pytest.raises(review.ReviewError, match="Stale decision"):
        act(reg, case, "accept", expected_evidence_sha256=old_hash)
    review.refresh_evidence(reg, case, "final added")
    assert act(reg, case, "accept")["state"] == "accepted_for_research"


def test_accept_guards(setup):
    reg, case, _ = setup
    act(reg, case, "start_review")
    with pytest.raises(review.ReviewError, match="final-holdout"):
        act(reg, case, "accept", final=False)
    with pytest.raises(review.ReviewError, match="stale"):
        act(reg, case, "accept", stale=True)
    f = review.add_finding(reg, case, "Reviewer A", "calibration", "high", "Miscalibrated top decile",
                           "Observed rate exceeds predictions", "development.json#/evaluation", "Developer")
    review.refresh_evidence(reg, case, "noop")  # version bump only
    with pytest.raises(review.ReviewError, match="critical/high"):
        act(reg, case, "accept")
    with pytest.raises(review.ReviewError, match="requires a comment"):
        review.update_finding(reg, f["finding_id"], "Developer", status="resolved")
    review.update_finding(reg, f["finding_id"], "Developer", status="resolved", remediation="Documented limitation")
    assert act(reg, case, "accept")["state"] == "accepted_for_research"
    hist = reg.all("SELECT * FROM finding_events WHERE finding_id=?", (f["finding_id"],))
    assert [h["event_type"] for h in hist] == ["created", "updated"]


def test_finding_validation_and_duplicate_check_finding(setup):
    reg, case, _ = setup
    with pytest.raises(review.ReviewError):
        review.add_finding(reg, case, "A", "nonsense", "high", "t", "d", "e", "o")
    with pytest.raises(review.ReviewError):
        review.add_finding(reg, case, "A", "calibration", "catastrophic", "t", "d", "e", "o")
    review.add_finding(reg, case, "A", "calibration", "low", "t", "d", "e", "o", source_check="ece_baseline")
    with pytest.raises(review.ReviewError, match="already exists"):
        review.add_finding(reg, case, "A", "calibration", "low", "t", "d", "e", "o", source_check="ece_baseline")


def test_append_only_tables_and_hash_chain_detects_tampering(setup):
    reg, case, _ = setup
    act(reg, case, "start_review")
    with pytest.raises(sqlite3.IntegrityError):
        reg.execute("UPDATE review_events SET actor='someone else'")
    with pytest.raises(sqlite3.IntegrityError):
        reg.execute("DELETE FROM review_events")
    # The triggers are not a security boundary: whoever owns the file can drop them.
    reg.execute("DROP TRIGGER review_events_no_update")
    reg.execute("UPDATE review_events SET actor='someone else' WHERE event_type='start_review'")
    assert not review.verify_event_chain(reg, case)["ok"]


def test_evidence_tampering_detected(setup):
    reg, _, tmp = setup
    (tmp / "development.json").write_text('{"x": 999}', encoding="utf-8")
    with pytest.raises(IntegrityError):
        reg.read_evidence("s1", "development")


def test_checks_never_decide():
    checks = review.deterministic_checks({"review_checks": {"calibration_in_the_large_abs_max": .02, "ece_max": .03,
                                                            "stress_auc_drop_max": .05,
                                                            "final_vs_development_auc_drop_max": .03}},
                                         None, None, None, None, None, ["policy changed"])
    assert {c["status"] for c in checks} <= {"ok", "flag", "pending"}
    assert any(c["check_id"] == "study_identity_current" and c["status"] == "flag" for c in checks)
