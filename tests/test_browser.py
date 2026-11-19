"""Real-browser UI test (Playwright driving installed Microsoft Edge).

Starts a separate uvicorn process on a free port with its own temporary data,
artifact and policy directories and a SYNTHETIC dataset, then exercises the
critical flows: page load without training, starting a study, review start,
the final-holdout audit, a findings record, the accept decision and the report
download. Skipped when the frontend is not built or Edge/Playwright is missing.

    python -m pytest -m browser -v
"""
import json
import re
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.browser
playwright = pytest.importorskip("playwright.sync_api")
if not (ROOT / "frontend" / "dist" / "index.html").exists():
    pytest.skip("frontend not built (npm run build)", allow_module_level=True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    from tests.fixtures import small_policy, synthetic_uci_frame
    tmp = Path(tempfile.mkdtemp(prefix="modelscope-browser-"))
    (tmp / "data").mkdir()
    (tmp / "artifacts").mkdir()
    synthetic_uci_frame(1600, seed=0).to_csv(tmp / "data" / "uci_credit_default.csv", index=False)
    (tmp / "policy.json").write_text(json.dumps(small_policy()), encoding="utf-8")
    env = {**os.environ, "MODELSCOPE_DATA_DIR": str(tmp / "data"), "MODELSCOPE_ARTIFACT_DIR": str(tmp / "artifacts"),
           "MODELSCOPE_DB": str(tmp / "artifacts" / "b.db"), "MODELSCOPE_POLICY": str(tmp / "policy.json")}
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def page(server):
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge", headless=True)
        except Exception as exc:  # Edge not installed
            pytest.skip(f"Edge unavailable: {exc}")
        pg = browser.new_page(viewport={"width": 1280, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.errors = errors
        yield pg
        browser.close()


def wait_api(server, predicate, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        status = json.load(urllib.request.urlopen(server + "/api/status"))
        if predicate(status):
            return status
        time.sleep(0.5)
    raise AssertionError("timed out waiting for API state")


def test_full_ui_flow(server, page):
    from playwright.sync_api import expect
    page.goto(server + "/#/overview")
    expect(page.get_by_role("heading", name="Overview", level=1)).to_be_visible()
    expect(page.get_by_text("No study yet").first).to_be_visible()
    assert json.load(urllib.request.urlopen(server + "/api/status"))["studies"] == []  # opening never trains

    page.get_by_role("link", name="Data quality").click()
    expect(page.get_by_role("heading", name="Schema and domain checks")).to_be_visible()

    page.get_by_role("link", name="Reports & runs").click()
    start = page.get_by_role("button", name="Start development study")
    expect(start).to_be_disabled()  # needs a name and confirmation
    page.get_by_label("Your name").first.fill("Browser Tester")
    page.get_by_label("Start training now").check()
    start.click()
    wait_api(server, lambda s: s["studies"] and s["studies"][0]["status"] == "development_complete")
    page.reload()

    page.get_by_role("link", name="Model comparison").click()
    expect(page.get_by_role("heading", name="Metrics (calibrated probabilities)")).to_be_visible()
    expect(page.get_by_role("img", name="ROC curves")).to_be_visible()
    page.get_by_role("button", name="Final holdout audit").click()
    expect(page.get_by_text("Final holdout still sealed")).to_be_visible()

    for link, heading in [("Segment analysis", "ROC-AUC by group with 95% bootstrap interval"),
                          ("Stress & shift lab", "Shift diagnostics"), ("Explanations", "Local explanation")]:
        page.get_by_role("link", name=link).click()
        expect(page.get_by_role("heading", name=heading)).to_be_visible()
    page.get_by_label("Customer ID (development explanation sample)").select_option(index=1)
    expect(page.get_by_text("probability", exact=False).first).to_be_visible()

    page.get_by_role("link", name="Review workspace").click()
    page.get_by_label("Your name (recorded with every action; not authenticated)").fill("Browser Reviewer")
    page.get_by_label("Rationale (at least 20 characters; cite evidence and limitations)").fill("Starting review of development evidence.")
    page.get_by_label("I confirm this is my decision").check()
    page.get_by_role("button", name="Start review").click()
    expect(page.get_by_text("Recorded: case is now in review.")).to_be_visible()

    page.get_by_role("link", name="Reports & runs").click()
    page.get_by_label("Type an acknowledgement containing the word “frozen”").fill("I accept the selection is frozen")
    page.get_by_label("I confirm final-holdout access").check()
    page.get_by_role("button", name="Run final audit").click()
    wait_api(server, lambda s: s["studies"][0]["final_evaluated"] and s["active_job"] is None)
    page.reload()
    expect(page.get_by_text("Already evaluated once for this study.")).to_be_visible()
    with page.expect_download() as dl:
        page.get_by_role("link", name="Validation report (Markdown)").click()
    assert "Final holdout audit (evaluated once)" in Path(dl.value.path()).read_text(encoding="utf-8")

    page.get_by_role("link", name="Review workspace").click()
    page.get_by_label("Title").fill("Browser-recorded finding")
    page.get_by_label("Description").fill("Recorded through the UI test.")
    page.get_by_label("Evidence reference").fill("development.json")
    page.get_by_label("Owner").fill("Developer")
    page.get_by_role("button", name="Record finding").click()
    expect(page.get_by_text("Finding #1 recorded.")).to_be_visible()
    page.get_by_role("combobox", name=re.compile(r"^Action")).select_option("accept")
    page.get_by_label("Rationale (at least 20 characters; cite evidence and limitations)").fill("Accepted for research use with documented limitations.")
    page.get_by_label("I confirm this is my decision").check()
    page.get_by_role("button", name="Accept for research").click()
    expect(page.get_by_text("Recorded: case is now accepted for research.")).to_be_visible()
    assert page.errors == [], page.errors


def test_narrow_viewport_has_no_horizontal_scroll(server, page):
    page.set_viewport_size({"width": 390, "height": 800})
    for route in ("overview", "comparison", "review", "runs"):
        page.goto(server + f"/#/{route}")
        page.wait_for_timeout(600)
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        assert overflow <= 1, f"{route} overflows by {overflow}px"
