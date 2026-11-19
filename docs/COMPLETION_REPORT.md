# Completion report (2026-11-18)

Environment: Windows 11 Home (10.0.26200), Python 3.11.9 (`py -3.11`),
Node v24.22.1 / npm 11.19.1, Microsoft Edge (for browser tests). Key
packages: fastapi 0.163.2, uvicorn 0.54.0, numpy 2.4.6, pandas 3.1.1,
scikit-learn 1.9.1, scipy 1.17.1, shap 0.51.0, pytest 9.2.0, playwright 1.65.2;
React 18.3.1, Vite 5.4.11, TypeScript 5.6.3. Full lock: `requirements-lock.txt`,
`frontend/package-lock.json`.

## Commands actually run, with outcomes
| Command | Exit | Outcome |
|---|---|---|
| `py -3.11 -m venv .venv`; `pip install -r requirements.txt` (starter) | 0 | installed |
| `python -m unittest discover -s tests -v` (starter) | 0 | 8/8 OK |
| `python scripts/download_data.py` (starter, ucimlrepo) | 1 | DNS failure on CSV host |
| `python scripts/download_data.py` (rewritten) | 0 | 30,000 rows; SHA-256 `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0`; 10 pass / 2 info / 8 warn / 0 fail |
| `python scripts/run_study.py development --actor "dev-check"` | 0 | check run, now stale (code changed afterwards); final never opened |
| `python -m pytest -q` (during build) | 1 → 0 | first failure exposed the final-guard ordering bug; fixed; then a header-mapping bug; fixed |
| `npm install`; `npm run build` | 0 | 66 kB gzipped JS |
| `python scripts/run_study.py development --actor "Claude Code (AI-assisted run)"` | 0 | `study-20261118T163406-a420d0fc`, 188 s, champion challenger |
| `python -m pytest -m browser` (first) | 1 | strict-locator test issue; 176 px mobile overflow (real UI bug) |
| `python scripts/run_study.py final --study study-20261118T163406-a420d0fc --actor "Claude Code (AI-assisted run)" --confirm-final-holdout` | 0 | one-shot audit; access log: 1 granted |
| `python scripts/trace_record.py --study study-20261118T163406-a420d0fc` | 0 | guide section L |
| `python -m pytest -q -m "not browser"` (final) | 0 | 71 passed (70 fixture + 1 official-data) |
| `python -m pytest -q -m browser` (final) | 0 | 2 passed (Edge) |
| `powershell -File .\run.ps1 -Port 8011` | n/a (stopped) | served `/health`, dashboard and `/api/status` |

## Official results (final holdout, n = 4,500, prevalence 22.11%)
| Metric | Baseline LR | Challenger HGB | Difference, 95% paired CI |
|---|---|---|---|
| ROC-AUC | 0.7657 | 0.7819 | +0.0162 [0.0087, 0.0240] |
| Average precision | 0.5302 | 0.5561 | +0.0259 [0.0126, 0.0386] |
| Log loss | 0.4396 | 0.4276 | −0.0120 [−0.0161, −0.0080] |
| Brier | 0.1378 | 0.1342 | −0.0036 [−0.0050, −0.0022] |

Champion chosen beforehand on tuning by the frozen log-loss rule: challenger.
See EVALUATION_REPORT.md for everything else.

## Unverified paths
Python 3.12; `run.ps1 -Build`; `npm ci` on a clean checkout; manual-file
acquisition; non-Edge browsers and assistive technology; Docker (not
provided). A human review decision has not been made.

## Not complete
Nothing in the required feature list is missing. The open items are those
under "Deferred" in BUILD_STATUS.md, plus the student-owned tasks: the review
decision, the exercises and independent reproduction.
