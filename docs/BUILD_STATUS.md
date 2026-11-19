# Build status (2026-11-18)

Evidence references point to tests (T), run artifacts (A) or commands in
[COMPLETION_REPORT.md](COMPLETION_REPORT.md).

## Implemented
- Official UCI 350 acquisition with exact-bytes SHA-256, live metadata mapping check, retries, manual route, manifest with citation/licence (A: `data/raw/source_manifest.json`).
- Schema/type/ID/target/domain validation with documented vs undocumented codes (T: `test_data_and_splits.py`).
- Pre-registered policy with dataset and split-manifest hashes; refusal on mismatch (T: `test_official_data.py`).
- Disjoint stratified 60/15/10/15 customer split with manifest hash (T: split tests).
- Train-only preprocessing pipelines; LR baseline and HGB challenger; bounded CV search inside train with timings (A: `development.json`).
- Platt calibration fitted on the calibration partition, none/sigmoid chosen on tuning (T: calibration tests).
- Discrimination, proper scoring rules, calibration curve/ECE/slope, threshold grid, user-supplied cost scenario.
- Paired bootstrap with failure counting; repeated development CV.
- Slices with train-fitted bands, support warnings, bootstrap CIs, unavailable single-class slices.
- Seeded stress suite (10 scenarios) and SIMULATED drift batches with PSI/KS.
- SHAP for both models with additivity gate in calibrated log-odds and ranking-stability refits.
- SQLite registry with hash-verified evidence, trusted model loading, identity-based staleness.
- Review state machine, findings with history, deterministic checks, append-only hash-chained events, holdout access log.
- Single-flight background jobs with identity cache and interrupted-job recovery.
- FastAPI API (see API.md) and React/TypeScript dashboard with the 8 required sections, loading/empty/error states, light/dark themes.
- Monitoring-batch upload with validation, drift and out-of-training-range counts (not persisted).
- Markdown report and JSON results download.

## Tested
- 73 automated tests, all passing: 70 offline tests on synthetic fixtures, 1 official-data check, 2 real-browser tests (Playwright driving Microsoft Edge).
- One official-data experiment: development study + one-shot final audit, study `study-20261118T163406-a420d0fc` (A: `docs/EVALUATION_REPORT.md`).
- Every dashboard page visually checked in Edge at 1360 px (light and dark comparison page) and four pages at 390 px for horizontal overflow.

## Not verified
- Python 3.12 (pins declare support; only 3.11.9 was run).
- `run.ps1 -Build` (the rebuild branch); plain `run.ps1 -Port 8011` was run and served /health, the dashboard and /api/status.
- `npm ci` from a clean checkout (`npm install` was used; package-lock.json exists).
- Manual acquisition route (`--manual`); only the API route was run.
- Browsers other than Edge; screen-reader behaviour; keyboard-only operation beyond focus styles.
- Docker: not provided.

## Known defects / deviations
- The `limit_in_thousands_10` stress note (and the official study's stress evidence) wrongly says the data validator flags those rows. See RISK_REGISTER R11.
- Starlette TestClient deprecation warning (httpx); see DECISIONS D-018.
- The repository folder is not a git checkout, so `git_revision` is null in study identities.
- No human review decision has been recorded for the official study (case state `evaluated`); this is for the student to do.

## Deferred (extensions)
- PostgreSQL backend, authentication/roles, tamper-proof audit storage.
- Isotonic calibration with a separate selection partition; BCa or retraining-inclusive intervals.
- Validator range checks against training ranges inside `validate_dataset` (needs a new study).
- Multiple split seeds; out-of-time data; fairness analysis with legal input.
- Docker support.
