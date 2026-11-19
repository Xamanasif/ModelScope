# Implementation plan (Phase 0) and how it was carried out

## Starting point (inspected 2026-11-18)
Working: generic frame validation (`validate_frame`, `feature_target`), 4-way
stratified customer split with integer counts, `binary_metrics` with explicit
input errors, a FastAPI health/status/starter page, a `ucimlrepo` acquisition
script, 8 unit tests. Starter tests on this machine: `python -m unittest discover -s tests -v`
→ 8/8 OK on Windows 11 / Python 3.11.9 (one Starlette deprecation warning).

Gaps: no source-specific schema/semantics, preprocessing, models, calibration,
experiment enforcement, registry, review workflow, uncertainty, slices, stress,
drift, SHAP, dashboard or real-data run. The acquisition script hashed a pandas
re-serialisation rather than the source file.

Assumptions checked against the source rather than trusted: column order and
names (from live UCI metadata), positive class (variable_info), licence
(landing page), code semantics (variable_info; undocumented codes found by
profiling).

## Architecture (smallest coherent)
See DECISIONS.md D-001. Python modules for each concern → `study.py`
orchestration → JSON evidence + SQLite registry → FastAPI → React dashboard.

## Plan as executed
1. Data acquisition with exact-bytes hashing, manual route, schema/domain validation. ✔
2. Pre-register the policy, including the split manifest hash, before modelling. ✔
3. Pipelines (train-only fitting), bounded CV search, Platt calibration with a development-only selection rule. ✔
4. Metrics (discrimination / probability / decision), paired bootstrap, repeated development CV. ✔
5. Slices with support warnings, seeded stress suite, PSI drift with simulated batches, SHAP with additivity gates and ranking stability. ✔
6. Registry (hash-verified evidence), state machine, findings, append-only hash-chained events, holdout access log. ✔
7. FastAPI endpoints, single-flight job runner with identity cache, React dashboard (8 sections). ✔
8. Tests: unit, API integration, official-data check, Playwright browser. ✔
9. Official-data run: development, then the one-shot final audit. ✔
10. Documentation and learning guide. ✔

Deviations from the prompt's suggestions are listed in DECISIONS.md (notably D-006,
hyperparameters tuned by CV inside train, and D-009, the SHAP algorithm).
