# Risk and limitations register

Severity uses the review rubric in `app/review.py::SEVERITIES`. "Status" says
what the project does about each risk; most are *documented*, not *resolved*.

| # | Risk / limitation | Severity | Evidence | Mitigation in ModelScope | Status |
|---|---|---|---|---|---|
| R1 | No temporal validation: cross-sectional 2005 sample | high (for any deployment claim) | dataset card | split described as in-distribution only; no out-of-time claim | documented, unresolvable with this data |
| R2 | Results don't transfer to other populations or products | high | dataset card | scope statements in UI and reports | documented |
| R3 | Target is a one-month delinquency label, not loss | medium | source variable_info | no ECL/IFRS 9/capital language; cost scenarios labelled user-supplied | documented |
| R4 | Undocumented codes (PAY −2/0, EDUCATION 0/5/6, MARRIAGE 0) | medium | data_quality checks (8 warnings) | kept as codes, not reinterpreted; warned in UI | documented |
| R5 | Leakage of target/ID into features | critical if present | `test_target_cannot_leak_into_features` | explicit feature list; ColumnTransformer drops everything else; shuffled-label canary test | controlled and tested |
| R6 | Selection bias from re-using evaluation data | high if present | protocol | CV inside train; clean tuning partition; frozen selection; one-shot final with access log | controlled; lock is local and bypassable (R13) |
| R7 | Calibration overfitting | medium | `calibration.json` fields | separate calibration partition; in-sample losses labelled as such; isotonic not used | controlled |
| R8 | Within-segment ranking is weak for delinquent customers | medium | slices (PAY_0 ≥ 2 AUC 0.54–0.58 final) | shown with CIs and caution text; flagged by check | open: candidate finding for the reviewer |
| R9 | Small slices give unstable estimates | low | slice warnings (EDUCATION others n=20, MARRIAGE code 0 n=5) | support/event warnings; single-class AUC shown as unavailable | documented |
| R10 | Slice charts misread as fairness evidence | medium | — | explicit caution copy; no recommendation language; no multiplicity correction claimed | documented |
| R11 | Sensitivity to input unit errors (LR) | medium | stress `limit_in_thousands_10` ΔAUC −0.018 (LR) | stress evidence counts rows below the training minimum; monitoring-batch uploads report out-of-training-range counts per feature. **Correction:** the scenario note in `app/stress.py` (and in that study's stress evidence) wrongly says the data validator flags these rows; `validate_dataset` only rejects limits ≤ 0 | documented; validator range check deferred (would change study identity) |
| R12 | SHAP misinterpreted as causal or as reasons for a decision | medium | explanations limitations | UI/report limitation text; not an adverse-action notice | documented |
| R13 | Local lock and audit log not tamper-proof | medium | `test_append_only_tables_and_hash_chain_detects_tampering` | triggers + hash chain detect casual edits only | documented |
| R14 | Reviewer identity is self-entered | medium | `identity_verified = 0` | labelled "not authenticated" everywhere | documented; real auth deferred |
| R15 | Pickle artifacts execute code on load | high if untrusted files loaded | `load_trusted_model` | load only registered, hash-matched files inside artifacts/; no model upload | controlled |
| R16 | Library behaviour drift (e.g. TreeSHAP additivity) | medium | D-009, version-pinned test | pins + lock file; additivity measured per run | controlled for pinned versions |
| R17 | Same person develops and reviews | medium | event log | separate UI workflows and recorded roles; not independent validation | documented |
| R18 | 0.5 threshold misread as policy | low | threshold rationale | labelled illustrative; trade-off grid and cost scenario shown | documented |
| R19 | Simulated drift mistaken for monitoring | medium | drift evidence | SIMULATED label in data, UI and report | documented |
