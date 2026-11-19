# Resume evidence audit: ModelScope

Audit date 2026-11-18. Scope: model validation, statistical testing,
robustness, explainability, reproducibility and review controls, for a
model-risk internship application (DBOI resume).

**Method.** Every figure below was read from the registered evidence files of
study `study-20261118T163406-a420d0fc` through `app.registry.Registry.read_evidence`,
which re-hashes each file against the SHA-256 recorded in
`artifacts/modelscope.db`. All 10 evidence files and both model artifacts
passed that check during this audit. README text was not used as proof. No
code was changed, no model was retrained and the final holdout was not
re-evaluated. Saved final-holdout results were *read*; the holdout access log
still shows exactly one access. Tests were re-run (synthetic fixtures in
temporary directories, plus one read-only check of the official file's hash
and split manifest).

Evidence locations (all git-ignored, local to this machine):
`artifacts/studies/study-20261118T163406-a420d0fc/{data_quality,split_manifest,development,selection,final,slices_tuning,slices_final,stress,drift,explanations}.json`,
`artifacts/modelscope.db`, `data/raw/source_manifest.json`. A rendered copy:
`docs/EVALUATION_REPORT.md`.

---

## 0. Project status

**This is not a starter.** The full workflow runs end to end on the official
data. Important context for every claim: the platform was **implemented by an
AI coding assistant (Claude Code) from your written specification**, and the
official experiment was run by that assistant (the holdout access log records
the actor "Claude Code (AI-assisted run)"). Your resume wording must reflect
this (section C).

### 0.1 Implemented and tested
| Capability | Test or run evidence |
|---|---|
| Official data acquisition with exact-bytes SHA-256 and metadata mapping check | run: `source_manifest.json` (route `uci_api`, 2026-11-18T09:26Z); `tests/test_official_data.py` re-checks hash and split |
| Schema, type, ID, target and domain validation | `tests/test_data_and_splits.py` (18 tests incl. 8 parametrised hard failures) |
| Disjoint stratified customer split with manifest hash | `test_split_disjoint_complete_reproducible_and_seed_sensitive`, `test_manifest_hash_changes_with_seed_fraction_or_data` |
| Train-only preprocessing; leakage canary | `test_preprocessing_statistics_come_from_training_rows_only`, `test_target_cannot_leak_into_features` (LR and HGB) |
| Bounded CV hyperparameter search inside train | run: `development.json → models.*.search_results` |
| Platt calibration on the calibration partition, chosen on tuning | `test_calibration_fit_uses_only_calibration_rows`, `test_platt_recovers_known_map`; run: `development.json → models.*.calibration` |
| Metrics incl. calibration diagnostics; exact fast AUC/AP | `test_fast_metrics_match_sklearn_with_ties`, `test_calibration_diagnostics_behave`, `test_metric_edge_cases` |
| Paired bootstrap with failed-resample counting | `test_paired_bootstrap_deterministic_and_paired`, `test_one_class_resamples_are_counted_not_zeroed`; run: 1,000 resamples, 0 failed |
| Frozen champion rule and one-shot final audit with access log | `test_final_holdout_requires_confirmation_and_is_one_shot`; run: `selection.json`, `final.json`, `holdout_access_log` (1 row, "granted") |
| Slices, stress suite, simulated drift | `test_stress_reproducible_and_labels_fixed_for_corruption`, `test_psi_properties_and_concept_drift_invisible_to_psi`; run: `slices_*.json`, `stress.json`, `drift.json` |
| SHAP with additivity checks | `test_shap_additivity_in_calibrated_logit_space` (both models), `test_demographic_onehot_explanations_aggregate_to_original_columns`; run: `explanations.json` |
| Registry integrity, review state machine, stale/duplicate rejection, append-only events | `tests/test_review_registry.py` (12 tests), API integration tests |
| FastAPI + React dashboard | `tests/test_api_integration.py` (11 tests), `tests/test_browser.py` (2 tests in Microsoft Edge) |

Per-file counts (collected): test_data_and_splits 18, test_modelling 21,
test_review_registry 12, test_api_integration 11, test_foundations 8,
test_official_data 1, test_browser 2.

**Test results re-run during this audit:** `python -m pytest -q -m "not browser"`
→ **71 passed** (70 synthetic-fixture tests + 1 official-data check), exit 0;
`python -m pytest -q -m browser` → **2 passed**, exit 0; `npm run typecheck` → exit 0.
Environment: Windows 11, Python 3.11.9, scikit-learn 1.9.1, shap 0.51.0.

### 0.2 Implemented but not verified
- Python 3.12 (dependency pins allow it; never run).
- `run.ps1 -Build`, `npm ci` from a clean checkout, the manual-download acquisition route.
- Browsers other than Edge; screen-reader and keyboard-only use.
- **Range validation of inputs.** A note inside `app/stress.py` (and therefore inside `stress.json`) says the data validator flags credit limits entered in thousands. That is **false**: `app/data.py::validate_dataset` only rejects limits ≤ 0. The monitoring-batch upload does detect out-of-training-range values (`app/main.py::monitoring_batch`, tested).
- **Reproducibility on another machine:** seeds and pins exist, but the development study has not been re-run from scratch to confirm identical numbers.

### 0.3 Planned, incomplete or not done
- **No human review decision exists.** Case `case-study-20261118T163406-a420d0fc` is in state `evaluated` (version 3). Its 3 events are all system events. There are **0 findings** and no `in_review`/accept/reject event.
- The project is **not under version control** (`git rev-parse` fails), so study identities record `git_revision = null`; code identity relies on a source hash.
- Not built: authentication, PostgreSQL, Docker, isotonic calibration, retraining-inclusive uncertainty, multiple split seeds, out-of-time validation.
- None of the guide's 14 exercises has been completed by you.

---

## 1. Data and experiment design

| Claim | Exact value / finding | Evaluation population | Evidence | Limitation |
|---|---|---|---|---|
| Dataset | UCI ML Repository ID 350 "Default of Credit Card Clients" (Yeh, 2009; DOI 10.24432/C55S3H), CC BY 4.0 per landing page | — | `data/raw/source_manifest.json` | one Taiwanese bank, Apr–Sep 2005 |
| Records used | **30,000 customers**, 25 columns, 0 missing cells, 30,000 unique IDs; file SHA-256 `45bcf4df…3dd0` matches the pre-registered hash | full file | `data_quality.json`; `configs/validation_policy.json → dataset.expected_sha256` | dataset size ≠ evaluation size (see splits) |
| Target / positive class | `Y` = "default payment next month"; **1 = default**, per source text "default payment (Yes = 1, No = 0)"; prevalence 22.12% (6,636) | full file | `data_quality.json → target_binary`; `app/data.py` docstring | one-month delinquency label, not a loss or regulatory default definition |
| Predictors | **19**: LIMIT_BAL, PAY_0, PAY_2–PAY_6, BILL_AMT1–6, PAY_AMT1–6 | — | `development.json → features`; `app/features.py::feature_columns` | — |
| Excluded columns | ID and target always; SEX, EDUCATION, MARRIAGE, AGE excluded by policy (audit slices only) | — | `validation_policy.json → features.demographic_predictors = false` | proxies (e.g. credit limit) may still carry demographic information |
| Data-quality issues found | 10 pass / 2 info / 8 warn / 0 fail; undocumented codes kept as-is (e.g. PAY_0 code 0 on 14,737 rows, −2 on 2,759; EDUCATION 0/5/6; MARRIAGE 0); 35 rows identical except ID | full file | `data_quality.json` | code meanings unknown from the source |
| Split sizes | **train 18,000 / tuning 4,500 / calibration 3,000 / final 4,500**; seed 42; stratified by customer ID; prevalence 0.2212 / 0.2211 / 0.2213 / 0.2211 | — | `split_manifest.json` (SHA `a35d7f50…`, equal to the pre-registered hash) | in-distribution only; rows are not chronological, so no out-of-time claim |
| Preprocessing | LR: median impute, signed log1p + standardise amounts, PAY codes clipped to [−2,3] and one-hot (49 inputs). HGB: raw values (19 inputs) | — | `app/features.py::build_logistic/build_hgb`; `development.json → preprocessing_version` | — |
| Leakage controls | all learned preprocessing inside sklearn Pipelines fitted on train; hyperparameters by 5-fold CV **inside train**; calibration map fitted on calibration rows; champion rule applied on tuning; selection hashed before final access; leakage canary test (shuffled labels + target copy → held-out AUC must stay < 0.62) | — | `app/study.py::run_development_study`; tests in §0.1 | 35 near-duplicate rows could sit in different partitions (distinct IDs) |
| Result provenance | Development results: tuning partition (n = 4,500). Audit results: final holdout (n = 4,500), evaluated once at 2026-11-18T16:38Z. Stress: perturbed copies of tuning (n = 4,500 each). Drift: SIMULATED batches of 1,500 from tuning. Tests: synthetic fixtures | as stated | file names above | — |

---

## 2. Models and results

### 2.1 Models and tuning
| Claim | Exact value | Population | Evidence | Limitation |
|---|---|---|---|---|
| Baseline | L2 logistic regression; grid C ∈ {0.001, 0.01, 0.1, 1, 10}; selected **C = 0.1** (CV log loss 0.43566 vs untuned C = 1: 0.43574); search 12.5 s, refit 0.65 s | 5-fold CV inside train (18,000) | `development.json → models.baseline`; `models` table | tuning made no measurable difference; on tuning the untuned model was marginally better (log loss 0.43057 vs 0.43080) |
| Challenger | sklearn HistGradientBoosting; 12 random configs + sklearn default; selected **lr 0.03, 200 trees, 15 leaves, min leaf 50, L2 1.0**; CV log loss 0.42962 vs untuned 0.43646; search 109.7 s, refit 0.95 s | 5-fold CV inside train | `development.json → models.challenger` | single random-search budget; one seed |
| Class imbalance | no weighting or resampling | — | policy `models.class_weighting` | — |

### 2.2 Performance (calibrated probabilities)
| Metric | LR tuning | HGB tuning | LR final | HGB final |
|---|---|---|---|---|
| ROC-AUC | 0.7753 | 0.7853 | **0.7657** | **0.7819** |
| Average precision (step-wise, not trapezoidal PR-AUC) | 0.5531 | 0.5822 | 0.5302 | 0.5561 |
| Brier score | 0.1345 | 0.1316 | 0.1378 | 0.1342 |
| Log loss | 0.4308 | 0.4218 | 0.4396 | 0.4276 |
| ECE (10 quantile bins) | 0.0132 | 0.0174 | 0.0162 | 0.0114 |
| Calibration slope | 0.989 | 1.025 | 0.943 | 0.984 |
| Obs − mean predicted | −0.0018 | +0.0004 | +0.0018 | +0.0012 |

Populations: tuning n = 4,500, final n = 4,500 (prevalence 0.2211 each).
Evidence: `development.json → evaluation.*.calibrated`; `final.json → evaluation`.

### 2.3 Calibration before/after (on data separate from the calibration fit)
| Model | Platt map (fitted on 3,000 calibration rows) | Tuning log loss none → sigmoid | Tuning Brier none → sigmoid | Decision |
|---|---|---|---|---|
| LR | a = 1.0355, b = 0.0548 | 0.43080 → 0.43078 | 0.13451 → 0.13449 | sigmoid (rule: keep unless tuning log loss rises) |
| HGB | a = 1.0416, b = 0.0567 | 0.42215 → 0.42184 | 0.13168 → 0.13158 | sigmoid |

Evidence: `development.json → models.*.calibration`, `evaluation.*.uncalibrated_metrics`.
**Limitation:** the effect is negligible (≤ 0.0003 log loss); the models were
already nearly calibrated. Uncalibrated final-holdout metrics were **not
measured** (only the frozen calibrated models were evaluated there).
Calibration-partition losses are in-sample and are not used as evidence.

### 2.4 Paired differences (challenger − baseline; 1,000 paired bootstrap resamples, 0 failed)
| Metric | Tuning difference [95% CI] | Final difference [95% CI] | Share of final resamples favouring challenger |
|---|---|---|---|
| ROC-AUC | +0.0099 [+0.0026, +0.0167] | **+0.0162 [+0.0087, +0.0240]** | 1.000 |
| Average precision | +0.0290 [+0.0158, +0.0414] | +0.0259 [+0.0126, +0.0386] | 1.000 |
| Log loss | −0.0089 [−0.0129, −0.0049] | −0.0120 [−0.0161, −0.0080] | 1.000 |
| Brier | −0.0029 [−0.0042, −0.0015] | −0.0036 [−0.0050, −0.0022] | 1.000 |

Evidence: `development.json → paired_bootstrap_tuning`; `final.json → paired_bootstrap_final`; `app/bootstrap.py::paired_bootstrap`.
Express AUC differences in **points** (e.g. "0.016 higher"), never as a
percentage. Limitation: percentile intervals conditional on the fitted models
and this sample; they exclude retraining variability. Repeated 5×2 CV on
train+tuning (uncalibrated): AUC difference mean +0.0092 (sd 0.0029, range
0.0034–0.0122); challenger lower log loss in 10/10 folds (descriptive; folds overlap).

**Which model is better on what:** the challenger is better on all four
metrics on both tuning and final. Calibration quality is similar: HGB had the
lower ECE on final, LR on tuning; LR's final slope of 0.943 indicates mild
over-confidence.

### 2.5 Model-selection decision
**Recorded: yes.** `selection.json` (SHA `9510bcd4…`, frozen 2026-11-18T16:36:20Z,
about 2 minutes before the single final access at 16:38:08Z). Rule (pre-registered):
choose the challenger only if the upper bound of the paired 95% CI of the
tuning log-loss difference is below 0. Observed CI [−0.0129, −0.0049] → **champion =
challenger**. The final access log entry references the same selection hash.
**Limitation:** a margin of only 0.0049 log loss separated the decision from
retaining the baseline, and the decision was taken by a rule, not by a human
reviewer (no human review recorded).

### 2.6 Most important negative findings and trade-offs
1. **Weak ranking within delinquency groups.** Final holdout, customers with PAY_0 ≥ 2 (n = 472, 333 defaults): AUC 0.537 [0.478, 0.591] LR and 0.575 [0.526, 0.628] HGB, against 0.766 / 0.782 overall. Most discriminative power comes from separating repayment groups, not from ranking within them.
2. **Low recall at the illustrative threshold.** At p ≥ 0.5 on final, HGB flags 12.2% of customers, with precision 0.677 and recall 0.373 (624 of 995 defaulters missed). At 0.2, recall is 0.656 with precision 0.436.
3. **The gain is modest** (+0.016 AUC points), and hyperparameter search added little (none for LR).
4. **Interpretability vs performance:** the better model is the less interpretable one.

---

## 3. Robustness and explainability

### 3.1 Stress scenarios executed (tuning partition, n = 4,500 each, one seed each 99–108)
Clean reference: LR AUC 0.7753 / log loss 0.4308; HGB AUC 0.7853 / log loss 0.4218.
Δ = absolute change in metric units; "rel." = relative change in log loss (Δ ÷ clean value).

| Scenario | Kind | Severity | LR ΔAUC | HGB ΔAUC | LR Δlog loss (rel.) | HGB Δlog loss (rel.) |
|---|---|---|---|---|---|---|
| missing_bill_10 | input corruption | 10% of BILL_AMT cells missing (2,656 cells) | +0.0005 | −0.0001 | +0.0000 (0.0%) | +0.0001 (0.0%) |
| missing_pay_status_10 | input corruption | 10% of PAY_* cells missing (2,730) | −0.0079 | −0.0061 | +0.0083 (+1.9%) | +0.0069 (+1.6%) |
| missing_amounts_30 | input corruption | 30% of amount cells missing (16,146) | −0.0049 | −0.0041 | +0.0031 (+0.7%) | +0.0025 (+0.6%) |
| **limit_in_thousands_10** | input corruption | 10% of LIMIT_BAL ÷ 1000 (469 rows below train minimum) | **−0.0182** | **−0.0041** | **+0.0225 (+5.2%)** | +0.0024 (+0.6%) |
| amount_noise_10pct | input corruption | lognormal σ = 0.10 on amounts | −0.0001 | −0.0001 | +0.0000 (0.0%) | −0.0002 (−0.1%) |
| pay0_reporting_plus1_20 | input corruption | 20% of delinquent PAY_0 +1 | −0.0016 | −0.0011 | +0.0054 (+1.3%) | +0.0030 (+0.7%) |
| composition_delinquent_x2 | composition | PAY_0 ≥ 1 weighted ×2 (prevalence 27.7%) | +0.0483 | +0.0509 | +0.0099 (+2.3%) | +0.0060 (+1.4%) |
| composition_young_x2 | composition | age < 30 weighted ×2 | −0.0028 | +0.0033 | +0.0040 (+0.9%) | +0.0033 (+0.8%) |
| prevalence_10pct | composition | class-conditional resample to 10% | +0.0007 | +0.0046 | −0.1202 (−27.9%) | −0.1187 (−28.1%) |
| prevalence_35pct | composition | resample to 35% | +0.0064 | +0.0072 | +0.1229 (+28.5%) | +0.1200 (+28.4%) |

Evidence: `stress.json`; `app/stress.py::apply_scenario/run_stress_suite`.
**Worst scenario:** credit-limit unit error for LR (ΔAUC −0.018). HGB's worst was
missing repayment status (−0.006). Neither crossed the 0.05 review limit.
**Limitations:** one seed per scenario, no confidence intervals on the deltas
(not measured). Input-corruption scenarios hold outcomes fixed: robustness
probes, not economic stress. Composition scenarios re-use historical rows; the
AUC *rise* under delinquent oversampling shows AUC depends on case mix, not that
the model improved. The log-loss swings under prevalence shift are mostly a
base-rate effect (outcome entropy changes), not only miscalibration. The
in-evidence note claiming the validator flags the unit error is wrong (§0.2).

### 3.2 Weakest segments (final holdout; 200 within-slice bootstrap resamples)
| Segment | n | Defaults | LR AUC [CI] | HGB AUC [CI] | Note |
|---|---|---|---|---|---|
| PAY_0 ≥ 2 | 472 | 333 | 0.537 [0.478, 0.591] | 0.575 [0.526, 0.628] | adequate support; weakest meaningful segment |
| PAY_0 = 1 | 492 | 157 | 0.613 [0.563, 0.663] | 0.660 [0.607, 0.704] | |
| 0 delayed months | 3,062 | 376 | 0.622 [0.589, 0.652] | 0.670 [0.631, 0.698] | |
| High school (EDUCATION 3) | 748 | 193 | 0.726 [0.685, 0.769] | 0.729 [0.690, 0.769] | demographic audit slice |
| EDUCATION undocumented code | 41 | 3 | 0.228 [0.000, 0.575] | 0.263 [0.100, 0.443] | below support/event minimums: not interpretable |
| MARRIAGE code 0 | 5 | 0 | unavailable (single class) | unavailable | shown as unavailable, not 0 |

Evidence: `slices_final.json` (tuning equivalents in `slices_tuning.json`, e.g.
PAY_0 ≥ 2: n = 466, AUC 0.570 / 0.634). The challenger's AUC point estimate was
≥ the baseline's in every supported final segment. **Limitations:** descriptive, no
multiplicity correction, not a fairness assessment; low within-group AUC is
partly range restriction (PAY_0 is constant-ish inside the group).

### 3.3 Simulated shift (not observed deployment drift)
All batches are labelled **SIMULATED**, 1,500 rows each, drawn from the tuning
partition; PSI reference = train. No real monitoring data exists.

| Batch | Construction | Max feature PSI | HGB AUC | HGB mean predicted vs observed |
|---|---|---|---|---|
| S1 | baseline sampling | 0.015 (LIMIT_BAL) | 0.790 | 0.219 vs 0.229 |
| S5 | PAY_0 ≥ 1 oversampled ×3 | **0.262 (PAY_0, review)** | 0.844 | 0.307 vs 0.303 |
| S6 | 40% of PAY_0 ≥ 2 defaults relabelled (concept drift) | **0.020 (stable)** | **0.717** | 0.221 vs 0.195 |

Evidence: `drift.json`; `app/drift.py::simulated_batches/psi`. Finding: PSI
detected the covariate shift (S5) but not the simulated concept drift (S6),
which was visible only through labelled performance (LR AUC 0.707). Train-vs-tuning
sanity PSI ≤ 0.0043. **Limitation:** synthetic perturbations of historical data;
no claim about real drift.

### 3.4 Explainability (SHAP)
| Claim | Value | Population | Evidence | Limitation |
|---|---|---|---|---|
| Implementation | LR: `shap.LinearExplainer` (Independent masker); HGB: `shap.TreeExplainer`, interventional; output = calibrated log-odds (raw SHAP × Platt slope a) | 500 tuning customers explained; background 200 train customers | `explanations.json`; `app/explain.py::explain_model` | not additive in probability space |
| Additivity verified | max error 8.9e−16 (LR), 2.6e−8 (HGB); tolerance 1e−6 | same 500 | `explanations.json → models.*.additivity_*` | — |
| Top drivers | LR: PAY_0 (mean abs SHAP 0.567), LIMIT_BAL 0.170, PAY_AMT2, PAY_3; HGB: PAY_0 0.411, LIMIT_BAL 0.184, BILL_AMT1 0.147, PAY_3, PAY_2 | same 500 | `global` arrays | attribution of the model, not causes; correlated features share credit |
| Stability | 5 refits on bootstrap resamples of train: Spearman rank correlation 0.86–0.93 (LR), 0.83–0.90 (HGB); top-5 overlap 4/5 (LR, every refit), 3–4/5 (HGB) | same 500 | `explanations.json → stability` | only 5 refits; background fixed |
| Library issue found | interventional TreeSHAP (shap 0.51.0 + sklearn 1.9.1) violates additivity on non-integer inputs; additive on this integer data | synthetic test | `tests/test_modelling.py::test_hgb_interventional_condition`; DECISIONS D-009 | root cause inside shap not established |

---

## 4. Engineering and review

| Claim | Exact finding | Evidence | Limitation |
|---|---|---|---|
| Working API | FastAPI endpoints for status, data quality, upload validation, jobs, evidence, local explanations, cost scenarios, monitoring batches, review, Markdown/JSON reports | `app/main.py`; `tests/test_api_integration.py` (11 passed) | local, no auth, binds 127.0.0.1 |
| Working UI | React/TypeScript dashboard, 8 sections; full flow (start study → review → final audit → finding → accept) driven in Edge; no horizontal overflow at 390 px | `tests/test_browser.py` (2 passed, synthetic data) | Edge only; accessibility not formally tested |
| Versioning | study identity = hash(dataset SHA, policy SHA `bf56d020…`, code-source SHA `58abb56f…`); stale studies cannot reach a final audit or acceptance (the earlier check study is correctly shown stale: "backend code changed") | `app/study.py::current_identity/stale_reasons`; `test_policy_change_makes_study_stale_and_new_identity` | no git revision recorded |
| Evidence integrity | evidence and model artifacts hashed at write, verified at read; tampered model refused (500) | `test_evidence_tampering_detected`, `test_tampered_model_artifact_is_refused` | detects changes; not tamper-proof |
| Holdout control | final access refused without confirmation, after a previous access, before frozen selection, or when stale; every attempt logged | `test_final_holdout_requires_confirmation_and_is_one_shot`; real log: 1 granted | local lock; files readable directly |
| Review controls | state machine; confirmation + ≥20-char rationale; stale version, stale evidence and duplicate request_id → 409; acceptance blocked by open high/critical findings or a missing final audit; append-only triggers; hash-chained events | `tests/test_review_registry.py`; real chain verified ok (3 events) | reviewer identity self-entered (`identity_verified = 0`) |
| Validation findings recorded | **none on the official study** (0 findings, no human events) | registry query | must be done by you |
| Deterministic checks | for the official study: flags for 4 low-support slices and 14 slice/model pairs with AUC CI below overall; all calibration, stress, SHAP and final-vs-dev checks "ok" | `GET /api/studies/{id}/review` | flags are prompts, not decisions |

---

## A. Five strongest verified facts (model-risk internship)

1. **Leakage-controlled, pre-registered protocol on real data.** 30,000 customers were split into 18,000/4,500/3,000/4,500 for train, tuning, calibration and final, with hashes fixed before modelling. Preprocessing and hyperparameters were fitted on train only, and a leakage-canary test enforces it.
2. **Decision before audit.** The challenger was selected by a pre-registered paired-bootstrap rule on development data (log-loss difference CI [−0.0129, −0.0049]). The selection record was hashed before the single logged final-holdout access.
3. **Quantified, paired comparison on the holdout.** ROC-AUC 0.782 vs 0.766, a difference of +0.016 points (95% paired bootstrap CI 0.009–0.024, n = 4,500). Log loss 0.428 vs 0.440. Both models well calibrated (ECE ≤ 0.016).
4. **Weaknesses surfaced, not hidden.**
   - Within already-delinquent customers (PAY_0 ≥ 2, n = 472), AUC falls to 0.54–0.58.
   - A 10% credit-limit unit error costs logistic regression 0.018 AUC versus 0.004 for boosting.
   - PSI missed a simulated concept drift that cut AUC to about 0.71.
5. **Auditable controls verified by tests.** The tests cover:
   - hash-verified evidence and model artifacts;
   - stale-study detection;
   - review decisions rejected when stale or duplicated;
   - append-only, hash-chained events.

   73 automated tests pass (70 synthetic, 1 official-data, 2 real-browser).

---

## B. Resume bullets (≈25–35 words)

1. Directed an AI-assisted build of a credit-default model-validation prototype on UCI’s 30,000-customer dataset, comparing logistic regression with gradient boosting under a pre-registered protocol with separate tuning, calibration and holdout partitions.
2. On a single 4,500-customer holdout evaluation, gradient boosting ranked defaulters slightly better than logistic regression (ROC-AUC 0.782 vs 0.766; paired-bootstrap difference +0.016, 95% CI 0.009–0.024), consistent with the pre-registered development-data selection.
3. Verified reproducibility and review controls—hashed data, policy and code versions, a logged single holdout evaluation, and human decisions rejected when stale or duplicated—through 73 passing automated tests, including real-browser tests.

*Alternative bullet 3 (robustness instead of controls):* Ran seeded stress tests showing a 10% credit-limit unit-entry error reduced logistic-regression ROC-AUC by 0.018 versus 0.004 for gradient boosting; both models ranked weakly among customers already two-plus months late.
(This has two findings in one bullet; split it if space allows.)

---

## C. Defending each bullet

### Bullet 1: problem, dataset, approach
**You must be able to demonstrate:** open `configs/validation_policy.json` and
explain each partition's job; show `split_manifest.json` counts; explain why
tuning was done by CV inside train; show `test_target_cannot_leak_into_features`
and explain what it would catch; state clearly what the AI built and what you specified.

**Technical reasoning:** separate partitions prevent the same data being used
to fit, select and judge. Calibration needs its own data because a map fitted
on evaluation rows looks perfect there. A pre-registered rule stops
"choosing the metric after seeing results". Rows aren't chronological, so this
is an in-distribution study only.

**Likely follow-ups:**
1. *Why not just train/test?* — Each extra choice (hyperparameters, calibration, champion) needs data that the final estimate never sees. Otherwise the final number is optimistic. Here hyperparameters used CV inside the 18,000 train rows, calibration used 3,000 separate rows, and selection used the 4,500 tuning rows.
2. *Can this show performance on future customers?* — No. It is a 2005 cross-sectional sample without chronology. The holdout measures performance on other customers from the same population; out-of-time validation would need dated cohorts.
3. *Why exclude demographics, and does that make it fair?* — They were excluded by policy and used only for audit slices. Exclusion doesn't guarantee fairness, because proxies such as credit limit remain. The slice results are descriptive, not a fairness test.

**Overstatement risks:** "built", "designed and implemented" (the code was AI-written); "bank", "deployed", "production"; "validated model" (no human review decision exists); "real-time monitoring".

### Bullet 2: finding and decision
**You must be able to demonstrate:** read the numbers from `final.json` or the
dashboard; explain paired bootstrap and why the CI excludes 0; show
`selection.json` and the access-log timestamp ordering; explain why the
log-loss rule, not AUC, made the selection.

**Technical reasoning:** pairing resamples the same customers for both models,
so correlated errors cancel and the difference CI is tight. AUC measures
ranking only, while log loss scores the probabilities, which is what the model
outputs. The final audit is confirmatory, so it can't change the decision.

**Likely follow-ups:**
1. *Is +0.016 AUC meaningful?* — It's statistically clear on this sample (CI 0.009–0.024; 1,000/1,000 resamples favour boosting) but practically modest. At the 0.5 threshold recall only rises from 0.350 to 0.373. Whether it's worth the lost interpretability is a business judgement.
2. *How were probabilities checked?* — Sigmoid calibration was fitted on 3,000 separate rows and evaluated on tuning and final. ECE was 0.011 (HGB) and 0.016 (LR) on final, with calibration slopes 0.984 and 0.943. Calibration changed log loss by at most 0.0003, because the models were already close.
3. *What doesn't the interval cover?* — Retraining variability, other populations and time. It is conditional on the fitted models and this 4,500-customer sample. Repeated CV gives a rough retraining view (AUC difference sd 0.0029), and that view overlaps.

**Overstatement risks:** "significantly improved", "1.6% better" or any percentage for AUC points; "outperformed on all segments" (several segments are too small to judge); "the model was approved".

### Bullet 3: controls and reproducibility
**You must be able to demonstrate:** run `python -m pytest -q` and
`python -m pytest -m browser` yourself; show a stale-version or
duplicate-request 409 (guide Exercise 12); show `verify_event_chain` detecting
an edited event; explain `current_identity` and why a policy edit makes a study
stale; explain why joblib artifacts are only loaded when their hash matches.

**Technical reasoning:** version identities tie each result to the exact
data, configuration and code. Optimistic concurrency (expected version and
evidence hash) stops a reviewer approving evidence that changed after they read
it. Hash chains make silent edits detectable. Pickles execute code on load, so
only trusted, hash-verified local artifacts are loaded.

**Likely follow-ups:**
1. *Is the audit trail tamper-proof?* — No. SQLite triggers and a hash chain detect accidental or casual edits. Anyone controlling the database file can drop the triggers and rewrite the chain. Reviewer names are self-entered and unauthenticated.
2. *How do you know results are reproducible?* — Seeds, pinned dependencies (`requirements-lock.txt`) and identity hashes are in place, and tests check determinism of splits, bootstrap and stress. A from-scratch re-run of the official development study has **not** been done yet (see D).
3. *What did the tests actually catch?* — An ordering bug in the holdout guard: a second access was refused, but logged with the wrong reason. Also a header-mapping bug for unlabelled batches, and a 176-pixel horizontal overflow at phone width. The fixes went into the code; no test was weakened.

**Overstatement risks:** "tamper-proof", "secure", "audited", "independent validation", "authentication"; implying you wrote the tests yourself. "Verified" is only safe once you have run them (D1).

---

## D. Smallest remaining checks before using the bullets

1. **Run the tests yourself:** `.\.venv\Scripts\python.exe -m pytest -q` and `-m browser`. Expected: 71 + 2 passed. No holdout access is involved. Required before "Verified" in bullet 3.
2. **Read the numbers yourself** from `docs/EVALUATION_REPORT.md` or the dashboard. Reading saved results does not re-access the holdout.
3. **Reproduce the development study in a separate registry.** This confirms the tuning-partition numbers and the champion. Set `MODELSCOPE_ARTIFACT_DIR` and `MODELSCOPE_DB` to an empty temp folder, then run `python scripts/run_study.py development --actor "<you>"` (about 3 minutes). In the real registry the same command returns "Cached" and does not retrain. This uses train, tuning and calibration rows only.
   **Do not** run `final` in that temp registry unless you accept the consequence: it is a second evaluation of the same holdout rows. The numbers would only repeat the audit, but it would not appear in the original access log. It weakens the "evaluated once" statement and must then be disclosed as a reproducibility re-run. The safest course is to skip it.
4. **Make the review decision yourself** in the Review workspace. Record at least the within-delinquent weakness (R8) as a finding. None of the bullets claims a decision, but interviewers will ask, and "human review controls" are stronger once you have used them.
5. **Put the folder under git** before further changes, so later identities carry a revision.
6. **Prepare the disclosure sentence:** "Implemented with an AI coding assistant from my specification; I verified it by …". List only checks you have actually done.
7. **Optional, for the robustness bullet:** understand and be ready to state the R11 defect. The stress note wrongly claims the validator catches unit errors.
