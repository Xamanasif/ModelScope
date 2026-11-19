<!-- Generated from registered evidence by GET /api/studies/study-20261118T163406-a420d0fc/report.md on 2026-11-18. Do not hand-edit numbers. -->
# ModelScope validation report — study-20261118T163406-a420d0fc

> Local research prototype. Not a production deployment, bank approval, regulatory compliance statement or independent organisational validation.

- Status: **final_evaluated**; review state: **evaluated**
- Identity: dataset `45bcf4df62ff2e23…`, policy `bf56d02019e5d485…`, code `58abb56fa05e8275…`, git `not a git checkout`
- Stale: no
- Split manifest: `a35d7f50d563f9e195542264fc9546f7b010af0c3430efae58f28b165933e37e`

## Data quality
| check | status | detail |
|---|---|---|
| schema | pass | 25 expected columns present, none unexpected |
| row_count | pass | 30000 rows |
| numeric_integer_types | pass | all columns numeric and integer-valued |
| missing_values | pass | no missing cells |
| unique_customer_ids | pass | 30000 unique IDs |
| target_binary | pass | positive class = 1 (source: 'default payment (Yes = 1, No = 0)'); invalid values=0; prevalence=0.2212 |
| domain_SEX | pass | all values documented |
| domain_EDUCATION | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {5: 280, 6: 51, 0: 14} |
| domain_MARRIAGE | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 54} |
| domain_PAY_0 | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 14737, -2: 2759} |
| domain_PAY_2 | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 15730, -2: 3782} |
| domain_PAY_3 | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 15764, -2: 4085} |
| domain_PAY_4 | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 16455, -2: 4348} |
| domain_PAY_5 | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 16947, -2: 4546} |
| domain_PAY_6 | warn | undocumented codes present (kept as distinct codes, not reinterpreted): {0: 16286, -2: 4895} |
| domain_AGE | pass | 0 values outside (18, 100); observed 21-79 |
| domain_LIMIT_BAL | pass | 0 non-positive credit limits |
| domain_PAY_AMT | pass | 0 negative payment amounts |
| domain_BILL_AMT_negative | info | 3932 negative bill-statement cells; plausible credit balances, kept as-is |
| duplicate_feature_rows | info | 35 rows identical to another row except ID (including target). IDs are distinct so they are treated as distinct customers; cannot be confirmed from the source. |

## Development evidence (tuning partition)
Partition sizes: {'train': 18000, 'tuning': 4500, 'calibration': 3000, 'final': 4500}; prevalence: train 0.2212, tuning 0.2211, calibration 0.2213, final 0.2211

| model | selected params | calibration | AUC | AP | log loss | Brier | ECE | CITL | slope |
|---|---|---|---|---|---|---|---|---|---|
| baseline (logistic_regression) | `{"C": 0.1}` | sigmoid (a=1.036, b=0.055) | 0.7753 | 0.5531 | 0.4308 | 0.1345 | 0.0132 | -0.0018 | 0.989 |
| challenger (hist_gradient_boosting) | `{"l2_regularization": 1.0, "learning_rate": 0.03, "max_iter": 200, "max_leaf_nodes": 15, "min_samples_leaf": 50}` | sigmoid (a=1.042, b=0.057) | 0.7853 | 0.5822 | 0.4218 | 0.1316 | 0.0174 | 0.0004 | 1.025 |

Untuned vs selected (uncalibrated, tuning partition):

| model | untuned AUC | untuned log loss | selected AUC | selected log loss | search s | configs |
|---|---|---|---|---|---|---|
| baseline | 0.7751 | 0.4306 | 0.7753 | 0.4308 | 12.48 | 5 |
| challenger | 0.7844 | 0.4241 | 0.7853 | 0.4221 | 109.73 | 13 |

Paired bootstrap (tuning; 1000 used, 0 failed resamples), challenger − baseline:

| metric | baseline | challenger | difference | 95% CI |
|---|---|---|---|---|
| roc_auc | 0.7753 | 0.7853 | 0.0099 | [0.0026, 0.0167] |
| average_precision | 0.5531 | 0.5822 | 0.0290 | [0.0158, 0.0414] |
| log_loss | 0.4308 | 0.4218 | -0.0089 | [-0.0129, -0.0049] |
| brier | 0.1345 | 0.1316 | -0.0029 | [-0.0042, -0.0015] |

Repeated development CV (5-fold x 2 repeats on train+tuning (uncalibrated, selected params)): AUC difference mean 0.0092 (sd 0.0029); challenger better on log loss in 10/10 folds.

## Frozen selection
- Rule: Select the challenger only if the upper bound of the paired 95% bootstrap CI for (challenger log loss - baseline log loss) on the tuning partition is below 0; otherwise retain the interpretable baseline.
- Log-loss difference CI: [-0.0129, -0.0049] → champion **challenger** (hist_gradient_boosting)
- Frozen at 2026-11-18T16:36:20+00:00; threshold 0.5 (illustrative)

## Final holdout audit (evaluated once)
Accessed by Claude Code (AI-assisted run) at 2026-11-18T16:38:13+00:00; n=4500, prevalence 0.2211

| model | metric | tuning | final |
|---|---|---|---|
| baseline | roc_auc | 0.7753 | 0.7657 |
| baseline | average_precision | 0.5531 | 0.5302 |
| baseline | log_loss | 0.4308 | 0.4396 |
| baseline | brier | 0.1345 | 0.1378 |
| challenger | roc_auc | 0.7853 | 0.7819 |
| challenger | average_precision | 0.5822 | 0.5561 |
| challenger | log_loss | 0.4218 | 0.4276 |
| challenger | brier | 0.1316 | 0.1342 |

| metric | difference (challenger − baseline) | 95% paired CI |
|---|---|---|
| roc_auc | 0.0162 | [0.0087, 0.0240] |
| average_precision | 0.0259 | [0.0126, 0.0386] |
| log_loss | -0.0120 | [-0.0161, -0.0080] |
| brier | -0.0036 | [-0.0050, -0.0022] |

- baseline @ 0.5: precision 0.6641, recall 0.3497, specificity 0.9498, CM {'tn': 3329, 'fp': 176, 'fn': 647, 'tp': 348}
- challenger @ 0.5: precision 0.6770, recall 0.3729, specificity 0.9495, CM {'tn': 3328, 'fp': 177, 'fn': 624, 'tp': 371}

## Stress suite (tuning partition)
Input-corruption scenarios hold outcomes fixed: robustness probes, not causal simulations of economic stress. Composition scenarios resample historical rows.

| scenario | kind | baseline ΔAUC | baseline Δlog loss | challenger ΔAUC | challenger Δlog loss |
|---|---|---|---|---|---|
| missing_bill_10 | input_corruption | +0.0005 | +0.0000 | -0.0001 | +0.0001 |
| missing_pay_status_10 | input_corruption | -0.0079 | +0.0083 | -0.0061 | +0.0069 |
| missing_amounts_30 | input_corruption | -0.0049 | +0.0031 | -0.0041 | +0.0025 |
| limit_in_thousands_10 | input_corruption | -0.0182 | +0.0225 | -0.0041 | +0.0024 |
| amount_noise_10pct | input_corruption | -0.0001 | +0.0000 | -0.0001 | -0.0002 |
| pay0_reporting_plus1_20 | input_corruption | -0.0016 | +0.0054 | -0.0011 | +0.0030 |
| composition_delinquent_x2 | composition | +0.0483 | +0.0099 | +0.0509 | +0.0060 |
| composition_young_x2 | composition | -0.0028 | +0.0040 | +0.0033 | +0.0033 |
| prevalence_10pct | composition | +0.0007 | -0.1202 | +0.0046 | -0.1187 |
| prevalence_35pct | composition | +0.0064 | +0.1229 | +0.0072 | +0.1200 |

## Explanations (SHAP, development sample)
- baseline: LinearExplainer (Independent masker, interventional); additivity ok=True (max err 8.9e-16); top mean|SHAP|: PAY_0 (0.567), LIMIT_BAL (0.170), PAY_AMT2 (0.155), PAY_3 (0.153), PAY_AMT1 (0.140), PAY_AMT3 (0.133)
- challenger: TreeExplainer (interventional, train background); additivity ok=True (max err 2.6e-08); top mean|SHAP|: PAY_0 (0.411), LIMIT_BAL (0.184), BILL_AMT1 (0.147), PAY_3 (0.118), PAY_2 (0.116), PAY_AMT1 (0.105)
  - SHAP attributes the model's output, not real-world causes.
  - Correlated features (BILL_AMT1..6, PAY_0..PAY_6) can share or swap attribution.
  - Values depend on the background sample (200 train rows); a different background gives different values.
  - Explanations are not a fairness certificate.
  - Additive in calibrated log-odds, not in probability.

## Deterministic checks (flags for a human; never decisions)
- [ok] study_identity_current: Dataset, policy and code identity match
- [ok] citl_baseline: baseline: observed - mean predicted on tuning = -0.0018 (limit ±0.02)
- [ok] ece_baseline: baseline: ECE (10 quantile bins) on tuning = 0.0132 (limit 0.03)
- [ok] citl_challenger: challenger: observed - mean predicted on tuning = +0.0004 (limit ±0.02)
- [ok] ece_challenger: challenger: ECE (10 quantile bins) on tuning = 0.0174 (limit 0.03)
- [ok] stress_baseline: baseline: largest AUC drop 0.0182 under 'limit_in_thousands_10' (limit 0.05)
- [ok] stress_challenger: challenger: largest AUC drop 0.0061 under 'missing_pay_status_10' (limit 0.05)
- [flag] slice_support: 4 slice(s) below support/event minimums: EDUCATION=others, EDUCATION=undocumented_code, MARRIAGE=others, MARRIAGE=undocumented_code
- [flag] slice_auc_below_overall: 14 slice/model pair(s) with AUC CI wholly below the overall AUC (expected for slices that restrict the range of PAY_0): credit_limit_band=Q1 <= 50,000 (baseline), credit_limit_band=Q1 <= 50,000 (challenger), recent_repayment_status=PAY_0 <= 0 (no delay code) (baseline), recent_repayment_status=PAY_0 <= 0 (no delay code) (challenger) …
- [ok] shap_additivity_baseline: baseline: max additivity error 8.88e-16
- [ok] shap_additivity_challenger: challenger: max additivity error 2.55e-08
- [ok] final_vs_dev_baseline: baseline: AUC tuning 0.7753 vs final 0.7657
- [ok] final_vs_dev_challenger: challenger: AUC tuning 0.7853 vs final 0.7819

## Findings
None recorded.

## Review events (append-only; actor names are self-entered, not authenticated)
Hash chain intact: True (3 events). A local SQLite file is not tamper-proof.

- 2026-11-18T16:34:06+00:00 system (system): case_created None → draft
- 2026-11-18T16:37:14+00:00 system (system): evidence_complete draft → evaluated
- 2026-11-18T16:38:18+00:00 system (system): evidence_changed evaluated → evaluated

## Limitations
- Cross-sectional 2005 Taiwan sample; no out-of-time or real deployment-drift evidence.
- Bootstrap intervals are conditional on the fitted models and evaluation sample.
- Slice results are descriptive; not a legal fairness assessment.
- Stress and drift batches are simulations derived from the same historical sample.
- No expected-credit-loss, IFRS 9 or capital conclusions can be drawn from this dataset.
