# Model cards (study `study-20261118T163406-a420d0fc`)

Both models are research artifacts trained on UCI 350 for a validation
exercise. **Intended use:** studying validation methods. **Not intended for:**
credit decisions about real people, production scoring, regulatory submissions.
Numbers below come from `docs/EVALUATION_REPORT.md` (rendered from registered
evidence).

## Baseline: logistic regression
| | |
|---|---|
| Family | scikit-learn `LogisticRegression` (L2, lbfgs), `app/features.py::build_logistic` |
| Preprocessing | `lr-pre-v1`: median impute; signed log1p + standardise LIMIT_BAL, BILL_AMT*, PAY_AMT*; PAY_* clipped to [−2, 3] then one-hot (49 inputs) |
| Selected | C = 0.1 (grid 0.001–10, 5-fold CV log loss inside train; untuned C = 1 nearly identical) |
| Calibration | sigmoid, a = 1.036, b = 0.055 (fitted on 3,000 calibration rows; changes tuning log loss only in the 5th decimal) |
| Training time | search 12.5 s (5 configs × 5 folds), refit 0.65 s |
| Tuning (dev) | AUC 0.7753 · AP 0.5531 · log loss 0.4308 · Brier 0.1345 · ECE 0.0132 · slope 0.989 |
| Final audit | AUC 0.7657 · AP 0.5302 · log loss 0.4396 · Brier 0.1378 · ECE 0.0162 · slope 0.943 |
| At p ≥ 0.5 (final) | precision 0.664, recall 0.350, specificity 0.950 |
| Top SHAP features (tuning sample) | PAY_0, LIMIT_BAL, PAY_AMT2, PAY_3, PAY_AMT1, PAY_AMT3 |
| Weaknesses seen | sensitive to unit errors in LIMIT_BAL (ΔAUC −0.018 when 10% of limits are divided by 1000), since the log-linear term extrapolates; slightly over-confident on final (slope 0.943) |

## Challenger: histogram gradient boosting (champion under the frozen rule)
| | |
|---|---|
| Family | scikit-learn `HistGradientBoostingClassifier`, `app/features.py::build_hgb` |
| Preprocessing | `hgb-pre-v1`: raw values passthrough (19 inputs); NaN handled natively |
| Selected | learning_rate 0.03, max_iter 200, max_leaf_nodes 15, min_samples_leaf 50, l2 1.0 (13 configs incl. sklearn default) |
| Calibration | sigmoid, a = 1.042, b = 0.057 |
| Training time | search 109.7 s (13 configs × 5 folds), refit 0.95 s |
| Tuning (dev) | AUC 0.7853 · AP 0.5822 · log loss 0.4218 · Brier 0.1316 · ECE 0.0174 · slope 1.025 |
| Final audit | AUC 0.7819 · AP 0.5561 · log loss 0.4276 · Brier 0.1342 · ECE 0.0114 · slope 0.984 |
| At p ≥ 0.5 (final) | precision 0.677, recall 0.373, specificity 0.950 |
| Paired difference vs baseline (final) | AUC +0.0162 [0.0087, 0.0240]; log loss −0.0120 [−0.0161, −0.0080] |
| Top SHAP features | PAY_0, LIMIT_BAL, BILL_AMT1, PAY_3, PAY_2, PAY_AMT1 |
| Weaknesses seen | less interpretable; within-segment ranking for already-delinquent customers is weak (PAY_0 ≥ 2: final AUC 0.575); explanations are background-dependent |

## Shared caveats
- The advantage is real on this sample but modest (+0.016 AUC). Both models
  flag only ~12% of customers at 0.5 and miss ~63% of defaulters there; the
  threshold is illustrative.
- Demographics are not inputs, but proxies (e.g. credit limit) may carry
  demographic information. Slice results are descriptive only.
- No out-of-time, out-of-population or real-drift evidence exists.
