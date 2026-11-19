# Experiment protocol (pre-registered)

Machine-readable version: [`configs/validation_policy.json`](../configs/validation_policy.json),
policy version 1.0.0, SHA-256 of canonical JSON
`bf56d02019e5d485…` (shown in full on the dashboard's Reports & runs page).
The policy JSON was written before any model was fitted on the official data
and has not changed since (every study records its hash). The only data
operations before registration were the schema/domain profile and computing
the split manifest hash (no model, no metric). This prose version was written
afterwards from that JSON and adds no rule the JSON does not contain.

## 1. Question
On UCI 350, does a gradient-boosting challenger that ranks better than a
logistic-regression baseline also give (a) reliable probabilities and (b)
stable performance across pre-specified customer segments and stress probes?

## 2. Data and target
- Source file SHA-256 `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0`
  (30,000 rows). A mismatch is refused.
- Target `DEFAULT_NEXT_MONTH` (source `Y`), positive label 1 = default.
- `ID` is never a predictor; it is used for the split only.

## 3. Splits
| Partition | Share | Customers | Used for |
|---|---|---|---|
| train | 60% | 18,000 | preprocessing fit, hyperparameter CV, model fit, slice band edges, SHAP background, drift reference |
| tuning | 15% | 4,500 | calibration-method choice, champion rule, development metrics, bootstrap, slices, stress, simulated drift, SHAP explain sample |
| calibration | 10% | 3,000 | fitting the sigmoid map only |
| final | 15% | 4,500 | one-shot audit after selection is frozen |

Stratified on the target, seed 42, manifest SHA-256
`a35d7f50d563f9e195542264fc9546f7b010af0c3430efae58f28b165933e37e`.
Changing seed or fractions changes the hash and makes all existing artifacts
incompatible (a new study). Repeated development CV uses train+tuning only.

**Temporal limit:** the sample is cross-sectional (Taiwan, April–September
2005). Neither out-of-time generalisation nor real deployment drift can be
established from it.

## 4. Features
19 predictors by default: `LIMIT_BAL`, `PAY_0, PAY_2..PAY_6`,
`BILL_AMT1..6`, `PAY_AMT1..6`. Demographics are excluded (configurable).
- Logistic regression: median imputation → signed log1p → standardisation for
  amounts; repayment codes clipped to [-2, 3] and one-hot encoded.
- HistGradientBoosting: raw values (missing values handled natively).
All learned preprocessing statistics are fitted inside the sklearn Pipeline on
train only.

## 5. Models and search
- Baseline: L2 logistic regression, C ∈ {0.001, 0.01, 0.1, 1, 10} (+ untuned C = 1).
- Challenger: HistGradientBoosting, 12 random configurations (seed 7) from
  learning_rate {0.03, 0.05, 0.1} × max_iter {100, 200, 400} × max_leaf_nodes
  {15, 31} × min_samples_leaf {20, 50, 100} × l2 {0, 1}, plus the sklearn-like
  untuned scikit-learn default configuration; early stopping off.
- Selection: lowest mean 5-fold stratified CV log loss inside train. No class
  weighting or resampling.

## 6. Calibration
Sigmoid fitted on the calibration partition; per model, sigmoid is kept unless
it raises tuning-partition log loss (then none). Calibration-partition losses
are in-sample for the calibrator and are not reported as quality estimates.

## 7. Champion rule (frozen)
Challenger is champion only if the upper bound of the paired 95% bootstrap CI
of (challenger − baseline) calibrated log loss on tuning is below 0; otherwise
the interpretable baseline stays. The selection record (`selection.json`) is
hashed before any final access.

## 8. Threshold
0.5 on calibrated probability ("more likely than not"): illustrative, not an
operational cut-off. Grid 0.1–0.7 reported. A cost threshold c_FP/(c_FP+c_FN)
is available as a user-supplied scenario only.

## 9. Uncertainty
Paired nonparametric percentile bootstrap over customers, 1,000 resamples,
seed 2026, 95%. One-class resamples are counted and excluded. Intervals are
conditional on the fitted models and the evaluation sample. Repeated 5-fold ×2
CV on train+tuning is descriptive.

## 10. Slices
Credit-limit quartiles (edges from train), recent repayment status (PAY_0),
six-month delay count, SEX, EDUCATION, MARRIAGE, AGE bands {18–29, 30–39,
40–49, 50+}. Warnings below 100 customers or 20 events; single-class slices
show AUC as unavailable. 200 within-slice bootstrap resamples. Descriptive, no
multiplicity correction, no fairness determination.

## 11. Stress, drift, explanations
Ten seeded scenarios (app/stress.py) on tuning; six SIMULATED batches; PSI
watch 0.10 / review 0.25 (review triggers, not regulatory limits). SHAP: 500
tuning rows, 200 train background rows, additivity tolerance 1e-6, 5
train-bootstrap refits for ranking stability.

## 12. Permitted after the final audit
Recording findings, documenting limitations, a human review decision. **Not
permitted:** changing models, features, calibration, thresholds, champion rule
or check limits in response to final results. A redesign is a new study needing
fresh unseen data, or must be labelled exploratory.

## 13. Holdout lock and its limits
The application refuses final access unless the development study is complete,
the selection record exists and verifies, the study is not stale, the user
confirms explicitly, and no final evaluation exists yet. Every attempt, granted
or refused, is appended to `holdout_access_log`. This is a local,
student-operated lock: anyone with file access can read `data/raw` directly. It
documents and discourages; it cannot prevent.

## 14. Execution record
- 2026-11-18: an earlier development-only check run
  (`study-20261118T093652-d13ae357`) was made while code was still changing; it
  is stale by code hash and its final holdout was never opened.
- 2026-11-18: evidence study `study-20261118T163406-a420d0fc` development
  complete (188 s); champion challenger; final audit run once at
  16:38:08 UTC (access log: 1 granted, 0 refused). Results:
  [EVALUATION_REPORT.md](EVALUATION_REPORT.md).
