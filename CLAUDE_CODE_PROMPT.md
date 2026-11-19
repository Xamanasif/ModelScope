# Complete ModelScope and teach me the implementation

You are working in the ModelScope repository opened in VS Code. Complete this project as a rigorous, reproducible local research application, and prepare an unusually thorough learning guide for me. I am a student building understanding, not just requesting a demo. Read the existing files before editing. Preserve working foundations unless you document and justify a change.

## Objective

Build **ModelScope: Credit Risk Model Validation & Monitoring Platform**. A reviewer must be able to compare a logistic-regression baseline and a gradient-boosting challenger, examine calibration and failure modes, inspect reproducible evidence, record findings and make an explicitly human model-review decision. The central question is whether better ranking accuracy also produces reliable probabilities and stable performance across customer segments.

This is a portfolio research prototype relevant to model risk and AI/ML validation. Do not claim production deployment, regulatory compliance, bank approval or independent organisational validation. The model developer and reviewer can be the same student, with distinct UI workflows. Do not describe financial-model validation as equivalent to AI model validation.

## Working rules

- Implement the actual end-to-end workflow. A static dashboard and generated scores do not count as completion.
- Work autonomously through ordinary reversible implementation choices. Do not stop after a plan or each phase. Do not delete existing user work, publish externally or create paid resources.
- Use free public data and CPU-friendly open-source packages. No paid APIs or GPU is required. An LLM is unnecessary for the core product; do not add one merely for appearance.
- Be transparent about inaccessible data, failed experiments, unrun paths and unfinished features. Never manufacture results, percentages, confidence intervals or completion claims.
- Treat this repository's existing status and tests as a starting point, not proof that the full platform works.
- Keep secrets out of source control. Never load arbitrary uploaded pickle/joblib models or execute uploaded code. Only load trusted locally generated artifacts.
- Build for Windows PowerShell and Python 3.11/3.12. Verify compatibility against official package documentation before choosing and pinning tested dependencies.
- Do not use my final evaluation data to improve models, thresholds, features, calibration or acceptance criteria. See the evaluation protocol below.

## Phase 0: inspect and establish a plan

Read README.md, START_HERE.md, docs/BUILD_STATUS.md, existing source and tests. Run the starter tests. Document existing working functions, gaps and assumptions. Create docs/IMPLEMENTATION_PLAN.md. Select the smallest coherent architecture; avoid infrastructure whose value you cannot explain. Log substantive decisions in docs/DECISIONS.md.

## Phase 1: data ingestion and provenance

Use the official UCI **Default of Credit Card Clients**, dataset ID 350:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

The starter provides an acquisition script using ucimlrepo; inspect the returned schema and current source metadata rather than trusting inferred column positions. Add a manual-file route if acquisition fails, and explain exactly where the user can obtain the official file. Record dataset source, citation, access date, license information and checksum. Do not redistribute source data without checking its terms. Do not silently substitute synthetic data.

Implement schema validation, numeric-type checks, target checks, unique customer IDs, unexpected columns, missing values and domain diagnostics. Remove the ID from predictors. Distinguish unexpected values from documented category codes: repayment status can legitimately contain negative codes. Document actual semantics and treatment of education/marital-status codes. Never invent a positive-class interpretation: confirm default=1 against the source.

This is a historical cross-sectional sample with several monthly features, not a collection of successive customer cohorts. Do not manufacture chronological splits from row order. Use disjoint customer-level stratified splits and explicitly state that genuine out-of-time validation and real deployment drift cannot be established from this dataset alone. Use demographic attributes for audit slices; make their predictor inclusion/exclusion explicit and configurable, with a documented default. Do not imply legal fairness from a slice chart.

## Phase 2: pre-register the experiment

Before modelling, write configs/validation_policy.json and docs/EXPERIMENT_PROTOCOL.md. Record split IDs/hashes, random seeds, target, feature policy, selection metric, calibration selection rule, operational threshold rationale, uncertainty method and permitted experiments. Proposed splits: 60% train, 15% tuning, 10% calibration, 15% final evaluation. Changing them must invalidate incompatible artifacts.

Fit encoders, imputers, scalers, feature selection and resampling on training data only. Use sklearn pipelines. Tune on the tuning partition or an inner CV scheme within development data; never tune on calibration/final partitions. Decide calibration family using development-only evidence, fit it on calibration data and evaluate once on the final holdout. Do not use the calibration fitting results as an unbiased estimate of calibration quality. A reasonable first implementation is a pre-specified sigmoid method; add isotonic only with a properly separated selection protocol.

Choose the champion on development evidence under a frozen rule. Run the final evaluation after selection is frozen, and retain it as an audit. If final results reveal a weakness, record it; a redesign becomes a new study requiring fresh unseen data or a clearly labelled exploratory analysis. Log holdout access and explain the limits of a local student-operated lock.

## Phase 3: baseline and challenger

Implement an interpretable logistic-regression baseline with appropriate scaling/encoding and a CPU-friendly gradient-boosting challenger. sklearn HistGradientBoosting is sufficient; choose XGBoost/LightGBM only with a justified dependency. Use a bounded, reproducible search budget. Do not use class weighting/oversampling by default without discussing probability and calibration consequences. Report untuned and selected configurations and training time.

## Phase 4: evaluation and uncertainty

Report ROC-AUC, average precision (name it accurately rather than conflating it with trapezoidal PR-AUC), log loss, Brier score, calibration curve, and threshold-specific confusion matrix, precision, recall and specificity. Show sample size and default prevalence. AUC ranking, probability calibration and threshold decisions answer different questions; explain all three.

Add paired bootstrap confidence intervals for model differences using the same resampled customers. Handle failed/one-class resamples visibly. State intervals are conditional on the fitted models and evaluation sample, and do not cover all training uncertainty. Add repeated development resamples if practical, keeping the final holdout sealed. Avoid significance hunting and excessive slice comparisons. Display empty/single-class slices as unavailable with a reason, not as zero.

Implement slices for credit-limit bands, repayment-history groups and selected demographic groups with minimum-support warnings. Band boundaries must be pre-specified or fitted on development data. Report uncertainty and avoid discriminatory recommendation language. If decision-cost simulation is included, label costs, exposure and loss assumptions as user-supplied scenarios rather than observed banking losses. This dataset does not establish production expected credit loss or IFRS9 provisioning.

## Phase 5: stress, shift and explanations

Create a separate stress evaluation suite using seeded, documented perturbations: missing values, plausible numeric corruption and customer-composition reweighting/subsampling. Keep valid ranges and transformations explicit. Compare both models on identical scenarios. Separate input corruption from business stress: changing a predictor while holding outcomes fixed is a robustness probe, not a causal simulation of default under recession.

Track distribution-distance diagnostics and performance changes where labels exist. Distinguish covariate shift, prevalence shift and concept drift; do not claim to identify concept drift from unlabeled data. Simulated batches must be visibly labelled SIMULATED, and must not appear to be a live monitoring feed. Drift thresholds are configurable review triggers, not universal regulatory limits.

Provide global and local SHAP explanations on a bounded, documented sample and background drawn only from development data. Support the chosen model implementations and output scale correctly; validate explanation additivity within numerical tolerance where applicable. Explain correlated-feature limitations, background dependence and that SHAP is neither causal proof nor a fairness certificate. Compare explanation rankings across development resamples only if implemented, otherwise mark that extension unfinished.

## Phase 6: registry, findings and human review

Use SQLite for a zero-setup local default; allow PostgreSQL later. Store dataset fingerprints, split manifests, preprocessing versions, configuration, code revision when available, model artifacts, evaluation runs and evidence references. Use MLflow locally if it adds value, but avoid fragile UI dependency on the MLflow server. Do not duplicate conflicting sources of truth.

Implement a documented state machine such as draft -> evaluated -> in_review -> accepted_for_research / changes_requested / rejected. Acceptance is for this local research prototype, not bank or production approval. Store findings with category, severity rubric, evidence, reviewer comment, owner, status and remediation history. Require explicit review confirmation. Reject duplicate submissions and stale decisions if the underlying candidate/evidence changed. Keep append-only review events; state clearly that an ordinary local database is not tamper-proof. Distinguish a locally entered reviewer identity from verified authentication.

Add challenger comparisons, historical runs and an auditable policy summary. Do not automatically accept a model because a single metric exceeds a threshold. A configured deterministic check can flag a concern, while a human decides with documented limitations.

## Phase 7: product and backend

Build a polished responsive UI. Prefer React + TypeScript + Vite if Node is available; a carefully built server-rendered alternative is acceptable if documented. Keep the FastAPI API. Provide:
1. Overview: dataset/version, development vs final results, major findings and actual experiment status.
2. Data quality: schema checks, provenance and split isolation.
3. Model comparison: discrimination, calibration, threshold trade-offs and paired uncertainty.
4. Segment analysis: support counts, prevalence and caution labels.
5. Stress/shift lab: scenario definitions and baseline comparisons.
6. Explanations: global/local explanations with limitations.
7. Review workspace: evidence-linked findings, human decisions and event history.
8. Reports/runs: downloadable validation report and machine-readable results.

Use clear charts, accessible labels, proper loading/empty/error states and readable tables. No decorative fake KPIs. Do not call an upload successful until validation completes. Long-running training must expose job status and return failures clearly. Keep jobs bounded and prevent accidental concurrent duplicate training. Cache by data/config/code identity; invalidate stale results. Bind locally by default.

## Phase 8: verification

Add meaningful unit, integration and browser tests. Cover split disjointness/completeness, ID exclusion, duplicate customers, bad target values, training-only preprocessing, calibration separation, metric edge cases, deterministic seeds, stress reproducibility, stale artifact rejection, API invalid input and review-state transitions. Include a test that fails if target information enters features. Do not turn a failing substantive assertion into an easier assertion merely to get green tests; document any justified correction.

Run one actual official-data experiment if the data and environment are available. Run tests offline with tiny clearly labelled fixtures, not full downloads. Record exact commands, dataset hash, tested versions, exit codes, actual model results and unverified paths. Verify critical UI flows in a real browser if available; otherwise provide a manual checklist and say browser verification was not run. No pretending screenshots or manual checks occurred.

## Deliverables

Deliver working code, bounded configs, setup/run commands for PowerShell, dependency locks for the tested environment, optional Docker support only if verified, API docs, dataset card, model cards, evaluation report, risk/limitations register, experiment protocol and a 5-minute demo walkthrough. Training must not happen automatically just by opening a dashboard. Do not commit data, model binaries, secrets, caches or virtual environments.

Update BUILD_STATUS.md with Implemented / Tested / Not verified / Deferred entries supported by evidence. Write a concise final completion report listing actual commands and outcomes. Do not claim that all features are complete unless verified.

## My mastery documentation: essential, not optional

Create docs/MODELSCOPE_MASTER_GUIDE.md and a linked docs/README.md. Make the guide accurate to the final repository. Write in simple English first, then mathematical/technical detail. Separate facts, experimental findings, assumptions and proposed extensions. Include:

A. Business problem, stakeholders, decisions supported and why a prediction API alone is insufficient.
B. Full architecture, directory map, entities and end-to-end request/data/training/review flow with a compact Mermaid diagram.
C. Every dataset column actually used, source semantics, preprocessing decisions, exclusions and dataset limitations.
D. A complete walk-through of logistic regression and gradient boosting: intuition, equations, objective, hyperparameters, regularisation, class imbalance and alternative choices.
E. Train/tune/calibration/test separation, all leakage routes, preprocessing fit locations and why this dataset cannot prove temporal generalisation.
F. Every metric: formula, worked numerical example, interpretation, edge cases and misleading uses. Explain calibration, proper scoring rules, discrimination and decisions separately.
G. Bootstrap pairing, confidence intervals, assumptions and remaining uncertainty.
H. Shift/stress construction, causal limitations, slice analysis and demographic audit limits.
I. SHAP algorithm choice, background, output space and limitations, tied to the code actually used.
J. Registry/schema, version identities, review state machine, duplicate/stale submissions and audit limitations.
K. Function-by-function walk-through of critical code, with paths and symbol names; show where each important decision happens. Do not invent line numbers.
L. One actual customer-record path through preprocessing, prediction, calibration, explanation and review, avoiding personal information and showing no fabricated output.
M. Actual results: tables, failures, what changed, negative findings and what cannot be concluded.
N. Alternatives rejected, trade-offs, production extensions, monitoring with and without labels, and what more data would be needed.
O. A complete execution log: action, why, implementation location, command, result and what could be improved. Include substantive assistant-made changes, not a transcript of every keystroke.
P. At least 80 non-repetitive interview questions with answers and follow-up challenges, spanning business, ML, statistics, data leakage, software, testing and limitations. Include 30-second, 2-minute and 5-minute explanations.
Q. At least 12 hands-on exercises with separate solutions, spanning data splits, calibration, metrics, drift, failure reproduction, API and review controls. Mark which exercises I have not personally completed.
R. Honest contribution statement: AI-assisted implementation versus the requirements, code, experiments and changes I personally verified. Explain how I can demonstrate independent understanding rather than claiming to have authored everything unaided.
S. Resume evidence matrix: each possible claim -> source code -> test or run artifact -> actual finding -> limitation. Draft three concise resume bullets using only verified completed work, with no invented improvements.

Maintain the documentation alongside implementation. Teach enough detail that I can reproduce the key results, change an assumption and explain why the result changes. Completeness means a working, tested, understandable project with honest limits, not an artificially perfect benchmark.
