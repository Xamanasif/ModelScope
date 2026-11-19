# Decision log

Substantive design decisions, in the order they were made (2026-11-18). Each
records the alternatives and why one was chosen. "Revisit if" says what
evidence would change it.

### D-001 Smallest coherent architecture
**Decision:** one Python package (`app/`) with plain modules, a FastAPI process, an
in-process single-worker job runner, SQLite via the standard library, JSON
evidence files on disk, and a static React build served by FastAPI.
**Rejected:** Celery/Redis job queue, Docker Compose, MLflow server, PostgreSQL,
ORM. Each adds a service or dependency whose value at one user and one dataset
can't be shown.
**Revisit if:** several users, several machines or long jobs need queuing.

### D-002 Direct UCI download instead of `ucimlrepo`'s DataFrame
The starter used `ucimlrepo.fetch_ucirepo(350)` and hashed `frame.to_csv()`,
which hashes a pandas re-serialisation, not the published file. The script now
calls the same official UCI API, downloads the exact `data_url` bytes, hashes
those bytes and re-checks the X1..X23 → name mapping against live metadata. The
`ucimlrepo` dependency was dropped. A `--manual` route registers a browser
download of the same CSV. The first `ucimlrepo` attempt failed with a transient
DNS error (`getaddrinfo failed`) on the CSV host; the retry logic covers that.

### D-003 Positive class and code semantics
Positive class = `Y = 1` = default, per the source's "default payment (Yes = 1,
No = 0)". Repayment codes -2 and 0 and EDUCATION 0/5/6, MARRIAGE 0 occur but
are undocumented. They are kept as distinct codes and reported as warnings,
never reinterpreted. Common folklore meanings (e.g. "-2 = no consumption") are
not used because the source does not state them.

### D-004 Customer-level stratified split, not time-based
Rows are cross-sectional and not chronological, so a chronological split would
be fake. A 60/15/10/15 stratified split by unique ID (seed 42) was used, with
integer counts allocated up front (starter fix kept). The manifest hash is
pre-registered in the policy; any change of seed, fractions or file produces
a different hash and is refused for the existing study.

### D-005 Demographics excluded from predictors by default
SEX, EDUCATION, MARRIAGE and AGE (UCI flags them as demographic) are audit-slice
variables only; `features.demographic_predictors` toggles this and changes the
study identity. Excluding them does not make a model "fair": proxies such as
credit limit remain.

### D-006 Hyperparameters by CV inside train; the tuning partition is clean
"Tune on the tuning partition" was replaced by 5-fold CV *inside* train. The
tuning partition is then untouched by every fitted parameter, so it can serve
for (a) the calibration-method choice, (b) the champion rule, (c) slices,
stress, drift and SHAP, without the optimism of re-using the data the
hyperparameters were picked on. Budget: 5 LR configs, 12 random HGB configs
plus the untuned default.

### D-007 No class weighting or resampling
Re-weighting or oversampling defaults shifts predicted probabilities away from
the observed 22% base rate and would then need calibration to undo it. The
policy asks for probabilities, so the models are fitted on the natural
distribution.

### D-008 Calibration: none vs sigmoid, selected on tuning, fitted on calibration
Platt (sigmoid) on the raw log-odds score, fitted by maximum likelihood on the
3,000 calibration rows. Platt's (1999) target smoothing is not used; it matters
mainly for tiny calibration sets. Selection rule (pre-registered): keep sigmoid
unless it raises tuning-partition log loss. Isotonic was not used: 3,000 rows
and no extra partition to select it without bias. Implemented by hand
(`app/calibration.py`) rather than `CalibratedClassifierCV` so the affine map
`a * s + b` is explicit, which makes SHAP values exactly transferable to the
calibrated scale.

### D-009 SHAP algorithm (revised during the build)
**First version:** HGB used `tree_path_dependent` TreeSHAP after an initial
check showed interventional TreeSHAP (shap 0.51.0, scikit-learn 1.9.1)
violating additivity by up to ~2.5 log-odds.
**Investigation:** the violation occurs with non-integer continuous inputs
(Gaussian or 2-decimal data) and does not occur with integer-valued inputs
(error ~1e-7), including the official dataset (all columns are integers).
**Final decision:** interventional TreeSHAP for HGB with the same 200-row train
background as the logistic LinearExplainer, so both models' explanations
answer the same question. Additivity is measured on every explained row
(`additivity_ok`) and flagged by a review check if violated. The condition is
pinned in `tests/test_modelling.py::test_hgb_interventional_condition` (skipped
on other library versions). The root cause inside shap was not established.

### D-010 Champion rule favours the interpretable baseline
The challenger wins only if the upper bound of the paired 95% bootstrap CI of
the log-loss difference on tuning is below 0. Log loss was chosen because the
product is a probability; AUC is reported but not decisive.

### D-011 No MLflow
A local SQLite registry plus SHA-256-indexed JSON evidence is the single source
of truth. Adding MLflow would duplicate run records (two places to disagree)
and add a server for no capability the review workflow needs.

### D-012 Evidence immutability and model loading
Evidence files are hashed at write time and verified at every read. Model
artifacts are joblib pickles, which execute code on load, so the loader accepts
only files registered for the study, inside `artifacts/`, with a matching
SHA-256. Uploaded models are never accepted; uploads are CSV-only and parsed
by pandas.

### D-013 Code identity limited to result-affecting modules
The study identity hashes `app/{bootstrap,calibration,config,data,drift,explain,
features,metrics,models,slices,splitting,stress,study}.py`. Hashing all of
`app/` would have made every API/UI edit invalidate the evidence. The trade-off
is that an unrelated edit inside one of those modules (even a comment) still
marks studies stale: deliberately conservative.

### D-014 Exact fast AUC/AP for bootstrap loops
Profiling showed ~9 ms per `roc_auc_score` call, all in sklearn's input
validation. `fast_auc` (Mann–Whitney with mid-ranks) and
`fast_average_precision` (step-wise, ties grouped) are used inside resampling
loops and tested equal to sklearn to ~1e-16, including heavily tied scores.

### D-015 Stress and drift run on the tuning partition
Both are development evidence, available before the final holdout is opened,
and therefore never involve sealed rows. Simulated batches are labelled
SIMULATED in data and UI.

### D-016 Hand-drawn SVG charts, no chart library
Three chart forms are needed (lines, interval/forest, bars). Hand-written
components keep the bundle small (66 kB gzipped), make light/dark tokens and
accessible labels straightforward, and avoid a dependency. Series colours: slot 1
blue for the baseline, slot 2 orange for the challenger, validated for colour-vision
deficiency in both themes.

### D-017 Replacing one starter test
`test_starter_api_is_honest` asserted the starter page text and
`model_loaded: false`. Those described a page that no longer exists. It was
replaced by `test_api_is_honest_and_never_trains_on_page_load`, which keeps the
substantive properties: no prediction endpoint, and opening the app creates no
study or job. No other substantive assertion was weakened.

### D-018 Not installing `httpx2`
Starlette's TestClient warns that `httpx` is deprecated in favour of a package
named `httpx2`. That package name could not be verified as the official
successor, so it was not installed; the warning remains.
