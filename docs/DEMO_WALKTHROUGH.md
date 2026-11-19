# Five-minute demo walkthrough

Preparation (before the audience arrives): data downloaded, a completed study
with a final audit (e.g. `study-20261118T163406-a420d0fc`), `.\run.ps1`
running, browser at http://127.0.0.1:8000. Don't train live; it takes ~3 minutes.

**0:00 – Overview (40 s).** "The question is whether a better-ranking model also
gives trustworthy probabilities and stable segment performance." Point at the
experiment-status timeline: data validated → development study → selection
frozen → final audit once → human decision still open. Say plainly that this is
a local research prototype, not an approved model.

**0:40 – Data quality (40 s).** Show the SHA-256 and licence, the eight warnings
for undocumented codes ("kept, not reinterpreted"), and split isolation:
overlaps all 0, union equals the dataset, manifest hash matches the
pre-registered one. One sentence: "Rows aren't chronological, so this can't
show out-of-time performance."

**1:20 – Model comparison (80 s).** Read the callout on three questions. Tuning:
challenger AUC 0.785 vs 0.775 and lower log loss. Show the calibration curve:
both close to the diagonal. Toggle to the final audit: AUC 0.782 vs 0.766;
paired difference +0.016 [0.009, 0.024]. Open the frozen selection record:
"The rule was fixed before the holdout was opened." Mention that at 0.5 recall
is only ~0.37, which is why the threshold is labelled illustrative.

**2:40 – Segments (40 s).** Repayment status slice: within PAY_0 ≥ 2 the AUC
falls to about 0.54–0.58. Most of the ranking comes from separating groups,
not from within them. Show a tiny slice marked "unavailable", not zero.

**3:20 – Stress & shift (40 s).** LIMIT_BAL unit error hurts logistic
regression more (−0.018) than boosting (−0.004). Pure prevalence shift leaves
AUC flat but moves log loss by ±0.12. SIMULATED batch S6: PSI stays near 0
while AUC falls, so concept drift is invisible without labels.

**4:00 – Explanations (25 s).** Global: PAY_0 and LIMIT_BAL dominate both
models. Pick a customer and show that base + Σ SHAP equals the calibrated
log-odds. "Attribution of the model, not the customer's causes."

**4:25 – Review workspace (35 s).** Show the deterministic checks (flags, not
decisions), raise a finding from a flag, show the decision form: version and
evidence hash travel with the request; stale or duplicate submissions are
rejected; the event chain is verified. End on Reports & runs: download the
Markdown report and JSON.

Likely questions are in the master guide, section P.
