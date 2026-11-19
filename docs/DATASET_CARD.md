# Dataset card: UCI 350 "Default of Credit Card Clients"

| Field | Value |
|---|---|
| Source | UCI Machine Learning Repository, ID 350, https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients |
| File used | `https://archive.ics.uci.edu/static/public/350/data.csv` (the API's `data_url`), saved byte-for-byte |
| SHA-256 | `45bcf4df62ff2e237a74eb155cabfb4bbbc171219a0637daef44fdad07503dd0` (2,897,080 bytes) |
| Accessed | 2026-11-18 09:26 UTC via `scripts/download_data.py` (route `uci_api`) |
| Source metadata last updated | Fri Mar 29 2024 (UCI API field) |
| Creator / citation | Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H |
| Intro paper | Yeh, I.-C. & Lien, C. (2009), *Expert Systems with Applications*, doi:10.1016/j.eswa.2007.12.020 |
| Licence | CC BY 4.0, as stated on the UCI landing page when checked on 2026-11-18. Attribution required; re-check before redistributing. The file is git-ignored and not redistributed by this repository. |
| Size | 30,000 rows × 25 columns (ID, X1..X23, Y); no missing cells |
| Population | Credit-card clients of a bank in Taiwan; repayment history April–September 2005 |

## Columns and treatment
| Source | Name used | Source semantics | Treatment in ModelScope |
|---|---|---|---|
| ID | ID | row identifier | split bookkeeping only; never a predictor |
| X1 | LIMIT_BAL | credit given (NT$), incl. family/supplementary | predictor; LR: signed log1p + standardise; slice bands |
| X2 | SEX | 1 male, 2 female | audit slice only (default) |
| X3 | EDUCATION | 1 graduate school, 2 university, 3 high school, 4 others | audit slice; 0/5/6 observed but undocumented → "undocumented_code" |
| X4 | MARRIAGE | 1 married, 2 single, 3 others | audit slice; 0 observed but undocumented → "undocumented_code" |
| X5 | AGE | years | audit slice (bands 18–29, 30–39, 40–49, 50+) |
| X6–X11 | PAY_0, PAY_2..PAY_6 | repayment status Sep → Apr 2005: −1 pay duly, 1–8 months' delay, 9 = nine+ | predictor; −2 and 0 observed but undocumented, kept as distinct codes; LR clips to [−2, 3] and one-hot encodes; HGB uses the raw code |
| X12–X17 | BILL_AMT1..6 | bill statement amount (NT$), Sep → Apr | predictor; negative values (3,932 cells) kept as plausible credit balances |
| X18–X23 | PAY_AMT1..6 | amount paid (NT$), Sep → Apr | predictor; negative values would fail validation (none observed) |
| Y | DEFAULT_NEXT_MONTH | "default payment (Yes = 1, No = 0)" | target; positive class 1; prevalence 22.12% (6,636 defaults) |

Note the source's naming jump: there is no `PAY_1`; the September status is `PAY_0`.

## Observed code counts (full file)
- EDUCATION: 0→14, 1→10,585, 2→14,030, 3→4,917, 4→123, 5→280, 6→51
- MARRIAGE: 0→54, 1→13,659, 2→15,964, 3→323
- PAY_0: −2→2,759, −1→5,686, 0→14,737, 1→3,688, 2→2,667, 3+→463; no code 9 appears in any PAY column
- 35 rows are identical to another row in every column except ID (56 if the target is also ignored). IDs differ, so they are treated as different customers; the source can't confirm this either way.

## Known limitations
- Cross-sectional sample from one bank, one country and a six-month window in
  2005 (around Taiwan's 2005–06 card-debt crisis). Not representative of other
  markets, products or periods.
- Row order is not chronological: no out-of-time validation is possible.
- "Default" means default payment next month: a short-horizon delinquency
  label, not a regulatory default definition and not a loss amount. No exposure
  or recovery data → no expected-credit-loss, IFRS 9 or capital conclusions.
- Undocumented codes (−2, 0, EDUCATION 0/5/6, MARRIAGE 0) limit interpretation.
- Demographic attributes allow descriptive slices only; group sizes for some
  codes are tiny and no legal fairness conclusion follows.
