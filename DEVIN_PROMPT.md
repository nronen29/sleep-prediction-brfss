# Handoff Prompt for Devin — Data Science Final Project
**Authors: Neta Ronen (318595956) & Elav Cohen (208698100)**
**Course: Introduction to Data Science**

---

## WHAT THIS PROJECT IS

We are building a data science final project that predicts insufficient sleep among U.S. adults using the BRFSS 2022 survey dataset. The deliverables are:
1. A revised project proposal (Word document) — **DONE**
2. A Jupyter Notebook with the full analysis — **TO DO**

The project is submitted to a university course. All code must be understandable, well-commented, and reproducible.

---

## RESEARCH QUESTION

**How well can insufficient sleep be predicted from structural characteristics alone, and how much does predictive performance improve when self-reported lifestyle and health variables are added?**

This is answered by training and comparing **two feature sets** within the same algorithms:
- **Group A (structural only):** age, sex, race/ethnicity, income, education, employment, marital status, geography (Census region)
- **Group B (extended):** Group A + general health, mental/physical health days, BMI, activity, smoking, alcohol, chronic conditions

The **performance gap between Group A and Group B within each algorithm** is the central finding. This is NOT a causal study — BRFSS is cross-sectional.

**Outcome:** `insufficient_sleep = 1` if `SLEPTIM1 < 7`, else 0. Exclude codes 77, 99, missing.

---

## FILES AND THEIR LOCATIONS

All files are in the GitHub repository (connect it). Local paths on Neta's machine:

```
final project/
├── LLCP2022XPT/
│   └── LLCP2022.XPT          ← Main data file (1.1 GB, BRFSS 2022 SAS transport format)
├── LLCP2022XPT.zip            ← Original ZIP (can ignore)
├── Data_Science_Proposal_Final.docx   ← Final revised proposal (Elav's version, USE THIS)
├── Data_Science_Proposal_Updated.docx ← Earlier draft by Claude (superseded)
├── PROJECT_PROMPT.md          ← Full context prompt (reference)
├── DEVIN_PROMPT.md            ← This file
└── [lecture PDFs]             ← Course lecture slides (see list below)
```

**Lecture PDFs available (use as method reference):**
- Introduction to Data Science (general intro)
- Data Science — Lecture 3
- Introduction to Data Science — Data Cleansing
- Linear Regression Models
- Decision Tree Models
- K-Nearest Neighbors and Supervised Learning
- Advanced Classification Models and Hyperparameter Tuning
- Beyond Accuracy — Making Better Classification Decisions
- Imbalanced Data — Why Accuracy Lies and How to Fix It
- Mastering Explainable AI (XAI)
- Multidimensional Unsupervised Learning
- Unsupervised Learning and Clustering

---

## VARIABLES

### Target
| Variable | Description |
|---|---|
| `SLEPTIM1` | Hours of sleep per 24 hours. Valid: 1–24. Exclude: 77 (don't know), 99 (refused), missing |

### Group A — Structural (both feature sets)
| Variable | Description | Notes |
|---|---|---|
| `_AGEG5YR` | Age group | 13 substantive categories; code 14 = missing |
| `SEXVAR` | Sex | |
| `_IMPRACE` | Race/ethnicity | CDC-imputed — disclose this |
| `_INCOMG1` | Income level | |
| `_EDUCAG` | Education | |
| `EMPLOY1` | Employment status | |
| `MARITAL` | Marital status | |
| `_STATE` | State → map to 4 Census regions + territories category | Do NOT one-hot encode 50+ states |

**Census region mapping:**
- Northeast: CT, ME, MA, NH, RI, VT, NJ, NY, PA
- Midwest: IL, IN, MI, OH, WI, IA, KS, MN, MO, NE, ND, SD
- South: DE, FL, GA, MD, NC, SC, VA, DC, WV, AL, KY, MS, TN, AR, LA, OK, TX
- West: AZ, CO, ID, MT, NV, NM, UT, WY, AK, CA, HI, OR, WA
- Territories: GU, PR, VI

### Group B — Extended (added in Model 2)
| Variable | Description | Notes |
|---|---|---|
| `GENHLTH` | General health status | |
| `MENTHLTH` | Poor mental health days (past 30) | Code 88 = 0 days |
| `PHYSHLTH` | Poor physical health days (past 30) | Code 88 = 0 days |
| `_BMI5CAT` | BMI category | |
| `_TOTINDA` | Physical activity | |
| `_SMOKER3` | Smoking status | |
| `_DRNKWK2` / `_RFDRHV8` | Alcohol measures | Verify exact encoding from codebook before use |
| `DIABETE4` | Diabetes | |
| `ASTHMA3` | Asthma | |
| `ADDEPEV3` | Diagnosed depression | |

### Survey weight (descriptive only — NOT a predictor)
| Variable | Description |
|---|---|
| `_LLCPWT` | Survey weight — use ONLY for weighted prevalence estimates in EDA. Models are unweighted. |

---

## TECHNICAL NOTES

### Loading the data
`pandas.read_sas` does **NOT support `usecols`**. Read in chunks and keep only needed columns:
```python
import pandas as pd

needed_cols = ['SLEPTIM1', '_AGEG5YR', 'SEXVAR', '_IMPRACE', '_INCOMG1',
               '_EDUCAG', 'EMPLOY1', 'MARITAL', '_STATE', '_LLCPWT',
               'GENHLTH', 'MENTHLTH', 'PHYSHLTH', '_BMI5CAT', '_TOTINDA',
               '_SMOKER3', '_DRNKWK2', '_RFDRHV8', 'DIABETE4', 'ASTHMA3', 'ADDEPEV3']

chunks = []
for chunk in pd.read_sas('LLCP2022XPT/LLCP2022.XPT', chunksize=10000):
    chunks.append(chunk[[c for c in needed_cols if c in chunk.columns]])
df = pd.concat(chunks, ignore_index=True)
```

### _STATE mapping
Map `_STATE` FIPS codes to Census regions. Do not one-hot encode individual states.

### Special codes to handle
- `MENTHLTH`, `PHYSHLTH`: code `88` → recode to `0` (means "none")
- All variables: codes for "don't know" (usually 7/77/777) and "refused" (usually 9/99/999) → `NaN`
- Verify each variable's special codes in the BRFSS 2022 codebook

---

## NOTEBOOK STRUCTURE (step by step)

Build the notebook in this order. After each section is complete, commit to GitHub.

### Section 0 — Setup
Library imports, version logging, random seed, file path constants.

### Section 1 — Load data
Chunk-based loading, verify record count (~445,132), verify required columns exist.

### Section 2 — Pipeline validation (BEFORE modeling)
Reproduce CDC's published crude weighted national prevalence of insufficient sleep (~32.7%) from Pankowska et al. (2023) / BRFSS 2020, using a BRFSS 2020 extract if available, or note that 2022 prevalence is analyzed separately without requiring it to match 2020.
Label clearly: **this is validation, not a novel finding.**

### Section 3 — Cleaning & preprocessing
- Recode special values per codebook
- Create binary target `insufficient_sleep`
- Examine missingness
- Drop or impute features
- Map `_STATE` to Census regions
- Document all decisions

### Section 4 — EDA
- Weighted and unweighted prevalence
- Sleep-hour distribution
- Prevalence by age, sex, income, education, race/ethnicity, region
- Missingness visualization
- Appropriate association measures (chi-square labeled as exploratory sample comparison, not survey-adjusted inference)

### Section 5 — Train/validation/test split
- **Stratified 60/20/20 split, fixed seed**
- **Same split used for both feature sets** (critical — so comparison is fair)
- Fit all preprocessing (imputation, encoding, scaling) on training data only
- Apply to validation and test sets

### Section 6 — Baseline
- Majority-class classifier (always predicts 0)
- Report: accuracy, precision, recall, F1, ROC-AUC, Average Precision

### Section 7 — Model 1 (Group A — structural only)
Three algorithms: Logistic Regression, Random Forest, Gradient Boosting
- Tune hyperparameters on train+validation
- Evaluate on validation set
- Select best settings before touching test set

### Section 8 — Model 2 (Group A + Group B — extended)
Same three algorithms, same split, same procedure.

### Section 9 — Final test evaluation
- Both feature sets, all three algorithms
- Metrics: accuracy, precision, recall, F1, ROC-AUC, Average Precision, confusion matrices
- Compare Group A vs Group B **within each algorithm**
- Class weights or threshold adjustments decided before this step
- Expected ROC-AUC: 0.60–0.68 for Model 1; modest improvement for Model 2
- Modest performance is a valid finding — do not force good numbers

### Section 10 — Interpretation
- Logistic Regression coefficients (direction of association)
- Permutation importance across algorithms (note: caution with correlated features)
- Subgroup performance by age, sex, income

### Section 11 (optional) — Clustering
- K-means or hierarchical on full feature set
- Compare insufficient-sleep rates between clusters
- Label as supplementary/exploratory

### Section 12 — Discussion and limitations
- Self-report bias
- Missingness
- CDC imputation in `_IMPRACE`
- Unweighted models describe the sample, not the U.S. population
- Reverse causation: Group B variables may be outcomes of poor sleep, not only predictors
- Separating feature groups reduces but does not remove confounding

---

## RELATED LITERATURE

1. **Grandner et al. (2015)**, Frontiers in Neurology, 6:112 — BRFSS 2009, predicted perceived insufficient rest (different target from ours), compared models with/without mental health using gradient boosting.
2. **DelRosso & Vodapally (2025)**, Clocks & Sleep, 7(4):59 — BRFSS 2022, weighted logistic regression for short sleep prevalence.
3. **Pankowska et al. (2023)**, Preventing Chronic Disease, 20:220400 — BRFSS 2020 weighted prevalence report (32.7% crude, 33.2% age-adjusted).

We do not claim to be the first to compare feature sets in BRFSS.

---

## WORKING RULES

1. **Never invent results, numbers, or conclusions.** Everything must come from actual code output.
2. **If code fails or output is unexpected, say so clearly.** Do not hide errors.
3. **Work one section at a time.** After each section ask: "Ready to move to the next section?"
4. **Preprocessing fitted on training data only — no data leakage.**
5. **Test set is untouched until Section 9.**
6. **Both feature sets use the same split and same respondents.**
7. **Survey weights are not predictors** — only used in EDA for descriptive estimates.
8. **Commit to GitHub after each completed section.**

---

## AI USAGE (for the final report)

Claude (Anthropic) assisted with the earlier proposal drafts. ChatGPT/Codex assisted with proposal revisions, source checks, code generation, and debugging. The authors are responsible for verifying and understanding all work. The final report will describe actual uses and include prompts from both tools.

---

## WHAT IS ALREADY DONE

- ✅ Revised proposal: `Data_Science_Proposal_Final.docx`
- ✅ Data downloaded: `LLCP2022XPT/LLCP2022.XPT` (1.1 GB)
- ✅ Research question finalized (two-model framework)
- ✅ Variable list finalized (Group A / Group B)
- ✅ Technical decisions made (_STATE → regions, chunk loading, weight handling)

## WHAT STILL NEEDS TO BE DONE

- ⬜ Jupyter Notebook — Sections 0–12 above
