# Data Science Final Project — Shared Context Prompt
**Use this as the opening message in any new Claude session for this project.**
**Authors: Neta Ronen (318595956) & Elav Cohen (208698100)**

---

## WHO WE ARE AND WHAT WE ARE DOING

We are two students completing a final project for an Introduction to Data Science course. The project has two deliverables:
1. A **Word document** (revised proposal — already written)
2. A **Jupyter Notebook** (the actual analysis)

We work step by step together. After each section of code is agreed upon, we move to the next. **Do not skip ahead.** At the end of each step, ask us: "Ready to move to the next section?"

---

## PROJECT TOPIC

**Research question:** How much of insufficient sleep can be explained by **structural characteristics alone** — age, sex, race/ethnicity, income, education, employment, marital status, and geography — and how much additional explanatory power is gained by adding self-reported lifestyle and health variables?

This is answered by training and comparing **two models**:
- **Model 1:** Structural/demographic variables only (Group A)
- **Model 2:** Group A + lifestyle and health variables (Group B)

The **performance gap** between the two models is the main finding.

**Outcome variable:** Insufficient sleep = SLEPTIM1 < 7 hours (CDC's official adult definition), binary: 1 = insufficient, 0 = sufficient.

**Intended audience:** Public-health organizations, healthcare professionals, policy makers.

**Causal claims are NOT made** — BRFSS is cross-sectional.

---

## DATASET

- **Name:** Behavioral Risk Factor Surveillance System (BRFSS) 2022
- **Source:** U.S. CDC — https://www.cdc.gov/brfss/annual_data/annual_2022.html
- **Format:** SAS Transport file (.XPT)
- **Size:** ~445,000 observations, 300+ variables
- **Coverage:** All 50 U.S. states, D.C., and several territories

### Group A — Structural variables (both models):
`_AGEG5YR`, `SEXVAR`, `_IMPRACE`, `_INCOMG1`, `_EDUCAG`, `EMPLOY1`, `MARITAL`, `_STATE` (mapped to 4 Census regions)

### Group B — Lifestyle & health variables (Model 2 only):
`GENHLTH`, `MENTHLTH`, `PHYSHLTH`, `_BMI5CAT`, `_TOTINDA`, `_SMOKER3`, alcohol variables, selected chronic conditions (diabetes, asthma, depression indicator)

### Survey weight:
`_LLCPWT` — used ONLY for descriptive/EDA weighted estimates. **Models are trained WITHOUT weights.** This must be noted in the discussion.

---

## PLANNED METHODS (step by step)

### Step 0 — Load data efficiently
- Use `pandas.read_sas` with `usecols` to load only needed columns (file is large)
- Do NOT load all 300+ columns

### Step 1 — Pipeline validation (BEFORE modeling)
- Reproduce CDC's published weighted prevalence of insufficient sleep from their BRFSS 2020 report (stratified by age, sex, race/ethnicity, income, education, region)
- If our weighted numbers match CDC's, the cleaning pipeline and weights are correct
- Label this clearly as VALIDATION, not a novel finding

### Step 2 — Cleaning & preprocessing
- Recode refusal/unknown/invalid codes as NaN
- Remove rows with missing target
- Examine missingness patterns per feature
- Impute: median for numeric, mode for categorical — fitted on training data only
- One-hot encode categoricals (fitted on training only — no data leakage)
- `_STATE`: map to 4 U.S. Census regions (Northeast, Midwest, South, West) — do NOT one-hot encode 50+ states

### Step 3 — EDA
- Weighted and unweighted prevalence of insufficient sleep
- Distribution across subgroups (age, sex, income, education, race, region)
- Missing value heatmap
- Correlation matrix
- Chi-square tests for categorical comparisons

### Step 4 — Train/test split
- Stratified 60% train / 20% validation / 20% test
- Hold out test set completely until final evaluation

### Step 5 — Baseline
- Report majority-class classifier performance (always predicts 0 = sufficient sleep)
- Expected: ~65% accuracy, near-zero recall for class 1
- All other models compared against this

### Step 6 — Model 1 (Structural only)
- Logistic Regression, Random Forest, Gradient Boosting
- Hyperparameter tuning on train+validation (GridSearchCV or similar)
- Evaluate on validation set

### Step 7 — Model 2 (Structural + Health/Lifestyle)
- Same three algorithms with Group A + Group B features
- Same tuning procedure

### Step 8 — Final evaluation on test set
- Both models, best algorithm from each
- Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix
- Emphasis on recall and PR-AUC for the minority class (insufficient sleep)
- Class-weight adjustment or threshold tuning if needed (decided on validation set)
- Expected ROC-AUC range: 0.60–0.68 for Model 1, modest improvement for Model 2

### Step 9 — Interpretation
- Logistic Regression coefficients (direction)
- Permutation importance across models (magnitude)
- Subgroup performance analysis (age bands, income quartiles, etc.)

### Step 10 — Optional: Clustering
- K-means or hierarchical on full feature set
- Compare insufficient-sleep rates between clusters
- Labeled as supplementary/exploratory

---

## RELATED LITERATURE (already cited in proposal)

1. **Grandner et al. (2015), Frontiers in Neurology** — BRFSS, same binary outcome, four predictor groups, gradient boosting, train/test split, 10-fold CV. Their limitation: did not separate structural from health predictors. Our two-model design addresses this.
2. **Clocks & Sleep (2025)** — Weighted logistic regression on BRFSS 2022. Our project adds predictive modeling and the two-model comparison.
3. **CDC, Preventing Chronic Disease (BRFSS 2020)** — Published weighted prevalence by demographics. We replicate this as a validation step.

**Our contribution:** No BRFSS study has framed the question as a two-stage comparison where the performance gap between a structural-only and a full model is the primary finding.

---

## WORKING RULES — FOLLOW THESE AT ALL TIMES

1. **Never guess or invent information.** If something is unclear, ask.
2. **Never invent results or conclusions.** All conclusions must come from actual output we run.
3. **If code fails or produces unexpected output, say so clearly.** Do not hide errors.
4. **Base all statistical methods on the course lecture slides** (uploaded PDFs). If a method is not covered in the lectures, flag it and ask before using it.
5. **Tell us which lecture and topic you are drawing from** when applying a method (e.g., "This uses weighted sampling, from the Data Cleansing lecture").
6. **If you need something not in the lectures, ask permission** before using an external source, and cite the source.
7. **Work one step at a time.** After each step, stop and ask: "Ready to move to the next section?"
8. **The submission format** is: a Jupyter Notebook + a Word document. We write code here and copy it ourselves.
9. **Do not add emojis or unnecessary filler.** Be direct.

---

## COURSE LECTURE MATERIALS AVAILABLE

The following PDF lecture slides are in the project folder and should be used as the primary reference for all methods:

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

## WHAT IS ALREADY DONE

- ✅ Revised proposal Word document created (`Data_Science_Proposal_Updated.docx`) with all reviewer feedback addressed
- ✅ Research question updated to two-model framework
- ✅ Dataset changed to BRFSS 2022
- ✅ Literature section added (Grandner 2015, Clocks & Sleep 2025, CDC report)
- ✅ Baseline defined
- ✅ Survey weight handling clarified
- ✅ _STATE → Census regions decision made
- ✅ Pipeline validation step added
- ✅ AI usage declaration written

## WHAT STILL NEEDS TO BE DONE

- ⬜ Download BRFSS 2022 XPT file
- ⬜ Jupyter Notebook: all 10 steps above

---

## AI USAGE DECLARATION (for the final report)

We use Claude (Anthropic) as an AI coding and writing assistant. AI is used for: drafting text, writing and debugging code, structuring analysis. All research decisions, variable choices, and interpretations are made by the authors. No AI tool generates data or fabricates results.
