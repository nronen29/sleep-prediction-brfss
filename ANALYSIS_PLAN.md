# Analysis Plan — Predicting Insufficient Sleep (BRFSS 2022)

Step-by-step plan for the notebook `sleep_prediction_brfss.ipynb`.
Sources: the revised proposal (`Data_Science_Proposal_Final.docx`), the course instructions (`Final_project_instructions_1.pdf`) and the lecture slides in this folder.

---

## The question

> How well can insufficient sleep be predicted from structural characteristics alone — age, sex, race/ethnicity, income, education, employment, marital status, and geography — before any health variable is added, and how much does prediction improve when lifestyle and health variables are added?

| Item | Definition |
|---|---|
| Target | `insufficient_sleep = 1` if `SLEPTIM1` < 7, else `0`. Valid values 1–24 only. Drop 77, 99 and missing. |
| Group A (structural) | `_AGEG5YR`, `SEXVAR`, `_IMPRACE`, `_INCOMG1`, `_EDUCAG`, `EMPLOY1`, `MARITAL`, `region` (from `_STATE`) |
| Group B (extended) | Group A + `GENHLTH`, `MENTHLTH`, `PHYSHLTH`, `_BMI5CAT`, `_TOTINDA`, `_SMOKER3`, `_DRNKWK2`, `_RFDRHV8`, `DIABETE4`, `ASTHMA3`, `ADDEPEV3` |
| Survey weight | `_LLCPWT`. Used only for descriptive (EDA) estimates. Never a predictor. |
| Main result | Test-set performance of Group B minus Group A, **within each algorithm** (LR, RF, GB). |

---

## Packages

Installed in `.venv` (see `requirements.txt`).

| Package | Version | Used for |
|---|---|---|
| pandas | 3.0.6 | Loading the XPT file (`read_sas`), cleaning, tables |
| numpy | 2.5.4 | Numeric operations |
| scikit-learn | 1.9.1 | Split, preprocessing, models, tuning, metrics, permutation importance, clustering |
| matplotlib | 3.11.2 | All figures |
| seaborn | 0.13.2 | Optional: nicer bar plots and heatmaps |
| scipy | 1.18.1 | `chi2_contingency` (EDA), `randint` / `uniform` for random search |
| jupyter / ipykernel | 1.1.1 / 7.4.0 | Running the notebook |

Optional, not installed: `shap` (only if we add SHAP to the interpretation step).

## Global parameters

| Parameter | Value | Why |
|---|---|---|
| `RANDOM_STATE` | `42` | Reproducibility. Used in every split, model and search (KNN and Advanced Classification lectures). |
| `DATA_PATH` | `LLCP2022XPT/LLCP2022.XPT` | Main data file |
| `DATA_2020_PATH` | `LLCP2020XPT/LLCP2020.XPT` | Only for the validation step. Downloaded by the cell in the notebook's Data section. |
| `CHUNKSIZE` | `50_000` | Rows per chunk when reading the XPT |
| Split | 60% train / 20% validation / 20% test, `stratify=y` | Proposal |
| Main tuning metric | `average_precision` | Imbalanced target (Beyond Accuracy and Imbalanced Data lectures) |

---

## Notebook structure

The course asks for the notebook to follow the report order: **background/EDA → methods → results → discussion**, with markdown cells as section headers. Each step below includes the markdown cell to paste into the notebook (a header and a short note on what the step does and why).

| Notebook header | Plan step |
|---|---|
| `## 1. Background and Exploratory Data Analysis` | Part 1 |
| `### 1.1 Research question` | already in the notebook |
| `### 1.2 Background` | already in the notebook |
| `### 1.3 Data` (+ download cell) | already in the notebook |
| `### 1.4 Setup` | Step 0 |
| `### 1.5 Loading the data` | Step 1 |
| `### 1.6 Pipeline validation (BRFSS 2020)` | Step 2 |
| `### 1.7 Data cleaning` | Step 3 |
| `### 1.8 Exploratory data analysis` | Step 4 |
| `## 2. Methods` | Part 2 |
| `### 2.1 Data split and preprocessing` | Step 5 |
| `### 2.2 Baseline` | Step 6 |
| `### 2.3 Model 1: structural variables only (Group A)` | Step 7 |
| `### 2.4 Model 2: structural + health and lifestyle variables (Group B)` | Step 8 |
| `## 3. Results` | Part 3 |
| `### 3.1 Test results: Model 1 vs. Model 2` | Step 9 |
| `### 3.2 Interpretation` | Step 10 |
| `### 3.3 Clustering (exploratory)` | Step 11 |
| `## 4. Discussion` | Step 12 |
| `## 5. AI usage statement` | Step 13 |

---

## PART 1 — Background and EDA

**In the notebook:**
```markdown
## 1. Background and Exploratory Data Analysis
This part introduces the topic, the research question and the data. It then loads, checks and cleans the data and explores it with tables and plots. We do this first so that we understand the data, and fix its problems, before building any model.
```

## Step 0 — Setup

> **Prompt:** "Add a setup cell at the top of the analysis. Import all the libraries we will use (pandas, numpy, scikit-learn, matplotlib, seaborn, scipy) and print their versions so the results can be reproduced. Set the random seed to 42 and define the paths to the 2022 and 2020 data files in one place."

**In the notebook:**
```markdown
### 1.4 Setup
Imports all libraries, prints their versions and sets the random seed and file paths in one place, so the notebook can be reproduced exactly.
```

- Import all packages and print their versions (the proposal says to record versions).
- Set `RANDOM_STATE = 42` and the file paths.
- Set the figure style once (for example `plt.rcParams["figure.dpi"] = 110`).

## Step 1 — Load the data

> **Prompt:** "Load the BRFSS 2022 XPT file. It's over 1 GB and read_sas can't select columns, so read it in chunks and keep only the columns we need: the sleep question, the Group A and Group B variables, the state and the survey weight. Then check that we got 445,132 rows, that the raw file has 328 columns, and that none of our columns are missing."

**In the notebook:**
```markdown
### 1.5 Loading the data
Reads the BRFSS 2022 file in chunks and keeps only the columns we need, because the file is over 1 GB and `read_sas` cannot select columns. We then check the number of rows and columns to make sure the file loaded correctly.
```

- `pd.read_sas(DATA_PATH, format="xport", chunksize=CHUNKSIZE)`. `read_sas` has no `usecols`, so keep the needed columns from each chunk and `pd.concat` at the end.
- Columns to keep: target, Group A, Group B, `_STATE` and `_LLCPWT`.
- Checks (print the result of each):
  - Number of rows (expected 445,132).
  - Number of columns in the raw file (expected 328).
  - All needed columns exist.
- Lecture: Data Cleansing ("Removing unnecessary data").

## Step 2 — Pipeline validation (validation only, not a finding)

> **Prompt:** "Before we build any model, I want to check that our cleaning and weighting are correct. Load the BRFSS 2020 file, keep only the 50 states and DC, build the insufficient-sleep variable the same way as for 2022, and calculate the weighted percentage with the survey weight. Compare it to the CDC's published 32.7% (the crude number, not the age-adjusted 33.2%). Make it clear in the notebook that this is only a validation check, not a result."

**In the notebook:**
```markdown
### 1.6 Pipeline validation (BRFSS 2020)
Applies our target definition and survey weights to BRFSS 2020 and compares the result with the CDC's published 32.7%. If the numbers match, our cleaning and weighting are likely correct. This is a check of our pipeline, not a finding.
```

- Load the BRFSS 2020 file the same way. The download cell in the Data section gets it from the CDC.
- Keep 50 states + DC only: drop `_STATE` 66 (Guam), 72 (Puerto Rico) and 78 (Virgin Islands).
- Build the same target from `SLEPTIM1`.
- Crude weighted prevalence = `sum(_LLCPWT * y) / sum(_LLCPWT)`.
- Compare with the published **32.7%** (crude). Do **not** compare with 33.2%, which is age-adjusted.
- Label it in the notebook: *"Validation of the cleaning and weighting pipeline, not a result."*
- If the 2020 file is not available, say so in the notebook and skip this step. Do not invent a number.

## Step 3 — Cleaning and preprocessing (before the split)

> **Prompt:** "Clean the data using the codebook. Replace all the 'don't know', 'refused' and missing codes with NaN, and turn 88 into 0 days for the mental and physical health questions. Create the target: 1 if someone sleeps less than 7 hours, 0 otherwise, and drop people with no valid sleep answer. Group the states into the 4 Census regions plus a territories group. Then show me how much is missing in each column with a bar chart, check for duplicates, and write down every decision we make. Don't impute or encode yet. That comes after the split."

**In the notebook:**
```markdown
### 1.7 Data cleaning
Turns the survey's special codes ("don't know", "refused") into missing values, creates the target, groups states into regions and checks how much data is missing. Models need clean inputs, and we document every decision here because it can change the results.
```

Recode special codes to `NaN`. **Every code below must be checked against the BRFSS 2022 codebook before running.**

| Variable | Valid values | Set to NaN | Other recoding | Type |
|---|---|---|---|---|
| `SLEPTIM1` | 1–24 | 77, 99, blank | → target | — |
| `_AGEG5YR` | 1–13 | 14 | — | categorical (ordered) |
| `SEXVAR` | 1, 2 | — | — | binary |
| `_IMPRACE` | 1–6 | — | CDC-imputed: disclose | categorical |
| `_INCOMG1` | 1–7 | 9 | — | categorical (ordered) |
| `_EDUCAG` | 1–4 | 9 | — | categorical (ordered) |
| `EMPLOY1` | 1–8 | 9 | — | categorical |
| `MARITAL` | 1–6 | 9 | — | categorical |
| `_STATE` | FIPS codes | — | → `region` (see below) | categorical |
| `GENHLTH` | 1–5 | 7, 9 | — | categorical (ordered) |
| `MENTHLTH` | 1–30, 88 | 77, 99 | **88 → 0 days** | numeric 0–30 |
| `PHYSHLTH` | 1–30, 88 | 77, 99 | **88 → 0 days** | numeric 0–30 |
| `_BMI5CAT` | 1–4 | blank | — | categorical (ordered) |
| `_TOTINDA` | 1, 2 | 9 | — | binary |
| `_SMOKER3` | 1–4 | 9 | — | categorical |
| `_RFDRHV8` | 1, 2 | 9 | — | binary |
| `_DRNKWK2` | 0–98999 | 99900 | divide by 100 (2 implied decimals) | numeric |
| `DIABETE4` | 1–4 | 7, 9 | — | categorical |
| `ASTHMA3` | 1, 2 | 7, 9 | — | binary |
| `ADDEPEV3` | 1, 2 | 7, 9 | — | binary |

- `_STATE` → `region`: Northeast, Midwest, South, West (4 Census regions, DC in South), plus `Territories` (GU, PR, VI). No one-hot encoding of single states.
- Drop rows with a missing target. Keep **one** cleaned table, used by both feature sets.
- Missingness: print the % missing for each feature and plot it (bar chart).
  - Rule from the Data Cleansing lecture: consider dropping a column with more than 40% missing.
  - Write down every decision (drop, impute, or keep "missing" as its own category).
- Check for duplicate rows.
- **Do not impute or encode here.** That happens inside the model pipeline, fitted on training data only (Step 5).
- Lecture: Data Cleansing (missing values, encoding, outliers).

## Step 4 — Exploratory data analysis

> **Prompt:** "Now let's explore the data. Show what percent of people sleep less than 7 hours, both with and without the survey weights. Plot a histogram of sleep hours with a line at 7. Make bar charts of the insufficient-sleep rate by age, sex, income, education, race and region, and also by general health and depression. Check the numeric columns for values that look wrong, make a correlation heatmap, and run chi-square tests. Label the chi-square tests as exploratory only. Finally, tell me how balanced the two classes are."

**In the notebook:**
```markdown
### 1.8 Exploratory data analysis
Describes how common short sleep is and how it differs between groups, using tables and plots. This shows the main patterns in the data and how balanced the two classes are, which guides our choice of metrics and models.
```

- Prevalence of insufficient sleep: unweighted (`y.mean()`) and weighted (with `_LLCPWT`).
- Sleep-hours distribution: histogram of `SLEPTIM1`, with a line at 7 hours.
- Prevalence by age, sex, income, education, race/ethnicity and region: bar charts (weighted).
- Prevalence by Group B variables, for example general health and depression.
- Numeric features (`MENTHLTH`, `PHYSHLTH`, `_DRNKWK2`): `describe()` (min/max vs. mean/std) and box plots, to find values that look wrong.
- Associations:
  - Pearson correlation heatmap for the numeric and ordered features (Lecture 3).
  - `scipy.stats.chi2_contingency` for each categorical feature vs. the target. Label it *"exploratory sample comparison, not survey-adjusted inference."*
- Class balance: % of class 1. This decides the metric and threshold strategy (Imbalanced Data lecture).
- Lectures: Lecture 3 (EDA, correlation), Data Cleansing (outliers).

---

## PART 2 — Methods

**In the notebook:**
```markdown
## 2. Methods
This part describes how we answer the question. We split the data, build a preprocessing pipeline, set a baseline, and train three algorithms on two feature sets with the same settings. All choices are made on the training and validation data, so the test set stays untouched for a fair final evaluation.
```

## Step 5 — Split and preprocessing pipeline

> **Prompt:** "Split the data into 60% train, 20% validation and 20% test, stratified by the target, with seed 42. Split only once, so Group A and Group B use exactly the same people. Then build a preprocessing pipeline that fills missing values (most frequent for categories, median for numbers), one-hot encodes the categories and scales the numbers. It must be fitted on the training data only. Don't touch the test set again until the final evaluation."

**In the notebook:**
```markdown
### 2.1 Data split and preprocessing
Splits the data once into train (60%), validation (20%) and test (20%), so both feature sets use exactly the same people. Missing values are filled and categories are encoded inside a pipeline that is fitted on the training data only, to avoid data leakage.
```

- Split once on the cleaned table, then select columns for A or B. Both feature sets therefore use the **same people in the same split**.

```python
X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y, test_size=0.20, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)  # 0.25 * 0.80 = 0.20
```

- Preprocessing with `ColumnTransformer` inside a `Pipeline`, fitted on training data only:
  - Categorical: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`.
  - Numeric: `SimpleImputer(strategy="median")` → `StandardScaler()` (needed for LR; harmless for trees).
- The test set is not used again until Step 9.
- Lectures: KNN (stratify, random seed), Advanced Classification ("wrap preprocessing in a Pipeline"), Data Cleansing.

## Step 6 — Baseline

> **Prompt:** "Add a simple baseline that always predicts 'sufficient sleep', the most common answer. Show its accuracy, precision, recall, F1, ROC-AUC and Average Precision on the validation set, and explain in one or two sentences why its accuracy looks good even though it never finds anyone who sleeps too little."

**In the notebook:**
```markdown
### 2.2 Baseline
A model that always predicts "sufficient sleep", the most common answer. It shows the score we get without learning anything, so we can tell whether our models really help.
```

- `DummyClassifier(strategy="most_frequent")`. It always predicts 0 ("sufficient sleep").
- Report on validation: accuracy, precision, recall, F1, ROC-AUC, Average Precision.
- Expected: accuracy ≈ share of class 0, recall = 0, ROC-AUC = 0.5.
- Lecture: Imbalanced Data ("Baseline first", the accuracy paradox).

## Step 7 — Model 1: Group A (structural only)

> **Prompt:** "Train Model 1 using only the Group A background variables. Use three algorithms: Logistic Regression, Random Forest and Gradient Boosting. Tune each one with a random search of 20 combinations, training on the train set, scoring on the validation set, and using Average Precision as the score. Use the parameter ranges from the plan. Show the 6 metrics on the validation set for each model, pick the best decision threshold on the validation set by maximizing F1, and check that the models aren't overfitting by comparing train and validation scores."

**In the notebook:**
```markdown
### 2.3 Model 1: structural variables only (Group A)
Trains and tunes Logistic Regression, Random Forest and Gradient Boosting using only the background variables. The best settings and decision thresholds are chosen on the validation set, never on the test set.
```

Three algorithms, each in the same pipeline from Step 5. **The same search budget is used for Group A and Group B.**

Tuning:
- `RandomizedSearchCV(n_iter=20, scoring="average_precision", random_state=42, n_jobs=-1)`.
- `cv=PredefinedSplit` (train = fit, validation = score). This means we tune on train/validation only, as the proposal says.
- `refit=False`. Then refit the best settings on train.

| Algorithm | scikit-learn class | Search space |
|---|---|---|
| Logistic Regression | `LogisticRegression(max_iter=1000)` | `C`: [0.01, 0.1, 1, 10]; `class_weight`: [None, "balanced"] |
| Random Forest | `RandomForestClassifier` | `n_estimators`: [200, 400]; `max_depth`: [6, 10, 14, 20]; `max_features`: ["sqrt", "log2"]; `min_samples_leaf`: [5, 20, 50]; `class_weight`: [None, "balanced"] |
| Gradient Boosting | `HistGradientBoostingClassifier` (see Decisions) | `max_iter`: [200, 500, 800]; `learning_rate`: [0.01, 0.05, 0.1]; `max_depth`: [3, 5, 7]; `class_weight`: [None, "balanced"] |

Then, on the validation set:
- Report the 6 metrics for each algorithm.
- **Choose the decision threshold** for each model on validation by sweeping 0.05–0.95 and picking the value that maximizes F1 (Imbalanced Data lecture: "threshold tuning first"). Fix it now.
- Train vs. validation gap: check for overfitting (Decision Tree lecture).
- Lectures: Advanced Classification (RF, GB, RandomizedSearchCV, ranges), Decision Tree (`class_weight`, `max_depth`), Imbalanced Data, Beyond Accuracy.

## Step 8 — Model 2: Group A + Group B (extended)

> **Prompt:** "Now do exactly the same thing for Model 2, using Group A plus the Group B health and lifestyle variables. Keep everything else the same as in Model 1: the same three algorithms, the same parameter ranges, the same number of search tries, the same split and the same way of choosing the threshold. That way the only difference between the two models is the extra variables."

**In the notebook:**
```markdown
### 2.4 Model 2: structural + health and lifestyle variables (Group B)
Repeats exactly the same procedure with the health and lifestyle variables added. Because everything else is identical, any difference from Model 1 comes from the added variables.
```

- Exactly the same as Step 7: same algorithms, search spaces, `n_iter`, split, threshold method.
- Only the input columns change.

---

## PART 3 — Results

**In the notebook:**
```markdown
## 3. Results
This part evaluates all models once on the test set and compares Model 1 and Model 2 within each algorithm, which answers the research question. It then looks at which variables drive the predictions, and adds an exploratory clustering.
```

## Step 9 — Final test evaluation (the test set is used once)

> **Prompt:** "Now evaluate all 6 models (3 algorithms × 2 feature sets) on the test set, once, with the settings and thresholds we already chose. Show accuracy, precision, recall, F1, ROC-AUC, Average Precision and a confusion matrix for each. Plot ROC curves and precision-recall curves comparing Model 1 and Model 2 for each algorithm. Make one main table with the Model 1 score, the Model 2 score and the difference for each algorithm. Report the numbers honestly, even if the improvement is small."

**In the notebook:**
```markdown
### 3.1 Test results: Model 1 vs. Model 2
Evaluates all six models once on the untouched test set. The gap between Model 2 and Model 1 within each algorithm is the answer to our research question.
```

- For each algorithm × feature set (6 models), using the settings and thresholds fixed in Steps 7–8:
  - accuracy, precision, recall, F1 (`classification_report`)
  - ROC-AUC (`roc_auc_score`) and Average Precision (`average_precision_score`), computed from `predict_proba`
  - confusion matrix (`ConfusionMatrixDisplay`)
- Figures: ROC curves (`RocCurveDisplay`) and precision–recall curves (`PrecisionRecallDisplay`), A vs. B on the same plot for each algorithm.
- **Main table:** for each algorithm, the Group A score, the Group B score and the gap (B − A) for ROC-AUC and Average Precision.
- Optional: bootstrap 95% confidence interval for the gap (1,000 resamples of the test set). This shows whether the gap is larger than random noise.
- Report the result honestly, even if the gap is small.
- Lecture: Beyond Accuracy (PR/ROC curves, AUC, PR curves for imbalanced data).

## Step 10 — Interpretation

> **Prompt:** "Help us understand what the models learned. Show the Logistic Regression results as odds ratios, and explain that they show direction of association, not causes. Calculate permutation importance for all three Model 2 models on the test set, so we can see which variables matter most, and note that related variables share importance. Then check whether the models work equally well for different age groups, men and women, and income groups."

**In the notebook:**
```markdown
### 3.2 Interpretation
Shows which variables the models rely on (odds ratios and permutation importance) and whether the models work equally well for different groups. This helps explain the results, but it does not show what causes short sleep.
```

- **Logistic Regression coefficients** (Group B model): `exp(coef_)` = odds ratios. These give the direction of association, **not** causes (Linear Regression lecture: "coefficients describe associations").
- **Permutation importance** for all 3 algorithms on Group B:
  - `sklearn.inspection.permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=10, random_state=42)`.
  - Run it on the full pipeline, so each original variable is shuffled as one unit.
  - Use a sample of ~20,000 test rows if it is too slow.
  - Note that correlated features (for example `GENHLTH` and `PHYSHLTH`) share importance.
- **Subgroup performance:** ROC-AUC and Average Precision on the test set by age group, sex and income group.
- Optional: SHAP with `TreeExplainer` on the GB model (needs `pip install shap`).
- Lectures: Explainable AI (permutation importance, SHAP), Linear Regression (coefficients).

## Step 11 — Clustering (optional, supplementary)

> **Prompt:** "As an extra exploration, cluster people by their health and lifestyle answers using K-Means. Scale the data first, try 2 to 10 clusters, and choose the number with the elbow method and the silhouette score (on a sample, since the data is large). Then show the insufficient-sleep rate in each cluster and plot the clusters in 2D with PCA. Label this section as exploratory."

**In the notebook:**
```markdown
### 3.3 Clustering (exploratory)
Groups people with similar health and lifestyle profiles and compares short-sleep rates between the groups. This is an extra exploration of the data's structure, not part of the main answer.
```

- Features: the Group B health/lifestyle variables (not the target), scaled with `StandardScaler`.
- `KMeans(n_clusters=k, n_init=10, random_state=42)` for k = 2…10.
- Choose k with the elbow method (inertia) and `silhouette_score`. Compute silhouette on a sample of ~20,000 rows, because it is slow on large data.
- Insufficient-sleep rate per cluster (the target is used only after clustering).
- Optional: `PCA(n_components=2)` scatter plot colored by cluster.
- Label it *"exploratory, supplementary."*
- Lectures: Unsupervised Learning and Clustering (K-Means, silhouette, elbow), Multidimensional Unsupervised Learning (scaling, PCA).

---

## PART 4 — Discussion

## Step 12 — Discussion and limitations

> **Prompt:** "Write a short discussion based only on the results we actually got. Answer our research question: how well do the background variables predict short sleep, and how much do the health variables add? Then list the limitations: self-reported answers, missing data, the CDC-imputed race variable, unweighted models, the chance that poor sleep causes some of the health answers, and confounding. End with what we would do differently and what is still unknown."

**In the notebook:**
```markdown
## 4. Discussion
Explains what the results mean for our research question, whether our approach was appropriate, the limitations of the data and methods, what we would do differently, and what remains unknown.
```

- Answer the question: how good is Group A, and how much does Group B add?
- Limitations:
  - Self-reported data.
  - Missing data and the decisions made about it.
  - CDC imputation in `_IMPRACE`.
  - Models are unweighted, so results describe the sample, not the U.S. population.
  - **Reverse causation:** poor sleep may cause some Group B answers (for example poor mental health).
  - Splitting features into groups does not remove confounding.
- What we would do differently, and what is still unknown.

## Step 13 — AI usage statement

> **Prompt:** "Update the AI usage section at the end of the notebook. Say which AI tools we used and what we used them for, like planning, writing code, debugging and drafting text, and list all the prompts we gave, in order."

**In the notebook:**
```markdown
## 5. AI usage statement
Lists the AI tools we used, what we used them for, and the prompts we gave, as required by the course.
```

- Tools used, what they were used for, and the prompts (required by the course instructions).

---

## Decisions to confirm before coding

1. **Gradient Boosting class.** The lecture uses `GradientBoostingClassifier`. With ~270,000 training rows and 800 trees it is very slow. `HistGradientBoostingClassifier` is the same boosting idea, much faster, and handles NaN by itself. Recommended. We must explain it in the report.
2. **Tuning method.** A single train/validation split (`PredefinedSplit`, as in the proposal) instead of 5-fold CV (as in the lecture). It is faster, and it matches the proposal's 60/20/20 split.
3. **Missing income.** If `_INCOMG1` has many missing values, either impute it with the mode or keep "missing" as its own category. Decide after Step 3.
4. ~~**BRFSS 2020 file** for the validation step.~~ Done: downloaded by the notebook's Data cell.

## Methods not covered in the lectures (explain briefly in the report)

The instructions allow these if we explain the idea and why we chose them.

- Logistic Regression. The lectures cover linear regression; LR appears only in the XAI code example.
- Chi-square test (`chi2_contingency`).
- Average Precision as a single number for the PR curve. The PR curve itself is covered.
- `HistGradientBoostingClassifier`, if chosen.
- Bootstrap confidence interval, if used.

## Submission checklist (from the course instructions)

- [ ] Notebook runs top to bottom with no errors (Kernel → Restart & Run All).
- [ ] Notebook order matches the report: background/EDA → methods → results → discussion.
- [ ] Every non-trivial step has a short markdown note: what it does and why.
- [ ] No abandoned attempts or leftover debug output.
- [ ] Every figure and table in the report can be reproduced from the notebook.
- [ ] Report (3–5 pages, PDF or Word) has Background, Question (and what changed and why), Methods (with packages and parameters), Results, Discussion, AI usage statement.
- [ ] The report says which notebook section produced each result.
