# Analysis Plan — Predicting Insufficient Sleep (BRFSS 2022)

Step-by-step plan for the notebook `sleep_prediction_brfss.ipynb`.
Sources: the revised proposal (`Data_Science_Proposal_Final.docx`), the course instructions (`Final_project_instructions_1.pdf`) and the lecture slides in this folder.

Each step ends with a **Results** block: what we got (numbers from the notebook output), the conclusion, decisions we made that were not in the plan, and what is important to remember for the report and the Q&A. It is updated after every step.

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

#### Results — Step 0
- **What we got:** pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2, scipy 1.18.1. `RANDOM_STATE = 42`. `DATA_2022_PATH` and `DATA_2020_PATH` are defined once.
- **Conclusion:** the environment is fixed and recorded, so the results can be reproduced.
- **Decisions not in the plan:**
  - seaborn is not imported. It was not in our list, and all plots use matplotlib.
  - No global figure style was set.
  - **A download cell was added after the setup cell.** It downloads the 2022 and 2020 files from the CDC if they are missing, because the files (about 1 GB each) are too large to submit and the notebook must run top to bottom on any computer. It reuses the paths from the setup cell.
- **Important:** the notebook must use the `.venv` kernel (Python 3.14.5), where these packages are installed.

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

#### Results — Step 1
- **What we got:** 445,132 rows (as expected), 328 columns in the raw file (as expected), 21 columns kept, none missing. Loading takes about 1 minute.
- **Conclusion:** the file is complete and has the structure the CDC describes.
- **Decisions not in the plan:** the column lists (`TARGET_COL`, `GROUP_A_COLS`, `GROUP_B_COLS`, `WEIGHT_COL`) are defined here and reused in all later steps. The cell stops with an error if the row or column counts are wrong.
- **Important — found while loading:**
  - `_DRNKWK2` shows 0 as `5.397605e-79`. This is a known problem when pandas reads SAS XPT files. It affects 188,832 rows and is fixed in Step 3.
  - The file inside each CDC zip ends with a space (`"LLCP2022.XPT "`). The download cell handles this.

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

#### Results — Step 2
- **What we got:** 51 states + DC kept. 390,193 respondents used (4,638 excluded: 77, 99 or missing). Unweighted 30.5%, **weighted 32.7%**, CDC published 32.7%. Difference: +0.02 percentage points.
- **Conclusion:** our target definition and our use of the survey weight reproduce the CDC's published number. This supports the pipeline. It is a validation check, not a finding.
- **Decisions not in the plan:** the function `make_insufficient_sleep()` is defined here and reused in Step 3, so 2020 and 2022 use exactly the same target definition.
- **Important:**
  - The weights matter: without them the result is 30.5%; with them it matches the CDC.
  - The 2020 file has no U.S. Virgin Islands (code 78), only Guam and Puerto Rico.

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
| `_DRNKWK2` | 0–98999 | 99900 | **tiny values (~5.4e-79) → 0** (SAS XPT read artifact; 188,832 rows in the loaded data); divide by 100 (2 implied decimals) | numeric |
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

#### Results — Step 3
- **What we got:**
  - Every code in the table above was checked against the 2022 codebook and applied. After cleaning, a check confirms that no invalid code is left.
  - 5,453 people had no valid sleep answer and were dropped → **439,679 remain**.
  - Regions: South 123,276, Midwest 114,570, West 110,175, Northeast 82,557, Territories 9,101.
  - Missing values: income 21.3% (highest), heavy drinking and drinks per week 11.0%, BMI 10.8%, smoking 7.9%, all others under 2.5%. **33.8% of rows** have at least one missing feature.
  - 71 fully identical rows (0.016%). Ignoring the survey weight, 605 rows are identical.
- **Conclusion:** the data is clean and every feature is kept. Missing values will be filled after the split.
- **Decisions (with reasons):**
  - Keep all features: income (21.3%) is below the 40% drop-column rule and is a key structural variable.
  - Do not drop rows with missing values: that would lose a third of the data and could bias the sample.
  - Fill missing values after the split (most frequent / median), on training data only, to avoid leakage.
  - Keep the 71 identical rows: there is no respondent ID, people can give the same answers, and they are too few to matter.
  - `_STATE` is removed after mapping to `region`.
- **Decisions not in the plan:**
  - `_DRNKWK2`: tiny values → 0 with `round(2)`, then divided by 100 (the codebook confirms 0 = "did not drink" and 2 implied decimals).
  - `GROUP_A_FEATURES`, `GROUP_B_FEATURES` and `NUMERIC_FEATURES` are defined here for the models.
  - The "Cleaning decisions" table was removed from the notebook; the decisions are recorded here and must go into the report.
- **Important:** `_IMPRACE` contains CDC-imputed values. This is not in the notebook yet and must be mentioned in the limitations.

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

#### Results — Step 4
- **What we got:**
  - **Prevalence 2022:** 32.8% unweighted, **36.1% weighted**. This includes the 3 territories, and it is higher than 2020 (32.7%).
  - **Sleep hours:** most people report 7 or 8 hours; 6 hours is the third most common answer.
  - **Weighted short-sleep rate by group:**
    - Age: peaks at 35–39 (42%), lowest at 80+ (24%).
    - Sex: male 37%, female 35%.
    - Income: 42% (<$15k) down to 30% ($200k+).
    - Education: college graduates 31%, other groups 37–39%.
    - Race: Black 46%, Other 46%, American Indian / Alaska Native 42%, Hispanic 36%, Asian 35%, White 33%.
    - Region: 34–38%.
    - General health: 29% (excellent) up to 54% (poor).
    - Depression: 45% (yes) vs. 34% (no).
  - **Numeric check:** health days stay within 0–30. Drinks per week: mean 3.05, max 532; 344 people report more than 100 a week, and 55.8% of them sleep less than 7 hours (vs. 32.7% of everyone else who answered).
  - **Correlation with the target (all weak):** poor mental health days 0.18, general health 0.14, poor physical health days 0.12, age −0.10, income −0.08, education −0.08, BMI 0.07, drinks 0.01. Strongest between features: general health – physical health days 0.49, income – education 0.42.
  - **Chi-square (exploratory only):** every p-value is tiny because of the large sample. Cramér's V: general health 0.144, employment 0.137, age 0.125, race 0.104, smoking 0.098, depression 0.092 … sex 0.008, heavy drinking 0.007.
  - **Class balance:** 67.2% vs. 32.8% (about 2 to 1).
- **Conclusion:** short sleep differs between groups, but every single association is weak, so we expect only modest prediction. Health variables (general health, depression, mental health days) show some of the larger differences, which suggests Group B may add information. The data is moderately imbalanced, so accuracy is not a useful metric. We focus on recall, F1, ROC-AUC and Average Precision, and tune the threshold.
- **Decisions not in the plan:**
  - Added **Cramér's V** next to the chi-square tests, because p-values are meaningless with ~440,000 people. It is not covered in the lectures, so it must be explained in the report.
  - Readable category labels from the codebook in the charts.
  - **Extreme drink values are kept** (team decision): they are real survey answers (7 days × up to 76 drinks), very few (344), carry signal (55.8% short sleep) and hardly affect the models.
  - The EDA is split into 7 cells instead of one, so each chart has its own explanation.
- **Important:** the weighted prevalence (36.1%) is a descriptive estimate. The models are unweighted.

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

#### Results — Step 5
- **What we got:** train 263,807 rows (60%), validation 87,936 (20%), test 87,936 (20%). All three have 32.8% short sleepers. A check confirms that the sets do not share any respondent. After preprocessing: Group A 8 features → 51 model columns; Group B 19 features → 79 model columns; 0 missing values left.
- **Conclusion:** both models use exactly the same people, the class balance is the same in every set, and the preprocessing leaves no missing values.
- **Decisions not in the plan:**
  - `make_preprocessor(features)` builds the preprocessing for either feature list, so Model 1 and Model 2 use identical steps.
  - `OneHotEncoder(sparse_output=False)`: dense output, which Gradient Boosting needs.
- **Important:** the test set has not been used since this step and must stay untouched until Step 9.

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

#### Results — Step 6
- **What we got (validation set):** accuracy 0.672, precision 0.000, recall 0.000, F1 0.000, ROC-AUC 0.500, Average Precision 0.328.
- **Conclusion:** the baseline's accuracy equals the share of people who sleep enough (67.2%), but it finds none of the short sleepers. This is the accuracy paradox from the Imbalanced Data lecture. A useful model must beat ROC-AUC 0.5 and Average Precision 0.328, and have recall above 0.
- **Decisions not in the plan:** the function `evaluate()` is defined here and used for every model, so all models are measured the same way. Precision and F1 are set to 0 when a model never predicts class 1.
- **Important:** Average Precision 0.328 is the lowest possible value for any model (it equals the share of short sleepers).

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

#### Results — Step 7
- **What we got (validation set; accuracy, precision, recall and F1 use the tuned threshold):**

  | Algorithm | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision | Train AP | AP gap |
  |---|---|---|---|---|---|---|---|---|---|
  | Logistic Regression | 0.25 | 0.459 | 0.364 | 0.866 | 0.513 | 0.626 | 0.434 | 0.435 | 0.000 |
  | Random Forest | 0.27 | 0.490 | 0.374 | 0.819 | 0.513 | 0.632 | 0.444 | 0.467 | 0.024 |
  | Gradient Boosting | 0.41 | 0.479 | 0.371 | 0.839 | 0.514 | 0.633 | 0.444 | 0.449 | 0.005 |
  | Baseline (always 0) | – | 0.672 | 0.000 | 0.000 | 0.000 | 0.500 | 0.328 | – | – |

  Best settings: Logistic Regression `C=10, class_weight=None`; Random Forest `n_estimators=400, max_depth=14, max_features="log2", min_samples_leaf=20, class_weight=None`; Gradient Boosting `max_iter=200, max_depth=3, learning_rate=0.1, class_weight="balanced"`. Run time: about 14 minutes for this cell.
- **Conclusion:** background variables alone carry real but modest signal: ROC-AUC about 0.63 and Average Precision about 0.44, vs. 0.50 and 0.33 for the baseline. This is within the 0.60–0.68 range the proposal expected. The three algorithms are almost tied, so more complex models add little over Logistic Regression. There is no meaningful overfitting (largest train–validation gap: 0.024, Random Forest).
- **Decisions not in the plan:**
  - **Tuning sample:** each search combination is trained on a fixed stratified sample of **50,000 training rows** (the same sample for Model 1 and Model 2), then the best one is retrained on all 263,807 training rows. Without it, tuning both models would take hours.
  - **Gradient Boosting = `HistGradientBoostingClassifier`** with `early_stopping=False` (decision 1 below).
  - Logistic Regression has only 8 possible combinations, so it tries all 8 instead of 20.
  - Each algorithm runs through one function, `tune_models(features)`, so Model 2 uses exactly the same procedure.
- **Important (for the report and Q&A):**
  - Maximizing F1 gives **low thresholds (0.25–0.41)** → high recall (82–87%), low precision (about 37%), and accuracy below the baseline. Flagging everyone would give F1 ≈ 0.49, so the models' F1 (0.51) is only slightly better.
  - Two best settings are at the edge of their ranges: Logistic Regression `C=10` (largest value) and Gradient Boosting's simplest option (200 rounds, depth 3). The lecture warns that the true best value may lie outside the range. Scores barely change between settings, so this is unlikely to matter, but it is worth a sentence in the discussion.
  - These validation scores are slightly optimistic, because the threshold was chosen on the same validation set. The honest scores come from the test set in Step 9.

## Step 8 — Model 2: Group A + Group B (extended)

> **Prompt:** "Now do exactly the same thing for Model 2, using Group A plus the Group B health and lifestyle variables. Keep everything else the same as in Model 1: the same three algorithms, the same parameter ranges, the same number of search tries, the same split and the same way of choosing the threshold. That way the only difference between the two models is the extra variables."

**In the notebook:**
```markdown
### 2.4 Model 2: structural + health and lifestyle variables (Group B)
Repeats exactly the same procedure with the health and lifestyle variables added. Because everything else is identical, any difference from Model 1 comes from the added variables.
```

- Exactly the same as Step 7: same algorithms, search spaces, `n_iter`, split, threshold method.
- Only the input columns change.

#### Results — Step 8
- **What we got (validation set; accuracy, precision, recall and F1 use the tuned threshold):**

  | Algorithm | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision | Train AP | AP gap |
  |---|---|---|---|---|---|---|---|---|---|
  | Logistic Regression | 0.26 | 0.530 | 0.392 | 0.788 | 0.524 | 0.660 | 0.486 | 0.486 | 0.000 |
  | Random Forest | 0.43 | 0.536 | 0.396 | 0.787 | 0.527 | 0.664 | 0.491 | 0.591 | 0.100 |
  | Gradient Boosting | 0.41 | 0.531 | 0.394 | 0.799 | 0.528 | 0.667 | 0.495 | 0.501 | 0.006 |
  | Baseline (always 0) | – | 0.672 | 0.000 | 0.000 | 0.000 | 0.500 | 0.328 | – | – |

  **Model 2 vs. Model 1 (validation):**

  | Algorithm | ROC-AUC M1 → M2 | Gain | AP M1 → M2 | Gain |
  |---|---|---|---|---|
  | Logistic Regression | 0.626 → 0.660 | +0.033 | 0.434 → 0.486 | +0.052 |
  | Random Forest | 0.632 → 0.664 | +0.033 | 0.444 → 0.491 | +0.048 |
  | Gradient Boosting | 0.633 → 0.667 | +0.033 | 0.444 → 0.495 | +0.051 |

  Best settings: Logistic Regression `C=10, class_weight=None`; Random Forest `n_estimators=400, max_depth=14, max_features="sqrt", min_samples_leaf=5, class_weight="balanced"`; Gradient Boosting `max_iter=500, max_depth=3, learning_rate=0.05, class_weight="balanced"`. Run time: about 19 minutes for this cell; the whole notebook now takes about 39 minutes.
- **Conclusion:** adding the health and lifestyle variables improves every algorithm by about the same amount: ROC-AUC +0.033 and Average Precision about +0.05. The gain is consistent but modest. Model 2 is still far from perfect (ROC-AUC about 0.66). The three algorithms remain almost tied, so the improvement comes from the added information, not from the choice of algorithm. These are validation results; the final answer comes from the test set (Step 9).
- **Decisions not in the plan:** none. The cell calls the same `tune_models()` function as Model 1, so the procedure is identical. It also prints a first validation comparison of Model 2 vs. Model 1.
- **Important (for the report and Q&A):**
  - **Random Forest overfits more in Model 2:** train AP 0.591 vs. validation AP 0.491 (gap 0.100, vs. 0.024 in Model 1). Its best settings use small leaves (`min_samples_leaf=5`), so it memorizes more of the training data. Its validation score is still about the same as the other two, so it is not worse, but this should be mentioned. Logistic Regression and Gradient Boosting show almost no gap.
  - Logistic Regression `C=10` is again at the edge of its range.
  - Thresholds are still low (0.26–0.43), so recall stays high (79–80%) and precision low (about 39%), as in Model 1.
  - Reverse causation: some Group B variables (for example poor mental health days) may partly be caused by poor sleep, so part of the gain may reflect that. This belongs in the discussion.

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

#### Results — Step 9
- **What we got (TEST set, used once; accuracy, precision, recall and F1 use the thresholds chosen on validation):**

  | Algorithm | Model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. Precision |
  |---|---|---|---|---|---|---|---|---|
  | Logistic Regression | 1 | 0.25 | 0.456 | 0.362 | 0.862 | 0.510 | 0.625 | 0.435 |
  | Logistic Regression | 2 | 0.26 | 0.531 | 0.393 | 0.789 | 0.525 | 0.660 | 0.488 |
  | Random Forest | 1 | 0.27 | 0.490 | 0.374 | 0.816 | 0.513 | 0.632 | 0.446 |
  | Random Forest | 2 | 0.43 | 0.537 | 0.397 | 0.788 | 0.528 | 0.666 | 0.494 |
  | Gradient Boosting | 1 | 0.41 | 0.478 | 0.370 | 0.839 | 0.514 | 0.633 | 0.446 |
  | Gradient Boosting | 2 | 0.41 | 0.531 | 0.394 | 0.801 | 0.529 | 0.667 | 0.496 |
  | Baseline (always 0) | – | – | 0.672 | 0.000 | 0.000 | 0.000 | 0.500 | 0.328 |

  **Main result — gain from Model 1 to Model 2 (test):**

  | Algorithm | ROC-AUC gain | AP gain | F1 gain | Recall change | Precision gain |
  |---|---|---|---|---|---|
  | Logistic Regression | +0.034 | +0.053 | +0.014 | −0.074 | +0.031 |
  | Random Forest | +0.034 | +0.048 | +0.015 | −0.028 | +0.023 |
  | Gradient Boosting | +0.034 | +0.050 | +0.015 | −0.038 | +0.024 |

  Confusion matrix example (Model 2, Gradient Boosting): finds 23,135 of 28,880 short sleepers (5,745 missed) and flags 35,521 of 59,056 people who sleep enough. The notebook also shows ROC and precision–recall curves; Model 2's curve is above Model 1's for every algorithm.
- **Conclusion (answer to the research question):** structural characteristics alone predict insufficient sleep only modestly (ROC-AUC about 0.63, Average Precision about 0.44, vs. 0.50 and 0.33 for the baseline). Adding the health and lifestyle variables gives a **consistent but modest** improvement with all three algorithms: ROC-AUC +0.034 and Average Precision about +0.05. Even Model 2 reaches only about 0.66 ROC-AUC. Because the three algorithms are almost tied, the improvement comes from the added information, not from the algorithm.
- **Decisions not in the plan:**
  - The optional bootstrap confidence interval for the gap was not done (not requested). The gain is the same for all three algorithms and the test set has 87,936 people, but there is no formal uncertainty estimate.
  - In this scikit-learn version, line colors in the ROC and precision–recall plots are set with `curve_kwargs` (the first run failed with `color=` and was fixed).
  - A short markdown summary of the test results was added under the plots.
- **Important (for the report and Q&A):**
  - **Test matches validation:** all ROC-AUC and Average Precision values are within 0.003 of the validation results, so the choices made on the validation set were not over-fitted.
  - **Compare with ROC-AUC and Average Precision**, not accuracy, recall or F1: each model has its own threshold, so recall drops in Model 2 while precision, accuracy and F1 rise.
  - With F1-based thresholds every model flags many people (many false positives). In a real screening use, the threshold would depend on the cost of each type of error.
  - Reverse causation: part of the gain may come from health answers that are partly caused by poor sleep.
  - Running the whole notebook now takes about 31–39 minutes.

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

#### Results — Step 10
*To be added after this step.*

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

#### Results — Step 11
*To be added after this step.*

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



## Decisions to confirm before coding

1. **Gradient Boosting class.** Done: `HistGradientBoostingClassifier` is used (Step 7). The lecture uses `GradientBoostingClassifier`, which is very slow with ~260,000 training rows and 800 trees; the Hist version is the same boosting idea and much faster. We must explain it in the report.
2. **Tuning method.** Done: a single train/validation split (`PredefinedSplit`, as in the proposal) instead of 5-fold CV (as in the lecture), with the search trained on a 50,000-row sample of the training set (Step 7).
3. **Missing income.** Done: income (21.3% missing) is kept and filled with the most frequent value inside the pipeline, after the split (Steps 3 and 5).
4. ~~**BRFSS 2020 file** for the validation step.~~ Done: downloaded by the notebook's Data cell.

## Methods not covered in the lectures (explain briefly in the report)

The instructions allow these if we explain the idea and why we chose them.

- Logistic Regression. The lectures cover linear regression; LR appears only in the XAI code example.
- Chi-square test (`chi2_contingency`).
- Cramér's V (strength of association next to the chi-square test, Step 4).
- Average Precision as a single number for the PR curve. The PR curve itself is covered.
- `HistGradientBoostingClassifier` (used, Step 7).
- Bootstrap confidence interval, if used.

## Submission checklist (from the course instructions)

- [ ] Notebook runs top to bottom with no errors (Kernel → Restart & Run All).
- [ ] Notebook order matches the report: background/EDA → methods → results → discussion.
- [ ] Every non-trivial step has a short markdown note: what it does and why.
- [ ] No abandoned attempts or leftover debug output.
- [ ] Every figure and table in the report can be reproduced from the notebook.
- [ ] Report (3–5 pages, PDF or Word) has Background, Question (and what changed and why), Methods (with packages and parameters), Results, Discussion, AI usage statement.
- [ ] The report says which notebook section produced each result.
