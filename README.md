# LightHR Dataset — Hiring Prediction Analysis

A end-to-end machine learning pipeline for predicting whether a job applicant will be hired within six months, using the `LightHR_dataset.csv` dataset. The notebook covers four major stages: clustering, association rule mining, classification, and fairness evaluation.

---

## Overview

The notebook explores the question: *what applicant characteristics predict successful hiring?* It approaches this from multiple angles — unsupervised clustering to find natural applicant groupings, association rules to surface interpretable hiring patterns, and supervised classifiers to predict outcomes — before critically examining the fairness and explainability of those predictions.

---

## Dataset

**File:** `LightHR_dataset.csv`

**Key features:**
- Demographic: Gender, Age, Address (county)
- Academic: University Name, Education level, A-levels scores
- Employment: Current Employment, Current Employment Sector, Current Annual Income
- References: Refree 1, Refree 2, Refree 3, References Submitted
- Application: Year of Application
- **Target:** `Was Hired by a Company within 6 months` (binary: 0/1)

---

## Notebook Structure

### 1. Clustering

Segments applicants into distinct profiles using **K-Medoids** clustering with Manhattan distance, evaluated via silhouette score.

**Key steps:**
- Data cleaning: university name normalisation, gender/education/address standardisation, missing value imputation
- Feature engineering: five dataset variants tested (raw binary employment flag, binned university tier, scaled versions)
- Dimensionality reduction: PCA variance analysis (85% and 95% thresholds)
- Cluster profiling: hire rates, education mix, and employment breakdown per cluster

**Best variant selected** based on silhouette score, then visualised in PCA space.

---

### 2. Association Rule Mining

Discovers interpretable if-then rules linking applicant attributes to hiring outcomes using the **Apriori** algorithm.

**Key steps:**
- Discretisation of continuous features (A-levels, Income, Age, Year) using uniform and quantile strategies with dynamic bin labelling
- University names binned into Elite / Strong / Other tiers
- Referee employment types extracted and encoded
- Transaction matrix built (one record per applicant, items = feature=value pairs)
- Sensitivity analysis across confidence thresholds (0.40–0.70) to select `min_confidence = 0.55` and `min_lift = 1.2`
- Rules filtered separately for `Hired=Yes` and `Hired=No` consequents

---

### 3. Classification

Trains and evaluates four classifiers to predict the hiring outcome on both a raw dataset and an engineered dataset.

**Models:**
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

**Key steps:**
- 60/20/20 train/validation/test split (stratified)
- Baseline training with default parameters
- Hyperparameter tuning via `RandomizedSearchCV` (5-fold stratified CV)
- Evaluation on Accuracy, F1-Score, and ROC-AUC
- Confusion matrix analysis
- Feature removal experiment (A-levels scores dropped) to test model robustness

**Two dataset variants compared:** raw (minimal preprocessing) vs. engineered (scaled, binned, feature-engineered).

---

### 4. Explainability Analysis

Explains the best-performing model's predictions using two complementary methods.

**SHAP (global + local):**
- Global feature importance (bar and beeswarm plots)
- Feature effect plots (violin/summary)
- Dependence plots for the top 4 features
- Force plots for individual hired vs. not-hired candidates
- Sensitivity analysis: prediction probability as a function of top feature value

**LIME (local):**
- 10 random test samples explained individually
- Top 5 contributing features per sample visualised as bar charts

**SHAP vs LIME comparison** table covering theoretical foundation, speed, consistency, and visualization types.

---

### 5. Fairness Evaluation

Audits the model for group and individual fairness across four sensitive attributes: Gender, Age, Education, and Address.

**Group fairness metrics:**
- Disparate Impact (DI)
- Statistical Parity Difference (SPD)
- Equal Opportunity Difference (EOD)
- Average Odds Difference (AOD)
- False Positive Rate Difference (FPRD)

**Individual fairness metrics:**
- Consistency Score (k-NN agreement)
- Pairwise Fairness (similar applicants, similar outcomes)
- Cross-group Fairness (fairness across group boundaries)
- FN/FP disparity between privileged and unprivileged groups

Results visualised as hire-rate bar charts, TPR/FPR comparisons, confusion matrices per group, radar charts, and a summary fairness heatmap.

---

## Dependencies

```
numpy==1.26.4
pandas==2.1.4
scikit-learn==1.3.2
scikit-learn-extra==0.3.0
scipy==1.11.4
matplotlib==3.8.2
seaborn==0.13.2
xgboost==2.0.3
mlxtend==0.23.1
shap==0.44.1
lime==0.2.0.1
```

Install all dependencies by running the first cell in the notebook, which executes:

```bash
pip install numpy==1.26.4 pandas==2.1.4 scikit-learn==1.3.2 ...
```

---

## Usage

1. Place `LightHR_dataset.csv` in the same directory as the notebook.
2. Install dependencies (first cell).
3. Run cells sequentially — each section depends on objects defined in the previous one.

> **Note:** SHAP and LIME are imported with graceful fallbacks. If either library fails to load, the relevant cells will print a warning and skip those sections.

---

## File Structure

```
.
├── final_notebook.ipynb   # Main analysis notebook
├── LightHR_dataset.csv    # Input dataset (required)
└── README.md              # This file
```
