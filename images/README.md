# 🖼️ Credit Risk Analysis - Visual Artifacts

This folder contains the key visualizations from the credit risk and loan approval workflow.
The figures are numbered in the same order as the [main README](../README.md): target distribution, exploratory analysis, then model evaluation.

| Figure | Chart | Stage |
|---|---|---|
| 1 | Approval flag distribution | Dataset |
| 2 | Credit score distribution | EDA |
| 3 | Feature box plots by approval status | EDA |
| 4 | Top 20 columns with missing values | EDA |
| 5 | Correlation heat map | EDA |
| 6 | Model performance comparison | Modeling |
| 7 | Confusion matrices, all models | Evaluation |

---

## 1. Approval Flag Distribution

<p align="center">
  <img src="ApprovalFlagDistribution.png" alt="Pie chart of approval flags: P2 62.7%, P3 14.5%, P4 11.5%, P1 11.3%" width="460">
</p>

<p align="center"><em>Figure 1: Distribution of the <code>Approved_Flag</code> target.</em></p>

**Insight:**
P2 accounts for 62.7% of applications, with P3, P4 and P1 each between 11% and 15%. Accuracy alone would reward predicting P2, so macro F1 and balanced accuracy are used for evaluation.

---

## 2. Credit Score Distribution

<p align="center">
  <img src="CreditScoreDistribution.png" alt="Histogram of credit scores peaking around 670 to 690" width="700">
</p>

<p align="center"><em>Figure 2: Credit score distribution.</em></p>

**Insight:**
Scores are tightly concentrated between roughly 650 and 710, with a long thin tail toward 500, so small score differences carry a lot of signal.

---

## 3. Approval Status Box Plots

<p align="center">
  <img src="ApprovalStatusBoxPlot.png" alt="Box plots of credit score, age, total trade lines and log net monthly income by approval flag" width="850">
</p>

<p align="center"><em>Figure 3: Credit score, age, total trade lines and log(net monthly income) by approval category.</em></p>

**Insight:**
Credit score separates P1 to P4 almost cleanly and age trends downward with risk, while income barely differs across classes.

---

## 4. Top 20 Columns With Missing Values

<p align="center">
  <img src="Top20ColumnsWithMissingValues.png" alt="Bar chart of the 20 columns with the most missing values, led by CC_utilization and PL_utilization" width="750">
</p>

<p align="center"><em>Figure 4: Top 20 columns by missing-value count.</em></p>

**Insight:**
Utilization and delinquency-history fields are missing for most applicants, often because the applicant has no such product or event. That makes missingness informative rather than noise.

---

## 5. Correlation Heat Map

<p align="center">
  <img src="CorrelationHeatMap.png" alt="Lower-triangle correlation heat map of numeric features" width="750">
</p>

<p align="center"><em>Figure 5: Correlation matrix of numeric features.</em></p>

**Insight:**
Trade-line counts, delinquency counts and enquiry counts each form strongly correlated blocks, which motivates reducing the feature set before modeling.

---

## 6. Model Performance Comparison

<p align="center">
  <img src="ModelPerformanceComparison.png" alt="Grouped bar chart of accuracy, precision, recall and F1 for eight models" width="850">
</p>

<p align="center"><em>Figure 6: Accuracy, precision, recall and F1 for the eight base models.</em></p>

**Insight:**
CatBoost and XGBoost lead at around 0.80 accuracy, the tree ensembles and logistic regression follow closely, and Naive Bayes trails far behind.

---

## 7. Confusion Matrices - All Models

<p align="center">
  <img src="ConfusionMatrixAllModels.png" alt="Confusion matrices for eight models on the four risk classes" width="900">
</p>

<p align="center"><em>Figure 7: Confusion matrices for all eight base models (classes 0–3 correspond to P1–P4).</em></p>

**Insight:**
Every model predicts the majority P2 class well. The hardest class is P3, which is mostly misclassified as P2, and CatBoost and XGBoost recover the most P1 and P4 applicants.
