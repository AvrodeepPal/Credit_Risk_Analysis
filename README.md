# Credit Risk Analysis and Loan Approval Prediction

## Overview

This repository contains a comprehensive Jupyter notebook ([CreditRiskEDA_LoanApproval.ipynb](https://github.com/AvrodeepPal/Credit_Risk_Analysis/blob/main/CreditRiskEDA_LoanApproval.ipynb)) and accompanying visualizations (in `/images`) that demonstrate end-to-end credit risk modeling for loan approval.

---

## Dataset Description

- **Records:** 51,336 loan applications  
- **Features:** 87 attributes including demographics, financial metrics, credit history, and trade line details  
- **Target:** `Approved_Flag` with four risk categories (P1–P4)  
- **Class Imbalance:** P2 (62.7%), P3 (14.5%), P4 (11.5%), P1 (11.3%)

---

## 1. Exploratory Data Analysis (EDA)

![Approval Status Distribution](https://github.com/AvrodeepPal/Credit_Risk_Analysis/raw/main/images/ApprovalStatusBoxPlot.png)

- Visualized class distribution and feature histograms  
- Identified demographic and credit-behavior patterns  
- **Summary:** Credit score and age clearly stratify risk; product enquiry behavior differs by category  

---

## 2. Feature Selection & Engineering

- Applied ANOVA F-tests to rank 81 numeric features  
- Selected top 20 features (e.g., `Credit_Score`, `enq_L3m`, `Age_Oldest_TL`)  
- Encoded 19 categorical features via one-hot encoding  
- **Summary:** Reduced feature set by 55% while preserving predictive power; 42 common features ensure deployment compatibility  

---

## 3. Model Development & Optimization

![Model Performance Comparison](https://github.com/AvrodeepPal/Credit_Risk_Analysis/raw/main/images/ModelPerformanceComparison.png)

- Trained and compared 8 algorithms (Logistic Regression, Random Forest, XGBoost, CatBoost, etc.)  
- Addressed class imbalance with sample weights and SMOTE-Tomek  
- Implemented ensemble methods: soft voting, weighted voting, stacking  

---

## 4. Confusion Matrix Analysis

![Confusion Matrices](https://github.com/AvrodeepPal/Credit_Risk_Analysis/raw/main/images/ConfusionMatrixAllModels.png)

- Generated confusion matrices for all models  
- Evaluated class-wise performance for each risk category  
- **Summary:** Stacking ensemble shows the best balance across P1–P4 classes  

---

## 5. Advanced Model Optimization & Ensembles

- **XGBoost (Class Weights):** F1 (Macro)=70.59%, Balanced Acc=76.21%  
- **Logistic Regression (SMOTE-Tomek):** F1 (Macro)=67.22%, Balanced Acc=72.81%  
- **Stacking Ensemble:** F1 (Macro)=70.69%, Balanced Acc=69.23%  
- **Hybrid Ensemble (Stacking + Weighted Voting):** F1 (Macro)=70.06%, Balanced Acc=68.54%  

**Conclusion:** The **stacking classifier** remains the optimal model, offering the best macro F1 and balanced accuracy for credit risk prediction. 
