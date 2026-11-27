# Machine Learning-Based Credit Card Fraud Detection

**A comparative evaluation of six classifiers and the effect of SMOTE class balancing on imbalanced transaction data**

This repository contains the code and outputs for an MSc project that compares six supervised machine learning algorithms for credit card fraud detection, and tests how SMOTE oversampling, at two different ratios, changes their performance on highly imbalanced data.

---

## Overview

Fraud makes up less than 0.2% of transactions in the dataset, so a model can be over 99% accurate while catching no fraud at all. This project therefore:

- compares **six classifiers**: Logistic Regression, Decision Tree, Random Forest, XGBoost, Support Vector Machine (SVM) and Naive Bayes
- trains each one under **three conditions**: no resampling, SMOTE at **50:50** and SMOTE at **80:20**
- evaluates all **18 models** on the same untouched test set using **Precision, Recall, F1-score, ROC-AUC and PR-AUC**, plus confusion-matrix counts
- uses **SHAP** to explain which features drive the XGBoost model's predictions

## Key results

| Finding | Result |
|---|---|
| Best model (F1-score) | Random Forest: 0.8313 without SMOTE, 0.8324 with SMOTE |
| Best PR-AUC | XGBoost without SMOTE: 0.8264 |
| Effect of SMOTE | Lowered Precision for **every** model; improved F1 for only two (SVM at 80:20, and Random Forest negligibly) |
| Precision collapse at 50:50 | Logistic Regression 0.053, SVM 0.107 |
| 80:20 vs 50:50 (all models) | False alarms fell from 3,644 to 1,883, with only 6 fewer frauds detected |
| ROC-AUC vs PR-AUC | SMOTE raised ROC-AUC for Logistic Regression and SVM even where PR-AUC fell |
| Top SHAP features | V14 and V4, followed by V11, V8, V12 and V1 |

**Main conclusion:** tree ensembles trained on the original data are the most reliable choice. If resampling is used, a mild ratio such as 80:20 is clearly safer than full 50:50 balancing, and Precision and PR-AUC should always be reported.

## Dataset

The project uses the public **Credit Card Fraud Detection** dataset from the Machine Learning Group, Université Libre de Bruxelles (ULB):

- Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions by European cardholders over two days in September 2013, of which 492 are fraud
- Features: `Time`, `Amount`, 28 anonymised PCA components (`V1` to `V28`) and the label `Class` (1 = fraud)

The dataset is **not included** in this repository because of its size (about 150 MB) and licence. Download `creditcard.csv` from Kaggle and place it in the project folder before running the code.

## Method

1. **Duplicate removal:** 1,081 exact duplicate rows are removed *before* splitting, leaving 283,726 transactions (473 fraud). This prevents identical transactions from appearing in both the training and test sets.
2. **Stratified 80:20 split** (random seed 42): 226,980 training rows (378 fraud) and 56,746 test rows (95 fraud). A leakage check confirms zero overlap between training and test.
3. **Scaling:** `Time` and `Amount` are standardised with `StandardScaler`, fitted on the training set only.
4. **Resampling:** SMOTE is applied to the **training set only**, at `sampling_strategy=1.0` (50:50) and `0.25` (80:20).
5. **Models:** six classifiers with fixed settings across all conditions. SVM is trained on a 30,000-row stratified subsample because of its computational cost.
6. **Evaluation:** all models are tested on the same untouched test set, using the default 0.5 threshold.
7. **Interpretability:** SHAP (TreeExplainer) is applied to XGBoost trained with SMOTE 80:20, and compared with XGBoost's built-in gain importance.

