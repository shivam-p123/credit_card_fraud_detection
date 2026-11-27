# %% [markdown]
# Credit Card Fraud Detection: six models, no SMOTE vs SMOTE 50:50 vs SMOTE 80:20, plus SHAP
#
# How to run:
#   1. Download creditcard.csv from Kaggle ("Credit Card Fraud Detection", ULB / mlg-ulb)
#      and put it in the same folder as this script.
#   2. pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib seaborn
#   3. Run it in VS Code / Jupyter cell by cell (each "# %%" is a cell), or: python fraud_analysis.py
#
# Outputs go into ./results (CSV tables) and ./figures (PNG images for the dissertation).

# %% Imports and settings
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             precision_recall_curve, ConfusionMatrixDisplay)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = os.environ.get("FRAUD_DATA", "creditcard.csv")
SVM_TRAIN_ROWS = 30000      # SVM scales roughly quadratically, so it is trained on a stratified subsample
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# %% 1. Load and explore the data
df = pd.read_csv(DATA_PATH)
print(df.shape)
counts = df["Class"].value_counts()
print(counts)
print(f"Fraud share: {counts[1] / len(df) * 100:.4f}%")

plt.figure(figsize=(5, 4))
sns.countplot(x="Class", data=df)
plt.yscale("log")
plt.title("Class distribution (log scale)")
plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.tight_layout()
plt.savefig("figures/01_class_distribution.png", dpi=300)
plt.close()

# %% 2. Preprocessing: scale Time and Amount (V1-V28 are already PCA components), stratified split
X = df.drop(columns="Class")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train[["Time", "Amount"]] = scaler.fit_transform(X_train[["Time", "Amount"]])
X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])
print("Train:", X_train.shape, "fraud =", y_train.sum(), "| Test:", X_test.shape, "fraud =", y_test.sum())

# %% 3. Three training scenarios. SMOTE is applied to the TRAINING set only; the test set stays untouched.
def make_scenarios(X_tr, y_tr):
    scenarios = {"No SMOTE": (X_tr, y_tr)}
    # sampling_strategy = minority/majority after resampling: 1.0 -> 50:50, 0.25 -> 80:20
    for name, ratio in [("SMOTE 50:50", 1.0), ("SMOTE 80:20", 0.25)]:
        Xs, ys = SMOTE(sampling_strategy=ratio, random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
        scenarios[name] = (Xs, ys)
    return scenarios

scenarios = make_scenarios(X_train, y_train)
for name, (Xs, ys) in scenarios.items():
    vc = ys.value_counts()
    print(f"{name}: legit={vc[0]:,} fraud={vc[1]:,} ({vc[1] / len(ys) * 100:.1f}% fraud)")

# %% 4. The six models
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, eval_metric="logloss",
                                 n_jobs=-1, random_state=RANDOM_STATE),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }

def stratified_subsample(X_tr, y_tr, n_rows):
    if len(X_tr) <= n_rows:
        return X_tr, y_tr
    X_sub, _, y_sub, _ = train_test_split(X_tr, y_tr, train_size=n_rows, stratify=y_tr,
                                          random_state=RANDOM_STATE)
    return X_sub, y_sub

# %% 5. Train and evaluate every model in every scenario
rows, fitted = [], {}
for scen_name, (Xs, ys) in scenarios.items():
    for model_name, model in get_models().items():
        Xfit, yfit = (stratified_subsample(Xs, ys, SVM_TRAIN_ROWS) if model_name == "SVM" else (Xs, ys))
        t0 = time.time()
        model.fit(Xfit, yfit)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        rows.append({
            "Scenario": scen_name, "Model": model_name,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "PR_AUC": average_precision_score(y_test, y_prob),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Train_rows": len(Xfit), "Train_time_s": round(train_time, 1),
        })
        fitted[(scen_name, model_name)] = (model, y_pred, y_prob)
        print(f"{scen_name:12s} | {model_name:20s} | F1={rows[-1]['F1']:.3f} "
              f"P={rows[-1]['Precision']:.3f} R={rows[-1]['Recall']:.3f} ({train_time:.0f}s)")

results = pd.DataFrame(rows)
results.to_csv("results/all_results.csv", index=False)
results.round(4)

# %% 6. Summary tables (one per metric, models x scenarios) for the results chapter
order = ["No SMOTE", "SMOTE 50:50", "SMOTE 80:20"]
for metric in ["Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
    table = results.pivot(index="Model", columns="Scenario", values=metric)[order].round(4)
    table.to_csv(f"results/table_{metric}.csv")
    print(f"\n{metric}\n{table}")

# %% 7. Figures: grouped bars per metric
for metric in ["Precision", "Recall", "F1", "PR_AUC"]:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results, x="Model", y=metric, hue="Scenario", hue_order=order)
    plt.title(f"{metric} by model and training scenario")
    plt.xticks(rotation=20)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"figures/02_{metric.lower()}_by_scenario.png", dpi=300)
    plt.close()

# %% 8. Figure: false positives per model and scenario (shows the precision collapse directly)
plt.figure(figsize=(9, 5))
sns.barplot(data=results, x="Model", y="FP", hue="Scenario", hue_order=order)
plt.yscale("log")
plt.title("Legitimate transactions wrongly flagged as fraud (log scale)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/03_false_positives.png", dpi=300)
plt.close()

# %% 9. Figure: precision-recall curves for each scenario
for scen_name in order:
    plt.figure(figsize=(6, 5))
    for model_name in get_models():
        _, _, y_prob = fitted[(scen_name, model_name)]
        p, r, _ = precision_recall_curve(y_test, y_prob)
        plt.plot(r, p, label=f"{model_name} (AP={average_precision_score(y_test, y_prob):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall curves: {scen_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/04_pr_curves_{scen_name.replace(' ', '_').replace(':', '-')}.png", dpi=300)
    plt.close()

# %% 10. Figure: confusion matrices for the best model (by F1) in each scenario
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, scen_name in zip(axes, order):
    best = results[results.Scenario == scen_name].sort_values("F1", ascending=False).iloc[0]
    _, y_pred, _ = fitted[(scen_name, best.Model)]
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Legit", "Fraud"]).plot(
        ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"{scen_name}\n{best.Model}")
plt.tight_layout()
plt.savefig("figures/05_confusion_matrices_best.png", dpi=300)
plt.close()

# %% 11. SHAP explanations for XGBoost trained with SMOTE 80:20
import shap

xgb_model = fitted[("SMOTE 80:20", "XGBoost")][0]
explainer = shap.TreeExplainer(xgb_model)
shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=RANDOM_STATE)
shap_values = explainer(shap_sample)

# Global view: which PCA components push predictions towards fraud
shap.summary_plot(shap_values.values, shap_sample, show=False, max_display=15)
plt.title("SHAP summary: XGBoost (SMOTE 80:20)")
plt.tight_layout()
plt.savefig("figures/06_shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

shap.summary_plot(shap_values.values, shap_sample, plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig("figures/07_shap_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# Local view: explain one correctly caught fraud case (case-by-case auditing)
_, y_pred_xgb, _ = fitted[("SMOTE 80:20", "XGBoost")]
caught = np.where((y_test.values == 1) & (y_pred_xgb == 1))[0]
if len(caught):
    idx = caught[0]
    single = explainer(X_test.iloc[[idx]])
    shap.plots.waterfall(single[0], max_display=12, show=False)
    plt.title("SHAP explanation of one detected fraud transaction")
    plt.tight_layout()
    plt.savefig("figures/08_shap_waterfall_single_fraud.png", dpi=300, bbox_inches="tight")
    plt.close()

# Compare SHAP ranking with XGBoost's built-in feature importance
importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_abs_SHAP": np.abs(shap_values.values).mean(axis=0),
    "XGB_gain_importance": xgb_model.feature_importances_,
}).sort_values("Mean_abs_SHAP", ascending=False)
importance.to_csv("results/shap_vs_xgb_importance.csv", index=False)
print(importance.head(10))

print("\nDone. Tables are in ./results and figures in ./figures")
