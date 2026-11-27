"""
diagrams.py - recreates the concept diagrams used in Chapters 1-3 of the dissertation.

These are illustrations of general concepts, not results. All results figures
(Chapter 4) are produced by fraud_analysis.py.

Run:  python diagrams.py
Output (saved in ./diagrams):
    fig1_uk_fraud_losses.png      Figure 1  - UK payment fraud losses, 2024 and 2025 (UK Finance data)
    fig2_random_forest.png        Figure 2  - Structure of a Random Forest classifier
    fig3_xgboost.png              Figure 3  - Sequential tree building in XGBoost
    fig4_svm.png                  Figure 4  - Maximum-margin hyperplane and support vectors in an SVM
    fig5_confusion_matrix.png     Figure 5  - Confusion matrix for fraud detection (illustration)
    fig6_pipeline.png             Figure 6  - Experimental pipeline used in this study
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = "diagrams"
os.makedirs(OUT, exist_ok=True)


# ---------- helpers ----------
def box(ax, x, y, w, h, text, fc, fs=10, pad=0.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad={pad}",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))


def small_tree(ax, x, y, s=1.0):
    nodes = [(x, y), (x - 3 * s, y - 3.5 * s), (x + 3 * s, y - 3.5 * s), (x - 4.5 * s, y - 7 * s),
             (x - 1.5 * s, y - 7 * s), (x + 1.5 * s, y - 7 * s), (x + 4.5 * s, y - 7 * s)]
    for i, j in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
        ax.plot([nodes[i][0], nodes[j][0]], [nodes[i][1], nodes[j][1]], color="#555", lw=1, zorder=1)
    for k, (a, b) in enumerate(nodes):
        ax.scatter(a, b, s=90 if k < 3 else 60, color="#7fb27f" if k < 3 else "#b9d9b9",
                   edgecolors="#333", zorder=2)


# ---------- Figure 1: UK fraud losses (data: UK Finance 2025; 2026) ----------
def fig1():
    cats = ["Unauthorised fraud\n(incl. card fraud)", "Authorised push\npayment (APP) fraud", "Total payment\nfraud"]
    y24 = [722, 450.7, 1170]
    y25 = [703.4, 576.4, 1280]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - w / 2, y24, w, label="2024", color="#4c78a8")
    b2 = ax.bar(x + w / 2, y25, w, label="2025", color="#e45756")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, f"£{b.get_height():,.0f}m", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("Losses (£ million)"); ax.set_ylim(0, 1450)
    ax.legend(); ax.set_title("UK payment fraud losses, 2024 and 2025")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig1_uk_fraud_losses.png", dpi=200, facecolor="white"); plt.close()


# ---------- Figure 2: Random Forest ----------
def fig2():
    fig, ax = plt.subplots(figsize=(11.26, 6)); ax.set_xlim(0, 100); ax.set_ylim(0, 53); ax.axis("off")
    box(ax, 50, 48, 50, 4, "Training data (transactions labelled fraud / legitimate)", "#dce8f5", 11)
    for x, n, pred in zip([14, 38, 62, 86], ["1", "2", "3", "n"], ["Fraud", "Legitimate", "Fraud", "Fraud"]):
        arrow(ax, 50, 45.6, x, 39.6); box(ax, x, 37.5, 18, 3.4, f"Bootstrap sample {n}", "#f3f3f3", 9); arrow(ax, x, 35.3, x, 31.8)
        small_tree(ax, x, 30, 1.0)
        ax.text(x, 17.8, f"Tree {n}\n(random feature subset at each split)", ha="center", va="center", fontsize=8)
        arrow(ax, x, 15.6, x, 13.2); box(ax, x, 11.5, 14, 2.6, "Vote: " + pred, "#fff4d6", 9); arrow(ax, x, 9.4, 50, 6.4)
    ax.text(74, 30, "...", fontsize=22, ha="center", va="center")
    box(ax, 50, 4, 56, 3.6, "Final prediction = majority vote across all trees  →  Fraud", "#f9dcdc", 11)
    plt.savefig(f"{OUT}/fig2_random_forest.png", dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white"); plt.close()


# ---------- Figure 3: XGBoost ----------
def fig3():
    fig, ax = plt.subplots(figsize=(11.5, 5.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.axis("off")
    box(ax, 50, 41, 44, 3.6, "Training data (features + fraud labels)", "#dce8f5", 11, pad=0.3)
    xs = [14, 40, 66, 90]; names = ["Tree 1", "Tree 2", "Tree 3", "Tree m"]
    for k, (x, n) in enumerate(zip(xs, names)):
        small_tree(ax, x, 33)
        ax.text(x, 23.3, n, ha="center", fontsize=10, weight="bold")
        if k < 3:
            mid = (x + xs[k + 1]) / 2
            box(ax, mid, 30, 9.5, 4.6, "Errors\n(residuals)", "#fff4d6", 8.5, pad=0.3)
            arrow(ax, x + 5.2, 30, mid - 5.3, 30); arrow(ax, mid + 5.3, 30, xs[k + 1] - 5.2, 30)
    arrow(ax, 50, 39, 14, 34.2)
    ax.text(78, 26.5, "...", fontsize=20, ha="center")
    for x in xs:
        arrow(ax, x, 22, 50, 12.6)
    box(ax, 50, 10, 60, 4, "Prediction = learning rate × (Tree 1 + Tree 2 + Tree 3 + … + Tree m)", "#f9dcdc", 10.5, pad=0.3)
    ax.text(50, 3.2, "Each new tree is fitted to the errors left by the trees before it; regularisation (L1/L2) penalises overly complex trees.",
            ha="center", fontsize=9, style="italic")
    plt.savefig(f"{OUT}/fig3_xgboost.png", dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white"); plt.close()


# ---------- Figure 4: SVM ----------
def fig4():
    rng = np.random.default_rng(3)
    a = rng.normal([2, 2], 0.55, (25, 2)); b = rng.normal([5, 5], 0.55, (25, 2))
    sa = a.sum(1); sb = b.sum(1); hi = sa.max(); lo = sb.min(); c = (hi + lo) / 2
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.scatter(a[:, 0], a[:, 1], c="#4c78a8", label="Legitimate", edgecolors="k", s=40)
    ax.scatter(b[:, 0], b[:, 1], c="#e45756", marker="^", label="Fraud", edgecolors="k", s=50)
    xx = np.linspace(0, 8, 10)
    ax.plot(xx, c - xx, "k-", lw=2, label="Optimal hyperplane")
    ax.plot(xx, hi - xx, "k--", lw=1, label="Margin boundaries"); ax.plot(xx, lo - xx, "k--", lw=1)
    for P, s, v in ((a, sa, hi), (b, sb, lo)):
        i = np.argmin(abs(s - v)); ax.scatter(P[i, 0], P[i, 1], s=230, facecolors="none", edgecolors="#2a9d3a", lw=2)
    ax.scatter([], [], s=180, facecolors="none", edgecolors="#2a9d3a", lw=2, label="Support vectors")
    m = c / 2; d = (lo - hi) / 4
    ax.annotate("", xy=(m + d, m + d), xytext=(m - d, m - d), arrowprops=dict(arrowstyle="<->", color="#555"))
    ax.text(m + 0.25, m - 0.45, "margin", fontsize=9, color="#555", rotation=45)
    ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2"); ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig4_svm.png", dpi=200, facecolor="white"); plt.close()


# ---------- Figure 5: confusion matrix illustration ----------
def fig5():
    fig, ax = plt.subplots(figsize=(7, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    cells = [(3, 3.5, "True Positive (TP)\nFraud correctly flagged", "#cfe8cf"),
             (6.5, 3.5, "False Negative (FN)\nFraud missed", "#f6d2d2"),
             (3, 0.5, "False Positive (FP)\nGenuine wrongly flagged", "#f6d2d2"),
             (6.5, 0.5, "True Negative (TN)\nGenuine correctly passed", "#cfe8cf")]
    for x, y, t, col in cells:
        ax.add_patch(plt.Rectangle((x, y), 3.5, 3, fc=col, ec="#333", lw=1.3))
        ax.text(x + 1.75, y + 1.5, t, ha="center", va="center", fontsize=10)
    ax.text(4.75, 6.75, "Predicted: Fraud", ha="center", fontsize=10.5, weight="bold")
    ax.text(8.25, 6.75, "Predicted: Legitimate", ha="center", fontsize=10.5, weight="bold")
    ax.text(2.8, 5, "Actual:\nFraud", ha="right", va="center", fontsize=10.5, weight="bold")
    ax.text(2.8, 2, "Actual:\nLegitimate", ha="right", va="center", fontsize=10.5, weight="bold")
    plt.savefig(f"{OUT}/fig5_confusion_matrix.png", dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white"); plt.close()


# ---------- Figure 6: experimental pipeline ----------
def fig6():
    fig, ax = plt.subplots(figsize=(11, 6.2)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    b = lambda *args, **kw: box(ax, *args, fs=kw.get("fs", 9.5))
    box(ax, 50, 55, 46, 5, "European credit card dataset\n284,807 transactions, 492 fraud (0.173%)", "#dce8f5", 10)
    arrow(ax, 50, 52.2, 50, 48.6)
    box(ax, 50, 46, 52, 4, "Remove 1,081 duplicate rows → 283,726 rows (473 fraud)", "#f3f3f3", 9.5)
    arrow(ax, 50, 43.7, 50, 40.4)
    box(ax, 50, 38, 52, 4, "Stratified 80:20 split (random_state = 42); leakage check = 0 overlaps", "#f3f3f3", 9)
    arrow(ax, 44, 35.8, 24, 31.6); arrow(ax, 56, 35.8, 78, 31.6)
    box(ax, 24, 29, 36, 4.6, "Training set: 226,980 rows (378 fraud)\nscale Time and Amount (fit here only)", "#f3f3f3", 9.5)
    box(ax, 78, 29, 36, 4.6, "Test set: 56,746 rows (95 fraud)\nkept untouched, original class ratio", "#fde9c9", 9.5)
    for x, t in zip([8, 24, 40], ["No SMOTE\n(0.17% fraud)", "SMOTE 50:50", "SMOTE 80:20"]):
        arrow(ax, 24, 26.5, x, 22.2); box(ax, x, 20, 13.5, 3.8, t, "#fff4d6", 8.8); arrow(ax, x, 17.9, 24, 14.2)
    box(ax, 24, 12, 40, 4, "Six classifiers trained per condition (18 models)\nLR, DT, RF, XGBoost, SVM*, Naive Bayes", "#e2f0d9", 9)
    arrow(ax, 44.5, 12, 58.5, 12); arrow(ax, 78, 26.5, 78, 14.3)
    box(ax, 78, 12, 36, 4, "Evaluation on test set: Precision, Recall,\nF1, ROC-AUC, PR-AUC, confusion matrix", "#f9dcdc", 9)
    arrow(ax, 78, 9.8, 78, 6.3)
    box(ax, 78, 4.2, 36, 3.4, "SHAP analysis (XGBoost, SMOTE 80:20)", "#e8dcf5", 9)
    ax.text(1, 1.2, "* SVM trained on a 30,000-row stratified subsample of each training set.", fontsize=8.5, style="italic")
    plt.savefig(f"{OUT}/fig6_pipeline.png", dpi=200, bbox_inches="tight", pad_inches=0.12, facecolor="white"); plt.close()


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
        print("Saved", f.__name__)
    print(f"Done. Diagrams are in ./{OUT}")
