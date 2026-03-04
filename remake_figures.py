"""Regenerate all figures as slide-friendly single plots."""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# Template colors
BLUE = "#389AFF"
YELLOW = "#F9A81A"
RED = "#EA1D64"
DARK = "#3A3838"
GREEN = "#2ECC71"
LIGHT_BG = "#F5F7FA"

out_dir = "./figures/slides"
os.makedirs(out_dir, exist_ok=True)

# ─── Load Data ────────────────────────────────────────────────────────────────
hier_meta = joblib.load("./outputs/hierarchical_predictions_meta.joblib")
hier_preds = pd.read_parquet("./outputs/hierarchical_predictions.parquet")
power_results = joblib.load("./outputs/power_analysis_results.joblib")

tier_names = ["Low (0-5)", "Medium (6-50)", "High (51-500)", "Viral (500+)"]
tier_short = ["Low", "Medium", "High", "Viral"]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: ROC Curve — Popularity Detection
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 1: ROC Curve")

y_actual_popular = (hier_preds["actual_tier"] >= 2).astype(int)
y_proba_popular = hier_preds["popular_probability"]

fpr, tpr, _ = roc_curve(y_actual_popular, y_proba_popular)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, color=BLUE, linewidth=3, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color=DARK, linewidth=1.5, linestyle="--", alpha=0.4, label="Random")
ax.fill_between(fpr, tpr, alpha=0.08, color=BLUE)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Recall)")
ax.set_title("Popularity Detection: ROC Curve", fontweight="bold", color=DARK)
ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=False)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.grid(True, alpha=0.15)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Confusion Matrix (Counts) — 4-Tier
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 2: Confusion Matrix (Counts)")

cm = confusion_matrix(hier_preds["actual_tier"], hier_preds["predicted_tier"], labels=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt=",", cmap="Blues", ax=ax,
            xticklabels=tier_short, yticklabels=tier_short,
            annot_kws={"size": 18, "fontweight": "bold"},
            linewidths=2, linecolor="white",
            cbar_kws={"label": "Count", "shrink": 0.8})
ax.set_xlabel("Predicted Tier", fontweight="bold")
ax.set_ylabel("Actual Tier", fontweight="bold")
ax.set_title("4-Tier Confusion Matrix", fontweight="bold", color=DARK)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_counts.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Confusion Matrix (Recall %) — 4-Tier
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 3: Confusion Matrix (Recall %)")

cm_recall = cm.astype(float) / cm.sum(axis=1, keepdims=True)
annot_labels = np.array([[f"{v:.1%}" for v in row] for row in cm_recall])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_recall, annot=annot_labels, fmt="", cmap="Greens", ax=ax,
            xticklabels=tier_short, yticklabels=tier_short,
            annot_kws={"size": 18, "fontweight": "bold"},
            linewidths=2, linecolor="white", vmin=0, vmax=1,
            cbar_kws={"label": "Recall", "shrink": 0.8})
ax.set_xlabel("Predicted Tier", fontweight="bold")
ax.set_ylabel("Actual Tier", fontweight="bold")
ax.set_title("4-Tier Confusion Matrix (Recall %)", fontweight="bold", color=DARK)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_recall.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Actual vs Predicted Tier Distribution
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 4: Tier Distribution")

actual_counts = [(hier_preds["actual_tier"] == t).sum() for t in range(4)]
pred_counts = [(hier_preds["predicted_tier"] == t).sum() for t in range(4)]

fig, ax = plt.subplots(figsize=(10, 7))
x = np.arange(4)
width = 0.35
bars1 = ax.bar(x - width/2, actual_counts, width, label="Actual", color=BLUE, alpha=0.85, edgecolor="white", linewidth=1.5)
bars2 = ax.bar(x + width/2, pred_counts, width, label="Predicted", color=RED, alpha=0.85, edgecolor="white", linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:,.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=13, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(tier_names)
ax.set_ylabel("Count")
ax.set_title("Actual vs Predicted Tier Distribution", fontweight="bold", color=DARK)
ax.legend(frameon=True, fancybox=True)
ax.grid(axis="y", alpha=0.15)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "tier_distribution.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Power Analysis — Learning Curve (Shares) — Top-3 Only
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 5: Learning Curve (Shares)")

shares_lc = pd.DataFrame(power_results["shares_learning_curve"])

target_labels = {
    "engager_job_title": "Job Role",
    "engager_industry": "Industry",
    "engager_company": "Company",
}
target_colors = {
    "engager_job_title": RED,
    "engager_industry": BLUE,
    "engager_company": GREEN,
}

fig, ax = plt.subplots(figsize=(11, 7))
for col in ["engager_job_title", "engager_industry", "engager_company"]:
    subset = shares_lc[shares_lc["target"] == col]
    grouped = subset.groupby("n_shares")["top3"]
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    ax.plot(means.index, means.values, "o-", label=target_labels[col],
            color=target_colors[col], linewidth=3, markersize=10)
    ax.fill_between(means.index, means.values - stds.values, means.values + stds.values,
                    alpha=0.12, color=target_colors[col])

ax.set_xlabel("Number of Shares Tracked")
ax.set_ylabel("Top-3 Accuracy")
ax.set_title("How Many Shares Do You Need?", fontweight="bold", color=DARK)
ax.set_xscale("log")
ax.set_ylim([0, 0.7])
ax.legend(frameon=True, fancybox=True, loc="upper left")
ax.grid(True, alpha=0.15)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "learning_curve_shares.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Power Analysis — Learning Curve (Engagements per Share)
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 6: Learning Curve (Engagements)")

engs_lc = pd.DataFrame(power_results["engs_learning_curve"])

fig, ax = plt.subplots(figsize=(11, 7))
for col in ["engager_job_title", "engager_industry", "engager_company"]:
    subset = engs_lc[engs_lc["target"] == col]
    grouped = subset.groupby("engs_per_share")["top3"]
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    ax.plot(means.index, means.values, "o-", label=target_labels[col],
            color=target_colors[col], linewidth=3, markersize=10)
    ax.fill_between(means.index, means.values - stds.values, means.values + stds.values,
                    alpha=0.12, color=target_colors[col])

ax.set_xlabel("Engagements Scraped per Share")
ax.set_ylabel("Top-3 Accuracy")
ax.set_title("How Many Engagements per Share?  (at 5,000 shares)", fontweight="bold", color=DARK)
ax.set_ylim([0, 0.7])
ax.legend(frameon=True, fancybox=True, loc="upper right")
ax.grid(True, alpha=0.15)
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "learning_curve_engagements.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Power Analysis — Heatmap (one combined, cleaner)
# ═══════════════════════════════════════════════════════════════════════════════
print("  Figure 7: Heatmap")

heatmap_df = pd.DataFrame(power_results["heatmap"])

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
for i, col in enumerate(["engager_job_title", "engager_industry", "engager_company"]):
    ax = axes[i]
    subset = heatmap_df[heatmap_df["target"] == col]
    pivot = subset.pivot_table(index="engs_per_share", columns="n_shares", values="top3", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                vmin=0, vmax=0.7, annot_kws={"size": 16, "fontweight": "bold"},
                linewidths=2, linecolor="white",
                cbar=i == 2, cbar_kws={"label": "Top-3 Accuracy", "shrink": 0.8})
    ax.set_xlabel("Number of Shares", fontsize=15)
    ax.set_ylabel("Eng. per Share" if i == 0 else "", fontsize=15)
    ax.set_title(target_labels[col], fontsize=20, fontweight="bold",
                 color=target_colors[col])

plt.suptitle("Top-3 Accuracy: Shares vs Engagements per Share",
             fontsize=22, fontweight="bold", color=DARK, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "heatmap.png"), dpi=200, bbox_inches="tight")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES 8-10: Distribution Sample Analysis — Regressor vs Classifier
# ═══════════════════════════════════════════════════════════════════════════════
dist_sa_results_path = "./outputs/distribution_sample_analysis_results.joblib"
dist_sa_out_dir = "./figures/distribution_sample_analysis"

if os.path.exists(dist_sa_results_path):
    print("\n  Figure 8-10: Distribution Sample Analysis")
    os.makedirs(dist_sa_out_dir, exist_ok=True)

    dist_sa = joblib.load(dist_sa_results_path)
    dist_sa_df = pd.DataFrame(dist_sa["results_df"])
    share_counts = dist_sa["share_counts"]
    dims = dist_sa["target_dimensions"]

    dim_colors_sa = {"job_title": RED, "industry": BLUE, "company": GREEN,
                      "job_level": "#9B59B6", "country": YELLOW}
    dim_labels_sa = {"job_title": "Job Role", "industry": "Industry", "company": "Company",
                      "job_level": "Job Level", "country": "Country"}
    approach_styles = {"regressor": "--", "classifier": "-"}
    approach_markers = {"regressor": "s", "classifier": "o"}

    # Figure 8: Top-3 learning curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    for i, dim in enumerate(dims):
        ax = axes[i // 3][i % 3]
        for approach in ["regressor", "classifier"]:
            subset = dist_sa_df[
                (dist_sa_df["dimension"] == dim) &
                (dist_sa_df["approach"] == approach) &
                (dist_sa_df["metric"] == "top3")
            ]
            grouped = subset.groupby("n_shares")["value"]
            means = grouped.mean()
            stds = grouped.std().fillna(0)

            ax.plot(means.index, means.values,
                    linestyle=approach_styles[approach],
                    marker=approach_markers[approach],
                    label=approach.title(),
                    color=dim_colors_sa[dim], linewidth=3, markersize=9)
            ax.fill_between(means.index, means.values - stds.values,
                            means.values + stds.values,
                            alpha=0.12, color=dim_colors_sa[dim])

        ax.set_xlabel("Number of Shares")
        ax.set_ylabel("Top-3 Accuracy" if i % 3 == 0 else "")
        ax.set_title(dim_labels_sa[dim], fontweight="bold", color=dim_colors_sa[dim])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    axes[1][2].set_visible(False)

    plt.suptitle("Top-3 Accuracy: Regressor vs Classifier", fontsize=22,
                 fontweight="bold", color=DARK, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(dist_sa_out_dir, "learning_curve_top3.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 9: Top-5 learning curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
    for i, dim in enumerate(dims):
        ax = axes[i // 3][i % 3]
        for approach in ["regressor", "classifier"]:
            subset = dist_sa_df[
                (dist_sa_df["dimension"] == dim) &
                (dist_sa_df["approach"] == approach) &
                (dist_sa_df["metric"] == "top5")
            ]
            grouped = subset.groupby("n_shares")["value"]
            means = grouped.mean()
            stds = grouped.std().fillna(0)

            ax.plot(means.index, means.values,
                    linestyle=approach_styles[approach],
                    marker=approach_markers[approach],
                    label=approach.title(),
                    color=dim_colors_sa[dim], linewidth=3, markersize=9)
            ax.fill_between(means.index, means.values - stds.values,
                            means.values + stds.values,
                            alpha=0.12, color=dim_colors_sa[dim])

        ax.set_xlabel("Number of Shares")
        ax.set_ylabel("Top-5 Accuracy" if i % 3 == 0 else "")
        ax.set_title(dim_labels_sa[dim], fontweight="bold", color=dim_colors_sa[dim])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=13)
        ax.grid(True, alpha=0.15)
        sns.despine(ax=ax)

    axes[1][2].set_visible(False)

    plt.suptitle("Top-5 Accuracy: Regressor vs Classifier", fontsize=22,
                 fontweight="bold", color=DARK, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(dist_sa_out_dir, "learning_curve_top5.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 10: Comparison bar chart at largest sample size
    largest_n = max(share_counts)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for plot_i, metric_name in enumerate(["top3", "top5"]):
        ax = axes[plot_i]
        x = np.arange(len(dims))
        width = 0.35
        reg_vals, cls_vals = [], []
        for dim in dims:
            for approach, vl in [("regressor", reg_vals), ("classifier", cls_vals)]:
                subset = dist_sa_df[
                    (dist_sa_df["n_shares"] == largest_n) &
                    (dist_sa_df["dimension"] == dim) &
                    (dist_sa_df["approach"] == approach) &
                    (dist_sa_df["metric"] == metric_name)
                ]
                vl.append(subset["value"].mean() if len(subset) > 0 else 0)

        bars1 = ax.bar(x - width/2, reg_vals, width, label="Regressor", color="#95a5a6", alpha=0.85)
        bars2 = ax.bar(x + width/2, cls_vals, width, label="Classifier", color=BLUE, alpha=0.85)
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=13, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Top-{metric_name[-1]} at {largest_n:,} Shares", fontweight="bold", color=DARK)
        ax.set_xticks(x)
        ax.set_xticklabels([dim_labels_sa[d] for d in dims])
        ax.set_ylim([0, 1.15])
        ax.legend(fontsize=13)
        ax.grid(axis="y", alpha=0.15)
        sns.despine(ax=ax)

    plt.suptitle("Regressor vs Classifier: Distribution Prediction", fontsize=22,
                 fontweight="bold", color=DARK, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(dist_sa_out_dir, "comparison_bar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"  Distribution sample analysis figures saved to {dist_sa_out_dir}/")
else:
    print(f"\n  Skipping distribution sample analysis figures (run distribution_sample_analysis.py first)")

print(f"\n  All figures saved to {out_dir}/")
