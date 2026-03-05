"""
Visualization script for hierarchical engagement prediction model
Focuses on popularity detection and tier classification performance
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import joblib

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ─── 0. Setup ───────────────────────────────────────────────────────────────
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("📊 Loading hierarchical predictions and metadata...")
results_path = "./outputs/hierarchical_predictions.parquet"
meta_path = "./outputs/hierarchical_predictions_meta.joblib"

df = pd.read_parquet(results_path)
metadata = joblib.load(meta_path)

# Extract data
y_actual_count = df["actual_count"].values
y_pred_count = df["predicted_count"].values
y_actual_tier = df["actual_tier"].values
y_pred_tier = df["predicted_tier"].values
y_pred_popular = df["predicted_popular"].values
y_popular_proba = df["popular_probability"].values

y_actual_popular = (y_actual_tier >= 2).astype(int)

tier_names = metadata["tier_names"]
tier_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']  # blue, green, orange, red

print(f"  Loaded {len(df):,} predictions")
print(f"  Popularity Recall: {metadata['popular_detection_recall']:.3f} ⭐")
print(f"  Popularity Precision: {metadata['popular_detection_precision']:.3f}")
print(f"  Tier Accuracy: {metadata['tier_accuracy']:.3f}")

# ─── 2. MOST IMPORTANT: Popularity Detection Performance ───────────────────
print("\n📈 Creating popularity detection visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: ROC Curve for popularity detection
fpr, tpr, thresholds = roc_curve(y_actual_popular, y_popular_proba)
roc_auc = auc(fpr, tpr)

axes[0].plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
axes[0].set_title('ROC Curve: Popular Post Detection', fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right", fontsize=11)
axes[0].grid(True, alpha=0.3)

# Panel 2: Confusion Matrix for popularity
cm_popular = confusion_matrix(y_actual_popular, y_pred_popular)
cm_popular_norm = cm_popular.astype('float') / cm_popular.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_popular_norm, annot=True, fmt='.2%', cmap='YlGnBu', ax=axes[1],
            xticklabels=['Not Popular', 'Popular'],
            yticklabels=['Not Popular', 'Popular'],
            cbar_kws={'label': 'Recall'})
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title('Popularity Confusion Matrix', fontsize=14, fontweight='bold')

# Panel 3: Probability distribution
axes[2].hist(y_popular_proba[y_actual_popular == 0], bins=50, alpha=0.6,
             label='Not Popular', color='steelblue', density=True)
axes[2].hist(y_popular_proba[y_actual_popular == 1], bins=50, alpha=0.6,
             label='Popular', color='coral', density=True)
axes[2].axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
axes[2].set_xlabel('Predicted Probability', fontsize=12)
axes[2].set_ylabel('Density', fontsize=12)
axes[2].set_title('Probability Distribution by Actual Class', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "popularity_detection.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 3. Tier Confusion Matrix ──────────────────────────────────────────────
print("📈 Creating tier confusion matrix...")
cm = confusion_matrix(y_actual_tier, y_pred_tier)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Absolute counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=tier_names, yticklabels=tier_names)
ax1.set_xlabel('Predicted Tier', fontsize=12)
ax1.set_ylabel('Actual Tier', fontsize=12)
ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

# Normalized (recall per tier)
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=ax2,
            xticklabels=tier_names, yticklabels=tier_names)
ax2.set_xlabel('Predicted Tier', fontsize=12)
ax2.set_ylabel('Actual Tier', fontsize=12)
ax2.set_title('Confusion Matrix (Recall %)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hierarchical_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 4. Stage-by-Stage Performance ─────────────────────────────────────────
print("📈 Creating stage-by-stage analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Stage 1: Overall popularity detection metrics
ax = axes[0, 0]
recall = metadata['popular_detection_recall']
precision = metadata['popular_detection_precision']
f1 = metadata['popular_detection_f1']
accuracy = metadata['popularity_accuracy']

metrics = ['Recall', 'Precision', 'F1 Score', 'Accuracy']
values = [recall, precision, f1, accuracy]
colors_bar = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6']

bars = ax.barh(metrics, values, color=colors_bar, alpha=0.8)
ax.set_xlim([0, 1])
ax.set_xlabel('Score', fontsize=12)
ax.set_title('Stage 1: Popularity Detection Metrics', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, values)):
    ax.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')

# Stage 2a: Low vs Medium performance (for not popular)
ax = axes[0, 1]
not_popular_mask = y_actual_tier <= 1
if not_popular_mask.sum() > 0:
    low_med_actual = y_actual_tier[not_popular_mask]
    low_med_pred = y_pred_tier[not_popular_mask]

    cm_low_med = confusion_matrix(low_med_actual, low_med_pred, labels=[0, 1])
    sns.heatmap(cm_low_med, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low', 'Medium'], yticklabels=['Low', 'Medium'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'Stage 2a: Low vs Medium\n(n={not_popular_mask.sum():,})',
                 fontsize=13, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No samples', ha='center', va='center', fontsize=16)
    ax.set_title('Stage 2a: Low vs Medium', fontsize=13, fontweight='bold')

# Stage 2b: High vs Viral performance (for popular)
ax = axes[1, 0]
popular_mask = y_actual_tier >= 2
if popular_mask.sum() > 0:
    high_viral_actual = y_actual_tier[popular_mask]
    high_viral_pred = y_pred_tier[popular_mask]

    cm_high_viral = confusion_matrix(high_viral_actual, high_viral_pred, labels=[2, 3])
    sns.heatmap(cm_high_viral, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['High', 'Viral'], yticklabels=['High', 'Viral'])
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(f'Stage 2b: High vs Viral\n(n={popular_mask.sum():,})',
                 fontsize=13, fontweight='bold')
else:
    ax.text(0.5, 0.5, 'No samples', ha='center', va='center', fontsize=16)
    ax.set_title('Stage 2b: High vs Viral', fontsize=13, fontweight='bold')

# Stage 3: Tier distribution comparison
ax = axes[1, 1]
x = np.arange(len(tier_names))
width = 0.35

actual_counts = [np.sum(y_actual_tier == i) for i in range(len(tier_names))]
pred_counts = [np.sum(y_pred_tier == i) for i in range(len(tier_names))]

bars1 = ax.bar(x - width/2, actual_counts, width, label='Actual',
               color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted',
               color='coral', alpha=0.8)

ax.set_xlabel('Tier', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Final Tier Distribution', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tier_names, rotation=15, ha='right', fontsize=9)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hierarchical_stages.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 5. Per-Tier Count Prediction Performance ──────────────────────────────
print("📈 Creating per-tier prediction plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for tier_id in range(len(tier_names)):
    ax = axes[tier_id]

    tier_mask = y_actual_tier == tier_id
    if tier_mask.sum() == 0:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center', fontsize=16)
        ax.set_title(tier_names[tier_id])
        continue

    actual = y_actual_count[tier_mask]
    pred = y_pred_count[tier_mask]

    # Scatter plot
    ax.scatter(actual, pred, alpha=0.4, s=20, color=tier_colors[tier_id])

    # Perfect prediction line
    min_val, max_val = actual.min(), actual.max()
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7, label='Perfect')

    # Calculate metrics
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs((actual - pred) / (actual + 1))) * 100

    ax.set_xlabel('Actual Count', fontsize=11)
    ax.set_ylabel('Predicted Count', fontsize=11)
    ax.set_title(f'{tier_names[tier_id]}\nMAE: {mae:.2f}, MAPE: {mape:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hierarchical_per_tier_predictions.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 6. Probability Calibration by Tier ────────────────────────────────────
print("📈 Creating probability calibration plot...")

fig, ax = plt.subplots(figsize=(12, 6))

# Bin by predicted probability
prob_bins = np.linspace(0, 1, 11)
bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2

observed_rates = []
predicted_probs = []
counts = []

for i in range(len(prob_bins) - 1):
    mask = (y_popular_proba >= prob_bins[i]) & (y_popular_proba < prob_bins[i+1])
    if mask.sum() > 0:
        observed_rates.append(y_actual_popular[mask].mean())
        predicted_probs.append(y_popular_proba[mask].mean())
        counts.append(mask.sum())

ax.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Perfect Calibration')
ax.plot(predicted_probs, observed_rates, 'o-', linewidth=2, markersize=8,
        label='Observed', color='steelblue')

# Add count labels
for x, y, c in zip(predicted_probs, observed_rates, counts):
    ax.annotate(f'n={c}', (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Observed Frequency', fontsize=12)
ax.set_title('Popularity Probability Calibration', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "probability_calibration.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 7. Feature Importance Comparison ──────────────────────────────────────
print("📈 Creating precision-recall by tier...")

fig, ax = plt.subplots(figsize=(12, 6))

precisions = []
recalls = []
f1_scores = []

for tier_id in range(len(tier_names)):
    tp = np.sum((y_actual_tier == tier_id) & (y_pred_tier == tier_id))
    fp = np.sum((y_actual_tier != tier_id) & (y_pred_tier == tier_id))
    fn = np.sum((y_actual_tier == tier_id) & (y_pred_tier != tier_id))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

x = np.arange(len(tier_names))
width = 0.25

bars1 = ax.bar(x - width, precisions, width, label='Precision', color='steelblue', alpha=0.8)
bars2 = ax.bar(x, recalls, width, label='Recall', color='coral', alpha=0.8)
bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color='forestgreen', alpha=0.8)

ax.set_xlabel('Tier', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision, Recall, and F1 Score by Tier', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(tier_names, rotation=15, ha='right')
ax.set_ylim([0, 1.1])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "hierarchical_precision_recall.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 8. Overall Predictions Colored by Confidence ──────────────────────────
print("📈 Creating confidence-colored predictions...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Color by confidence in popularity prediction
confidence = np.maximum(y_popular_proba, 1 - y_popular_proba)

# Raw scale
scatter1 = ax1.scatter(y_actual_count, y_pred_count, c=confidence,
                       cmap='RdYlGn', alpha=0.5, s=15, vmin=0.5, vmax=1.0)
max_val = max(y_actual_count.max(), y_pred_count.max())
ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7)
ax1.set_xlabel('Actual Count', fontsize=12)
ax1.set_ylabel('Predicted Count', fontsize=12)
ax1.set_title('Predictions Colored by Confidence', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Prediction Confidence')

# Log scale
scatter2 = ax2.scatter(np.log1p(y_actual_count), np.log1p(y_pred_count),
                       c=confidence, cmap='RdYlGn', alpha=0.5, s=15, vmin=0.5, vmax=1.0)
max_log = max(np.log1p(y_actual_count).max(), np.log1p(y_pred_count).max())
ax2.plot([0, max_log], [0, max_log], 'r--', linewidth=2, alpha=0.7)
ax2.set_xlabel('log1p(Actual Count)', fontsize=12)
ax2.set_ylabel('log1p(Predicted Count)', fontsize=12)
ax2.set_title('Predictions (Log Scale) Colored by Confidence', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Prediction Confidence')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "predictions_with_confidence.png"), dpi=300, bbox_inches='tight')
plt.close()

# ─── 9. Summary Statistics ─────────────────────────────────────────────────
print("📈 Creating summary statistics...")

summary_data = []
for tier_id in range(len(tier_names)):
    tier_mask = y_actual_tier == tier_id
    if tier_mask.sum() == 0:
        continue

    actual = y_actual_count[tier_mask]
    pred = y_pred_count[tier_mask]

    tp = np.sum((y_actual_tier == tier_id) & (y_pred_tier == tier_id))
    fp = np.sum((y_actual_tier != tier_id) & (y_pred_tier == tier_id))
    fn = np.sum((y_actual_tier == tier_id) & (y_pred_tier != tier_id))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    summary_data.append({
        'Tier': tier_names[tier_id],
        'Count': tier_mask.sum(),
        'Precision': precision,
        'Recall': recall,
        'Actual Mean': actual.mean(),
        'Pred Mean': pred.mean(),
        'MAE': np.mean(np.abs(actual - pred)),
        'RMSE': np.sqrt(np.mean((actual - pred) ** 2)),
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(fig_dir, "hierarchical_summary_statistics.csv"), index=False)
print("\n📊 Summary Statistics:")
print(summary_df.to_string(index=False))

# ─── 10. Generate Report ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("✨ HIERARCHICAL VISUALIZATION COMPLETE!")
print("=" * 70)
print(f"\n📁 All figures saved to: {fig_dir}/")
print("\n📋 Key Figures to Review:")
print("  1. popularity_detection.png - ⭐ MOST IMPORTANT! ROC, confusion, probabilities")
print("  2. hierarchical_stages.png - Performance at each stage")
print("  3. hierarchical_confusion_matrix.png - Overall tier classification")
print("  4. probability_calibration.png - Are probabilities well-calibrated?")
print("  5. predictions_with_confidence.png - Predictions colored by confidence")

print("\n🎯 Key Metrics:")
print(f"  Popular Detection Recall: {metadata['popular_detection_recall']:.3f} {'✅' if metadata['popular_detection_recall'] > 0.7 else '⚠️'}")
print(f"  Popular Detection Precision: {metadata['popular_detection_precision']:.3f}")
print(f"  Popular Detection F1: {metadata['popular_detection_f1']:.3f}")
print(f"  ROC-AUC: {metadata['popularity_auc']:.3f}")

if metadata['popular_detection_recall'] > 0.7:
    print("\n🎉 SUCCESS! Your model catches >70% of popular posts!")
    print("   Consider using popular_probability for ranking posts.")
else:
    print("\n💡 Recall needs improvement. Try:")
    print("   - Increase SAMPLE_SIZE for more popular examples")
    print("   - Add network features (follower count, connection count)")
    print("   - Adjust scale_pos_weight in popularity_classifier")