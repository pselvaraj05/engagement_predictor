"""
Complete Model Validation Suite
================================
Comprehensive validation combining:
- Ranking metrics (priority: can you identify viral posts?)
- Traditional metrics (MAE, distribution checks, temporal stability)
- Diagnostic tests (baselines, subgroups, errors)

Usage:
    python model_validation_suite.py --model hierarchical
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score, ndcg_score
)
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
fig_dir = "./figures/validation"
os.makedirs(fig_dir, exist_ok=True)

print("=" * 80)
print("COMPREHENSIVE MODEL VALIDATION SUITE")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("📊 Loading data...")
shares_df = pd.read_parquet(os.path.join(output_dir, "shares.parquet"))
eng_df = pd.read_parquet(os.path.join(output_dir, "engagements.parquet"))
profiles_df = pd.read_parquet(os.path.join(output_dir, "profiles.parquet"))

profile_to_user = dict(zip(eng_df["profile_id"], eng_df["es_user_id"]))
eng_df = eng_df.copy()
eng_df.rename(
    columns={
        "user_id": "sharer_user_id",
        "es_user_id": "engager_user_id",
        "created_at": "engagement_created_at",
        "client_id": "sharer_client_id",
    },
    inplace=True,
)
profiles_df = profiles_df.copy()
profiles_df["user_id"] = profiles_df["profile_id"].map(profile_to_user)

print(f"  Loaded: {len(shares_df):,} shares, {len(eng_df):,} engagements")


# ─── 2. Load Predictions ────────────────────────────────────────────────────
def load_predictions(model_type):
    """Load predictions based on model type"""
    if model_type == 'hierarchical':
        pred_path = os.path.join(save_dir, "hierarchical_predictions.parquet")
        meta_path = os.path.join(save_dir, "hierarchical_predictions_meta.joblib")
    elif model_type == 'tiered':
        pred_path = os.path.join(save_dir, "tiered_predictions.parquet")
        meta_path = os.path.join(save_dir, "tiered_predictions_meta.joblib")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(pred_path):
        print(f"❌ Predictions not found: {pred_path}")
        return None, None

    df = pd.read_parquet(pred_path)
    metadata = joblib.load(meta_path)
    return df, metadata


parser = argparse.ArgumentParser(description='Validate engagement prediction model')
parser.add_argument('--model', type=str, default='hierarchical',
                    choices=['hierarchical', 'tiered'],
                    help='Which model to validate')
args = parser.parse_args() if len(sys.argv) > 1 else argparse.Namespace(model='hierarchical')

model_type = args.model
pred_df, metadata = load_predictions(model_type)

if pred_df is None:
    print(f"❌ Cannot proceed without predictions")
    sys.exit(1)

y_test = pred_df['actual_count'].values
y_pred = pred_df['predicted_count'].values
y_tiers_test = pred_df['actual_tier'].values if 'actual_tier' in pred_df.columns else None

print(f"  Loaded {len(pred_df):,} predictions for {model_type} model\n")

# ─────────────────────────────────────────────────────────────────────────────
# PART A: RANKING METRICS (PRIMARY - BUSINESS VALUE)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PART A: RANKING METRICS (PRIMARY)")
print("=" * 80)
print("Focus: Can you identify high-performing posts?\n")

# ─── A1. Top-K Recall ───────────────────────────────────────────────────────
print("─" * 80)
print("A1. TOP-K RECALL ⭐⭐⭐⭐⭐ (Most Important)")
print("─" * 80)


def top_k_recall(y_true, y_pred, k=0.1):
    """What % of actual top-k posts did we catch?"""
    threshold = np.quantile(y_true, 1 - k)
    actual_top_k = y_true >= threshold
    pred_threshold = np.quantile(y_pred, 1 - k)
    pred_top_k = y_pred >= pred_threshold

    caught = (actual_top_k & pred_top_k).sum()
    total = actual_top_k.sum()

    return caught / total if total > 0 else 0


k_values = [0.01, 0.02, 0.05, 0.10, 0.20]
recalls = []

print("📊 Results:")
for k in k_values:
    recall = top_k_recall(y_test, y_pred, k)
    recalls.append(recall)

    if recall > 0.7:
        status = "✅ Excellent"
    elif recall > 0.5:
        status = "✅ Good"
    elif recall > 0.3:
        status = "⚠️  Fair"
    else:
        status = "❌ Poor"

    print(
        f"  Top {k * 100:4.1f}%: {recall:5.1%}  ({int(recall * len(y_test) * k):,}/{int(len(y_test) * k):,} caught)  {status}")

# ─── A2. NDCG Score ─────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("A2. NDCG (Ranking Quality) ⭐⭐⭐⭐")
print("─" * 80)

ndcg_scores = {}
print("📊 Results:")
for k in [10, 50, 100, 500, None]:
    score = ndcg_score([y_test], [y_pred], k=k)
    ndcg_scores[k] = score
    k_label = f"@{k}" if k else "(all)"

    if score > 0.85:
        status = "✅ Excellent"
    elif score > 0.75:
        status = "✅ Good"
    elif score > 0.65:
        status = "⚠️  Fair"
    else:
        status = "❌ Poor"

    print(f"  NDCG {k_label:6s}: {score:.4f}  {status}")

# ─── A3. Precision@K ────────────────────────────────────────────────────────
print("\n" + "─" * 80)
print("A3. PRECISION@K (Accuracy of Top Picks) ⭐⭐⭐")
print("─" * 80)


def precision_at_k(y_true, y_pred, k=100):
    """Of top K predicted posts, what % are actually in top 10%?"""
    top_k_idx = np.argsort(y_pred)[-k:]
    threshold = np.quantile(y_true, 0.9)
    correct = (y_true[top_k_idx] >= threshold).sum()
    return correct / k


print("📊 Results (% of picks in actual top 10%):")
for k in [10, 50, 100, 500]:
    precision = precision_at_k(y_test, y_pred, k)
    lift = precision / 0.10

    if precision > 0.6:
        status = "✅ Excellent"
    elif precision > 0.4:
        status = "✅ Good"
    elif precision > 0.2:
        status = "⚠️  Fair"
    else:
        status = "❌ Poor"

    print(f"  P@{k:4d}: {precision:5.1%} (lift: {lift:4.1f}x)  {status}")

# ─── A4. Spearman Correlation ───────────────────────────────────────────────
print("\n" + "─" * 80)
print("A4. SPEARMAN RANK CORRELATION ⭐⭐⭐")
print("─" * 80)

corr, pval = spearmanr(y_test, y_pred)

if corr > 0.8:
    status = "✅ Very Strong"
elif corr > 0.6:
    status = "✅ Strong"
elif corr > 0.4:
    status = "⚠️  Moderate"
else:
    status = "❌ Weak"

print(f"📊 Correlation: {corr:.4f}  (p={pval:.2e})  {status}")

# ─── A5. Visualization: Ranking Metrics ─────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Top-K Recall Curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot([k * 100 for k in k_values], [r * 100 for r in recalls],
         marker='o', linewidth=3, markersize=10, color='steelblue', label='Model')
ax1.plot([k * 100 for k in k_values], [k * 100 for k in k_values],
         linestyle='--', linewidth=2, color='red', alpha=0.7, label='Random')
ax1.fill_between([k * 100 for k in k_values], [r * 100 for r in recalls], [k * 100 for k in k_values],
                 alpha=0.3, color='green')
ax1.set_xlabel('Top K%', fontsize=11)
ax1.set_ylabel('Recall (%)', fontsize=11)
ax1.set_title('Top-K Recall Curve', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Lift Chart
ax2 = fig.add_subplot(gs[0, 1])
lift = [r / k for r, k in zip(recalls, k_values)]
colors_bar = ['#27ae60' if l > 1.5 else '#e67e22' if l > 1.2 else '#e74c3c' for l in lift]
bars = ax2.bar([f"Top {k * 100:.0f}%" for k in k_values], lift, color=colors_bar, alpha=0.8)
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Random')
ax2.set_ylabel('Lift vs Random', fontsize=11)
ax2.set_title('Lift Over Random Baseline', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, l in zip(bars, lift):
    ax2.text(bar.get_x() + bar.get_width() / 2, l + 0.05, f'{l:.2f}x',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Rank Correlation Scatter
ax3 = fig.add_subplot(gs[1, :])
n_samples = min(5000, len(y_test))
sample_idx = np.random.choice(len(y_test), n_samples, replace=False)
actual_ranks = pd.Series(y_test[sample_idx]).rank(pct=True)
pred_ranks = pd.Series(y_pred[sample_idx]).rank(pct=True)
scatter = ax3.scatter(actual_ranks, pred_ranks, alpha=0.3, s=15,
                      c=y_test[sample_idx], cmap='viridis',
                      norm=plt.Normalize(vmin=0, vmax=np.percentile(y_test, 95)))
ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Perfect')
ax3.set_xlabel('Actual Rank (percentile)', fontsize=11)
ax3.set_ylabel('Predicted Rank (percentile)', fontsize=11)
ax3.set_title(f'Rank Correlation (Spearman: {corr:.3f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax3, label='Actual Count', pad=0.01)

# Plot 4: Binary Classification
ax4 = fig.add_subplot(gs[2, 0])
threshold_actual = np.quantile(y_test, 0.9)
threshold_pred = np.quantile(y_pred, 0.9)
actual_viral = y_test >= threshold_actual
pred_viral = y_pred >= threshold_pred
tp = (actual_viral & pred_viral).sum()
fp = (~actual_viral & pred_viral).sum()
tn = (~actual_viral & ~pred_viral).sum()
fn = (actual_viral & ~pred_viral).sum()
cm = np.array([[tn, fp], [fn, tp]])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax4,
            xticklabels=['Not Top 10%', 'Top 10%'],
            yticklabels=['Not Top 10%', 'Top 10%'],
            cbar_kws={'label': 'Recall'})
ax4.set_ylabel('Actual', fontsize=11)
ax4.set_xlabel('Predicted', fontsize=11)
ax4.set_title('Binary Classification (Top 10%)', fontsize=12, fontweight='bold')

# Plot 5: Precision@K
ax5 = fig.add_subplot(gs[2, 1])
k_vals_prec = [10, 50, 100, 500]
prec_vals = [precision_at_k(y_test, y_pred, k) for k in k_vals_prec]
bars = ax5.bar([str(k) for k in k_vals_prec], prec_vals, color='coral', alpha=0.8)
ax5.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Random (10%)')
ax5.set_ylabel('Precision', fontsize=11)
ax5.set_xlabel('K', fontsize=11)
ax5.set_title('Precision@K', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)
for bar, p in zip(bars, prec_vals):
    ax5.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f'{p:.1%}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.savefig(os.path.join(fig_dir, 'ranking_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PART B: TRADITIONAL METRICS (SECONDARY - FOR COMPLETENESS)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("PART B: TRADITIONAL METRICS (SECONDARY)")
print("=" * 80)
print("Focus: Prediction accuracy and model diagnostics\n")

# ─── B1. Baseline Comparison ────────────────────────────────────────────────
print("─" * 80)
print("B1. BASELINE COMPARISON")
print("─" * 80)

# Prepare full dataframe for baselines
y_df = eng_df.groupby("share_id").size().reset_index(name="total_engagements")
share_owner = (
    eng_df.sort_values("engagement_created_at")
    .groupby("share_id")[["sharer_user_id", "sharer_client_id"]]
    .first()
    .reset_index()
)
full_df = (
    shares_df[["share_id", "shared_at", "share_content_type"]]
    .merge(y_df, on="share_id", how="left")
    .merge(share_owner, on="share_id", how="left")
)
full_df["total_engagements"] = full_df["total_engagements"].fillna(0).astype(np.int32)

test_share_ids = set(pred_df['share_id'])
train_mask = ~full_df['share_id'].isin(test_share_ids)

# Baseline 1: Always median
baseline_median = np.median(full_df.loc[train_mask, 'total_engagements'])
mae_baseline_median = mean_absolute_error(y_test, [baseline_median] * len(y_test))

# Baseline 2: Sharer average
sharer_avg = full_df[train_mask].groupby('sharer_user_id')['total_engagements'].mean()
test_sharer_avg = full_df.loc[~train_mask, 'sharer_user_id'].map(sharer_avg).fillna(baseline_median)
# Align with test set
test_df_aligned = full_df[full_df['share_id'].isin(test_share_ids)].set_index('share_id')
pred_df_aligned = pred_df.set_index('share_id')
test_sharer_avg_aligned = test_df_aligned.loc[pred_df_aligned.index, 'sharer_user_id'].map(sharer_avg).fillna(
    baseline_median).values
mae_baseline_sharer = mean_absolute_error(y_test, test_sharer_avg_aligned)

# Model MAE
mae_model = mean_absolute_error(y_test, y_pred)

improvement_median = (mae_baseline_median - mae_model) / mae_baseline_median * 100
improvement_sharer = (mae_baseline_sharer - mae_model) / mae_baseline_sharer * 100

print("📊 Mean Absolute Error (MAE):")
print(f"  Always Median:    {mae_baseline_median:8.2f}")
print(f"  Sharer Average:   {mae_baseline_sharer:8.2f}")
print(f"  Your Model:       {mae_model:8.2f}")
print(f"\n  Improvement over median:  {improvement_median:+6.1f}%")
print(f"  Improvement over sharer:  {improvement_sharer:+6.1f}%")

if improvement_median > 20:
    print("  ✅ Model significantly beats baseline")
elif improvement_median > 0:
    print("  ⚠️  Model marginally beats baseline")
else:
    print("  ❌ Model is worse than baseline!")

# ─── B2. Additional Traditional Metrics ─────────────────────────────────────
print("\n" + "─" * 80)
print("B2. ADDITIONAL METRICS")
print("─" * 80)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100

print("📊 Results:")
print(f"  RMSE:  {rmse:8.2f}")
print(f"  R²:    {r2:8.4f}")
print(f"  MAPE:  {mape:8.2f}%")

# ─── B3. Distribution Check ─────────────────────────────────────────────────
print("\n" + "─" * 80)
print("B3. DISTRIBUTION ANALYSIS")
print("─" * 80)

stats_dict = {
    'Metric': ['Mean', 'Median', 'Std', 'Min', 'Max'],
    'Actual': [
        np.mean(y_test), np.median(y_test), np.std(y_test),
        np.min(y_test), np.max(y_test)
    ],
    'Predicted': [
        np.mean(y_pred), np.median(y_pred), np.std(y_pred),
        np.min(y_pred), np.max(y_pred)
    ]
}
stats_df = pd.DataFrame(stats_dict)

print("📊 Distribution Statistics:")
print(stats_df.to_string(index=False))

# Check prediction quality
pred_mean_ratio = np.mean(y_pred) / np.mean(y_test)
if 0.9 <= pred_mean_ratio <= 1.1:
    print(f"\n  ✅ Predicted mean matches actual (ratio: {pred_mean_ratio:.2f})")
elif 0.7 <= pred_mean_ratio <= 1.3:
    print(f"\n  ⚠️  Predicted mean slightly off (ratio: {pred_mean_ratio:.2f})")
else:
    print(f"\n  ❌ Predicted mean very different from actual (ratio: {pred_mean_ratio:.2f})")

# Kolmogorov-Smirnov test
ks_stat, ks_pval = stats.ks_2samp(y_test, y_pred)
print(f"  K-S test p-value: {ks_pval:.4f}", end="")
if ks_pval >= 0.05:
    print(" ✅ (distributions similar)")
else:
    print(" ⚠️  (distributions differ)")

# ─── B4. Temporal Stability ─────────────────────────────────────────────────
print("\n" + "─" * 80)
print("B4. TEMPORAL STABILITY")
print("─" * 80)

test_df = pred_df.copy()
test_df['shared_at'] = pd.to_datetime(test_df['shared_at'])
test_df['month'] = test_df['shared_at'].dt.to_period('M')

monthly = test_df.groupby('month').apply(
    lambda x: pd.Series({
        'mae': mean_absolute_error(x['actual_count'], x['predicted_count']),
        'count': len(x),
        'mean_actual': x['actual_count'].mean()
    })
).reset_index()
monthly['month_str'] = monthly['month'].astype(str)

print("📊 Monthly Performance:")
for _, row in monthly.iterrows():
    print(f"  {row['month_str']}: MAE={row['mae']:6.2f}, n={row['count']:6,.0f}")

mae_std = monthly['mae'].std()
mae_mean = monthly['mae'].mean()
cv = mae_std / mae_mean

print(f"\n  Coefficient of Variation: {cv:.2%}", end="")
if cv < 0.1:
    print(" ✅ (very stable)")
elif cv < 0.2:
    print(" ✅ (stable)")
else:
    print(" ⚠️  (variable, may need retraining)")

# ─── B5. Subgroup Performance ───────────────────────────────────────────────
if y_tiers_test is not None:
    print("\n" + "─" * 80)
    print("B5. SUBGROUP PERFORMANCE")
    print("─" * 80)

    tier_names = metadata.get('tier_names', ['Low', 'Medium', 'High', 'Viral'])

    print("📊 Performance by Tier:")
    for tier_id, tier_name in enumerate(tier_names):
        mask = y_tiers_test == tier_id
        if mask.sum() > 0:
            tier_mae = mean_absolute_error(y_test[mask], y_pred[mask])
            tier_median = np.median(y_test[mask])
            pct_error = (tier_mae / (tier_median + 1)) * 100
            print(f"  {tier_name:15s}: MAE={tier_mae:7.2f}, % Error={pct_error:5.1f}%, n={mask.sum():6,}")

# ─── B6. Visualization: Traditional Metrics ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Baseline comparison
ax = axes[0, 0]
methods = ['Always\nMedian', 'Sharer\nAverage', 'Your\nModel']
maes = [mae_baseline_median, mae_baseline_sharer, mae_model]
colors = ['#e74c3c', '#e67e22', '#27ae60']
bars = ax.bar(methods, maes, color=colors, alpha=0.8)
ax.set_ylabel('MAE', fontsize=11)
ax.set_title('Model vs Baselines', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, mae in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width() / 2, mae + max(maes) * 0.02,
            f'{mae:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Distribution comparison
ax = axes[0, 1]
ax.hist(y_test, bins=50, alpha=0.6, label='Actual', color='steelblue', density=True)
ax.hist(y_pred, bins=50, alpha=0.6, label='Predicted', color='coral', density=True)
ax.set_xlabel('Engagement Count', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax.legend()
ax.set_yscale('log')

# Plot 3: Q-Q plot
ax = axes[0, 2]
residuals = y_pred - y_test
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 4: Temporal stability
ax = axes[1, 0]
ax.plot(range(len(monthly)), monthly['mae'], marker='o', linewidth=2, markersize=8, color='steelblue')
ax.axhline(y=mae_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {mae_mean:.1f}')
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly['month_str'], rotation=45, ha='right')
ax.set_ylabel('MAE', fontsize=11)
ax.set_title('MAE Over Time', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Predicted vs Actual scatter
ax = axes[1, 1]
sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
ax.scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.3, s=10)
max_val = max(y_test.max(), y_pred.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.7)
ax.set_xlabel('Actual', fontsize=11)
ax.set_ylabel('Predicted', fontsize=11)
ax.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 6: Residuals
ax = axes[1, 2]
ax.scatter(y_test[sample_idx], residuals[sample_idx], alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Actual', fontsize=11)
ax.set_ylabel('Residual', fontsize=11)
ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'traditional_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY: PRODUCTION READINESS SCORE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("FINAL SUMMARY: PRODUCTION READINESS")
print("=" * 80)

# Scoring (weighted)
score = 0
max_score = 100

# Ranking metrics (70% weight)
top10_recall = recalls[-2]  # 0.10 index
if top10_recall > 0.7:
    score += 30
    status1 = "✅"
elif top10_recall > 0.5:
    score += 20
    status1 = "⚠️ "
else:
    score += 10
    status1 = "❌"

ndcg_all = ndcg_scores[None]
if ndcg_all > 0.75:
    score += 20
    status2 = "✅"
elif ndcg_all > 0.65:
    score += 13
    status2 = "⚠️ "
else:
    score += 7
    status2 = "❌"

if corr > 0.6:
    score += 20
    status3 = "✅"
elif corr > 0.4:
    score += 13
    status3 = "⚠️ "
else:
    score += 7
    status3 = "❌"

# Traditional metrics (30% weight)
if improvement_median > 20:
    score += 15
    status4 = "✅"
elif improvement_median > 0:
    score += 10
    status4 = "⚠️ "
else:
    score += 0
    status4 = "❌"

if cv < 0.1:
    score += 15
    status5 = "✅"
elif cv < 0.2:
    score += 10
    status5 = "⚠️ "
else:
    score += 5
    status5 = "❌"

print("\n📊 Scoring Breakdown:")
print(f"\nRanking Metrics (70% weight):")
print(f"  {status1} Top 10% Recall:        {top10_recall:.1%}  (30 pts)")
print(f"  {status2} NDCG Score:             {ndcg_all:.3f}  (20 pts)")
print(f"  {status3} Spearman Correlation:   {corr:.3f}  (20 pts)")
print(f"\nTraditional Metrics (30% weight):")
print(f"  {status4} Improvement over baseline: {improvement_median:+.1f}%  (15 pts)")
print(f"  {status5} Temporal Stability (CV):   {cv:.2%}  (15 pts)")

final_score_pct = score / max_score
print(f"\n" + "=" * 80)
print(f"OVERALL SCORE: {score}/{max_score} ({final_score_pct:.0%})")
print("=" * 80)

if final_score_pct >= 0.80:
    verdict = "🎉 PRODUCTION READY"
    recommendation = "Model is excellent. Deploy with confidence!"
elif final_score_pct >= 0.65:
    verdict = "✅ ACCEPTABLE"
    recommendation = "Model works well. Deploy and iterate."
elif final_score_pct >= 0.50:
    verdict = "⚠️  NEEDS IMPROVEMENT"
    recommendation = "Model shows promise but needs tuning before production."
else:
    verdict = "❌ NOT READY"
    recommendation = "Model requires significant work. Do not deploy."

print(f"\n{verdict}")
print(f"{recommendation}")

# Save report
report_path = os.path.join(fig_dir, f'{model_type}_validation_summary.txt')
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write(f"VALIDATION SUMMARY: {model_type.upper()} MODEL\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Overall Score: {score}/{max_score} ({final_score_pct:.0%})\n")
    f.write(f"Verdict: {verdict}\n")
    f.write(f"Recommendation: {recommendation}\n\n")
    f.write("Key Metrics:\n")
    f.write(f"  Top 10% Recall: {top10_recall:.1%}\n")
    f.write(f"  NDCG: {ndcg_all:.3f}\n")
    f.write(f"  Spearman: {corr:.3f}\n")
    f.write(f"  MAE: {mae_model:.2f}\n")
    f.write(f"  Improvement: {improvement_median:+.1f}%\n")

print(f"\n✅ Validation complete!")
print(f"   Summary saved to: {report_path}")
print(f"   Figures saved to: {fig_dir}/")
print(f"     - ranking_metrics.png")
print(f"     - traditional_metrics.png")