"""
LGBMRanker for Engagement Prediction
====================================
Learning-to-rank approach optimized for identifying viral posts.

Key differences from regression:
- Optimizes ranking order (NDCG) not point estimates (MAE)
- Focuses on top positions (viral posts)
- Uses query groups (e.g., posts per month)

This is the RIGHT approach for your use case!
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ndcg_score, mean_absolute_error
from scipy.stats import spearmanr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from lightgbm import LGBMRanker
    HAS_LGBM = True
except:
    print("❌ LightGBM not installed. Run: pip install lightgbm")
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
os.makedirs(save_dir, exist_ok=True)

EMBED_COL_IN_SHARES = "comment_embedding"
EMBED_NPY_PATH = os.path.join(save_dir, "item_text_embeddings.npy")
EMBED_META_PATH = os.path.join(save_dir, "item_text_embeddings_meta.joblib")

# Ranking-specific settings
SAMPLE_FOR_TRAINING = True
SAMPLE_SIZE = 1000000
max_text_features = 2000
test_size = 0.2
random_state = 42

print("=" * 80)
print("🎯 LGBMRANKER ENGAGEMENT MODEL")
print("=" * 80)
print("Optimizing for: Ranking quality (NDCG), not point estimates (MAE)\n")

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("📊 Loading data...")
shares_df = pd.read_parquet(os.path.join(output_dir, "shares.parquet"))
eng_df = pd.read_parquet(os.path.join(output_dir, "engagements.parquet"))
profiles_df = pd.read_parquet(os.path.join(output_dir, "profiles.parquet"))

# Prepare data (same as before)
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

shares = shares_df[["share_id", "shared_at", "share_content_type", "user_commentary"]].copy()
engs = eng_df[["sharer_client_id", "sharer_user_id", "engagement_type", "share_id",
               "engagement_created_at", "engager_user_id"]].copy()
profiles = profiles_df[["user_id", "industry", "job_title_role", "job_title_levels",
                        "location_region", "location_country", "job_title_class"]].copy()
profiles = profiles.drop_duplicates(subset=["user_id"]).reset_index(drop=True)

print(f"  Loaded: {len(shares_df):,} shares, {len(eng_df):,} engagements")

# ─── 2. Feature Engineering (Same as Before) ────────────────────────────────
print("\n🔧 Building features...")

y_df = engs.groupby("share_id").size().reset_index(name="total_engagements")
share_owner = (
    engs.sort_values("engagement_created_at")
        .groupby("share_id")[["sharer_user_id", "sharer_client_id"]]
        .first()
        .reset_index()
)

# Sharer stats
sharer_stats = engs.groupby('sharer_user_id').agg({
    'share_id': 'nunique',
    'engagement_type': 'count',
}).reset_index()
sharer_stats.columns = ['sharer_user_id', 'sharer_total_shares', 'sharer_total_engagements']
sharer_stats['sharer_avg_engagement_per_share'] = (
    sharer_stats['sharer_total_engagements'] / sharer_stats['sharer_total_shares']
).fillna(0)

# Viral rate
viral_shares = engs.groupby('share_id').size()
viral_share_ids = viral_shares[viral_shares > 50].index
sharer_viral_rate = (
    engs[engs['share_id'].isin(viral_share_ids)]
    .groupby('sharer_user_id')['share_id']
    .nunique()
    .reset_index(name='viral_share_count')
)
sharer_viral_rate = sharer_stats[['sharer_user_id', 'sharer_total_shares']].merge(
    sharer_viral_rate, on='sharer_user_id', how='left'
)
sharer_viral_rate['viral_share_count'] = sharer_viral_rate['viral_share_count'].fillna(0)
sharer_viral_rate['sharer_viral_rate'] = (
    sharer_viral_rate['viral_share_count'] / sharer_viral_rate['sharer_total_shares']
).fillna(0)
sharer_stats = sharer_stats.merge(
    sharer_viral_rate[['sharer_user_id', 'sharer_viral_rate']],
    on='sharer_user_id', how='left'
)

# Build training dataframe
train_df = (
    shares.merge(y_df, on="share_id", how="left")
          .merge(share_owner, on="share_id", how="left")
          .merge(sharer_stats, on="sharer_user_id", how="left")
)
train_df["total_engagements"] = train_df["total_engagements"].fillna(0).astype(np.int32)

# Sharer profile
sharer_profiles = profiles.rename(columns={"user_id": "sharer_user_id"})
train_df = train_df.merge(sharer_profiles, on="sharer_user_id", how="left")

# Time features
train_df["shared_at"] = pd.to_datetime(train_df["shared_at"], errors="coerce", utc=True)
train_df["share_hour"] = train_df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
train_df["share_dow"] = train_df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)
train_df["is_weekend"] = (train_df["share_dow"] >= 5).astype(np.int8)
train_df["is_business_hours"] = ((train_df["share_hour"] >= 9) & (train_df["share_hour"] <= 17)).astype(np.int8)

# CRITICAL FOR RANKING: Add time-based query groups
# Use WEEK instead of MONTH to keep groups under 10K limit
train_df["year_week"] = train_df["shared_at"].dt.to_period('W').astype(str)

# Content features
train_df["user_commentary"] = train_df["user_commentary"].fillna("")
train_df["text_length"] = train_df["user_commentary"].str.len()
train_df["word_count"] = train_df["user_commentary"].str.split().str.len()
train_df["has_question"] = train_df["user_commentary"].str.contains(r'\?').astype(np.int8)
train_df["has_url"] = train_df["user_commentary"].str.contains(r'http').astype(np.int8)
train_df["exclamation_count"] = train_df["user_commentary"].str.count('!')
train_df["hashtag_count"] = train_df["user_commentary"].str.count('#')
train_df["mention_count"] = train_df["user_commentary"].str.count('@')

# Fill missing
cat_cols = ["share_content_type", "industry", "job_title_class", "location_country",
            "job_title_role", "job_title_levels", "location_region"]
for c in cat_cols:
    train_df[c] = train_df[c].fillna("unknown").astype(str)

num_cols = ["share_hour", "share_dow", "is_weekend", "is_business_hours",
            "text_length", "word_count", "has_question", "has_url",
            "exclamation_count", "hashtag_count", "mention_count",
            "sharer_avg_engagement_per_share", "sharer_total_shares", "sharer_viral_rate"]
for c in num_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna(0).astype(np.float32)

print(f"  Features: {len(cat_cols)} categorical, {len(num_cols)} numeric")

# ─── 3. Text Features ───────────────────────────────────────────────────────
print("📝 Processing text features...")

def _load_embeddings_from_shares_column(df: pd.DataFrame, col: str):
    if col is None or col not in df.columns:
        return None
    try:
        emb = np.vstack(df[col].apply(lambda x: np.array(x, dtype=np.float32)).to_numpy())
        return emb
    except:
        return None

def _load_embeddings_from_files(item_ids: np.ndarray):
    if not (os.path.exists(EMBED_NPY_PATH) and os.path.exists(EMBED_META_PATH)):
        return None
    meta = joblib.load(EMBED_META_PATH)
    if "item_ids" not in meta:
        return None
    saved_item_ids = np.array(meta["item_ids"])
    emb = np.load(EMBED_NPY_PATH)
    idx_map = {sid: i for i, sid in enumerate(saved_item_ids)}
    rows = []
    for sid in item_ids:
        rows.append(emb[idx_map[sid]] if sid in idx_map else np.zeros(emb.shape[1], dtype=np.float32))
    return np.vstack(rows).astype(np.float32)

item_ids = train_df["share_id"].to_numpy()
item_emb = _load_embeddings_from_shares_column(train_df, EMBED_COL_IN_SHARES)
if item_emb is None:
    item_emb = _load_embeddings_from_files(item_ids)

use_tfidf = item_emb is None
if use_tfidf:
    tfidf = TfidfVectorizer(max_features=max_text_features, stop_words="english", ngram_range=(1, 2))
    item_text_sparse = tfidf.fit_transform(train_df["user_commentary"]).astype(np.float32)
    print(f"  ℹ️  Using TF-IDF: {item_text_sparse.shape}")
else:
    print(f"  ✅ Using embeddings: {item_emb.shape}")

# ─── 4. Build Feature Matrix ───────────────────────────────────────────────
print("\n🔨 Building feature matrix...")
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_cat = ohe.fit_transform(train_df[cat_cols]).astype(np.float32)

scaler = StandardScaler(with_mean=False)
X_num = csr_matrix(scaler.fit_transform(train_df[num_cols].to_numpy(dtype=np.float32)))

X_text = item_text_sparse if use_tfidf else csr_matrix(item_emb)
X = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)

y = train_df["total_engagements"].to_numpy(dtype=np.float32)

print(f"  Feature matrix: {X.shape}")
print(f"  Target: min={y.min():.0f}, max={y.max():.0f}, mean={y.mean():.2f}")

# ─── 5. Stratified Sampling ─────────────────────────────────────────────────
if SAMPLE_FOR_TRAINING and X.shape[0] > SAMPLE_SIZE:
    print(f"\n⚡ Sampling {SAMPLE_SIZE:,} examples...")

    # Stratified by engagement tier
    def assign_tier(eng):
        if eng <= 5: return 0
        elif eng <= 50: return 1
        elif eng <= 500: return 2
        else: return 3

    y_tiers = np.array([assign_tier(e) for e in y])

    tier_sample_sizes = {
        0: int(SAMPLE_SIZE * 0.20),
        1: int(SAMPLE_SIZE * 0.30),
        2: int(SAMPLE_SIZE * 0.30),
        3: int(SAMPLE_SIZE * 0.20),
    }

    sample_idx = []
    np.random.seed(random_state)

    for tier_id, target_size in tier_sample_sizes.items():
        tier_mask = y_tiers == tier_id
        tier_indices = np.where(tier_mask)[0]
        if len(tier_indices) > 0:
            n_sample = min(target_size, len(tier_indices))
            sampled = np.random.choice(tier_indices, size=n_sample, replace=False)
            sample_idx.extend(sampled)

    sample_idx = np.array(sample_idx)
    np.random.shuffle(sample_idx)

    X = X[sample_idx]
    y = y[sample_idx]
    train_df = train_df.iloc[sample_idx].reset_index(drop=True)

    print(f"  Sampled to: {X.shape}")

# ─── 6. Time-based Split with Query Groups ─────────────────────────────────
print("\n📅 Creating time-based split with query groups...")

train_df = train_df.sort_values('shared_at').reset_index(drop=True)
split_idx = int(len(train_df) * (1 - test_size))
split_date = train_df.iloc[split_idx]['shared_at']

train_mask = (train_df['shared_at'] < split_date).to_numpy()
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# CRITICAL: Create query groups AFTER split to ensure continuity
# This prevents empty groups and ensures query IDs are contiguous
train_df_split = train_df[train_mask].copy()
test_df_split = train_df[~train_mask].copy()

# Assign query IDs within each split
train_df_split['query_id'] = train_df_split.groupby('year_week').ngroup()
test_df_split['query_id'] = test_df_split.groupby('year_week').ngroup()

query_train = train_df_split['query_id'].to_numpy()
query_test = test_df_split['query_id'].to_numpy()

# Count samples per query (these should now be contiguous!)
train_query_counts = np.bincount(query_train)
test_query_counts = np.bincount(query_test)

# Filter out any zero counts (shouldn't happen now, but be safe)
train_query_counts = train_query_counts[train_query_counts > 0]
test_query_counts = test_query_counts[test_query_counts > 0]

# Validate totals
assert train_query_counts.sum() == len(query_train), "Train query count mismatch!"
assert test_query_counts.sum() == len(query_test), "Test query count mismatch!"

# Validate query sizes (must be < 10,000)
max_query_size_train = train_query_counts.max()
max_query_size_test = test_query_counts.max()

print(f"  Train: {X_train.shape[0]:,} samples, {len(train_query_counts):,} query groups")
print(f"  Test:  {X_test.shape[0]:,} samples, {len(test_query_counts):,} query groups")
print(f"  Avg samples per query: train={train_query_counts.mean():.0f}, test={test_query_counts.mean():.0f}")
print(f"  Max samples per query: train={max_query_size_train:,}, test={max_query_size_test:,}")

if max_query_size_train >= 10000:
    print(f"  ❌ ERROR: Train has query group with {max_query_size_train:,} samples (limit: 10,000)")
    print("     Solution: Use daily groups instead of weekly")
    exit(1)
if max_query_size_test >= 10000:
    print(f"  ❌ ERROR: Test has query group with {max_query_size_test:,} samples (limit: 10,000)")
    print("     Solution: Use daily groups instead of weekly")
    exit(1)

print("  ✅ All query groups within 10K limit")

# ─── 7. Train LGBMRanker ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("🎯 TRAINING LGBMRANKER (LambdaRank Objective)")
print("=" * 80)

# CRITICAL: LGBMRanker expects RELEVANCE LABELS (ranks), not raw counts!
# Convert engagement counts to relevance labels within each query group

def counts_to_relevance_labels(counts, query_ids, n_bins=5):
    """
    Convert engagement counts to relevance labels (0-4) within each query group.

    Args:
        counts: Raw engagement counts
        query_ids: Query group IDs
        n_bins: Number of relevance levels (default: 5 = 0,1,2,3,4)

    Returns:
        Relevance labels (0 = lowest, n_bins-1 = highest)
    """
    labels = np.zeros(len(counts), dtype=np.int32)

    for qid in np.unique(query_ids):
        mask = query_ids == qid
        group_counts = counts[mask]

        if len(group_counts) == 1:
            # Single item in group
            labels[mask] = 0
        else:
            # Use percentile-based binning within this query group
            # This ensures all relevance levels are used
            labels[mask] = pd.qcut(
                group_counts,
                q=min(n_bins, len(group_counts)),  # Can't have more bins than samples
                labels=False,
                duplicates='drop'
            )

    return labels

print("\n🔄 Converting engagement counts to relevance labels...")
print("   (0 = lowest engagement, 4 = highest engagement within each week)")

y_train_labels = counts_to_relevance_labels(y_train, query_train, n_bins=5)
y_test_labels = counts_to_relevance_labels(y_test, query_test, n_bins=5)

print(f"   Train labels: min={y_train_labels.min()}, max={y_train_labels.max()}")
print(f"   Test labels:  min={y_test_labels.min()}, max={y_test_labels.max()}")
print(f"   Distribution: {np.bincount(y_train_labels)}")

ranker = LGBMRanker(
    objective='lambdarank',  # Optimizes NDCG directly!
    metric='ndcg',           # Primary metric
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=10,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    label_gain=list(range(5)),  # Gains for labels 0-4
    n_jobs=-1,
    random_state=random_state,
    importance_type='gain',
    verbose=-1
)

print("\nTraining ranker...")
ranker.fit(
    X_train,
    y_train_labels,  # Use labels, not raw counts!
    group=train_query_counts,
    eval_set=[(X_test, y_test_labels)],
    eval_group=[test_query_counts],
    eval_metric='ndcg',
)

print("✅ Training complete!")

# ─── 8. Evaluate with Ranking Metrics ──────────────────────────────────────
print("\n" + "=" * 80)
print("📊 EVALUATION (Ranking-Focused)")
print("=" * 80)

# Get predictions (these are ranking scores, not counts!)
y_pred = ranker.predict(X_test)

# ─── Ranking Metrics (PRIMARY) ──────────────────────────────────────────────

def top_k_recall(y_true, y_pred, k=0.1):
    threshold = np.quantile(y_true, 1 - k)
    actual_top_k = y_true >= threshold
    pred_threshold = np.quantile(y_pred, 1 - k)
    pred_top_k = y_pred >= pred_threshold
    caught = (actual_top_k & pred_top_k).sum()
    total = actual_top_k.sum()
    return caught / total if total > 0 else 0

print("\n🎯 PRIMARY METRICS (What Actually Matters):\n")

# Top-K Recall
print("Top-K Recall:")
for k in [0.01, 0.05, 0.10, 0.20]:
    recall = top_k_recall(y_test, y_pred, k)
    status = "✅" if recall > 0.7 else "⚠️ " if recall > 0.5 else "❌"
    print(f"  Top {k*100:4.1f}%: {recall:5.1%}  {status}")

# NDCG
print("\nNDCG (Ranking Quality):")
for k in [10, 50, 100, None]:
    score = ndcg_score([y_test], [y_pred], k=k)
    k_label = f"@{k:3d}" if k else "(all)"
    status = "✅" if score > 0.85 else "⚠️ " if score > 0.75 else "❌"
    print(f"  NDCG {k_label}: {score:.4f}  {status}")

# Spearman Correlation
corr, pval = spearmanr(y_test, y_pred)
status = "✅" if corr > 0.8 else "⚠️ " if corr > 0.6 else "❌"
print(f"\nSpearman Correlation: {corr:.4f}  {status}")

# ─── Traditional Metrics (SECONDARY - For Reference) ────────────────────────
print("\n📋 SECONDARY METRICS (For Reference Only):\n")

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"  ℹ️  Note: MAE is not optimized by ranker, so it may be high")
print(f"  ℹ️  This is EXPECTED and ACCEPTABLE for a ranking model")

# ─── 9. Monthly Rolling Validation ──────────────────────────────────────────
print("\n" + "=" * 80)
print("📅 MONTHLY ROLLING VALIDATION")
print("=" * 80)

test_df = train_df[~train_mask].copy()
test_df['predicted_score'] = y_pred
test_df['actual'] = y_test
test_df['month'] = pd.to_datetime(test_df['shared_at']).dt.to_period('M')

monthly_results = []
for month in test_df['month'].unique():
    month_mask = test_df['month'] == month
    month_actual = test_df.loc[month_mask, 'actual'].values
    month_pred = test_df.loc[month_mask, 'predicted_score'].values

    if len(month_actual) > 100:  # Need enough samples
        ndcg = ndcg_score([month_actual], [month_pred])
        top10_recall = top_k_recall(month_actual, month_pred, k=0.1)
        corr, _ = spearmanr(month_actual, month_pred)

        monthly_results.append({
            'month': str(month),
            'n_samples': len(month_actual),
            'ndcg': ndcg,
            'top10_recall': top10_recall,
            'spearman': corr
        })

monthly_df = pd.DataFrame(monthly_results)
print("\n📊 Performance by Month:")
print(monthly_df.to_string(index=False))

# Check stability
ndcg_cv = monthly_df['ndcg'].std() / monthly_df['ndcg'].mean()
print(f"\nNDCG Coefficient of Variation: {ndcg_cv:.2%}", end="")
if ndcg_cv < 0.05:
    print(" ✅ (very stable)")
elif ndcg_cv < 0.10:
    print(" ✅ (stable)")
else:
    print(" ⚠️  (variable - monitor for drift)")

# ─── 10. Feature Importance ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("📊 TOP FEATURES FOR RANKING")
print("=" * 80)

feature_names = (
    list(ohe.get_feature_names_out()) +
    num_cols +
    [f"text_{i}" for i in range(X_text.shape[1])]
)

importance = ranker.feature_importances_
top_idx = np.argsort(importance)[-20:]

print("\nTop 20 Most Important Features:")
for i in reversed(top_idx):
    feat_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
    print(f"  {feat_name[:50]:50s}: {importance[i]:.1f}")

# ─── 11. Save Model ─────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("💾 SAVING MODEL")
print("=" * 80)

artifact = {
    "model": ranker,
    "model_type": "ranker",
    "ohe": ohe,
    "scaler": scaler,
    "tfidf": tfidf if use_tfidf else None,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "use_tfidf": use_tfidf,
    "feature_names": feature_names,
}

joblib.dump(artifact, os.path.join(save_dir, "ranker_engagement_model.joblib"))
print("  ✅ Saved: ranker_engagement_model.joblib")

# Save predictions
results_df = pd.DataFrame({
    "share_id": test_df["share_id"].values,
    "actual_count": y_test,
    "ranking_score": y_pred,  # Note: scores, not counts!
    "shared_at": test_df["shared_at"].values,
})
results_df.to_parquet(os.path.join(save_dir, "ranker_predictions.parquet"), index=False)
print("  ✅ Saved: ranker_predictions.parquet")

# Save metadata
metadata = {
    "model_type": "ranker",
    "top_10_recall": float(top_k_recall(y_test, y_pred, 0.1)),
    "ndcg": float(ndcg_score([y_test], [y_pred])),
    "spearman": float(corr),
    "mae": float(mae),
    "split_date": str(split_date),
    "n_train": len(y_train),
    "n_test": len(y_test),
    "monthly_stability_cv": float(ndcg_cv),
}
joblib.dump(metadata, os.path.join(save_dir, "ranker_predictions_meta.joblib"))
print("  ✅ Saved: ranker_predictions_meta.joblib")

# ─── 12. Production Usage Example ───────────────────────────────────────────
print("\n" + "=" * 80)
print("🚀 PRODUCTION USAGE")
print("=" * 80)

print("""
# To use the ranker in production:

artifact = joblib.load('ranker_engagement_model.joblib')
ranker = artifact['model']

# For a batch of new posts:
X_new = preprocess_features(new_posts)  # Your feature pipeline
ranking_scores = ranker.predict(X_new)

# Sort by ranking score (higher = more likely to be viral)
ranked_idx = np.argsort(ranking_scores)[::-1]

# Get top 10% for promotion
top_10_pct = int(len(ranking_scores) * 0.10)
promote_idx = ranked_idx[:top_10_pct]

print(f"Promote these {len(promote_idx)} posts:")
for idx in promote_idx:
    print(f"  Post {new_posts[idx]['share_id']}: score={ranking_scores[idx]:.2f}")

# Note: Ranking scores are NOT engagement predictions!
# They're relative scores for ranking posts against each other.
# Higher score = more likely to be viral (relative to other posts).
""")

print("\n" + "=" * 80)
print("✨ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nKey Takeaways:")
print(f"  • Model optimized for RANKING, not point estimates")
print(f"  • Top 10% Recall: {metadata['top_10_recall']:.1%}")
print(f"  • NDCG: {metadata['ndcg']:.3f}")
print(f"  • Spearman: {metadata['spearman']:.3f}")
print(f"  • MAE: {metadata['mae']:.2f} (ignore - not relevant for ranking)")
print(f"\n  Focus on Top-K Recall and NDCG - these are what matter!")
