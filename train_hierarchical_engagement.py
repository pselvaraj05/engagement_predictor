"""
Hierarchical Engagement Prediction Model
=========================================
Uses a two-stage classification approach:

Stage 1: Binary Decision - Will this be POPULAR? (High/Viral vs Low/Medium)
    → This is the most important business decision
    → Focus on maximizing recall for popular posts

Stage 2a: For NOT POPULAR → Classify Low vs Medium
Stage 2b: For POPULAR → Classify High vs Viral

Stage 3: Tier-specific regressors for count estimates

This approach prioritizes identifying popular posts while still providing
fine-grained predictions within each group.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, accuracy_score, roc_auc_score
)
import warnings

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    print("⚠️  LightGBM not found. Install with: pip install lightgbm")
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
os.makedirs(save_dir, exist_ok=True)

EMBED_COL_IN_SHARES = "comment_embedding"
EMBED_NPY_PATH = os.path.join(save_dir, "item_text_embeddings.npy")
EMBED_META_PATH = os.path.join(save_dir, "item_text_embeddings_meta.joblib")

# Engagement tier definitions
TIER_BOUNDARIES = {
    'low': (0, 5),
    'medium': (6, 50),
    'high': (51, 500),
    'viral': (501, float('inf'))
}

POPULAR_THRESHOLD = 50  # High/Viral cutoff

# Performance settings
SAMPLE_FOR_TRAINING = True
SAMPLE_SIZE = 1000000  # Larger sample for better viral representation
max_text_features = 2000
test_size = 0.2
random_state = 42

print("=" * 70)
print("🎯 HIERARCHICAL ENGAGEMENT PREDICTION MODEL")
print("=" * 70)

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("\n📊 Loading data...")
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

shares = shares_df[["share_id", "shared_at", "share_content_type", "user_commentary"]].copy()
engs = eng_df[
    ["sharer_client_id", "sharer_user_id", "engagement_type", "share_id",
     "engagement_created_at", "engager_user_id"]
].copy()
profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "job_title_levels",
     "location_region", "location_country", "job_title_class"]
].copy()
profiles = profiles.drop_duplicates(subset=["user_id"]).reset_index(drop=True)

# ─── 2. Enhanced Feature Engineering ───────────────────────────────────────
print("🔧 Building enhanced features...")

# Target: total engagements per share
y_df = engs.groupby("share_id").size().reset_index(name="total_engagements")

# Share ownership
share_owner = (
    engs.sort_values("engagement_created_at")
        .groupby("share_id")[["sharer_user_id", "sharer_client_id"]]
        .first()
        .reset_index()
)

# Sharer historical performance
print("  → Computing sharer statistics...")
sharer_stats = engs.groupby('sharer_user_id').agg({
    'share_id': 'nunique',
    'engagement_type': 'count',
}).reset_index()
sharer_stats.columns = ['sharer_user_id', 'sharer_total_shares', 'sharer_total_engagements']
sharer_stats['sharer_avg_engagement_per_share'] = (
    sharer_stats['sharer_total_engagements'] / sharer_stats['sharer_total_shares']
).fillna(0)

# NEW: Sharer's historical "viral rate"
print("  → Computing viral rates...")
viral_shares = engs.groupby('share_id').size()
viral_share_ids = viral_shares[viral_shares > POPULAR_THRESHOLD].index
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
    on='sharer_user_id',
    how='left'
)

# Engagement type distribution per sharer
eng_type_dist = engs.groupby(['sharer_user_id', 'engagement_type']).size().unstack(fill_value=0)
eng_type_dist.columns = [f'sharer_eng_{col}_count' for col in eng_type_dist.columns]
eng_type_dist = eng_type_dist.reset_index()
sharer_stats = sharer_stats.merge(eng_type_dist, on='sharer_user_id', how='left')

# Build training dataframe
train_df = (
    shares.merge(y_df, on="share_id", how="left")
          .merge(share_owner, on="share_id", how="left")
          .merge(sharer_stats, on="sharer_user_id", how="left")
)
train_df["total_engagements"] = train_df["total_engagements"].fillna(0).astype(np.int32)

# CRITICAL: NO LOG TRANSFORMATION!
# Tweedie/Poisson objectives expect RAW counts, not log-transformed values

# Sharer profile features
sharer_profiles = profiles.rename(columns={"user_id": "sharer_user_id"})
train_df = train_df.merge(sharer_profiles, on="sharer_user_id", how="left")

# Time features
print("  → Extracting temporal features...")
train_df["shared_at"] = pd.to_datetime(train_df["shared_at"], errors="coerce", utc=True)
train_df["share_hour"] = train_df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
train_df["share_dow"] = train_df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)
train_df["is_weekend"] = (train_df["share_dow"] >= 5).astype(np.int8)
train_df["is_business_hours"] = (
    (train_df["share_hour"] >= 9) & (train_df["share_hour"] <= 17)
).astype(np.int8)

# Content signal features
print("  → Extracting content signals...")
train_df["user_commentary"] = train_df["user_commentary"].fillna("")
train_df["text_length"] = train_df["user_commentary"].str.len()
train_df["word_count"] = train_df["user_commentary"].str.split().str.len()
train_df["has_question"] = train_df["user_commentary"].str.contains(r'\?').astype(np.int8)
train_df["has_url"] = train_df["user_commentary"].str.contains(r'http').astype(np.int8)
train_df["exclamation_count"] = train_df["user_commentary"].str.count('!')
train_df["hashtag_count"] = train_df["user_commentary"].str.count('#')
train_df["mention_count"] = train_df["user_commentary"].str.count('@')

# ─── 2.5 Assign Labels ─────────────────────────────────────────────────────
print("  → Assigning engagement tiers and popularity labels...")

def assign_tier(engagement):
    if engagement <= 5:
        return 0  # low
    elif engagement <= 50:
        return 1  # medium
    elif engagement <= 500:
        return 2  # high
    else:
        return 3  # viral

train_df['engagement_tier'] = train_df['total_engagements'].apply(assign_tier)
train_df['is_popular'] = (train_df['total_engagements'] > POPULAR_THRESHOLD).astype(int)

# Show distribution
tier_names = ['Low (0-5)', 'Medium (6-50)', 'High (51-500)', 'Viral (500+)']
tier_dist = train_df['engagement_tier'].value_counts().sort_index()
popular_dist = train_df['is_popular'].value_counts()

print("\n  Full Tier Distribution:")
for tier_id, name in enumerate(tier_names):
    count = tier_dist.get(tier_id, 0)
    pct = count / len(train_df) * 100
    print(f"    {name:20s}: {count:8,} ({pct:5.2f}%)")

print("\n  Popularity Distribution:")
print(f"    Not Popular (≤{POPULAR_THRESHOLD}):  {popular_dist.get(0, 0):8,} ({popular_dist.get(0, 0)/len(train_df)*100:5.2f}%)")
print(f"    Popular (>{POPULAR_THRESHOLD}):     {popular_dist.get(1, 0):8,} ({popular_dist.get(1, 0)/len(train_df)*100:5.2f}%)")

# Fill missing values
cat_cols = [
    "share_content_type", "industry", "job_title_class",
    "location_country", "job_title_role", "job_title_levels", "location_region",
]
for c in cat_cols:
    train_df[c] = train_df[c].fillna("unknown").astype(str)

num_cols = [
    "share_hour", "share_dow", "is_weekend", "is_business_hours",
    "text_length", "word_count", "has_question", "has_url",
    "exclamation_count", "hashtag_count", "mention_count",
    "sharer_avg_engagement_per_share", "sharer_total_shares",
    "sharer_viral_rate",
]

for c in num_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna(0).astype(np.float32)

# ─── 3. Text Features ───────────────────────────────────────────────────────
print("\n📝 Processing text features...")

def _load_embeddings_from_shares_column(df: pd.DataFrame, col: str):
    if col is None or col not in df.columns:
        return None
    try:
        emb = np.vstack(df[col].apply(lambda x: np.array(x, dtype=np.float32)).to_numpy())
        return emb
    except Exception as e:
        warnings.warn(f"Failed to load embeddings: {e}")
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

y_tiers = train_df["engagement_tier"].to_numpy(dtype=np.int32)
y_popular = train_df["is_popular"].to_numpy(dtype=np.int32)
y_counts = train_df["total_engagements"].to_numpy(dtype=np.float32)  # RAW COUNTS!

print(f"  Feature matrix: {X.shape}")
print(f"  Target stats: min={y_counts.min():.0f}, max={y_counts.max():.0f}, mean={y_counts.mean():.2f}, median={np.median(y_counts):.0f}")

# Sanity check: ensure we're using raw counts, not log-transformed
if y_counts.mean() < 5:
    print("  ⚠️  WARNING: Target mean is suspiciously low. Are values log-transformed?")
elif y_counts.min() < 0:
    print("  ⚠️  WARNING: Negative target values detected!")
else:
    print("  ✅ Target values look correct (raw counts)")

# ─── 4.5 Stratified Sampling ────────────────────────────────────────────────
if SAMPLE_FOR_TRAINING and X.shape[0] > SAMPLE_SIZE:
    print(f"\n⚡ Stratified sampling to {SAMPLE_SIZE:,} examples...")

    # Heavily oversample popular posts (our key focus)
    tier_sample_sizes = {
        0: int(SAMPLE_SIZE * 0.20),  # Low: 20%
        1: int(SAMPLE_SIZE * 0.30),  # Medium: 30%
        2: int(SAMPLE_SIZE * 0.30),  # High: 30%
        3: int(SAMPLE_SIZE * 0.20),  # Viral: 20%
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
    y_tiers = y_tiers[sample_idx]
    y_popular = y_popular[sample_idx]
    y_counts = y_counts[sample_idx]
    train_df = train_df.iloc[sample_idx].reset_index(drop=True)

    print(f"  Sampled distribution:")
    for tier_id, name in enumerate(tier_names):
        count = (y_tiers == tier_id).sum()
        pct = count / len(y_tiers) * 100
        print(f"    {name:20s}: {count:7,} ({pct:5.2f}%)")

    popular_count = y_popular.sum()
    print(f"    Popular posts: {popular_count:,} ({popular_count/len(y_popular)*100:.2f}%)")

# ─── 5. Time-based Split ────────────────────────────────────────────────────
print("\n📅 Creating time-based train/test split...")
train_df = train_df.sort_values('shared_at').reset_index(drop=True)
split_idx = int(len(train_df) * (1 - test_size))
split_date = train_df.iloc[split_idx]['shared_at']

train_mask = (train_df['shared_at'] < split_date).to_numpy()
X_train, X_test = X[train_mask], X[~train_mask]
y_tiers_train, y_tiers_test = y_tiers[train_mask], y_tiers[~train_mask]
y_popular_train, y_popular_test = y_popular[train_mask], y_popular[~train_mask]
y_counts_train, y_counts_test = y_counts[train_mask], y_counts[~train_mask]

print(f"  Train: {X_train.shape[0]:,} samples (up to {split_date.date()})")
print(f"  Test:  {X_test.shape[0]:,} samples (from {split_date.date()})")
print(f"    - Popular in test: {y_popular_test.sum():,} ({y_popular_test.sum()/len(y_popular_test)*100:.2f}%)")

# ─── 6. STAGE 1: Train Popularity Classifier ───────────────────────────────
print("\n" + "=" * 70)
print("🎯 STAGE 1: Will This Be POPULAR? (Binary Classification)")
print("=" * 70)

popularity_classifier = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=10,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    scale_pos_weight=2.0,  # Boost popular class
    n_jobs=-1,
    random_state=random_state,
    verbose=-1
)

print("\nTraining popularity classifier...")
popularity_classifier.fit(X_train, y_popular_train)

y_popular_pred = popularity_classifier.predict(X_test)
y_popular_proba = popularity_classifier.predict_proba(X_test)[:, 1]

print("\n📊 Stage 1 Results:")
print(f"  Accuracy: {accuracy_score(y_popular_test, y_popular_pred):.3f}")
print(f"  ROC-AUC:  {roc_auc_score(y_popular_test, y_popular_proba):.3f}")

print("\nClassification Report (Popular vs Not Popular):")
print(classification_report(
    y_popular_test,
    y_popular_pred,
    target_names=['Not Popular', 'Popular'],
    digits=3
))

# ─── 7. STAGE 2a: Low vs Medium (for not popular) ──────────────────────────
print("\n" + "=" * 70)
print("🎯 STAGE 2a: Low vs Medium (for NOT POPULAR posts)")
print("=" * 70)

not_popular_mask_train = y_popular_train == 0
not_popular_mask_test = y_popular_pred == 0

print(f"\nTraining on {not_popular_mask_train.sum():,} not-popular samples...")

low_medium_classifier = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    class_weight='balanced',
    n_jobs=-1,
    random_state=random_state,
    verbose=-1
)

low_medium_classifier.fit(
    X_train[not_popular_mask_train],
    y_tiers_train[not_popular_mask_train]
)

# Evaluate on test samples predicted as not popular
if not_popular_mask_test.sum() > 0:
    y_low_med_pred = low_medium_classifier.predict(X_test[not_popular_mask_test])
    y_low_med_actual = y_tiers_test[not_popular_mask_test]

    print(f"\n📊 Stage 2a Results (n={not_popular_mask_test.sum():,}):")
    print(f"  Accuracy: {accuracy_score(y_low_med_actual, y_low_med_pred):.3f}")

# ─── 8. STAGE 2b: High vs Viral (for popular) ──────────────────────────────
print("\n" + "=" * 70)
print("🎯 STAGE 2b: High vs Viral (for POPULAR posts)")
print("=" * 70)

popular_mask_train = y_popular_train == 1
popular_mask_test = y_popular_pred == 1

print(f"\nTraining on {popular_mask_train.sum():,} popular samples...")

high_viral_classifier = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    class_weight='balanced',
    n_jobs=-1,
    random_state=random_state,
    verbose=-1
)

high_viral_classifier.fit(
    X_train[popular_mask_train],
    y_tiers_train[popular_mask_train]
)

# Evaluate on test samples predicted as popular
if popular_mask_test.sum() > 0:
    y_high_viral_pred = high_viral_classifier.predict(X_test[popular_mask_test])
    y_high_viral_actual = y_tiers_test[popular_mask_test]

    print(f"\n📊 Stage 2b Results (n={popular_mask_test.sum():,}):")
    print(f"  Accuracy: {accuracy_score(y_high_viral_actual, y_high_viral_pred):.3f}")

# ─── 9. Combine Hierarchical Predictions ───────────────────────────────────
print("\n" + "=" * 70)
print("🎯 STAGE 3: Combined Hierarchical Predictions")
print("=" * 70)

y_tiers_pred_hierarchical = np.zeros(len(y_popular_pred), dtype=int)

# Predict tiers for not-popular posts
if not_popular_mask_test.sum() > 0:
    y_tiers_pred_hierarchical[not_popular_mask_test] = low_medium_classifier.predict(
        X_test[not_popular_mask_test]
    )

# Predict tiers for popular posts
if popular_mask_test.sum() > 0:
    y_tiers_pred_hierarchical[popular_mask_test] = high_viral_classifier.predict(
        X_test[popular_mask_test]
    )

print("\n📊 Overall Tier Classification Results:")
print(f"  Accuracy: {accuracy_score(y_tiers_test, y_tiers_pred_hierarchical):.3f}")

print("\nClassification Report:")
print(classification_report(
    y_tiers_test,
    y_tiers_pred_hierarchical,
    target_names=tier_names,
    digits=3
))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_tiers_test, y_tiers_pred_hierarchical)
print("          Predicted →")
print("Actual ↓  ", "  ".join([f"{name[:6]:>6s}" for name in tier_names]))
for i, name in enumerate(tier_names):
    print(f"{name[:10]:10s}", "  ".join([f"{cm[i][j]:6d}" for j in range(4)]))

# Key metric: How well do we identify popular posts?
actual_popular = y_tiers_test >= 2
pred_popular = y_tiers_pred_hierarchical >= 2
popular_precision = (actual_popular & pred_popular).sum() / pred_popular.sum() if pred_popular.sum() > 0 else 0
popular_recall = (actual_popular & pred_popular).sum() / actual_popular.sum() if actual_popular.sum() > 0 else 0
popular_f1 = 2 * (popular_precision * popular_recall) / (popular_precision + popular_recall) if (popular_precision + popular_recall) > 0 else 0

print("\n✨ KEY METRIC - Popular Post Detection (High + Viral):")
print(f"  Precision: {popular_precision:.3f}")
print(f"  Recall:    {popular_recall:.3f}")
print(f"  F1 Score:  {popular_f1:.3f}")

# ─── 10. Train Tier-Specific Regressors ────────────────────────────────────
print("\n" + "=" * 70)
print("🎯 STAGE 4: Training Tier-Specific Count Regressors")
print("=" * 70)

tier_regressors = {}

for tier_id, tier_name in enumerate(tier_names):
    tier_mask_train = y_tiers_train == tier_id
    n_samples = tier_mask_train.sum()

    print(f"\n  Training {tier_name} regressor ({n_samples:,} samples)...")

    if n_samples < 100:
        print(f"    ⚠️  Skipping (too few samples)")
        continue

    # Adjust parameters for viral tier (extreme values)
    if tier_id == 3:  # Viral tier needs special handling
        tier_regressors[tier_id] = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.8,  # Higher for extreme values
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=10,
            min_child_samples=5,     # Lower for rare viral posts
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.5,          # Less regularization for viral
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )
    else:
        tier_regressors[tier_id] = LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1
        )

    # Train on RAW counts (NO log transformation!)
    tier_regressors[tier_id].fit(
        X_train[tier_mask_train],
        y_counts_train[tier_mask_train]  # RAW COUNTS!
    )

    # Evaluate on test set for this tier
    tier_mask_test = y_tiers_test == tier_id
    if tier_mask_test.sum() > 0:
        tier_pred_raw = tier_regressors[tier_id].predict(X_test[tier_mask_test])
        tier_pred = np.clip(tier_pred_raw, 0, None)  # Just clip negatives, NO expm1()!
        tier_actual = y_counts_test[tier_mask_test]
        tier_mae = mean_absolute_error(tier_actual, tier_pred)

        # Debug: check prediction range
        print(f"    ✅ MAE: {tier_mae:.2f}, pred range: [{tier_pred.min():.1f}, {tier_pred.max():.1f}]")

        # Sanity check for this tier
        tier_min, tier_max = TIER_BOUNDARIES[list(TIER_BOUNDARIES.keys())[tier_id]]
        if tier_pred.mean() < tier_min or tier_pred.mean() > tier_max * 2:
            print(f"    ⚠️  WARNING: Predictions outside expected range for {tier_name}")

# Combined count predictions
y_pred_counts = np.zeros(len(y_tiers_pred_hierarchical))
for tier_id, reg in tier_regressors.items():
    tier_mask = y_tiers_pred_hierarchical == tier_id
    if tier_mask.sum() > 0:
        tier_pred_raw = reg.predict(X_test[tier_mask])
        # Clip to reasonable bounds per tier (prevent extreme outliers)
        tier_min, tier_max = TIER_BOUNDARIES[list(TIER_BOUNDARIES.keys())[tier_id]]
        if tier_id == 3:  # Viral: allow higher ceiling
            y_pred_counts[tier_mask] = np.clip(tier_pred_raw, 0, 50000)
        else:
            y_pred_counts[tier_mask] = np.clip(tier_pred_raw, 0, tier_max * 3)

overall_mae = mean_absolute_error(y_counts_test, y_pred_counts)
print(f"\n📊 Overall Count Prediction MAE: {overall_mae:.2f}")

# Additional diagnostics
print(f"   Predicted counts - min: {y_pred_counts.min():.1f}, max: {y_pred_counts.max():.1f}, mean: {y_pred_counts.mean():.1f}")
print(f"   Actual counts    - min: {y_counts_test.min():.1f}, max: {y_counts_test.max():.1f}, mean: {y_counts_test.mean():.1f}")

# Check if predictions are reasonable
pred_mean_ratio = y_pred_counts.mean() / y_counts_test.mean()
if pred_mean_ratio > 1.5 or pred_mean_ratio < 0.7:
    print(f"   ⚠️  WARNING: Predicted mean is {pred_mean_ratio:.2f}x actual mean (should be close to 1.0)")

# ─── 11. Save Everything ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("💾 Saving Models and Results")
print("=" * 70)

artifact = {
    "type": "hierarchical",
    "popularity_classifier": popularity_classifier,
    "low_medium_classifier": low_medium_classifier,
    "high_viral_classifier": high_viral_classifier,
    "tier_regressors": tier_regressors,
    "ohe": ohe,
    "scaler": scaler,
    "tfidf": tfidf if use_tfidf else None,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "use_tfidf": use_tfidf,
    "tier_names": tier_names,
    "tier_boundaries": TIER_BOUNDARIES,
    "popular_threshold": POPULAR_THRESHOLD,
    "embed_col_in_shares": EMBED_COL_IN_SHARES if not use_tfidf else None,
}

joblib.dump(artifact, os.path.join(save_dir, "hierarchical_engagement_model.joblib"))
print("  ✅ Saved: hierarchical_engagement_model.joblib")

# Save predictions
results_df = pd.DataFrame({
    "share_id": train_df[~train_mask]["share_id"].values,
    "actual_count": y_counts_test,
    "predicted_count": y_pred_counts,
    "actual_tier": y_tiers_test,
    "predicted_tier": y_tiers_pred_hierarchical,
    "predicted_popular": y_popular_pred,
    "popular_probability": y_popular_proba,
    "shared_at": train_df[~train_mask]["shared_at"].values,
})
results_df.to_parquet(os.path.join(save_dir, "hierarchical_predictions.parquet"), index=False)
print("  ✅ Saved: hierarchical_predictions.parquet")

# Save metadata
metadata = {
    "overall_mae": float(overall_mae),
    "tier_accuracy": float(accuracy_score(y_tiers_test, y_tiers_pred_hierarchical)),
    "popularity_accuracy": float(accuracy_score(y_popular_test, y_popular_pred)),
    "popularity_auc": float(roc_auc_score(y_popular_test, y_popular_proba)),
    "popular_detection_precision": float(popular_precision),
    "popular_detection_recall": float(popular_recall),
    "popular_detection_f1": float(popular_f1),
    "split_date": str(split_date),
    "tier_names": tier_names,
    "tier_boundaries": TIER_BOUNDARIES,
    "popular_threshold": POPULAR_THRESHOLD,
    "n_train": len(y_tiers_train),
    "n_test": len(y_tiers_test),
}
joblib.dump(metadata, os.path.join(save_dir, "hierarchical_predictions_meta.joblib"))
print("  ✅ Saved: hierarchical_predictions_meta.joblib")

print("\n" + "=" * 70)
print("✨ HIERARCHICAL TRAINING COMPLETE!")
print("=" * 70)
print("\nKey Metrics:")
print(f"  Popular Detection Recall: {popular_recall:.3f} ⭐ (Most Important!)")
print(f"  Popular Detection Precision: {popular_precision:.3f}")
print(f"  Popular Detection F1: {popular_f1:.3f}")
print(f"  Overall Tier Accuracy: {metadata['tier_accuracy']:.3f}")
print(f"  Popularity ROC-AUC: {metadata['popularity_auc']:.3f}")
print(f"  Count MAE: {metadata['overall_mae']:.2f}")
