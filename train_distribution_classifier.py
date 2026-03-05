"""
Audience Distribution Classifier
=================================
Predicts the DOMINANT engager category for each share using per-dimension
LightGBM classifiers.

Unlike the earlier regressor approach which
requires complete profiles, this classifier maximizes data by training
each dimension independently — only requiring that dimension's label to be
non-null.

For each target dimension (job_title, industry, company, job_level, country):
  - Build training labels from the dominant category of engagers per share
  - Train a LGBMClassifier with class_weight="balanced"
  - Evaluate with top-k accuracy on a shared test set

Outputs:
  - distribution_model.joblib — trained model artifacts
  - figures/distribution/ — evaluation plots
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import jensenshannon, cosine
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import top_k_accuracy_score
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception:
    print("LightGBM not found. Install with: pip install lightgbm")
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
fig_dir = "./figures/distribution"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

MIN_ENGAGEMENTS_PER_SHARE = 3
TRAIN_SAMPLE_SIZE = 20_000
max_text_features = 2000
test_size = 0.2
random_state = 42
N_ESTIMATORS = 500

MIN_LABEL_FREQ_JOB = 500
MIN_LABEL_FREQ_INDUSTRY = 500
MIN_LABEL_FREQ_COMPANY = 200
MIN_LABEL_FREQ_LEVEL = 500
MIN_LABEL_FREQ_COUNTRY = 500

TARGET_DIMENSIONS = ["job_title", "industry", "company", "job_level", "country"]

sns.set_style("whitegrid")

print("=" * 70)
print("AUDIENCE DISTRIBUTION CLASSIFIER")
print("=" * 70)
print(f"\n  min_engagements_per_share: {MIN_ENGAGEMENTS_PER_SHARE}")
print(f"  train_sample_size:         {TRAIN_SAMPLE_SIZE}")
print(f"  max_text_features:         {max_text_features}")
print(f"  test_size:                 {test_size}")
print(f"  n_estimators:              {N_ESTIMATORS}")

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 1: Loading Data")
print("=" * 70)

shares_df = pd.read_parquet(os.path.join(output_dir, "shares.parquet"))
eng_df = pd.read_parquet(os.path.join(output_dir, "engagements.parquet"))
profiles_df = pd.read_parquet(os.path.join(output_dir, "profiles.parquet"))
clients_df = pd.read_parquet(os.path.join(output_dir, "clients.parquet"))

total_engagements = len(eng_df)
print(f"  Shares:      {len(shares_df):>12,}")
print(f"  Engagements: {total_engagements:>12,}")
print(f"  Profiles:    {len(profiles_df):>12,}")
print(f"  Clients:     {len(clients_df):>12,}")

client_name_map = dict(zip(clients_df["client_id"], clients_df["title"]))
del clients_df

profile_to_user = dict(zip(eng_df["profile_id"], eng_df["es_user_id"]))

eng_df.rename(
    columns={
        "user_id": "sharer_user_id",
        "es_user_id": "engager_user_id",
        "created_at": "engagement_created_at",
        "client_id": "sharer_client_id",
    },
    inplace=True,
)
profiles_df["user_id"] = profiles_df["profile_id"].map(profile_to_user)

# ─── 2. Build Engager Profiles ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 2: Building Engager Profiles")
print("=" * 70)

engager_profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "employer_client_id",
     "job_title_levels", "location_country"]
].drop_duplicates(subset=["user_id"]).reset_index(drop=True)
engager_profiles["company_name"] = engager_profiles["employer_client_id"].map(client_name_map)
engager_profiles.rename(columns={"user_id": "engager_user_id"}, inplace=True)

# Keep ALL profiles — no complete_mask filtering
print(f"  Total engager profiles: {len(engager_profiles):,}")

# ─── 3. Join ALL Engagements with Profiles ───────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: Joining ALL Engagements with Profiles")
print("=" * 70)

engs = eng_df[
    ["sharer_user_id", "engagement_type", "share_id",
     "engagement_created_at", "engager_user_id"]
].copy().reset_index(drop=True)

del eng_df
gc.collect()

# Left join — some target fields will be NaN, that's fine
engs = engs.merge(
    engager_profiles[["engager_user_id", "job_title_role", "industry", "company_name",
                      "job_title_levels", "location_country"]],
    on="engager_user_id",
    how="left",
)
print(f"  Total engagements:  {total_engagements:,}")
print(f"  After profile join: {len(engs):,}")

del engager_profiles
gc.collect()

# ─── 4. Collapse Rare Labels ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 4: Collapsing Rare Labels")
print("=" * 70)

label_configs = {
    "job_title": ("job_title_role", MIN_LABEL_FREQ_JOB),
    "industry": ("industry", MIN_LABEL_FREQ_INDUSTRY),
    "company": ("company_name", MIN_LABEL_FREQ_COMPANY),
    "job_level": ("job_title_levels", MIN_LABEL_FREQ_LEVEL),
    "country": ("location_country", MIN_LABEL_FREQ_COUNTRY),
}

kept_labels = {}
for dim, (raw_col, min_freq) in label_configs.items():
    # For each dimension, compute frequencies from engagements where that field is non-null
    non_null_mask = engs[raw_col].notna() & (engs[raw_col] != "")
    counts = engs.loc[non_null_mask, raw_col].value_counts()
    kept = set(counts[counts >= min_freq].index)
    kept_labels[dim] = sorted(kept)

    # Frequent → keep, rare → "other", NaN stays NaN
    engs[f"{dim}_label"] = np.where(
        ~non_null_mask,
        np.nan,
        np.where(engs[raw_col].isin(kept), engs[raw_col], "other"),
    )
    n_non_null = engs[f"{dim}_label"].notna().sum()
    n_classes = engs.loc[engs[f"{dim}_label"].notna(), f"{dim}_label"].nunique()
    print(f"  {dim}: {len(counts)} raw -> {n_classes} classes (min_freq={min_freq}, non-null={n_non_null:,})")

# ─── 5. Build Shared Test Set ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: Building Shared Test Set")
print("=" * 70)

# "Complete" engagement = all 5 {dim}_label columns are non-null
label_cols = [f"{dim}_label" for dim in TARGET_DIMENSIONS]
complete_mask = engs[label_cols].notna().all(axis=1)
complete_engs = engs[complete_mask].copy()
print(f"  Complete engagements (all 5 dims non-null): {len(complete_engs):,}")

# Count complete engagements per share → keep shares with >= MIN_ENGAGEMENTS_PER_SHARE
complete_per_share = complete_engs.groupby("share_id").size()
eligible_test_shares = complete_per_share[complete_per_share >= MIN_ENGAGEMENTS_PER_SHARE].index
print(f"  Shares with >= {MIN_ENGAGEMENTS_PER_SHARE} complete engagements: {len(eligible_test_shares):,}")

# Get shared_at for sorting
share_dates = shares_df[["share_id", "shared_at"]].copy()
share_dates["shared_at"] = pd.to_datetime(share_dates["shared_at"], errors="coerce", utc=True)

eligible_test_df = share_dates[share_dates["share_id"].isin(eligible_test_shares)].sort_values("shared_at")
split_idx = int(len(eligible_test_df) * (1 - test_size))
test_share_ids = set(eligible_test_df.iloc[split_idx:]["share_id"].values)

split_date = eligible_test_df.iloc[split_idx]["shared_at"]
print(f"  Test shares:   {len(test_share_ids):,} (from {split_date})")
print(f"  Non-test pool: {split_idx:,} shares (up to {split_date})")

del complete_engs, complete_per_share, eligible_test_df
gc.collect()

# ─── 6. Per-Dimension Training Data ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 6: Per-Dimension Training Data")
print("=" * 70)

dim_train_labels = {}  # dim -> {share_id: dominant_label}
dim_train_share_ids = {}  # dim -> set of share_ids

for dim in TARGET_DIMENSIONS:
    col = f"{dim}_label"

    # From non-test engagements, keep rows where {dim}_label is not NaN
    non_test_mask = ~engs["share_id"].isin(test_share_ids)
    dim_mask = engs[col].notna()
    dim_engs = engs[non_test_mask & dim_mask][["share_id", col]].copy()

    # Compute dominant category per share (most frequent label)
    dominant = (
        dim_engs.groupby("share_id")[col]
        .agg(lambda x: x.value_counts().index[0])
    )

    # Count qualifying engagements per share
    engs_per_share = dim_engs.groupby("share_id").size()
    valid_shares = engs_per_share[engs_per_share >= MIN_ENGAGEMENTS_PER_SHARE].index
    dominant = dominant[dominant.index.isin(valid_shares)]

    # Sample if more than TRAIN_SAMPLE_SIZE available
    if len(dominant) > TRAIN_SAMPLE_SIZE:
        np.random.seed(random_state)
        sampled_ids = np.random.choice(dominant.index, size=TRAIN_SAMPLE_SIZE, replace=False)
        dominant = dominant[dominant.index.isin(sampled_ids)]

    dim_train_labels[dim] = dominant.to_dict()
    dim_train_share_ids[dim] = set(dominant.index)

    print(f"  {dim}: {len(dominant):,} training shares, {dominant.nunique()} classes")

del dim_engs
gc.collect()

# ─── 7. Build Features (once) ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 7: Building Feature Matrix")
print("=" * 70)

# Union of all share_ids (all training + test)
all_share_ids = set()
for dim in TARGET_DIMENSIONS:
    all_share_ids |= dim_train_share_ids[dim]
all_share_ids |= test_share_ids
print(f"  Total unique shares (train + test): {len(all_share_ids):,}")

# Union of training share_ids across all dims (for fitting transformers)
all_train_share_ids = set()
for dim in TARGET_DIMENSIONS:
    all_train_share_ids |= dim_train_share_ids[dim]
print(f"  Training shares (union across dims): {len(all_train_share_ids):,}")

# Join share metadata
shares = shares_df[["share_id", "shared_at", "share_content_type", "user_commentary"]].copy()
shares["shared_at"] = pd.to_datetime(shares["shared_at"], errors="coerce", utc=True)

# Sharer profiles
sharer_profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "job_title_levels",
     "location_region", "location_country", "job_title_class"]
].drop_duplicates(subset=["user_id"]).reset_index(drop=True)
sharer_profiles.rename(
    columns={
        "user_id": "sharer_user_id",
        "industry": "sharer_industry",
        "job_title_role": "sharer_job_title_role",
        "job_title_levels": "sharer_job_title_levels",
        "location_region": "sharer_location_region",
        "location_country": "sharer_location_country",
        "job_title_class": "sharer_job_title_class",
    },
    inplace=True,
)
del profiles_df
gc.collect()

# Get sharer_user_id for each share
share_sharer_map = pd.read_parquet(
    os.path.join(output_dir, "engagements.parquet"),
    columns=["share_id", "user_id"],
).drop_duplicates(subset=["share_id"]).rename(columns={"user_id": "sharer_user_id"})

# Build feature dataframe for all needed shares
feat_df = shares[shares["share_id"].isin(all_share_ids)].copy().reset_index(drop=True)
feat_df = feat_df.merge(share_sharer_map, on="share_id", how="left")
feat_df = feat_df.merge(sharer_profiles, on="sharer_user_id", how="left")

del shares, shares_df, sharer_profiles, share_sharer_map
gc.collect()

print(f"  Feature dataframe: {len(feat_df):,} shares")

# Temporal features
feat_df["share_hour"] = feat_df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
feat_df["share_dow"] = feat_df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)
feat_df["is_weekend"] = (feat_df["share_dow"] >= 5).astype(np.int8)
feat_df["is_business_hours"] = (
    (feat_df["share_hour"] >= 9) & (feat_df["share_hour"] <= 17)
).astype(np.int8)

# Content signal features
feat_df["user_commentary"] = feat_df["user_commentary"].fillna("")
feat_df["text_length"] = feat_df["user_commentary"].str.len()
feat_df["word_count"] = feat_df["user_commentary"].str.split().str.len()
feat_df["has_question"] = feat_df["user_commentary"].str.contains(r'\?').astype(np.int8)
feat_df["has_url"] = feat_df["user_commentary"].str.contains(r'http').astype(np.int8)
feat_df["exclamation_count"] = feat_df["user_commentary"].str.count('!')
feat_df["hashtag_count"] = feat_df["user_commentary"].str.count('#')
feat_df["mention_count"] = feat_df["user_commentary"].str.count('@')

cat_cols = [
    "share_content_type",
    "sharer_industry", "sharer_job_title_class",
    "sharer_location_country", "sharer_job_title_role",
    "sharer_job_title_levels", "sharer_location_region",
]
for c in cat_cols:
    if c in feat_df.columns:
        feat_df[c] = feat_df[c].fillna("unknown").astype(str)

num_cols = [
    "share_hour", "share_dow", "is_weekend", "is_business_hours",
    "text_length", "word_count", "has_question", "has_url",
    "exclamation_count", "hashtag_count", "mention_count",
]
for c in num_cols:
    if c in feat_df.columns:
        feat_df[c] = feat_df[c].fillna(0).astype(np.float32)

print(f"  Categorical features: {len(cat_cols)}")
print(f"  Numeric features:     {len(num_cols)}")

# Build share_id → row_idx map
share_id_to_idx = dict(zip(feat_df["share_id"], range(len(feat_df))))
available_share_ids = set(share_id_to_idx.keys())

# Filter training labels to shares that exist in the feature dataframe
for dim in TARGET_DIMENSIONS:
    before = len(dim_train_labels[dim])
    dim_train_labels[dim] = {sid: lbl for sid, lbl in dim_train_labels[dim].items()
                             if sid in available_share_ids}
    dim_train_share_ids[dim] = set(dim_train_labels[dim].keys())
    after = len(dim_train_labels[dim])
    if before != after:
        print(f"  {dim}: filtered {before:,} -> {after:,} training shares (removed {before - after:,} missing from features)")

# Also filter test shares
test_share_ids = test_share_ids & available_share_ids

# Recompute union of training share_ids
all_train_share_ids = set()
for dim in TARGET_DIMENSIONS:
    all_train_share_ids |= dim_train_share_ids[dim]

# Identify training rows for fitting transformers
train_row_mask = feat_df["share_id"].isin(all_train_share_ids).values

# Fit TF-IDF on training shares only
tfidf = TfidfVectorizer(
    max_features=max_text_features, stop_words="english", ngram_range=(1, 2)
)
tfidf.fit(feat_df.loc[train_row_mask, "user_commentary"])
X_text = tfidf.transform(feat_df["user_commentary"]).astype(np.float32)
print(f"  TF-IDF shape: {X_text.shape}")

# Fit OHE on training shares only
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
ohe.fit(feat_df.loc[train_row_mask, cat_cols])
X_cat = ohe.transform(feat_df[cat_cols]).astype(np.float32)
print(f"  OHE shape:    {X_cat.shape}")

# Fit scaler on training shares only
scaler = StandardScaler(with_mean=False)
scaler.fit(feat_df.loc[train_row_mask, num_cols].to_numpy(dtype=np.float32))
X_num = csr_matrix(scaler.transform(feat_df[num_cols].to_numpy(dtype=np.float32)))

X_all = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)
print(f"  Final matrix: {X_all.shape}")

# ─── 8. Train LGBMClassifier per Dimension ──────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 8: Training Classifiers")
print("=" * 70)

models = {}
label_encoders = {}

for dim in TARGET_DIMENSIONS:
    print(f"\n  --- {dim} ---")

    # Get training share_ids and labels
    train_labels = dim_train_labels[dim]
    train_sids = sorted(train_labels.keys())

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform([train_labels[sid] for sid in train_sids])
    label_encoders[dim] = le

    # Get feature row indices
    train_idx = [share_id_to_idx[sid] for sid in train_sids]
    X_train = X_all[train_idx]

    print(f"    Training samples: {len(train_sids):,}")
    print(f"    Classes: {len(le.classes_)}")

    clf = LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=10,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    models[dim] = clf

    print(f"    Training complete.")

# ─── 9. Evaluate on Shared Test Set ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 9: Evaluation on Shared Test Set")
print("=" * 70)

# Compute dominant category per test share per dimension from complete engagements
test_engs = engs[engs["share_id"].isin(test_share_ids)].copy()

results = {}
for dim in TARGET_DIMENSIONS:
    col = f"{dim}_label"
    le = label_encoders[dim]

    # Dominant category per test share for this dimension
    dim_test_engs = test_engs[test_engs[col].notna()][["share_id", col]]
    dominant_test = (
        dim_test_engs.groupby("share_id")[col]
        .agg(lambda x: x.value_counts().index[0])
    )

    # Filter to shares that have enough engagements for this dimension
    test_engs_per_share = dim_test_engs.groupby("share_id").size()
    valid_test_shares = test_engs_per_share[test_engs_per_share >= MIN_ENGAGEMENTS_PER_SHARE].index
    dominant_test = dominant_test[dominant_test.index.isin(valid_test_shares)]

    # Filter to labels known by the encoder
    known_labels = set(le.classes_)
    valid_mask = dominant_test.isin(known_labels)
    dominant_test = dominant_test[valid_mask]

    test_sids = sorted(dominant_test.index)
    if len(test_sids) == 0:
        print(f"  {dim}: No valid test shares, skipping.")
        continue

    y_test = le.transform(dominant_test[test_sids].values)

    # Get feature rows
    test_idx = [share_id_to_idx[sid] for sid in test_sids if sid in share_id_to_idx]
    test_sids_valid = [sid for sid in test_sids if sid in share_id_to_idx]
    if len(test_idx) != len(test_sids):
        # Recompute y_test for valid sids only
        y_test = le.transform(dominant_test[test_sids_valid].values)

    X_test = X_all[test_idx]

    # Predict
    y_pred = models[dim].predict(X_test)
    y_proba = models[dim].predict_proba(X_test)

    # Top-k accuracy
    n_classes = len(le.classes_)
    top1 = top_k_accuracy_score(y_test, y_proba, k=1, labels=range(n_classes))
    top3 = top_k_accuracy_score(y_test, y_proba, k=min(3, n_classes), labels=range(n_classes))
    top5 = top_k_accuracy_score(y_test, y_proba, k=min(5, n_classes), labels=range(n_classes))

    # Supplementary: distribution-level metrics on test set
    # Build actual and predicted distributions across the test set
    actual_counts = pd.Series(le.inverse_transform(y_test)).value_counts(normalize=True).reindex(le.classes_, fill_value=0).values
    pred_counts = pd.Series(le.inverse_transform(y_pred)).value_counts(normalize=True).reindex(le.classes_, fill_value=0).values

    js_div = jensenshannon(actual_counts, pred_counts) if actual_counts.sum() > 0 and pred_counts.sum() > 0 else 1.0
    cos_sim = 1 - cosine(actual_counts, pred_counts) if actual_counts.sum() > 0 and pred_counts.sum() > 0 else 0.0

    print(f"\n  {dim}:")
    print(f"    Test shares:    {len(test_sids_valid):,}")
    print(f"    Top-1 Accuracy: {top1:.3f}")
    print(f"    Top-3 Accuracy: {top3:.3f}")
    print(f"    Top-5 Accuracy: {top5:.3f}")
    print(f"    JS Divergence:  {js_div:.3f}  (aggregate distribution)")
    print(f"    Cosine Sim:     {cos_sim:.3f}  (aggregate distribution)")

    results[dim] = {
        "test_share_ids": test_sids_valid,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "js_divergence": js_div,
        "cosine_similarity": cos_sim,
        "actual_dist": actual_counts,
        "pred_dist": pred_counts,
        "classes": le.classes_,
        "n_test": len(test_sids_valid),
    }

del test_engs, engs
gc.collect()

# ─── 10. Plots ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 10: Generating Plots")
print("=" * 70)

# Plot 1: Per-dimension actual vs predicted distribution bars
for dim in TARGET_DIMENSIONS:
    if dim not in results:
        continue
    r = results[dim]
    classes = r["classes"]
    actual_dist = r["actual_dist"]
    pred_dist = r["pred_dist"]

    # Sort by actual proportion, take top 15
    top_idx = np.argsort(actual_dist)[::-1][:15]
    top_names = [classes[i] for i in top_idx]
    top_actual = actual_dist[top_idx]
    top_pred = pred_dist[top_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_names))
    width = 0.35

    ax.bar(x - width / 2, top_actual, width, label="Actual", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, top_pred, width, label="Predicted", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Proportion", fontsize=12)
    ax.set_title(f"{dim.replace('_', ' ').title()} — Actual vs Predicted Distribution",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(top_names, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"dist_comparison_{dim}.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: dist_comparison_{dim}.png")

# Plot 2: 2x3 scatter plot grid — predicted vs actual top-1 probability
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
dim_colors = {"job_title": "#e74c3c", "industry": "#3498db", "company": "#2ecc71",
              "job_level": "#9b59b6", "country": "#f39c12"}

for i, dim in enumerate(TARGET_DIMENSIONS):
    if dim not in results:
        continue
    ax = axes[i // 3][i % 3]
    r = results[dim]

    actual_top1_prob = r["y_proba"][np.arange(len(r["y_test"])), r["y_test"]]
    pred_top1_prob = r["y_proba"].max(axis=1)

    ax.scatter(actual_top1_prob, pred_top1_prob, alpha=0.1, s=5, color=dim_colors[dim])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("P(true class)", fontsize=11)
    ax.set_ylabel("P(predicted class)", fontsize=11)
    ax.set_title(f"{dim.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

axes[1][2].set_visible(False)

plt.suptitle("Predicted vs Actual Class Probabilities",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "scatter_top1_proportion.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: scatter_top1_proportion.png")

# Plot 3: Summary metrics bar chart (top-1/3/5)
fig, ax = plt.subplots(figsize=(10, 5))

metrics_data = []
for dim in TARGET_DIMENSIONS:
    if dim not in results:
        continue
    r = results[dim]
    metrics_data.append({
        "Dimension": dim.replace("_", " ").title(),
        "Top-1 Accuracy": r["top1"],
        "Top-3 Accuracy": r["top3"],
        "Top-5 Accuracy": r["top5"],
    })

metrics_df = pd.DataFrame(metrics_data).set_index("Dimension")
metrics_df.plot(kind="bar", ax=ax, width=0.7, alpha=0.85)

ax.set_ylabel("Accuracy (higher = better)", fontsize=12)
ax.set_title("Classification Accuracy by Target Dimension",
             fontsize=14, fontweight="bold")
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10, loc="lower right")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "metrics_summary.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: metrics_summary.png")

# ─── 11. Save Everything ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 11: Saving Models and Results")
print("=" * 70)

artifact = {
    "models": models,
    "label_encoders": label_encoders,
    "ohe": ohe,
    "scaler": scaler,
    "tfidf": tfidf,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "target_dimensions": TARGET_DIMENSIONS,
    "kept_labels": kept_labels,
    "min_label_freq": {
        "job": MIN_LABEL_FREQ_JOB,
        "industry": MIN_LABEL_FREQ_INDUSTRY,
        "company": MIN_LABEL_FREQ_COMPANY,
        "level": MIN_LABEL_FREQ_LEVEL,
        "country": MIN_LABEL_FREQ_COUNTRY,
    },
}
joblib.dump(artifact, os.path.join(save_dir, "distribution_model.joblib"))
print("  Saved: distribution_model.joblib")

# Save metadata
metadata = {}
for dim in TARGET_DIMENSIONS:
    if dim not in results:
        continue
    r = results[dim]
    metadata[dim] = {
        "top1_accuracy": float(r["top1"]),
        "top3_accuracy": float(r["top3"]),
        "top5_accuracy": float(r["top5"]),
        "js_divergence": float(r["js_divergence"]),
        "cosine_similarity": float(r["cosine_similarity"]),
        "n_classes": len(r["classes"]),
        "classes": list(r["classes"]),
        "n_test": r["n_test"],
        "n_train": len(dim_train_labels[dim]),
    }

metadata["split_date"] = str(split_date)
metadata["config"] = {
    "min_engagements_per_share": MIN_ENGAGEMENTS_PER_SHARE,
    "train_sample_size": TRAIN_SAMPLE_SIZE,
    "max_text_features": max_text_features,
    "test_size": test_size,
    "n_estimators": N_ESTIMATORS,
}

joblib.dump(metadata, os.path.join(save_dir, "distribution_predictions_meta.joblib"))
print("  Saved: distribution_predictions_meta.joblib")

# ─── 12. Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\n  Split date: {split_date}")

print(f"\n  {'Dimension':<15s} {'Train':>8s} {'Test':>8s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'JS Div':>8s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
for dim in TARGET_DIMENSIONS:
    if dim not in results:
        continue
    r = results[dim]
    n_train = len(dim_train_labels[dim])
    print(f"  {dim:<15s} {n_train:>8,} {r['n_test']:>8,} {r['top1']:>7.3f} "
          f"{r['top3']:>7.3f} {r['top5']:>7.3f} {r['js_divergence']:>8.3f}")

print(f"\n  Figures saved to: {fig_dir}/")
print(f"  Models saved to:  {save_dir}")
