"""
Audience Distribution Predictor
================================
Predicts the DISTRIBUTION of engager profiles for a given share.

Instead of predicting individual engager profiles (classification),
this predicts what proportion of engagers will come from each category.

For example, a share might attract:
  - 40% Marketing, 30% Sales, 20% Engineering, 10% Other (job title)
  - 50% Computer Software, 25% Financial Services, 25% Other (industry)

This is a multi-output regression problem where each target is a
proportion vector that sums to 1.

Approach:
  1. Aggregate engagements at the share level → compute proportion vectors
  2. Use share-level features (content, sharer profile, temporal)
  3. Train separate multi-output regressors for each target dimension
  4. Evaluate with cosine similarity, JS divergence, top-K overlap

Outputs:
  - distribution_model.joblib — trained model artifacts
  - distribution_predictions.parquet — test set predictions
  - distribution_predictions_meta.joblib — metadata and metrics
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
except Exception:
    print("LightGBM not found. Install with: pip install lightgbm")
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
fig_dir = "./figures/distribution"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

SAMPLE_SIZE = 0                 # 0 = full dataset; set e.g. 50_000 for quick test
MIN_ENGAGEMENTS_PER_SHARE = 5   # shares with fewer engagements are too noisy
max_text_features = 2000
test_size = 0.2
random_state = 42
N_ESTIMATORS = 500

# Minimum frequency for a category to get its own column (otherwise → "other")
MIN_LABEL_FREQ_JOB = 500
MIN_LABEL_FREQ_INDUSTRY = 500
MIN_LABEL_FREQ_COMPANY = 200
MIN_LABEL_FREQ_LEVEL = 500
MIN_LABEL_FREQ_COUNTRY = 500

TARGET_DIMENSIONS = ["job_title", "industry", "company", "job_level", "country"]

sns.set_style("whitegrid")

print("=" * 70)
print("AUDIENCE DISTRIBUTION PREDICTOR")
print("=" * 70)
print(f"\n  sample_size:              {SAMPLE_SIZE if SAMPLE_SIZE > 0 else 'full dataset'}")
print(f"  min_engagements_per_share: {MIN_ENGAGEMENTS_PER_SHARE}")
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

# Filter to profiles with all three fields known
complete_mask = (
    engager_profiles["job_title_role"].notna() & (engager_profiles["job_title_role"] != "") &
    engager_profiles["industry"].notna() & (engager_profiles["industry"] != "") &
    engager_profiles["company_name"].notna() & (engager_profiles["company_name"] != "")
)
complete_profiles = engager_profiles[complete_mask].reset_index(drop=True)
valid_ids = set(complete_profiles["engager_user_id"].dropna())

print(f"  Total engager profiles:    {len(engager_profiles):,}")
print(f"  With all fields complete:  {len(complete_profiles):,} ({len(complete_profiles)/len(engager_profiles)*100:.1f}%)")

# ─── 3. Filter Engagements & Join Profiles ──────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: Filtering & Joining Engagements")
print("=" * 70)

engs = eng_df[eng_df["engager_user_id"].isin(valid_ids)][
    ["sharer_user_id", "engagement_type", "share_id",
     "engagement_created_at", "engager_user_id"]
].copy().reset_index(drop=True)

print(f"  Total engagements:        {total_engagements:,}")
print(f"  With complete profile:    {len(engs):,} ({len(engs)/total_engagements*100:.1f}%)")

del eng_df
gc.collect()

# Join engager profile attributes
engs = engs.merge(
    complete_profiles[["engager_user_id", "job_title_role", "industry", "company_name",
                       "job_title_levels", "location_country"]],
    on="engager_user_id",
    how="inner",
)
print(f"  After profile join:       {len(engs):,}")

del complete_profiles
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
    # Collapse NaN/blank to "other" before counting (job_level, country not in complete_mask)
    engs[raw_col] = engs[raw_col].fillna("other").replace("", "other")
    counts = engs[raw_col].value_counts()
    kept = set(counts[counts >= min_freq].index) - {"other"}
    kept_labels[dim] = sorted(kept)

    engs[f"{dim}_label"] = engs[raw_col].where(engs[raw_col].isin(kept), "other")
    n_classes = engs[f"{dim}_label"].nunique()
    print(f"  {dim}: {len(counts)} raw -> {n_classes} classes (min_freq={min_freq})")

# ─── 5. Aggregate to Share-Level Distributions ──────────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: Computing Share-Level Distributions")
print("=" * 70)

# Count engagements per share
share_eng_counts = engs.groupby("share_id").size()
eligible_shares = share_eng_counts[share_eng_counts >= MIN_ENGAGEMENTS_PER_SHARE].index
engs_eligible = engs[engs["share_id"].isin(eligible_shares)].reset_index(drop=True)

print(f"  Shares with >= {MIN_ENGAGEMENTS_PER_SHARE} complete-profile engagements: {len(eligible_shares):,}")
print(f"  Eligible engagement rows: {len(engs_eligible):,}")

del engs
gc.collect()

# Build proportion vectors per share for each target dimension
dist_frames = {}

for dim in TARGET_DIMENSIONS:
    col = f"{dim}_label"
    categories = kept_labels[dim] + ["other"]

    # Compute counts per share per category
    counts_df = (
        engs_eligible.groupby(["share_id", col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )

    # Normalize to proportions
    row_sums = counts_df.sum(axis=1)
    proportions = counts_df.div(row_sums, axis=0)

    # Rename columns with prefix
    proportions.columns = [f"{dim}__{c}" for c in proportions.columns]
    dist_frames[dim] = proportions

    print(f"  {dim}: {len(categories)} categories, {len(proportions):,} shares")

# Merge all distributions into one dataframe keyed by share_id
share_dists = dist_frames[TARGET_DIMENSIONS[0]]
for dim in TARGET_DIMENSIONS[1:]:
    share_dists = share_dists.join(dist_frames[dim], how="inner")

share_dists = share_dists.reset_index()
print(f"\n  Combined distribution table: {share_dists.shape[0]:,} shares x {share_dists.shape[1]-1} proportion columns")

# Also store the engagement count per share for weighting
share_eng_count_map = share_eng_counts[eligible_shares].to_dict()
share_dists["n_engagements"] = share_dists["share_id"].map(share_eng_count_map)

del engs_eligible, dist_frames
gc.collect()

# ─── 6. Build Share-Level Features ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 6: Building Share-Level Features")
print("=" * 70)

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

shares = shares_df[["share_id", "shared_at", "share_content_type", "user_commentary"]].copy()
del shares_df
gc.collect()

# Get sharer_user_id for each share from the first engagement
# (Need to reload a small piece for this mapping)
share_sharer_map = pd.read_parquet(
    os.path.join(output_dir, "engagements.parquet"),
    columns=["share_id", "user_id"],
).drop_duplicates(subset=["share_id"]).rename(columns={"user_id": "sharer_user_id"})

# Join features onto share_dists
train_df = share_dists.merge(shares, on="share_id", how="inner")
train_df = train_df.merge(share_sharer_map, on="share_id", how="left")
train_df = train_df.merge(sharer_profiles, on="sharer_user_id", how="left")

del shares, sharer_profiles, share_sharer_map, share_dists
gc.collect()

if SAMPLE_SIZE > 0 and len(train_df) > SAMPLE_SIZE:
    np.random.seed(random_state)
    sample_idx = np.random.choice(len(train_df), size=SAMPLE_SIZE, replace=False)
    train_df = train_df.iloc[sample_idx].reset_index(drop=True)
    print(f"  Sampled to {SAMPLE_SIZE:,} shares")

print(f"  Training shares: {len(train_df):,}")

# Temporal features
train_df["shared_at"] = pd.to_datetime(train_df["shared_at"], errors="coerce", utc=True)
train_df["share_hour"] = train_df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
train_df["share_dow"] = train_df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)
train_df["is_weekend"] = (train_df["share_dow"] >= 5).astype(np.int8)
train_df["is_business_hours"] = (
    (train_df["share_hour"] >= 9) & (train_df["share_hour"] <= 17)
).astype(np.int8)

# Content signal features
train_df["user_commentary"] = train_df["user_commentary"].fillna("")
train_df["text_length"] = train_df["user_commentary"].str.len()
train_df["word_count"] = train_df["user_commentary"].str.split().str.len()
train_df["has_question"] = train_df["user_commentary"].str.contains(r'\?').astype(np.int8)
train_df["has_url"] = train_df["user_commentary"].str.contains(r'http').astype(np.int8)
train_df["exclamation_count"] = train_df["user_commentary"].str.count('!')
train_df["hashtag_count"] = train_df["user_commentary"].str.count('#')
train_df["mention_count"] = train_df["user_commentary"].str.count('@')

cat_cols = [
    "share_content_type",
    "sharer_industry", "sharer_job_title_class",
    "sharer_location_country", "sharer_job_title_role",
    "sharer_job_title_levels", "sharer_location_region",
]
for c in cat_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna("unknown").astype(str)

num_cols = [
    "share_hour", "share_dow", "is_weekend", "is_business_hours",
    "text_length", "word_count", "has_question", "has_url",
    "exclamation_count", "hashtag_count", "mention_count",
]
for c in num_cols:
    if c in train_df.columns:
        train_df[c] = train_df[c].fillna(0).astype(np.float32)

print(f"  Categorical features: {len(cat_cols)}")
print(f"  Numeric features:     {len(num_cols)}")

# ─── 7. Feature Matrix ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 7: Building Feature Matrix")
print("=" * 70)

tfidf = TfidfVectorizer(
    max_features=max_text_features, stop_words="english", ngram_range=(1, 2)
)
X_text = tfidf.fit_transform(train_df["user_commentary"]).astype(np.float32)
print(f"  TF-IDF shape: {X_text.shape}")

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_cat = ohe.fit_transform(train_df[cat_cols]).astype(np.float32)
print(f"  OHE shape:    {X_cat.shape}")

scaler = StandardScaler(with_mean=False)
X_num = csr_matrix(scaler.fit_transform(train_df[num_cols].to_numpy(dtype=np.float32)))

X = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)
print(f"  Final matrix: {X.shape}")

# ─── 8. Extract Target Matrices ─────────────────────────────────────────────
target_columns = {}
for dim in TARGET_DIMENSIONS:
    cols = [c for c in train_df.columns if c.startswith(f"{dim}__")]
    target_columns[dim] = cols
    print(f"  {dim}: {len(cols)} proportion columns")

# ─── 9. Time-Based Train/Test Split ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 8: Train/Test Split")
print("=" * 70)

sort_order = train_df["shared_at"].argsort().values
train_df = train_df.iloc[sort_order].reset_index(drop=True)
X = X[sort_order]

split_idx = int(len(train_df) * (1 - test_size))
split_date = train_df.iloc[split_idx]["shared_at"]

train_mask = np.arange(len(train_df)) < split_idx
X_train, X_test = X[train_mask], X[~train_mask]

print(f"  Train: {X_train.shape[0]:,} shares (up to {split_date})")
print(f"  Test:  {X_test.shape[0]:,} shares (from {split_date})")

# ─── 10. Train Distribution Regressors ──────────────────────────────────────
models = {}
results = {}

for dim in TARGET_DIMENSIONS:
    cols = target_columns[dim]
    Y = train_df[cols].to_numpy(dtype=np.float32)
    Y_train, Y_test = Y[train_mask], Y[~train_mask]

    print("\n" + "=" * 70)
    print(f"TRAINING: {dim} distribution ({len(cols)} categories)")
    print("=" * 70)

    base_reg = LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=10,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )

    model = MultiOutputRegressor(base_reg, n_jobs=1)
    model.fit(X_train, Y_train)
    models[dim] = model

    # Predict
    Y_pred = model.predict(X_test)

    # Clip negatives and re-normalize to valid probability distributions
    Y_pred = np.clip(Y_pred, 0, None)
    row_sums = Y_pred.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    Y_pred = Y_pred / row_sums

    # ── Evaluation Metrics ──

    # 1. Cosine similarity (1 = perfect, 0 = orthogonal)
    cos_sims = []
    for i in range(len(Y_test)):
        if Y_test[i].sum() > 0 and Y_pred[i].sum() > 0:
            cos_sims.append(1 - cosine(Y_test[i], Y_pred[i]))
    mean_cos_sim = np.mean(cos_sims) if cos_sims else 0

    # 2. Jensen-Shannon divergence (0 = identical, ~0.83 = maximally different)
    js_divs = []
    for i in range(len(Y_test)):
        if Y_test[i].sum() > 0 and Y_pred[i].sum() > 0:
            js_divs.append(jensenshannon(Y_test[i], Y_pred[i]))
    mean_js_div = np.nanmean(js_divs) if js_divs else 1

    # 3. Top-K category overlap
    # Does the model correctly identify the top categories?
    category_names = [c.split("__")[1] for c in cols]
    top_k_overlaps = {1: [], 3: [], 5: []}

    for i in range(len(Y_test)):
        actual_order = np.argsort(Y_test[i])[::-1]
        pred_order = np.argsort(Y_pred[i])[::-1]

        for k in [1, 3, 5]:
            actual_top_k = set(actual_order[:k])
            pred_top_k = set(pred_order[:k])
            overlap = len(actual_top_k & pred_top_k) / k
            top_k_overlaps[k].append(overlap)

    mean_top_k = {k: np.mean(v) for k, v in top_k_overlaps.items()}

    # 4. Mean absolute error on proportions
    mae = np.mean(np.abs(Y_test - Y_pred))

    print(f"\n  Cosine Similarity:     {mean_cos_sim:.3f}  (1.0 = perfect)")
    print(f"  JS Divergence:         {mean_js_div:.3f}  (0.0 = perfect)")
    print(f"  Proportion MAE:        {mae:.4f}")
    print(f"  Top-1 Category Match:  {mean_top_k[1]:.3f}")
    print(f"  Top-3 Category Overlap:{mean_top_k[3]:.3f}")
    print(f"  Top-5 Category Overlap:{mean_top_k[5]:.3f}")

    # Show example: average predicted vs actual distribution for top categories
    avg_actual = Y_test.mean(axis=0)
    avg_pred = Y_pred.mean(axis=0)
    top_idx = np.argsort(avg_actual)[::-1][:10]

    print(f"\n  Top 10 categories (avg actual vs predicted proportion):")
    print(f"    {'Category':<35s} {'Actual':>8s} {'Predicted':>10s} {'Error':>8s}")
    print(f"    {'-'*35} {'-'*8} {'-'*10} {'-'*8}")
    for idx in top_idx:
        name = category_names[idx]
        print(f"    {name:<35s} {avg_actual[idx]:>8.3f} {avg_pred[idx]:>10.3f} {avg_pred[idx]-avg_actual[idx]:>+8.3f}")

    results[dim] = {
        "Y_test": Y_test,
        "Y_pred": Y_pred,
        "cosine_similarity": mean_cos_sim,
        "js_divergence": mean_js_div,
        "proportion_mae": mae,
        "top_k_overlap": mean_top_k,
        "category_names": category_names,
    }

# ─── 11. Plots ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Generating Plots")
print("=" * 70)

# Plot 1: Actual vs Predicted distribution (bar chart for each dimension)
for dim in TARGET_DIMENSIONS:
    r = results[dim]
    avg_actual = r["Y_test"].mean(axis=0)
    avg_pred = r["Y_pred"].mean(axis=0)
    names = r["category_names"]

    # Sort by actual proportion, take top 15
    top_idx = np.argsort(avg_actual)[::-1][:15]
    top_names = [names[i] for i in top_idx]
    top_actual = avg_actual[top_idx]
    top_pred = avg_pred[top_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, top_actual, width, label="Actual", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width/2, top_pred, width, label="Predicted", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Average Proportion", fontsize=12)
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

# Plot 2: Scatter plot of predicted vs actual top-1 proportion
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
dim_colors = {"job_title": "#e74c3c", "industry": "#3498db", "company": "#2ecc71",
              "job_level": "#9b59b6", "country": "#f39c12"}

for i, dim in enumerate(TARGET_DIMENSIONS):
    ax = axes[i // 3][i % 3]
    r = results[dim]
    actual_top1_prop = r["Y_test"].max(axis=1)
    pred_top1_prop = r["Y_pred"].max(axis=1)

    ax.scatter(actual_top1_prop, pred_top1_prop, alpha=0.1, s=5, color=dim_colors[dim])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Actual Top-1 Proportion", fontsize=11)
    ax.set_ylabel("Predicted Top-1 Proportion", fontsize=11)
    ax.set_title(f"{dim.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

# Hide unused subplot (position [1][2])
axes[1][2].set_visible(False)

plt.suptitle("Predicted vs Actual Dominant Category Proportion",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "scatter_top1_proportion.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: scatter_top1_proportion.png")

# Plot 3: Summary metrics bar chart
fig, ax = plt.subplots(figsize=(10, 5))

metrics_data = []
for dim in TARGET_DIMENSIONS:
    r = results[dim]
    metrics_data.append({
        "Dimension": dim.replace("_", " ").title(),
        "Cosine Sim": r["cosine_similarity"],
        "1 - JS Div": 1 - r["js_divergence"],
        "Top-1 Match": r["top_k_overlap"][1],
        "Top-3 Overlap": r["top_k_overlap"][3],
    })

metrics_df = pd.DataFrame(metrics_data).set_index("Dimension")
metrics_df.plot(kind="bar", ax=ax, width=0.7, alpha=0.85)

ax.set_ylabel("Score (higher = better)", fontsize=12)
ax.set_title("Distribution Prediction Quality by Target",
             fontsize=14, fontweight="bold")
ax.set_ylim([0, 1.05])
ax.legend(fontsize=10, loc="lower right")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "metrics_summary.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: metrics_summary.png")

# ─── 12. Save Everything ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Saving Models and Results")
print("=" * 70)

artifact = {
    "models": models,
    "ohe": ohe,
    "scaler": scaler,
    "tfidf": tfidf,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "target_dimensions": TARGET_DIMENSIONS,
    "target_columns": target_columns,
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

# Save predictions
pred_rows = {
    "share_id": train_df[~train_mask]["share_id"].values,
    "shared_at": train_df[~train_mask]["shared_at"].values,
    "n_engagements": train_df[~train_mask]["n_engagements"].values,
}

for dim in TARGET_DIMENSIONS:
    r = results[dim]
    for j, cat_name in enumerate(r["category_names"]):
        pred_rows[f"{dim}__{cat_name}__actual"] = r["Y_test"][:, j]
        pred_rows[f"{dim}__{cat_name}__predicted"] = r["Y_pred"][:, j]

pred_df = pd.DataFrame(pred_rows)
pred_df.to_parquet(os.path.join(save_dir, "distribution_predictions.parquet"), index=False)
print("  Saved: distribution_predictions.parquet")

# Save metadata
metadata = {}
for dim in TARGET_DIMENSIONS:
    r = results[dim]
    metadata[dim] = {
        "cosine_similarity": float(r["cosine_similarity"]),
        "js_divergence": float(r["js_divergence"]),
        "proportion_mae": float(r["proportion_mae"]),
        "top_k_overlap": {int(k): float(v) for k, v in r["top_k_overlap"].items()},
        "n_categories": len(r["category_names"]),
        "categories": r["category_names"],
    }

metadata["split_date"] = str(split_date)
metadata["n_train"] = int(train_mask.sum())
metadata["n_test"] = int((~train_mask).sum())
metadata["config"] = {
    "sample_size": SAMPLE_SIZE,
    "min_engagements_per_share": MIN_ENGAGEMENTS_PER_SHARE,
    "max_text_features": max_text_features,
    "test_size": test_size,
    "n_estimators": N_ESTIMATORS,
}

joblib.dump(metadata, os.path.join(save_dir, "distribution_predictions_meta.joblib"))
print("  Saved: distribution_predictions_meta.joblib")

# ─── 13. Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(f"\n  Train / Test:  {int(train_mask.sum()):,} / {int((~train_mask).sum()):,} shares")
print(f"  Split date:    {split_date}")

print(f"\n  {'Dimension':<15s} {'Cosine':>8s} {'JS Div':>8s} {'MAE':>8s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")
for dim in TARGET_DIMENSIONS:
    r = results[dim]
    print(f"  {dim:<15s} {r['cosine_similarity']:>8.3f} {r['js_divergence']:>8.3f} "
          f"{r['proportion_mae']:>8.4f} {r['top_k_overlap'][1]:>7.3f} "
          f"{r['top_k_overlap'][3]:>7.3f} {r['top_k_overlap'][5]:>7.3f}")

print(f"\n  Figures saved to: {fig_dir}/")
print(f"  Models saved to:  {save_dir}")
