"""
Distribution Predictor — Sample Size & Model Comparison
=========================================================
Runs the distribution predictor at 5K, 10K, 20K share sample sizes
using two approaches:

  A) Regressor:  MultiOutputRegressor(LGBMRegressor) predicting proportion
                 vectors, evaluated with top-K overlap via argsort.
  B) Classifier: LGBMClassifier predicting the dominant category per share,
                 evaluated with sklearn top_k_accuracy_score.

The classifier reframes the problem: instead of predicting every category's
proportion independently, it asks "which category dominates?" and uses
softmax probabilities for ranking — directly optimising for top-K.

Outputs:
  - figures/distribution_sample_analysis/  — learning curves & comparisons
  - outputs/distribution_sample_analysis_results.joblib — raw metrics
"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import top_k_accuracy_score
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
except Exception:
    print("LightGBM not found. Install with: pip install lightgbm")
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
fig_dir = "./figures/distribution_sample_analysis"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

SHARE_COUNTS = [5_000, 10_000, 20_000]
N_TRIALS = 3
MIN_ENGAGEMENTS_PER_SHARE = 5
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

print("=" * 70, flush=True)
print("DISTRIBUTION SAMPLE SIZE & MODEL COMPARISON", flush=True)
print("=" * 70, flush=True)
print(f"\n  share_counts:  {SHARE_COUNTS}")
print(f"  n_trials:      {N_TRIALS}")
print(f"  n_estimators:  {N_ESTIMATORS}")

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("STAGE 1: Loading Data", flush=True)
print("=" * 70, flush=True)

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
print("\n" + "=" * 70, flush=True)
print("STAGE 2: Building Engager Profiles", flush=True)
print("=" * 70, flush=True)

engager_profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "employer_client_id",
     "job_title_levels", "location_country"]
].drop_duplicates(subset=["user_id"]).reset_index(drop=True)
engager_profiles["company_name"] = engager_profiles["employer_client_id"].map(client_name_map)
engager_profiles.rename(columns={"user_id": "engager_user_id"}, inplace=True)

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
print("\n" + "=" * 70, flush=True)
print("STAGE 3: Filtering & Joining Engagements", flush=True)
print("=" * 70, flush=True)

engs = eng_df[eng_df["engager_user_id"].isin(valid_ids)][
    ["sharer_user_id", "engagement_type", "share_id",
     "engagement_created_at", "engager_user_id"]
].copy().reset_index(drop=True)

print(f"  Total engagements:        {total_engagements:,}")
print(f"  With complete profile:    {len(engs):,} ({len(engs)/total_engagements*100:.1f}%)")

del eng_df
gc.collect()

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
print("\n" + "=" * 70, flush=True)
print("STAGE 4: Collapsing Rare Labels", flush=True)
print("=" * 70, flush=True)

label_configs = {
    "job_title": ("job_title_role", MIN_LABEL_FREQ_JOB),
    "industry": ("industry", MIN_LABEL_FREQ_INDUSTRY),
    "company": ("company_name", MIN_LABEL_FREQ_COMPANY),
    "job_level": ("job_title_levels", MIN_LABEL_FREQ_LEVEL),
    "country": ("location_country", MIN_LABEL_FREQ_COUNTRY),
}

kept_labels = {}
for dim, (raw_col, min_freq) in label_configs.items():
    engs[raw_col] = engs[raw_col].fillna("other").replace("", "other")
    counts = engs[raw_col].value_counts()
    kept = set(counts[counts >= min_freq].index) - {"other"}
    kept_labels[dim] = sorted(kept)
    engs[f"{dim}_label"] = engs[raw_col].where(engs[raw_col].isin(kept), "other")
    n_classes = engs[f"{dim}_label"].nunique()
    print(f"  {dim}: {len(counts)} raw -> {n_classes} classes (min_freq={min_freq})")

# ─── 5. Aggregate to Share-Level Distributions ──────────────────────────────
print("\n" + "=" * 70, flush=True)
print("STAGE 5: Computing Share-Level Distributions", flush=True)
print("=" * 70, flush=True)

share_eng_counts = engs.groupby("share_id").size()
eligible_shares = share_eng_counts[share_eng_counts >= MIN_ENGAGEMENTS_PER_SHARE].index
engs_eligible = engs[engs["share_id"].isin(eligible_shares)].reset_index(drop=True)

print(f"  Shares with >= {MIN_ENGAGEMENTS_PER_SHARE} complete-profile engagements: {len(eligible_shares):,}")
print(f"  Eligible engagement rows: {len(engs_eligible):,}")

del engs
gc.collect()

dist_frames = {}
for dim in TARGET_DIMENSIONS:
    col = f"{dim}_label"
    categories = kept_labels[dim] + ["other"]
    counts_df = (
        engs_eligible.groupby(["share_id", col])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    row_sums = counts_df.sum(axis=1)
    proportions = counts_df.div(row_sums, axis=0)
    proportions.columns = [f"{dim}__{c}" for c in proportions.columns]
    dist_frames[dim] = proportions
    print(f"  {dim}: {len(categories)} categories, {len(proportions):,} shares")

share_dists = dist_frames[TARGET_DIMENSIONS[0]]
for dim in TARGET_DIMENSIONS[1:]:
    share_dists = share_dists.join(dist_frames[dim], how="inner")
share_dists = share_dists.reset_index()
print(f"\n  Combined: {share_dists.shape[0]:,} shares x {share_dists.shape[1]-1} proportion columns")

share_eng_count_map = share_eng_counts[eligible_shares].to_dict()
share_dists["n_engagements"] = share_dists["share_id"].map(share_eng_count_map)

del engs_eligible, dist_frames
gc.collect()

# ─── 6. Build Share-Level Features ──────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("STAGE 6: Building Share-Level Features", flush=True)
print("=" * 70, flush=True)

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

share_sharer_map = pd.read_parquet(
    os.path.join(output_dir, "engagements.parquet"),
    columns=["share_id", "user_id"],
).drop_duplicates(subset=["share_id"]).rename(columns={"user_id": "sharer_user_id"})

train_df = share_dists.merge(shares, on="share_id", how="inner")
train_df = train_df.merge(share_sharer_map, on="share_id", how="left")
train_df = train_df.merge(sharer_profiles, on="sharer_user_id", how="left")

del shares, sharer_profiles, share_sharer_map, share_dists
gc.collect()

print(f"  Total shares: {len(train_df):,}")

# Temporal features
train_df["shared_at"] = pd.to_datetime(train_df["shared_at"], errors="coerce", utc=True)
train_df["share_hour"] = train_df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
train_df["share_dow"] = train_df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)
train_df["is_weekend"] = (train_df["share_dow"] >= 5).astype(np.int8)
train_df["is_business_hours"] = (
    (train_df["share_hour"] >= 9) & (train_df["share_hour"] <= 17)
).astype(np.int8)

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

# ─── 7. Feature Matrix (built ONCE on all shares) ─────────────────────────
print("\n" + "=" * 70, flush=True)
print("STAGE 7: Building Feature Matrix (full dataset)", flush=True)
print("=" * 70, flush=True)

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

X_all = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)
print(f"  Final matrix: {X_all.shape}")

# ─── 8. Prepare Targets & Label Encoders ──────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("STAGE 8: Preparing Targets", flush=True)
print("=" * 70, flush=True)

target_columns = {}
proportion_arrays = {}
dominant_encoded = {}
fixed_label_encoders = {}

for dim in TARGET_DIMENSIONS:
    cols = [c for c in train_df.columns if c.startswith(f"{dim}__")]
    target_columns[dim] = cols
    category_names = [c.split("__")[1] for c in cols]

    Y = train_df[cols].to_numpy(dtype=np.float32)
    proportion_arrays[dim] = Y

    # Dominant category per share for classifier approach
    dominant_idx = Y.argmax(axis=1)
    dominant_labels = np.array([category_names[i] for i in dominant_idx])

    le = LabelEncoder()
    le.fit(category_names)
    dominant_encoded[dim] = le.transform(dominant_labels)
    fixed_label_encoders[dim] = le

    print(f"  {dim}: {len(cols)} categories, {len(le.classes_)} classes for classifier")

# Sort by time for consistent time-based splitting
shared_at = train_df["shared_at"].values
share_ids = train_df["share_id"].values
time_sort = np.argsort(shared_at)

X_all = X_all[time_sort]
shared_at = shared_at[time_sort]
share_ids = share_ids[time_sort]
for dim in TARGET_DIMENSIONS:
    proportion_arrays[dim] = proportion_arrays[dim][time_sort]
    dominant_encoded[dim] = dominant_encoded[dim][time_sort]

all_indices = np.arange(len(train_df))

# Cap SHARE_COUNTS to available data
max_possible = len(train_df)
SHARE_COUNTS = [s for s in SHARE_COUNTS if s <= max_possible]
print(f"\n  Shares available: {max_possible:,}")
print(f"  Share counts to test: {SHARE_COUNTS}")

# ─── 9. Sample Size Experiment ────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("STAGE 9: Running Sample Size Experiments", flush=True)
print("=" * 70, flush=True)

results = []

for n_shares in SHARE_COUNTS:
    print(f"\n{'─'*50}", flush=True)
    print(f"  n_shares = {n_shares:,}", flush=True)
    print(f"{'─'*50}", flush=True)

    for trial in range(N_TRIALS):
        rng = np.random.RandomState(random_state + trial)

        # Sample shares (indices in sorted order)
        sampled_idx = np.sort(rng.choice(max_possible, size=n_shares, replace=False))
        X_sample = X_all[sampled_idx]

        # Time-based split
        split_idx = int(n_shares * (1 - test_size))
        if split_idx < 50 or (n_shares - split_idx) < 50:
            continue

        X_train = X_sample[:split_idx]
        X_test = X_sample[split_idx:]

        for dim in TARGET_DIMENSIONS:
            Y_full = proportion_arrays[dim][sampled_idx]
            Y_train = Y_full[:split_idx]
            Y_test = Y_full[split_idx:]

            y_dom_full = dominant_encoded[dim][sampled_idx]
            y_dom_train = y_dom_full[:split_idx]
            y_dom_test = y_dom_full[split_idx:]

            n_classes = len(fixed_label_encoders[dim].classes_)
            n_cols = len(target_columns[dim])

            # ── Approach A: Regressor ──
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
            model_reg = MultiOutputRegressor(base_reg, n_jobs=1)
            model_reg.fit(X_train, Y_train)

            Y_pred = model_reg.predict(X_test)
            Y_pred = np.clip(Y_pred, 0, None)
            row_sums = Y_pred.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            Y_pred = Y_pred / row_sums

            # Top-K overlap for regressor
            reg_top_k = {1: [], 3: [], 5: []}
            for i in range(len(Y_test)):
                actual_order = np.argsort(Y_test[i])[::-1]
                pred_order = np.argsort(Y_pred[i])[::-1]
                for k in [1, 3, 5]:
                    actual_top_k = set(actual_order[:k])
                    pred_top_k = set(pred_order[:k])
                    overlap = len(actual_top_k & pred_top_k) / k
                    reg_top_k[k].append(overlap)

            for k in [1, 3, 5]:
                results.append({
                    "n_shares": n_shares,
                    "trial": trial,
                    "dimension": dim,
                    "approach": "regressor",
                    "metric": f"top{k}",
                    "value": np.mean(reg_top_k[k]),
                })

            del model_reg, Y_pred
            gc.collect()

            # ── Approach B: Classifier ──
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
            clf.fit(X_train, y_dom_train)
            y_pred_cls = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)

            labels = list(range(n_classes))
            top1_cls = np.mean(y_pred_cls == y_dom_test)
            try:
                top3_cls = top_k_accuracy_score(
                    y_dom_test, y_proba, k=min(3, n_classes), labels=labels
                )
                top5_cls = top_k_accuracy_score(
                    y_dom_test, y_proba, k=min(5, n_classes), labels=labels
                )
            except ValueError:
                top3_cls = top1_cls
                top5_cls = top1_cls

            for k, val in [(1, top1_cls), (3, top3_cls), (5, top5_cls)]:
                results.append({
                    "n_shares": n_shares,
                    "trial": trial,
                    "dimension": dim,
                    "approach": "classifier",
                    "metric": f"top{k}",
                    "value": val,
                })

            del clf, y_proba
            gc.collect()

        # Print progress
        for dim in TARGET_DIMENSIONS:
            reg_t3 = [r["value"] for r in results
                       if r["n_shares"] == n_shares and r["trial"] == trial
                       and r["dimension"] == dim and r["approach"] == "regressor"
                       and r["metric"] == "top3"]
            cls_t3 = [r["value"] for r in results
                       if r["n_shares"] == n_shares and r["trial"] == trial
                       and r["dimension"] == dim and r["approach"] == "classifier"
                       and r["metric"] == "top3"]
            if reg_t3 and cls_t3:
                print(f"    {dim:<12s} trial {trial}  reg_top3={reg_t3[0]:.3f}  cls_top3={cls_t3[0]:.3f}",
                      flush=True)

results_df = pd.DataFrame(results)

# ─── 10. Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 70, flush=True)

for metric_name in ["top3", "top5"]:
    print(f"\n  {metric_name.upper()} Accuracy (mean across trials):", flush=True)
    print(f"  {'Dimension':<12s} {'Approach':<12s}", end="", flush=True)
    for ns in SHARE_COUNTS:
        print(f"  {ns:>7,}", end="")
    print(flush=True)
    print(f"  {'-'*12} {'-'*12}", end="")
    for _ in SHARE_COUNTS:
        print(f"  {'-'*7}", end="")
    print(flush=True)

    for dim in TARGET_DIMENSIONS:
        for approach in ["regressor", "classifier"]:
            print(f"  {dim:<12s} {approach:<12s}", end="", flush=True)
            for ns in SHARE_COUNTS:
                subset = results_df[
                    (results_df["n_shares"] == ns) &
                    (results_df["dimension"] == dim) &
                    (results_df["approach"] == approach) &
                    (results_df["metric"] == metric_name)
                ]
                if len(subset) > 0:
                    print(f"  {subset['value'].mean():>7.3f}", end="")
                else:
                    print(f"  {'N/A':>7s}", end="")
            print(flush=True)

# ─── 11. Plots ────────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Generating Plots", flush=True)
print("=" * 70, flush=True)

dim_colors = {"job_title": "#e74c3c", "industry": "#3498db", "company": "#2ecc71",
              "job_level": "#9b59b6", "country": "#f39c12"}
dim_labels = {"job_title": "Job Role", "industry": "Industry", "company": "Company",
              "job_level": "Job Level", "country": "Country"}
approach_styles = {"regressor": "--", "classifier": "-"}
approach_markers = {"regressor": "s", "classifier": "o"}

# Plot 1: Learning curves — top-3 accuracy
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
for i, dim in enumerate(TARGET_DIMENSIONS):
    ax = axes[i // 3][i % 3]
    for approach in ["regressor", "classifier"]:
        subset = results_df[
            (results_df["dimension"] == dim) &
            (results_df["approach"] == approach) &
            (results_df["metric"] == "top3")
        ]
        grouped = subset.groupby("n_shares")["value"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        ax.plot(means.index, means.values,
                linestyle=approach_styles[approach],
                marker=approach_markers[approach],
                label=approach.title(),
                color=dim_colors[dim], linewidth=2, markersize=7)
        ax.fill_between(means.index, means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.12, color=dim_colors[dim])

    ax.set_xlabel("Number of Shares", fontsize=12)
    ax.set_ylabel("Top-3 Accuracy" if i % 3 == 0 else "", fontsize=12)
    ax.set_title(f"{dim_labels[dim]}", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

axes[1][2].set_visible(False)

plt.suptitle("Top-3 Accuracy: Regressor vs Classifier by Sample Size",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "learning_curve_top3.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: learning_curve_top3.png", flush=True)

# Plot 2: Learning curves — top-5 accuracy
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
for i, dim in enumerate(TARGET_DIMENSIONS):
    ax = axes[i // 3][i % 3]
    for approach in ["regressor", "classifier"]:
        subset = results_df[
            (results_df["dimension"] == dim) &
            (results_df["approach"] == approach) &
            (results_df["metric"] == "top5")
        ]
        grouped = subset.groupby("n_shares")["value"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        ax.plot(means.index, means.values,
                linestyle=approach_styles[approach],
                marker=approach_markers[approach],
                label=approach.title(),
                color=dim_colors[dim], linewidth=2, markersize=7)
        ax.fill_between(means.index, means.values - stds.values,
                        means.values + stds.values,
                        alpha=0.12, color=dim_colors[dim])

    ax.set_xlabel("Number of Shares", fontsize=12)
    ax.set_ylabel("Top-5 Accuracy" if i % 3 == 0 else "", fontsize=12)
    ax.set_title(f"{dim_labels[dim]}", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

axes[1][2].set_visible(False)

plt.suptitle("Top-5 Accuracy: Regressor vs Classifier by Sample Size",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "learning_curve_top5.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: learning_curve_top5.png", flush=True)

# Plot 3: Comparison bar chart — regressor vs classifier at largest sample size
largest_n = max(SHARE_COUNTS)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for plot_i, metric_name in enumerate(["top3", "top5"]):
    ax = axes[plot_i]
    x = np.arange(len(TARGET_DIMENSIONS))
    width = 0.35

    reg_vals = []
    cls_vals = []
    for dim in TARGET_DIMENSIONS:
        for approach, vals_list in [("regressor", reg_vals), ("classifier", cls_vals)]:
            subset = results_df[
                (results_df["n_shares"] == largest_n) &
                (results_df["dimension"] == dim) &
                (results_df["approach"] == approach) &
                (results_df["metric"] == metric_name)
            ]
            vals_list.append(subset["value"].mean() if len(subset) > 0 else 0)

    bars1 = ax.bar(x - width/2, reg_vals, width, label="Regressor", color="#95a5a6", alpha=0.85)
    bars2 = ax.bar(x + width/2, cls_vals, width, label="Classifier", color="#3498db", alpha=0.85)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{metric_name.replace('top', 'Top-')} Accuracy at {largest_n:,} Shares",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([dim_labels[d] for d in TARGET_DIMENSIONS])
    ax.set_ylim([0, 1.15])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Regressor vs Classifier: Distribution Prediction",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "comparison_bar.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: comparison_bar.png", flush=True)

# ─── 12. Save Results ─────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Saving Results", flush=True)
print("=" * 70, flush=True)

save_payload = {
    "results_df": results_df.to_dict(),
    "share_counts": SHARE_COUNTS,
    "n_trials": N_TRIALS,
    "target_dimensions": TARGET_DIMENSIONS,
    "n_estimators": N_ESTIMATORS,
    "total_eligible_shares": max_possible,
}
joblib.dump(save_payload, os.path.join(save_dir, "distribution_sample_analysis_results.joblib"))
print(f"  Saved: {save_dir}distribution_sample_analysis_results.joblib")
print(f"  Figures: {fig_dir}/")

print("\n" + "=" * 70, flush=True)
print("DONE", flush=True)
print("=" * 70, flush=True)
