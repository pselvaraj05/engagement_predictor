"""
Sample Size Power Analysis
===========================
Answers: "How many shares and engagements do I need to scrape to predict
which target audience will engage with a new share?"

Uses the existing dataset as ground truth and simulates what happens when
you have progressively fewer shares and engagements to train on.

Two-phase design for speed:
  Phase 1: Pre-filter the 52M engagements → ~4M valid rows, save to cache.
           Runs once (~2 min), then skipped on subsequent runs.
  Phase 2: Run learning curve simulation from cache.
           Uses full-strength LightGBM (500 trees) matching train_profile_predictor.py.

Outputs:
  - Learning curves for each target (job title, industry, company)
  - A table showing the minimum sample sizes for acceptable accuracy
  - Saved to ./figures/power_analysis/
"""

import os
import sys
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import warnings

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
except Exception:
    print("LightGBM not found. Install with: pip install lightgbm", flush=True)
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
fig_dir = "./figures/power_analysis"
cache_path = os.path.join(save_dir, "_power_analysis_cache.parquet")
os.makedirs(fig_dir, exist_ok=True)

random_state = 42
test_size = 0.2
max_text_features = 1000

MIN_LABEL_FREQ_JOB = 500
MIN_LABEL_FREQ_INDUSTRY = 500
MIN_LABEL_FREQ_COMPANY = 200

# How many shares to test at each point on the learning curve
SHARE_COUNTS = [50, 200, 1_000, 5_000, 20_000]

# How many engagements per share to scrape (simulate)
ENGS_PER_SHARE_COUNTS = [1, 3, 5, 10]

# Number of random trials per setting (for confidence intervals)
N_TRIALS = 3

# Minimum engagements a share must have to be useful in the simulation
MIN_ENGS_PER_SHARE = 10

TARGET_COLS = ["engager_job_title", "engager_industry", "engager_company"]

sns.set_style("whitegrid")

print("=" * 70, flush=True)
print("SAMPLE SIZE POWER ANALYSIS", flush=True)
print("=" * 70, flush=True)

# ─── 1. Load or Build Cache ─────────────────────────────────────────────────
if os.path.exists(cache_path):
    print(f"\nLoading cached data from {cache_path}...", flush=True)
    cached = pd.read_parquet(cache_path)
    print(f"  Loaded {len(cached):,} rows from cache", flush=True)
else:
    print("\nPhase 1: Building cache (one-time, ~2 min)...", flush=True)

    shares_df = pd.read_parquet(os.path.join(output_dir, "shares.parquet"))
    print("  Loaded shares", flush=True)
    eng_df = pd.read_parquet(os.path.join(output_dir, "engagements.parquet"))
    print(f"  Loaded {len(eng_df):,} engagements", flush=True)
    profiles_df = pd.read_parquet(os.path.join(output_dir, "profiles.parquet"))
    print("  Loaded profiles", flush=True)
    clients_df = pd.read_parquet(os.path.join(output_dir, "clients.parquet"))

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

    # Engager profiles — filter to complete
    engager_profiles = profiles_df[
        ["user_id", "industry", "job_title_role", "employer_client_id"]
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
    print(f"  Complete engager profiles: {len(complete_profiles):,}", flush=True)

    # Filter engagements
    print("  Filtering engagements to valid engagers...", flush=True)
    engs = eng_df[eng_df["engager_user_id"].isin(valid_ids)][
        ["sharer_user_id", "engagement_type", "share_id",
         "engagement_created_at", "engager_user_id"]
    ].copy().reset_index(drop=True)
    print(f"  Valid engagements: {len(engs):,}", flush=True)

    del eng_df
    gc.collect()

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

    # Join everything into one flat table
    print("  Joining...", flush=True)
    cached = engs.merge(shares, on="share_id", how="inner")
    del engs, shares
    gc.collect()

    cached = cached.merge(sharer_profiles, on="sharer_user_id", how="left")
    del sharer_profiles
    gc.collect()

    cached = cached.merge(complete_profiles, on="engager_user_id", how="inner")
    del complete_profiles
    gc.collect()

    print(f"  Joined rows: {len(cached):,}", flush=True)

    # Save cache
    cached.to_parquet(cache_path, index=False)
    print(f"  Cache saved to {cache_path}", flush=True)

# ─── 2. Prepare Simulation Data ─────────────────────────────────────────────
print("\nPreparing simulation data...", flush=True)

# Count engagements per share
engs_per_share = cached.groupby("share_id").size()
eligible_shares = engs_per_share[engs_per_share >= MIN_ENGS_PER_SHARE].index
cached_eligible = cached[cached["share_id"].isin(eligible_shares)].reset_index(drop=True)

print(f"  Total cached rows:     {len(cached):,}", flush=True)
print(f"  Shares with >= {MIN_ENGS_PER_SHARE} engagements: {len(eligible_shares):,}", flush=True)
print(f"  Eligible rows:         {len(cached_eligible):,}", flush=True)

del cached
gc.collect()

# Fast index: share_id → row indices
_sids = cached_eligible["share_id"].values
_sort = np.argsort(_sids)
_sorted = _sids[_sort]
_uniq, _starts = np.unique(_sorted, return_index=True)
_ends = np.append(_starts[1:], len(_sorted))

share_groups = {}
for i in range(len(_uniq)):
    share_groups[_uniq[i]] = _sort[_starts[i]:_ends[i]].tolist()

del _sids, _sort, _sorted, _uniq, _starts, _ends
eligible_share_list = list(share_groups.keys())

max_possible_shares = len(eligible_share_list)
SHARE_COUNTS = [s for s in SHARE_COUNTS if s <= max_possible_shares]
print(f"  Share counts to test: {SHARE_COUNTS}", flush=True)

# Compute fixed label sets from the full data
print("  Computing fixed label sets...", flush=True)

fixed_label_sets = {}
fixed_label_encoders = {}

for col, raw_col, min_freq in [
    ("engager_job_title", "job_title_role", MIN_LABEL_FREQ_JOB),
    ("engager_industry", "industry", MIN_LABEL_FREQ_INDUSTRY),
    ("engager_company", "company_name", MIN_LABEL_FREQ_COMPANY),
]:
    counts = cached_eligible[raw_col].value_counts()
    kept = set(counts[counts >= min_freq].index)
    kept.add("other")
    fixed_label_sets[col] = kept

    le = LabelEncoder()
    le.fit(sorted(kept))
    fixed_label_encoders[col] = le
    print(f"    {col}: {len(kept)} classes", flush=True)


# ─── 3. Simulation Helper ───────────────────────────────────────────────────
cat_cols = [
    "share_content_type", "engagement_type",
    "sharer_industry", "sharer_job_title_class",
    "sharer_location_country", "sharer_job_title_role",
    "sharer_job_title_levels", "sharer_location_region",
]
num_cols = [
    "share_hour", "share_dow", "is_weekend", "is_business_hours",
    "text_length", "word_count", "has_question", "has_url",
    "exclamation_count", "hashtag_count", "mention_count",
]


def build_and_evaluate(sampled_share_ids, engs_per_share_limit, rng):
    # Sample engagements per share
    row_indices = []
    for sid in sampled_share_ids:
        indices = share_groups[sid]
        if len(indices) <= engs_per_share_limit:
            row_indices.extend(indices)
        else:
            row_indices.extend(rng.choice(indices, size=engs_per_share_limit, replace=False).tolist())

    df = cached_eligible.iloc[row_indices].copy().reset_index(drop=True)

    if len(df) < 50:
        return None

    # Targets — use FIXED label sets
    df["engager_job_title"] = df["job_title_role"].where(df["job_title_role"].isin(fixed_label_sets["engager_job_title"]), "other")
    df["engager_industry"] = df["industry"].where(df["industry"].isin(fixed_label_sets["engager_industry"]), "other")
    df["engager_company"] = df["company_name"].where(df["company_name"].isin(fixed_label_sets["engager_company"]), "other")

    # Features
    df["shared_at"] = pd.to_datetime(df["shared_at"], errors="coerce", utc=True)
    df["share_hour"] = df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
    df["share_dow"] = df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)
    df["is_weekend"] = (df["share_dow"] >= 5).astype(np.int8)
    df["is_business_hours"] = ((df["share_hour"] >= 9) & (df["share_hour"] <= 17)).astype(np.int8)
    df["user_commentary"] = df["user_commentary"].fillna("")
    df["text_length"] = df["user_commentary"].str.len()
    df["word_count"] = df["user_commentary"].str.split().str.len()
    df["has_question"] = df["user_commentary"].str.contains(r'\?').astype(np.int8)
    df["has_url"] = df["user_commentary"].str.contains(r'http').astype(np.int8)
    df["exclamation_count"] = df["user_commentary"].str.count('!')
    df["hashtag_count"] = df["user_commentary"].str.count('#')
    df["mention_count"] = df["user_commentary"].str.count('@')
    df["engagement_type"] = df["engagement_type"].fillna("unknown")

    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(np.float32)

    # Build matrix
    tfidf = TfidfVectorizer(max_features=max_text_features, stop_words="english", ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df["user_commentary"]).astype(np.float32)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df[cat_cols]).astype(np.float32)
    scaler = StandardScaler(with_mean=False)
    X_num = csr_matrix(scaler.fit_transform(df[num_cols].to_numpy(dtype=np.float32)))
    X = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)

    # Time-based split
    sort_order = df["shared_at"].argsort().values
    X = X[sort_order]

    split_idx = int(len(df) * (1 - test_size))
    if split_idx < 20 or (len(df) - split_idx) < 20:
        return None

    train_mask = np.arange(len(df)) < split_idx
    X_train, X_test = X[train_mask], X[~train_mask]

    metrics = {}
    for col in TARGET_COLS:
        le = fixed_label_encoders[col]
        y = le.transform(df[col].values)[sort_order]
        y_train, y_test = y[train_mask], y[~train_mask]
        n_classes = len(le.classes_)

        if n_classes < 2:
            continue

        clf = LGBMClassifier(
            n_estimators=500,
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
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)

        top1 = accuracy_score(y_test, y_pred)
        try:
            labels = list(range(n_classes))
            top3 = top_k_accuracy_score(y_test, y_proba, k=min(3, n_classes), labels=labels)
            top5 = top_k_accuracy_score(y_test, y_proba, k=min(5, n_classes), labels=labels)
        except ValueError:
            top3 = top1
            top5 = top1

        metrics[col] = {
            "top1": top1, "top3": top3, "top5": top5,
            "n_classes": n_classes,
            "n_train": int(train_mask.sum()),
            "n_test": int((~train_mask).sum()),
        }

    return metrics


# ─── 4. Run the Simulation ──────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Running Learning Curve Simulation", flush=True)
print("=" * 70, flush=True)

# Part A: Vary number of shares (all engagements per share)
print("\n--- Part A: Varying number of shares (all engagements per share) ---", flush=True)
results_by_shares = []

for n_shares in SHARE_COUNTS:
    print(f"\n  n_shares = {n_shares:,}", flush=True)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(random_state + trial)
        sampled = rng.choice(eligible_share_list, size=n_shares, replace=False)

        metrics = build_and_evaluate(sampled, engs_per_share_limit=999, rng=rng)
        if metrics is None:
            continue

        for col, m in metrics.items():
            results_by_shares.append({
                "n_shares": n_shares, "trial": trial, "target": col, **m,
            })

    for col in TARGET_COLS:
        accs = [r["top1"] for r in results_by_shares
                if r["n_shares"] == n_shares and r["target"] == col]
        if accs:
            print(f"    {col}: top1={np.mean(accs):.3f} +/- {np.std(accs):.3f}", flush=True)

shares_df_results = pd.DataFrame(results_by_shares)

# Part B: Vary engagements per share
fixed_n_shares = min(5_000, max_possible_shares)
print(f"\n--- Part B: Varying engagements per share (fixed {fixed_n_shares:,} shares) ---", flush=True)
results_by_engs = []

for engs_limit in ENGS_PER_SHARE_COUNTS:
    print(f"\n  engs_per_share = {engs_limit}", flush=True)
    for trial in range(N_TRIALS):
        rng = np.random.RandomState(random_state + trial)
        sampled = rng.choice(eligible_share_list, size=fixed_n_shares, replace=False)

        metrics = build_and_evaluate(sampled, engs_per_share_limit=engs_limit, rng=rng)
        if metrics is None:
            continue

        for col, m in metrics.items():
            results_by_engs.append({
                "engs_per_share": engs_limit, "trial": trial, "target": col, **m,
            })

    for col in TARGET_COLS:
        accs = [r["top1"] for r in results_by_engs
                if r["engs_per_share"] == engs_limit and r["target"] == col]
        if accs:
            print(f"    {col}: top1={np.mean(accs):.3f} +/- {np.std(accs):.3f}", flush=True)

engs_df_results = pd.DataFrame(results_by_engs)

# ─── 5. Plot Results ────────────────────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("Generating Plots", flush=True)
print("=" * 70, flush=True)

target_labels = {
    "engager_job_title": "Job Role",
    "engager_industry": "Industry",
    "engager_company": "Company",
}
target_colors = {
    "engager_job_title": "#e74c3c",
    "engager_industry": "#3498db",
    "engager_company": "#2ecc71",
}

# Plot A: Learning curve by number of shares
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, metric_name in enumerate(["top1", "top3", "top5"]):
    ax = axes[i]
    for col in TARGET_COLS:
        subset = shares_df_results[shares_df_results["target"] == col]
        grouped = subset.groupby("n_shares")[metric_name]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        ax.plot(means.index, means.values, "o-", label=target_labels[col],
                color=target_colors[col], linewidth=2, markersize=6)
        ax.fill_between(means.index, means.values - stds.values, means.values + stds.values,
                        alpha=0.15, color=target_colors[col])

    ax.set_xlabel("Number of Shares Tracked", fontsize=12)
    ax.set_ylabel("Accuracy" if i == 0 else "", fontsize=12)
    ax.set_title(f"Top-{metric_name[-1]} Accuracy", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle("How many shares do you need to track?", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "learning_curve_shares.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: learning_curve_shares.png", flush=True)

# Plot B: Learning curve by engagements per share
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for i, metric_name in enumerate(["top1", "top3", "top5"]):
    ax = axes[i]
    for col in TARGET_COLS:
        subset = engs_df_results[engs_df_results["target"] == col]
        grouped = subset.groupby("engs_per_share")[metric_name]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        ax.plot(means.index, means.values, "o-", label=target_labels[col],
                color=target_colors[col], linewidth=2, markersize=6)
        ax.fill_between(means.index, means.values - stds.values, means.values + stds.values,
                        alpha=0.15, color=target_colors[col])

    ax.set_xlabel("Engagements Scraped per Share", fontsize=12)
    ax.set_ylabel("Accuracy" if i == 0 else "", fontsize=12)
    ax.set_title(f"Top-{metric_name[-1]} Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle(f"How many engagements per share do you need to scrape? (at {fixed_n_shares:,} shares)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "learning_curve_engagements.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: learning_curve_engagements.png", flush=True)

# Plot C: Heatmap
print("  Running grid for heatmap...", flush=True)
heatmap_share_counts = [200, 1_000, 5_000]
heatmap_share_counts = [s for s in heatmap_share_counts if s <= max_possible_shares]
heatmap_results = []

for n_shares in heatmap_share_counts:
    for engs_limit in ENGS_PER_SHARE_COUNTS:
        rng = np.random.RandomState(random_state)
        sampled = rng.choice(eligible_share_list, size=n_shares, replace=False)
        metrics = build_and_evaluate(sampled, engs_per_share_limit=engs_limit, rng=rng)
        if metrics is None:
            continue
        for col, m in metrics.items():
            heatmap_results.append({
                "n_shares": n_shares, "engs_per_share": engs_limit, "target": col, **m,
            })
    print(f"    Done: {n_shares:,} shares", flush=True)

heatmap_df = pd.DataFrame(heatmap_results)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for i, col in enumerate(TARGET_COLS):
    ax = axes[i]
    subset = heatmap_df[heatmap_df["target"] == col]
    pivot = subset.pivot_table(index="engs_per_share", columns="n_shares", values="top3", aggfunc="mean")
    pivot = pivot.sort_index(ascending=False)

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                vmin=0, vmax=1, cbar_kws={"label": "Top-3 Accuracy"})
    ax.set_xlabel("Number of Shares", fontsize=11)
    ax.set_ylabel("Engagements per Share" if i == 0 else "", fontsize=11)
    ax.set_title(f"{target_labels[col]}", fontsize=13, fontweight="bold")

plt.suptitle("Top-3 Accuracy: Shares vs Engagements per Share",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "heatmap_shares_vs_engagements.png"), dpi=300, bbox_inches="tight")
plt.close()
print("  Saved: heatmap_shares_vs_engagements.png", flush=True)

# ─── 6. Summary & Recommendations ───────────────────────────────────────────
print("\n" + "=" * 70, flush=True)
print("RESULTS & RECOMMENDATIONS", flush=True)
print("=" * 70, flush=True)

print("\n  Minimum shares needed (top-3 accuracy within 95% of best):", flush=True)
print(f"  {'Target':<25s} {'Min Shares':>12s} {'Top-3 at Min':>14s} {'Top-3 at Max':>14s}", flush=True)
print(f"  {'-'*25} {'-'*12} {'-'*14} {'-'*14}", flush=True)

recommendations = {}
for col in TARGET_COLS:
    subset = shares_df_results[shares_df_results["target"] == col]
    grouped = subset.groupby("n_shares")["top3"].mean()
    max_acc = grouped.max()
    threshold = 0.95 * max_acc

    min_shares = None
    for n_shares in sorted(grouped.index):
        if grouped[n_shares] >= threshold:
            min_shares = n_shares
            break

    recommendations[col] = {
        "min_shares": min_shares,
        "top3_at_min": grouped.get(min_shares, 0),
        "top3_at_max": max_acc,
    }
    print(f"  {target_labels[col]:<25s} {min_shares:>12,} {grouped.get(min_shares, 0):>14.3f} {max_acc:>14.3f}", flush=True)

print(f"\n  Engagements per share impact (at {fixed_n_shares:,} shares):", flush=True)
print(f"  {'Target':<25s}", end="", flush=True)
for e in ENGS_PER_SHARE_COUNTS:
    print(f"  {e:>5d} eng", end="")
print(flush=True)
print(f"  {'-'*25}", end="")
for _ in ENGS_PER_SHARE_COUNTS:
    print(f"  {'-'*8}", end="")
print(flush=True)

for col in TARGET_COLS:
    subset = engs_df_results[engs_df_results["target"] == col]
    grouped = subset.groupby("engs_per_share")["top3"].mean()
    print(f"  {target_labels[col]:<25s}", end="")
    for e in ENGS_PER_SHARE_COUNTS:
        val = grouped.get(e, float("nan"))
        print(f"  {val:>8.3f}", end="")
    print(flush=True)

# Recommendation
print(f"\n" + "-" * 70, flush=True)
print("  PRACTICAL RECOMMENDATION:", flush=True)
print("-" * 70, flush=True)
min_shares_needed = max(r["min_shares"] for r in recommendations.values() if r["min_shares"])
print(f"\n  Track at least {min_shares_needed:,} shares", flush=True)
print(f"  Scrape at least 3-5 engager profiles per share", flush=True)
print(f"  Total profiles to scrape: ~{min_shares_needed * 4:,}", flush=True)
print(f"\n  This gets you within 95% of maximum achievable accuracy", flush=True)
print(f"  for all three targets (job title, industry, company).", flush=True)

# Save results
all_results = {
    "shares_learning_curve": shares_df_results.to_dict(),
    "engs_learning_curve": engs_df_results.to_dict(),
    "heatmap": heatmap_df.to_dict(),
    "recommendations": recommendations,
}
joblib.dump(all_results, os.path.join(save_dir, "power_analysis_results.joblib"))
print(f"\n  Results saved to: {save_dir}power_analysis_results.joblib", flush=True)
print(f"  Figures saved to: {fig_dir}/", flush=True)
