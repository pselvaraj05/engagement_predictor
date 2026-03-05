"""
Engager Profile Prediction Model
=================================
Predicts the profile of the person who will engage with a given share.

Instead of predicting HOW MANY engagements a post will receive,
this model predicts WHO will engage — specifically:

  1. Job Title (job_title_role) — e.g., engineering, sales, marketing
  2. Industry — e.g., financial services, computer software
  3. Company (employer_client_id → client name) — e.g., adobe, ibm, workday

Approach:
  - Training data is at the engagement level (one row per share–engager pair)
  - Share features (content type, text, temporal, sharer profile) are inputs
  - Engager profile attributes are the classification targets
  - Separate LightGBM classifiers are trained per target

"""

import os
import gc
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

try:
    from lightgbm import LGBMClassifier
except Exception:
    print("LightGBM not found. Install with: pip install lightgbm")
    exit(1)

# ─── 0. Configuration ───────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
os.makedirs(save_dir, exist_ok=True)

SAMPLE_SIZE = 0                 # 0 = use full dataset; set e.g. 100_000 for a quick test
max_text_features = 2000
test_size = 0.2
random_state = 42
N_ESTIMATORS = 500

# Minimum frequency for a label to be kept (otherwise → "other")
MIN_LABEL_FREQ_JOB = 500
MIN_LABEL_FREQ_INDUSTRY = 500
MIN_LABEL_FREQ_COMPANY = 200

TARGET_COLS = ["engager_job_title", "engager_industry", "engager_company"]

print("=" * 70)
print("ENGAGER PROFILE PREDICTION MODEL")
print("=" * 70)
print(f"\n  sample_size:      {SAMPLE_SIZE if SAMPLE_SIZE > 0 else 'full dataset'}")
print(f"  max_text_features: {max_text_features}")
print(f"  test_size:         {test_size}")
print(f"  n_estimators:      {N_ESTIMATORS}")
print(f"  min_freq (job/ind/co): {MIN_LABEL_FREQ_JOB}/{MIN_LABEL_FREQ_INDUSTRY}/{MIN_LABEL_FREQ_COMPANY}")

# ─── 1. Load Data ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 1: Loading Data")
print("=" * 70)

shares_df = pd.read_parquet(os.path.join(output_dir, "shares.parquet"))
eng_df = pd.read_parquet(os.path.join(output_dir, "engagements.parquet"))
profiles_df = pd.read_parquet(os.path.join(output_dir, "profiles.parquet"))
clients_df = pd.read_parquet(os.path.join(output_dir, "clients.parquet"))

total_engagements = len(eng_df)
total_profiles = len(profiles_df)

print(f"  Shares:      {len(shares_df):>12,}")
print(f"  Engagements: {total_engagements:>12,}")
print(f"  Profiles:    {total_profiles:>12,}")
print(f"  Clients:     {len(clients_df):>12,}")

# Build employer_client_id → company name mapping
client_name_map = dict(zip(clients_df["client_id"], clients_df["title"]))
del clients_df

# Map profile_id → es_user_id
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

# ─── 2. Pre-Filter: Identify Engagers with Complete Profiles ────────────────
print("\n" + "=" * 70)
print("STAGE 2: Pre-Filtering Profiles (before joins)")
print("=" * 70)

# Build engager profiles with targets
engager_profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "employer_client_id"]
].copy()
engager_profiles = engager_profiles.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
engager_profiles["company_name"] = engager_profiles["employer_client_id"].map(client_name_map)
engager_profiles.rename(columns={"user_id": "engager_user_id"}, inplace=True)

total_engager_profiles = len(engager_profiles)

# Per-field coverage in the full profiles table
has_job_prof = engager_profiles["job_title_role"].notna() & (engager_profiles["job_title_role"] != "")
has_ind_prof = engager_profiles["industry"].notna() & (engager_profiles["industry"] != "")
has_co_prof = engager_profiles["company_name"].notna() & (engager_profiles["company_name"] != "")

print(f"\n  Total unique engager profiles: {total_engager_profiles:,}")
print(f"\n  Per-field coverage in profiles table:")
print(f"    job_title_role:  {has_job_prof.sum():>10,} / {total_engager_profiles:,}  ({has_job_prof.mean()*100:5.1f}%)")
print(f"    industry:        {has_ind_prof.sum():>10,} / {total_engager_profiles:,}  ({has_ind_prof.mean()*100:5.1f}%)")
print(f"    company_name:    {has_co_prof.sum():>10,} / {total_engager_profiles:,}  ({has_co_prof.mean()*100:5.1f}%)")

# Keep only profiles with ALL three fields known
complete_mask = has_job_prof & has_ind_prof & has_co_prof
complete_profiles = engager_profiles[complete_mask].reset_index(drop=True)
n_complete = len(complete_profiles)
n_incomplete = total_engager_profiles - n_complete

print(f"\n  Profiles with ALL three targets known:")
print(f"    Complete:   {n_complete:>10,} ({n_complete / total_engager_profiles * 100:5.1f}%)")
print(f"    Incomplete: {n_incomplete:>10,} ({n_incomplete / total_engager_profiles * 100:5.1f}%)")

# Get the set of valid engager user_ids
valid_engager_ids = set(complete_profiles["engager_user_id"].dropna())
print(f"    Valid engager IDs: {len(valid_engager_ids):,}")

# ─── 3. Filter Engagements to Valid Engagers ────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 3: Filtering Engagements")
print("=" * 70)

engs = eng_df[
    ["sharer_client_id", "sharer_user_id", "engagement_type", "share_id",
     "engagement_created_at", "engager_user_id"]
]

# Filter to only engagements from users with complete profiles
print(f"\n  Total engagements: {total_engagements:,}")
valid_mask = engs["engager_user_id"].isin(valid_engager_ids)
n_valid_engs = valid_mask.sum()
n_dropped_engs = total_engagements - n_valid_engs

print(f"  With complete engager profile: {n_valid_engs:>12,} ({n_valid_engs / total_engagements * 100:5.1f}%)")
print(f"  Dropped (no complete profile): {n_dropped_engs:>12,} ({n_dropped_engs / total_engagements * 100:5.1f}%)")

engs = engs[valid_mask].copy().reset_index(drop=True)

# Free the full engagements dataframe
del eng_df, valid_mask
gc.collect()

# Optional sampling AFTER filtering
if SAMPLE_SIZE > 0 and len(engs) > SAMPLE_SIZE:
    print(f"\n  Sampling {SAMPLE_SIZE:,} from {len(engs):,} valid engagements...")
    np.random.seed(random_state)
    sample_idx = np.random.choice(len(engs), size=SAMPLE_SIZE, replace=False)
    engs = engs.iloc[sample_idx].reset_index(drop=True)

print(f"  Engagements for training: {len(engs):,}")

# ─── 4. Build Training DataFrame ────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 4: Building Training Data")
print("=" * 70)

shares = shares_df[["share_id", "shared_at", "share_content_type", "user_commentary"]].copy()
del shares_df
gc.collect()

# Sharer profiles (features)
sharer_profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "job_title_levels",
     "location_region", "location_country", "job_title_class"]
].copy()
sharer_profiles = sharer_profiles.drop_duplicates(subset=["user_id"]).reset_index(drop=True)
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

print("  Joining engagements with share features...")
train_df = engs.merge(shares, on="share_id", how="inner")
del engs, shares
gc.collect()

print("  Joining sharer profiles...")
train_df = train_df.merge(sharer_profiles, on="sharer_user_id", how="left")
del sharer_profiles
gc.collect()

print("  Joining engager profiles (targets)...")
train_df = train_df.merge(complete_profiles, on="engager_user_id", how="inner")
del complete_profiles
gc.collect()

rows_after_join = len(train_df)
print(f"\n  Rows after all joins: {rows_after_join:,}")

# ─── 5. Define Targets & Collapse Rare Labels ───────────────────────────────
print("\n" + "=" * 70)
print("STAGE 5: Preparing Target Labels")
print("=" * 70)

train_df["engager_job_title"] = train_df["job_title_role"]
train_df["engager_industry"] = train_df["industry"]
train_df["engager_company"] = train_df["company_name"]

print(f"\n  Rows with all targets known: {len(train_df):,}")
print(f"  (All rows are complete — filtering happened in Stage 2)")


def collapse_rare_labels(series, min_freq, label="other"):
    counts = series.value_counts()
    rare = counts[counts < min_freq].index
    collapsed = series.where(~series.isin(rare), label)
    return collapsed, len(rare), counts


n_before = {}
raw_distributions = {}
for col in TARGET_COLS:
    n_before[col] = train_df[col].nunique()

train_df["engager_job_title"], n_rare_job, dist_job = collapse_rare_labels(
    train_df["engager_job_title"], MIN_LABEL_FREQ_JOB
)
train_df["engager_industry"], n_rare_ind, dist_ind = collapse_rare_labels(
    train_df["engager_industry"], MIN_LABEL_FREQ_INDUSTRY
)
train_df["engager_company"], n_rare_comp, dist_comp = collapse_rare_labels(
    train_df["engager_company"], MIN_LABEL_FREQ_COMPANY
)

print(f"\n  Label summary after collapsing rare classes:")
for col, n_rare, dist in [
    ("engager_job_title", n_rare_job, dist_job),
    ("engager_industry", n_rare_ind, dist_ind),
    ("engager_company", n_rare_comp, dist_comp),
]:
    n_classes = train_df[col].nunique()
    print(f"\n    {col}: {n_before[col]} raw -> {n_classes} classes ({n_rare} rare collapsed)")
    top10 = train_df[col].value_counts().head(10)
    for label, count in top10.items():
        pct = count / len(train_df) * 100
        print(f"      {label:35s} {count:>8,} ({pct:5.1f}%)")

print(f"\n  Final training data: {len(train_df):,} rows")

# ─── 6. Feature Engineering ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 6: Feature Engineering")
print("=" * 70)

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

# Engagement type as feature
train_df["engagement_type"] = train_df["engagement_type"].fillna("unknown")

cat_cols = [
    "share_content_type", "engagement_type",
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

# ─── 7. Text Features & Feature Matrix ──────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 7: Building Feature Matrix")
print("=" * 70)

print(f"  Fitting TF-IDF (max_features={max_text_features})...")
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

# ─── 8. Encode Targets ──────────────────────────────────────────────────────
label_encoders = {}
y_encoded = {}

for col in TARGET_COLS:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(train_df[col].values)
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# ─── 9. Time-Based Train/Test Split ─────────────────────────────────────────
print("\n" + "=" * 70)
print("STAGE 8: Train/Test Split")
print("=" * 70)

sort_order = train_df["shared_at"].argsort().values
train_df = train_df.iloc[sort_order].reset_index(drop=True)
X = X[sort_order]
for col in TARGET_COLS:
    y_encoded[col] = y_encoded[col][sort_order]

split_idx = int(len(train_df) * (1 - test_size))
split_date = train_df.iloc[split_idx]["shared_at"]

train_mask = np.arange(len(train_df)) < split_idx
X_train, X_test = X[train_mask], X[~train_mask]

print(f"  Train: {X_train.shape[0]:,} samples (up to {split_date})")
print(f"  Test:  {X_test.shape[0]:,} samples (from {split_date})")

# ─── 10. Train Classifiers ──────────────────────────────────────────────────
classifiers = {}
results = {}

for col in TARGET_COLS:
    y_train = y_encoded[col][train_mask]
    y_test_col = y_encoded[col][~train_mask]
    n_classes = len(label_encoders[col].classes_)

    print("\n" + "=" * 70)
    print(f"TRAINING: {col} ({n_classes} classes)")
    print("=" * 70)

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
    classifiers[col] = clf

    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test_col, y_pred)
    top3_acc = top_k_accuracy_score(y_test_col, y_proba, k=min(3, n_classes))
    top5_acc = top_k_accuracy_score(y_test_col, y_proba, k=min(5, n_classes))

    print(f"\n  Accuracy (top-1): {acc:.3f}")
    print(f"  Top-3 Accuracy:   {top3_acc:.3f}")
    print(f"  Top-5 Accuracy:   {top5_acc:.3f}")

    # Classification report — top 15 classes by support
    class_names = label_encoders[col].classes_
    print(f"\n  Classification Report (top 15 classes by support):")
    report = classification_report(
        y_test_col, y_pred,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    if "support" in report_df.columns:
        top_classes = (
            report_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
            .sort_values("support", ascending=False)
            .head(15)
        )
        print(top_classes.to_string())

    results[col] = {
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
        "y_test": y_test_col,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }

# ─── 11. Example Predictions ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Example Predictions (5 random test samples)")
print("=" * 70)

np.random.seed(random_state)
example_indices = np.random.choice(X_test.shape[0], size=min(5, X_test.shape[0]), replace=False)

for i, idx in enumerate(example_indices):
    print(f"\n--- Sample {i+1} ---")
    for col in TARGET_COLS:
        le = label_encoders[col]
        proba = results[col]["y_proba"][idx]
        top3_idx = np.argsort(proba)[-3:][::-1]
        actual_label = le.classes_[results[col]["y_test"][idx]]

        print(f"  {col}:")
        print(f"    Actual: {actual_label}")
        print(f"    Predicted top 3:")
        for rank, class_idx in enumerate(top3_idx, 1):
            print(f"      {rank}. {le.classes_[class_idx]} ({proba[class_idx]:.1%})")

# ─── 12. Save Everything ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Saving Models and Results")
print("=" * 70)

artifact = {
    "classifiers": classifiers,
    "label_encoders": label_encoders,
    "ohe": ohe,
    "scaler": scaler,
    "tfidf": tfidf,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "target_cols": TARGET_COLS,
    "min_label_freq": {
        "job": MIN_LABEL_FREQ_JOB,
        "industry": MIN_LABEL_FREQ_INDUSTRY,
        "company": MIN_LABEL_FREQ_COMPANY,
    },
}

joblib.dump(artifact, os.path.join(save_dir, "profile_prediction_model.joblib"))
print("  Saved: profile_prediction_model.joblib")

# Save predictions
pred_rows = {
    "share_id": train_df[~train_mask]["share_id"].values,
    "shared_at": train_df[~train_mask]["shared_at"].values,
}
for col in TARGET_COLS:
    le = label_encoders[col]
    pred_rows[f"{col}_actual"] = le.inverse_transform(results[col]["y_test"])
    pred_rows[f"{col}_predicted"] = le.inverse_transform(results[col]["y_pred"])

    proba = results[col]["y_proba"]
    for rank in range(min(3, proba.shape[1])):
        top_k_indices = np.argsort(proba, axis=1)[:, -(rank + 1)]
        pred_rows[f"{col}_top{rank+1}"] = le.inverse_transform(top_k_indices)
        pred_rows[f"{col}_top{rank+1}_prob"] = proba[
            np.arange(len(proba)), top_k_indices
        ]

pred_df = pd.DataFrame(pred_rows)
pred_df.to_parquet(
    os.path.join(save_dir, "profile_predictions.parquet"), index=False
)
print("  Saved: profile_predictions.parquet")

# Save metadata
metadata = {}
for col in TARGET_COLS:
    metadata[col] = {
        "accuracy": float(results[col]["accuracy"]),
        "top3_accuracy": float(results[col]["top3_accuracy"]),
        "top5_accuracy": float(results[col]["top5_accuracy"]),
        "n_classes": len(label_encoders[col].classes_),
        "classes": list(label_encoders[col].classes_),
    }
metadata["split_date"] = str(split_date)
metadata["n_train"] = int(train_mask.sum())
metadata["n_test"] = int((~train_mask).sum())
metadata["config"] = {
    "sample_size": SAMPLE_SIZE,
    "max_text_features": max_text_features,
    "test_size": test_size,
    "n_estimators": N_ESTIMATORS,
    "min_label_freq_job": MIN_LABEL_FREQ_JOB,
    "min_label_freq_industry": MIN_LABEL_FREQ_INDUSTRY,
    "min_label_freq_company": MIN_LABEL_FREQ_COMPANY,
}

joblib.dump(metadata, os.path.join(save_dir, "profile_predictions_meta.joblib"))
print("  Saved: profile_predictions_meta.joblib")

# ─── 13. Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("\n  Data Pipeline:")
print(f"    Total engagements:          {total_engagements:>12,}")
print(f"    With complete profile:      {n_valid_engs:>12,} ({n_valid_engs/total_engagements*100:.1f}%)")
print(f"    After joins:                {rows_after_join:>12,}")
print(f"    Train / Test:               {X_train.shape[0]:>12,} / {X_test.shape[0]:,}")

print("\n  Results:")
print(f"    {'Target':<25s} {'Top-1':>7s} {'Top-3':>7s} {'Top-5':>7s} {'Classes':>8s}")
print(f"    {'-'*25} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")
for col in TARGET_COLS:
    r = results[col]
    nc = len(label_encoders[col].classes_)
    print(f"    {col:<25s} {r['accuracy']:>7.3f} {r['top3_accuracy']:>7.3f} {r['top5_accuracy']:>7.3f} {nc:>8d}")
