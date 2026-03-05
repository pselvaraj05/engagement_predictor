"""
Predict engagement tier + audience distribution for a single share.

Usage:
  1. Edit the `share` dict below with your share's metadata.
  2. Run:  .venv/bin/python predict_share.py
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack

# ─── INPUT: Edit this dict ─────────────────────────────────────────────────
share = {
    "share_content_type": "link",
    "user_commentary": "Excited to share our Q4 results...",
    "shared_at": "2025-06-15 10:30:00",
    # Sharer profile
    "sharer_industry": "Computer Software",
    "sharer_job_title_role": "Marketing",
    "sharer_job_title_class": "professional",
    "sharer_job_title_levels": "Senior",
    "sharer_location_country": "United States",
    "sharer_location_region": "North America",
    # Optional (set 0 if unknown)
    "sharer_avg_engagement_per_share": 0,
    "sharer_total_shares": 0,
    "sharer_viral_rate": 0,
}

# ─── Load model artifacts ──────────────────────────────────────────────────
eng_artifact = joblib.load("outputs/hierarchical_engagement_model.joblib")
dist_artifact = joblib.load("outputs/distribution_model.joblib")

# ─── Helper: build derived features from the share dict ────────────────────
def _build_row(share_dict):
    """Return a single-row DataFrame with all derived features."""
    ts = pd.to_datetime(share_dict["shared_at"], utc=True)
    commentary = share_dict.get("user_commentary", "") or ""
    words = commentary.split()

    return {
        "share_content_type": share_dict.get("share_content_type", "unknown"),
        # Sharer profile fields (with prefix)
        "sharer_industry": share_dict.get("sharer_industry", "unknown"),
        "sharer_job_title_role": share_dict.get("sharer_job_title_role", "unknown"),
        "sharer_job_title_class": share_dict.get("sharer_job_title_class", "unknown"),
        "sharer_job_title_levels": share_dict.get("sharer_job_title_levels", "unknown"),
        "sharer_location_country": share_dict.get("sharer_location_country", "unknown"),
        "sharer_location_region": share_dict.get("sharer_location_region", "unknown"),
        # Same fields without prefix (for engagement model)
        "industry": share_dict.get("sharer_industry", "unknown"),
        "job_title_role": share_dict.get("sharer_job_title_role", "unknown"),
        "job_title_class": share_dict.get("sharer_job_title_class", "unknown"),
        "job_title_levels": share_dict.get("sharer_job_title_levels", "unknown"),
        "location_country": share_dict.get("sharer_location_country", "unknown"),
        "location_region": share_dict.get("sharer_location_region", "unknown"),
        # Temporal
        "share_hour": float(ts.hour),
        "share_dow": float(ts.dayofweek),
        "is_weekend": float(ts.dayofweek >= 5),
        "is_business_hours": float(9 <= ts.hour <= 17),
        # Content signals
        "text_length": float(len(commentary)),
        "word_count": float(len(words)),
        "has_question": float("?" in commentary),
        "has_url": float("http" in commentary),
        "exclamation_count": float(commentary.count("!")),
        "hashtag_count": float(commentary.count("#")),
        "mention_count": float(commentary.count("@")),
        # Sharer stats
        "sharer_avg_engagement_per_share": float(share_dict.get("sharer_avg_engagement_per_share", 0)),
        "sharer_total_shares": float(share_dict.get("sharer_total_shares", 0)),
        "sharer_viral_rate": float(share_dict.get("sharer_viral_rate", 0)),
        # Raw text
        "user_commentary": commentary,
    }


def _build_feature_vector(row, artifact):
    """Build a sparse feature matrix from a row dict using the artifact's transformers."""
    cat_cols = artifact["cat_cols"]
    num_cols = artifact["num_cols"]

    df = pd.DataFrame([row])
    for c in cat_cols:
        df[c] = df[c].fillna("unknown").astype(str)
    for c in num_cols:
        df[c] = df[c].fillna(0).astype(np.float32)

    X_cat = artifact["ohe"].transform(df[cat_cols]).astype(np.float32)
    X_num = csr_matrix(artifact["scaler"].transform(df[num_cols].to_numpy(dtype=np.float32)))
    X_text = artifact["tfidf"].transform(df["user_commentary"]).astype(np.float32)

    return hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)


# ─── Build feature vectors ─────────────────────────────────────────────────
row = _build_row(share)
X_eng = _build_feature_vector(row, eng_artifact)
X_dist = _build_feature_vector(row, dist_artifact)

# ─── Engagement prediction (hierarchical) ──────────────────────────────────
tier_names = eng_artifact["tier_names"]        # ['Low (0-5)', 'Medium (6-50)', ...]
tier_boundaries = eng_artifact["tier_boundaries"]  # {'low': (0,5), ...}

pop_proba = eng_artifact["popularity_classifier"].predict_proba(X_eng)[0, 1]
is_popular = pop_proba >= 0.5

if is_popular:
    tier = eng_artifact["high_viral_classifier"].predict(X_eng)[0]
else:
    tier = eng_artifact["low_medium_classifier"].predict(X_eng)[0]

tier_label = tier_names[tier]

# Count estimate from tier-specific regressor
tier_regressors = eng_artifact["tier_regressors"]
if tier in tier_regressors:
    raw_pred = tier_regressors[tier].predict(X_eng)[0]
    tier_key = list(tier_boundaries.keys())[tier]
    lo, hi = tier_boundaries[tier_key]
    if tier == 3:  # viral
        count_est = int(np.clip(raw_pred, 0, 50000))
    else:
        count_est = int(np.clip(raw_pred, 0, hi * 3))
else:
    # Fallback: midpoint of tier range
    tier_key = list(tier_boundaries.keys())[tier]
    lo, hi = tier_boundaries[tier_key]
    count_est = int((lo + min(hi, 1000)) / 2)

# ─── Distribution prediction ───────────────────────────────────────────────
dim_display = {
    "job_title": "Job Title",
    "industry": "Industry",
    "company": "Company",
    "job_level": "Job Level",
    "country": "Country",
}

dist_results = {}
for dim in dist_artifact["target_dimensions"]:
    clf = dist_artifact["models"][dim]
    le = dist_artifact["label_encoders"][dim]
    proba = clf.predict_proba(X_dist)[0]

    # Sort descending and take top 5
    top_idx = np.argsort(proba)[::-1][:5]
    labels = le.inverse_transform(top_idx)
    pcts = proba[top_idx] * 100
    dist_results[dim] = list(zip(labels, pcts))

# ─── Print results ──────────────────────────────────────────────────────────
print()
print("ENGAGEMENT PREDICTION")
print(f"  Tier:            {tier_label}")
print(f"  Estimated Count: ~{count_est} engagements")
print(f"  Popular Prob:    {pop_proba:.1%}")

print()
print("AUDIENCE DISTRIBUTION")
for dim in dist_artifact["target_dimensions"]:
    name = dim_display.get(dim, dim)
    parts = [f"{pct:.1f}% {lbl}" for lbl, pct in dist_results[dim]]
    print(f"  {name + ':':14s}{' | '.join(parts)}")
print()
