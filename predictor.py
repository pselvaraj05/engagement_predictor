"""
Simple "total engagements per post" predictor, structured similarly to your LightFM script.

Key design points (per your ask):
- Reuses existing text embeddings if available (from a parquet column or a saved .npy/.npz)
- If embeddings are not available, it falls back to TF-IDF (you can swap in your own embedding function)
- Uses the same shares/engagements/profiles/user_client_map inputs you already load
- Incorporates additional features from sharer and (optional) engager distributions if you want later

Output:
- Trains a regression model to predict total engagements for a share_id
- Saves model + preprocessors + feature config to outputs/
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings

# If you have lightgbm installed, this is a strong baseline for count prediction.
# If not installed, swap for sklearn's HistGradientBoostingRegressor or GradientBoostingRegressor.
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor


# ─── 0. Params ──────────────────────────────────────────────────────────────
output_dir = "./data/"
save_dir = "./outputs/"
os.makedirs(save_dir, exist_ok=True)

# Embedding reuse options:
# 1) If shares.parquet contains a column with per-post embeddings (list/np array), set this:
EMBED_COL_IN_SHARES = "comment_embedding"  # <- change to your actual column name or set to None

# 2) Or, if you save embeddings separately keyed by share_id:
#    expected format: npy with shape (n_items, d) aligned to item_ids order, and a joblib with item_ids
EMBED_NPY_PATH = os.path.join(save_dir, "item_text_embeddings.npy")       # optional
EMBED_META_PATH = os.path.join(save_dir, "item_text_embeddings_meta.joblib")  # optional; should include {"item_ids": [...]}

# TF-IDF fallback
max_text_features = 5000

# Model params
test_size = 0.2
random_state = 42

# For engagement counts, Poisson/Tweedie objectives often work well.
# We'll use LGBM with objective="poisson" if available.
lgbm_params = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=255,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=random_state,
)

# ─── 1. Load data (same as your script) ─────────────────────────────────────
shares_df = pd.read_parquet(os.path.join(output_dir, "shares.parquet"))
eng_df = pd.read_parquet(os.path.join(output_dir, "engagements.parquet"))
profiles_df = pd.read_parquet(os.path.join(output_dir, "profiles.parquet"))

# Align / rename as in your script
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
    ["sharer_client_id", "sharer_user_id", "engagement_type", "share_id", "engagement_created_at", "engager_user_id"]
].copy()
profiles = profiles_df[
    ["user_id", "industry", "job_title_role", "job_title_levels", "location_region", "location_country", "job_title_class"]
].copy()

profiles = profiles.drop_duplicates(subset=["user_id"]).reset_index(drop=True)


# ─── 2. Build training target: total engagements per share ──────────────────
# Note: This is "total engagements observed in engs" per share_id.
# If you need a time-based train/test split later, add time windows.
y_df = engs.groupby("share_id").size().reset_index(name="total_engagements")

# Merge to shares and add sharer attributes
# sharer_user_id lives in engs; take the (sharer_user_id, sharer_client_id) for each share
share_owner = (
    engs.sort_values("engagement_created_at")
        .groupby("share_id")[["sharer_user_id", "sharer_client_id"]]
        .first()
        .reset_index()
)

train_df = (
    shares.merge(y_df, on="share_id", how="left")
          .merge(share_owner, on="share_id", how="left")
)

train_df["total_engagements"] = train_df["total_engagements"].fillna(0).astype(np.int32)
train_df["target"] = np.log1p(train_df["total_engagements"].astype(np.float32))

# Sharer profile features
sharer_profiles = profiles.rename(columns={"user_id": "sharer_user_id"})
train_df = train_df.merge(
    sharer_profiles,
    on="sharer_user_id",
    how="left",
)

# Time features
train_df["shared_at"] = pd.to_datetime(train_df["shared_at"], errors="coerce", utc=True)
train_df["share_hour"] = train_df["shared_at"].dt.hour.fillna(-1).astype(np.int16)
train_df["share_dow"] = train_df["shared_at"].dt.dayofweek.fillna(-1).astype(np.int16)

# Basic text
train_df["user_commentary"] = train_df["user_commentary"].fillna("")

# Fill missing cats
cat_cols = [
    "share_content_type",
    "industry",
    "job_title_class",
    "location_country",
    "job_title_role",
    "job_title_levels",
    "location_region",
]
for c in cat_cols:
    train_df[c] = train_df[c].fillna("unknown").astype(str)

num_cols = ["share_hour", "share_dow"]


# ─── 3. Embeddings: reuse if available; else compute TF-IDF ─────────────────
def _load_embeddings_from_shares_column(df: pd.DataFrame, col: str):
    if col is None:
        return None
    if col not in df.columns:
        return None

    # Expect each row df[col] is a list/np array of fixed length
    # Convert to 2D float32 array aligned to df row order
    try:
        emb = np.vstack(df[col].apply(lambda x: np.array(x, dtype=np.float32)).to_numpy())
        return emb
    except Exception as e:
        warnings.warn(f"Found column '{col}' but failed to stack embeddings: {e}")
        return None


def _load_embeddings_from_files(item_ids: np.ndarray):
    if not (os.path.exists(EMBED_NPY_PATH) and os.path.exists(EMBED_META_PATH)):
        return None

    meta = joblib.load(EMBED_META_PATH)
    if "item_ids" not in meta:
        warnings.warn(f"{EMBED_META_PATH} missing 'item_ids'; cannot align embeddings.")
        return None

    saved_item_ids = np.array(meta["item_ids"])
    emb = np.load(EMBED_NPY_PATH)

    # Align: build index mapping from saved_item_ids -> row
    idx_map = {sid: i for i, sid in enumerate(saved_item_ids)}
    rows = []
    missing = 0
    for sid in item_ids:
        if sid in idx_map:
            rows.append(emb[idx_map[sid]])
        else:
            missing += 1
            rows.append(np.zeros((emb.shape[1],), dtype=np.float32))
    if missing > 0:
        warnings.warn(f"Embeddings missing for {missing}/{len(item_ids)} items; filled with zeros.")
    return np.vstack(rows).astype(np.float32)


# Align everything at share-level
item_ids = train_df["share_id"].to_numpy()

# Try: 1) embeddings column in shares.parquet (merged into train_df via shares)
item_emb = _load_embeddings_from_shares_column(train_df, EMBED_COL_IN_SHARES)

# Try: 2) embeddings from files keyed by share_id
if item_emb is None:
    item_emb = _load_embeddings_from_files(item_ids)

# Else: fallback to TF-IDF
use_tfidf = item_emb is None
tfidf = None
item_text_sparse = None

if use_tfidf:
    tfidf = TfidfVectorizer(max_features=max_text_features, stop_words="english", ngram_range=(1, 2))
    item_text_sparse = tfidf.fit_transform(train_df["user_commentary"]).astype(np.float32)
    print("ℹ️ No embeddings found; using TF-IDF.")
else:
    print(f"✅ Using existing embeddings with shape: {item_emb.shape}")


# ─── 4. Build feature matrix (sparse + optional dense embeddings) ───────────
# Categorical one-hot
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_cat = ohe.fit_transform(train_df[cat_cols]).astype(np.float32)

# Numeric
X_num = csr_matrix(train_df[num_cols].to_numpy(dtype=np.float32))

# Text features
if use_tfidf:
    X_text = item_text_sparse
else:
    # Wrap dense embeddings as sparse for hstack compatibility
    X_text = csr_matrix(item_emb)

# Combine all
X = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)
# y = train_df["total_engagements"].to_numpy(dtype=np.float32)
y = train_df["target"].to_numpy(dtype=np.float32)

print(f"Feature matrix: {X.shape}, target shape: {y.shape}")


# ─── 5. Train/test split and train model ───────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

if HAS_LGBM:
    # Poisson objective expects non-negative targets; engagement counts fit well
    # model = LGBMRegressor(objective="poisson", **lgbm_params)
    model = LGBMRegressor(objective="regression", **lgbm_params)
else:
    # Fallback (not Poisson-specific). Works, but less ideal for count data.
    model = HistGradientBoostingRegressor(random_state=random_state)

model.fit(X_train, y_train)

pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
pred = np.clip(pred, 0, None)

mae = mean_absolute_error(np.expm1(y_test), pred)
print(f"✅ Test MAE: {mae:.3f}")


# ─── 6. Save model & preprocessors (similar pattern to your script) ─────────
artifact = {
    "model": model,
    "ohe": ohe,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "use_tfidf": use_tfidf,
    "tfidf": tfidf,  # None if using embeddings
    "embed_col_in_shares": EMBED_COL_IN_SHARES if not use_tfidf else None,
    "embed_files": {
        "npy_path": EMBED_NPY_PATH if os.path.exists(EMBED_NPY_PATH) else None,
        "meta_path": EMBED_META_PATH if os.path.exists(EMBED_META_PATH) else None,
    },
}

joblib.dump(artifact, os.path.join(save_dir, "engagement_volume_model.joblib"))
print("✅ Saved: outputs/engagement_volume_model.joblib")


# ─── 6.5 Save predictions + metadata for plotting in a separate file ───────
results_df = pd.DataFrame({
    "actual_engagements": np.expm1(y_test),
    "predicted_engagements": pred
})
results_df.to_parquet(os.path.join(save_dir, "engagement_volume_predictions.parquet"), index=False)

joblib.dump({
    "mae": float(mae),
    "test_size": test_size,
    "random_state": random_state,
    "has_lgbm": HAS_LGBM,
    "use_tfidf": use_tfidf,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "embed_col_in_shares": EMBED_COL_IN_SHARES if not use_tfidf else None,
    "embed_files": {
        "npy_path": EMBED_NPY_PATH if os.path.exists(EMBED_NPY_PATH) else None,
        "meta_path": EMBED_META_PATH if os.path.exists(EMBED_META_PATH) else None,
    },
}, os.path.join(save_dir, "engagement_volume_predictions_meta.joblib"))

print("✅ Saved: outputs/engagement_volume_predictions.parquet")
print("✅ Saved: outputs/engagement_volume_predictions_meta.joblib")


# ─── 7. Inference helper (single post row -> predicted total engagements) ───
def predict_total_engagements(model_artifact, post_row: dict) -> float:
    """
    post_row should contain:
      - share_content_type
      - user_commentary
      - share_hour, share_dow
      - industry, job_title_role, job_title_levels, location_region, location_country, job_title_class
    Plus either:
      - embeddings available in post_row[EMBED_COL_IN_SHARES], OR
      - you accept TF-IDF fallback (if model trained with tfidf)
    """
    model = model_artifact["model"]
    ohe = model_artifact["ohe"]
    tfidf = model_artifact["tfidf"]
    use_tfidf = model_artifact["use_tfidf"]
    cat_cols = model_artifact["cat_cols"]
    num_cols = model_artifact["num_cols"]

    df1 = pd.DataFrame([post_row]).copy()
    for c in cat_cols:
        df1[c] = df1.get(c, "unknown")
        df1[c] = df1[c].fillna("unknown").astype(str)
    for c in num_cols:
        df1[c] = df1.get(c, -1)
        df1[c] = df1[c].fillna(-1).astype(np.float32)

    X_cat = ohe.transform(df1[cat_cols]).astype(np.float32)
    X_num = csr_matrix(df1[num_cols].to_numpy(dtype=np.float32))

    if use_tfidf:
        df1["user_commentary"] = df1.get("user_commentary", "")
        X_text = tfidf.transform(df1["user_commentary"].fillna("")).astype(np.float32)
    else:
        emb_col = model_artifact["embed_col_in_shares"]
        if emb_col and emb_col in df1.columns and df1[emb_col].notna().all():
            emb = np.vstack(df1[emb_col].apply(lambda x: np.array(x, dtype=np.float32)).to_numpy())
        else:
            # If the trained model used embeddings but you didn't provide them at inference time,
            # we must fail loudly (or you can implement your own embedder here).
            raise ValueError(
                "Model was trained with embeddings, but no embeddings were provided at inference time. "
                "Provide post_row[embedding_col] or implement an embedding fallback."
            )
        X_text = csr_matrix(emb)

    X1 = hstack([X_cat, X_num, X_text], format="csr").astype(np.float32)
    pred_log = float(model.predict(X1)[0])
    pred = np.expm1(pred_log)
    return max(pred, 0.0)


# Example inference:
# art = joblib.load("./outputs/engagement_volume_model.joblib")
# example = {
#     "share_content_type": "link",
#     "user_commentary": "Excited to share our latest launch...",
#     "share_hour": 10,
#     "share_dow": 2,
#     "industry": "software",
#     "job_title_role": "marketing",
#     "job_title_levels": "director",
#     "location_region": "north_america",
#     "location_country": "US",
#     "job_title_class": "marketing",
#     # If you trained with embeddings:
#     # "comment_embedding": np.random.randn(384).astype(np.float32),
# }
# print("Predicted engagements:", predict_total_engagements(art, example))
