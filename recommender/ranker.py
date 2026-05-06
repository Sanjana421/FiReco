"""
ranker.py  –  Semantic retrieval using sentence-transformers + FAISS
Replaces the old TF-IDF cosine approach.

Why this is better (for interviews):
  - Understands meaning, not just keyword overlap
  - "eating out" matches "dining cashback" even with zero shared words
  - Industry-standard approach used at every major tech/finance company
"""

from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# ── paths ──────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "products_clean.csv"
INDEX_DIR = ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "products.faiss"
IDS_FILE   = INDEX_DIR / "product_ids.npy"

# ── model (downloads once, ~90 MB, cached after that) ──────────────────────
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

# ── load data ──────────────────────────────────────────────────────────────
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Processed file not found: {DATA_FILE}. "
        "Run pipeline/preprocess.py first."
    )

df = pd.read_csv(DATA_FILE).fillna("")

if "annual_fee" not in df.columns:
    df["annual_fee"] = pd.to_numeric(
        df.get("annual_charges_in_inr", 0), errors="coerce"
    ).fillna(0)

if "combined_text" not in df.columns:
    df["combined_text"] = (
        df.get("description", "").astype(str) + " "
        + df.get("tags", "").astype(str) + " "
        + df.get("reward_type", "").astype(str) + " "
        + df.get("speciality", "").astype(str) + " "
        + df.get("primary_usage_category", "").astype(str) + " "
        + df.get("bank_name", "").astype(str)
    )

# ── build or load FAISS index ─────────────────────────────────────────────
def _build_index() -> faiss.IndexFlatIP:
    """Encode all products and store in a FAISS inner-product index."""
    model = _get_model()
    texts = df["combined_text"].astype(str).tolist()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])   # inner product = cosine when normalised
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    np.save(str(IDS_FILE), np.arange(len(df)))
    return index


def _load_index() -> faiss.IndexFlatIP:
    if INDEX_FILE.exists() and IDS_FILE.exists():
        return faiss.read_index(str(INDEX_FILE))
    return _build_index()


_index = _load_index()


# ── public API ────────────────────────────────────────────────────────────
def recommend(
    goal_text: str,
    preferred_category: str | None = None,
    max_annual_fee: float | None = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Retrieve top-k products semantically closest to goal_text.

    Scoring:
      1. Cosine similarity via FAISS (embedding-based)
      2. +0.10 bonus if product category matches preferred_category
         (configurable – see CATEGORY_BOOST below)
      3. Hard filter: drop products above max_annual_fee if specified
    """
    CATEGORY_BOOST = 0.10   # single config point – easy to explain in interviews

    goal_text = (goal_text or "").strip()
    query = goal_text
    if preferred_category and preferred_category.lower() != "all":
        query += f" {preferred_category}"

    model = _get_model()
    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # retrieve more than top_k so we have room after filtering
    fetch_k = min(len(df), top_k * 4)
    similarities, indices = _index.search(query_vec, fetch_k)

    results = df.iloc[indices[0]].copy()
    results["score"] = similarities[0].astype(float)

    # category boost (transparent, documented, not a magic number)
    if preferred_category and preferred_category.lower() != "all":
        cat_col = results["primary_usage_category"].astype(str).str.lower()
        results.loc[cat_col == preferred_category.lower(), "score"] += CATEGORY_BOOST

    # fee filter
    if max_annual_fee is not None:
        results = results[results["annual_fee"] <= float(max_annual_fee)]

    results = results.sort_values("score", ascending=False).head(top_k)

    keep_cols = [
        "product_name", "reward_type", "speciality",
        "points_accumulation_rate", "primary_usage_category",
        "bank_name", "annual_fee", "description", "tags", "score",
    ]
    existing_cols = [c for c in keep_cols if c in results.columns]
    return results[existing_cols].reset_index(drop=True)