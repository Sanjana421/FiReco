from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "database" / "finance.db"
CSV_FILE = ROOT / "data" / "processed" / "products_clean.csv"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_data() -> pd.DataFrame:
    if DB_FILE.exists():
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT * FROM financial_products", conn)
        conn.close()
        return df.fillna("")

    if CSV_FILE.exists():
        return pd.read_csv(CSV_FILE).fillna("")

    raise FileNotFoundError(
        "No processed data found. Run pipeline/preprocess.py first."
    )


df = load_data()

if "annual_fee" not in df.columns:
    if "annual_charges_in_inr" in df.columns:
        df["annual_fee"] = pd.to_numeric(
            df["annual_charges_in_inr"],
            errors="coerce"
        ).fillna(0)
    else:
        df["annual_fee"] = 0.0


if "combined_text" not in df.columns:
    df["combined_text"] = (
        df.get("description", "").astype(str) + " " +
        df.get("tags", "").astype(str) + " " +
        df.get("reward_type", "").astype(str) + " " +
        df.get("speciality", "").astype(str) + " " +
        df.get("primary_usage_category", "").astype(str) + " " +
        df.get("bank_name", "").astype(str)
    )


texts = df["combined_text"].astype(str).tolist()

product_embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)


def recommend(
    goal_text,
    preferred_category="all",
    max_annual_fee=None,
    top_k=5
):
    query = (goal_text or "").strip()

    if preferred_category.lower() != "all":
        query += f" {preferred_category}"

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarity_scores = product_embeddings @ query_embedding

    results = df.copy()
    results["score"] = similarity_scores

    # category boost
    if preferred_category.lower() != "all":
        category_match = (
            results["primary_usage_category"]
            .astype(str)
            .str.lower()
            == preferred_category.lower()
        )

        results.loc[category_match, "score"] += 0.15

    # annual fee boost
    if max_annual_fee is not None:
        fee_match = (
            pd.to_numeric(
                results["annual_fee"],
                errors="coerce"
            ).fillna(0)
            <= float(max_annual_fee)
        )

        results.loc[fee_match, "score"] += 0.10

    results = (
        results
        .sort_values("score", ascending=False)
        .head(int(top_k))
    )

    keep_cols = [
        "product_name",
        "reward_type",
        "speciality",
        "points_accumulation_rate",
        "primary_usage_category",
        "bank_name",
        "annual_fee",
        "description",
        "tags",
        "score",
    ]

    keep_cols = [c for c in keep_cols if c in results.columns]

    return results[keep_cols].reset_index(drop=True)