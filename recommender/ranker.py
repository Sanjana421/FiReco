from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
DB_FILE = ROOT / "database" / "finance.db"
CSV_FILE = ROOT / "data" / "processed" / "products_clean.csv"

model = None
df = None
product_embeddings = None


def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def load_data() -> pd.DataFrame:
    if DB_FILE.exists():
        conn = sqlite3.connect(DB_FILE)
        data = pd.read_sql_query("SELECT * FROM financial_products", conn)
        conn.close()
        return data.fillna("")

    if CSV_FILE.exists():
        return pd.read_csv(CSV_FILE).fillna("")

    raise FileNotFoundError(
        "No processed data found. Run pipeline/preprocess.py first."
    )


def build_dataframe() -> pd.DataFrame:
    data = load_data()

    if "annual_fee" not in data.columns:
        if "annual_charges_in_inr" in data.columns:
            data["annual_fee"] = pd.to_numeric(
                data["annual_charges_in_inr"],
                errors="coerce",
            ).fillna(0)
        else:
            data["annual_fee"] = 0.0

    if "combined_text" not in data.columns:
        description = data["description"] if "description" in data.columns else ""
        tags = data["tags"] if "tags" in data.columns else ""
        reward_type = data["reward_type"] if "reward_type" in data.columns else ""
        speciality = data["speciality"] if "speciality" in data.columns else ""
        primary_usage_category = (
            data["primary_usage_category"] if "primary_usage_category" in data.columns else ""
        )
        bank_name = data["bank_name"] if "bank_name" in data.columns else ""

        data["combined_text"] = (
            description.astype(str) + " " +
            tags.astype(str) + " " +
            reward_type.astype(str) + " " +
            speciality.astype(str) + " " +
            primary_usage_category.astype(str) + " " +
            bank_name.astype(str)
        )

    return data


def get_embeddings():
    global df, product_embeddings

    if df is None:
        df = build_dataframe()

    if product_embeddings is None:
        texts = df["combined_text"].astype(str).tolist()
        product_embeddings = get_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return product_embeddings


def recommend(
    goal_text,
    preferred_category="all",
    max_annual_fee=None,
    top_k=5,
):
    global df

    if df is None:
        df = build_dataframe()

    embeddings = get_embeddings()
    query = (goal_text or "").strip()

    if preferred_category and preferred_category.lower() != "all":
        query += f" {preferred_category}"

    query_embedding = get_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    similarity_scores = embeddings @ query_embedding

    results = df.copy()
    results["score"] = similarity_scores

    if preferred_category and preferred_category.lower() != "all" and "primary_usage_category" in results.columns:
        category_match = (
            results["primary_usage_category"]
            .astype(str)
            .str.lower()
            == preferred_category.lower()
        )
        results.loc[category_match, "score"] += 0.15

    if max_annual_fee is not None:
        fee_match = (
            pd.to_numeric(results["annual_fee"], errors="coerce")
            .fillna(0)
            <= float(max_annual_fee)
        )
        results.loc[fee_match, "score"] += 0.10

    results = (
        results.sort_values("score", ascending=False)
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