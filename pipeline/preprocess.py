from pathlib import Path
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "products_raw.csv"
OUT_FILE = ROOT / "data" / "processed" / "products_clean.csv"
DB_FILE = ROOT / "database" / "finance.db"


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_description(row: pd.Series) -> str:
    parts = [
        safe_str(row.get("credit_card_name")),
        safe_str(row.get("reward_type")),
        safe_str(row.get("speciality")),
        safe_str(row.get("primary_usage_category")),
        safe_str(row.get("bank_name")),
        safe_str(row.get("points_accumulation_rate")),
        f"annual fee {safe_str(row.get('annual_charges_in_inr'))}",
    ]
    return " ".join([p for p in parts if p])


def generate_tags(row: pd.Series) -> str:
    text = build_description(row).lower()
    tags = set()

    keyword_map = {
        "cashback": ["cashback"],
        "travel": ["travel", "airline", "miles"],
        "shopping": ["shopping", "online", "retail"],
        "fuel": ["fuel", "petrol", "diesel"],
        "student": ["student", "starter", "beginner"],
        "premium": ["premium", "lounge", "gold", "platinum", "signature"],
        "discounts": ["discount", "coupon", "movie", "dining"],
        "points": ["points"],
        "business": ["business"],
    }

    for tag, keywords in keyword_map.items():
        if any(k in text for k in keywords):
            tags.add(tag)

    try:
        fee = float(row.get("annual_charges_in_inr", 0) or 0)
        if fee == 0:
            tags.add("low_fee")
        elif fee <= 1000:
            tags.add("moderate_fee")
        else:
            tags.add("premium_fee")
    except Exception:
        pass

    return " ".join(sorted(tags))


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)
    df.columns = df.columns.str.strip().str.lower()
    df = df.fillna("")

    # Standardize columns
    df["product_name"] = df["credit_card_name"].astype(str).str.strip()
    df["product_type"] = "credit_card"
    df["annual_fee"] = pd.to_numeric(df["annual_charges_in_inr"], errors="coerce").fillna(0).astype(float)

    # Build text fields
    df["description"] = df.apply(build_description, axis=1)
    df["tags"] = df.apply(generate_tags, axis=1)
    df["combined_text"] = (
        df["description"].astype(str) + " " +
        df["tags"].astype(str) + " " +
        df["reward_type"].astype(str) + " " +
        df["speciality"].astype(str) + " " +
        df["primary_usage_category"].astype(str) + " " +
        df["bank_name"].astype(str)
    )

    # Save processed CSV
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    # Save to SQLite
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("financial_products", conn, if_exists="replace", index=False)
    conn.close()

    print(f"Processed data saved to: {OUT_FILE}")
    print(f"Saved data to SQLite database: {DB_FILE}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()