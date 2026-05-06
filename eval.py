"""
eval.py  –  Evaluation harness for the recommendation system
Run:  python eval.py

Outputs a results table + saves eval_results.json

Metrics explained (for interviews):
  Hit@k   – did the correct category appear in the top-k results?
  MRR     – Mean Reciprocal Rank; rewards finding the right answer early
  NDCG@k  – Normalised Discounted Cumulative Gain; standard IR metric
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── import the ranker (works whether you use TF-IDF or the new embeddings) ─
import sys
sys.path.insert(0, str(Path(__file__).parent))
from recommender.ranker import recommend

# ── golden test set ────────────────────────────────────────────────────────
# Each entry: a natural-language query + the category/reward_type we expect
# to appear in the top-3 results.
# These were written to reflect real user language, NOT the column names.
GOLDEN_QUERIES = [
    # (query_text, expected_category, expected_reward_type, max_fee)
    ("I want a card that gives me money back on groceries",          "groceries",     "cashback",          None),
    ("Best card for frequent flyers and airline miles",              "travel",        "travel_points",     None),
    ("I eat out a lot and want dining rewards",                      "dining",        "dining_rewards",    None),
    ("Card for petrol and fuel expenses",                            "fuel",          "fuel_rewards",      None),
    ("Premium card with lounge access and luxury perks",             "luxury",        "premium_points",    None),
    ("I shop online a lot, want cashback on e-commerce",             "online_spends", "cashback",          None),
    ("Business expenses card with high reward points",               "business",      "premium_points",    None),
    ("Weekend shopping card with good rewards",                      "shopping",      "shopping_rewards",  None),
    ("Zero annual fee card for everyday spending",                   None,            "cashback",          0),
    ("Travel card under 2000 rupees annual fee",                     "travel",        None,                2000),
    ("Dining card with discounts at restaurants",                    "dining",        "dining_rewards",    None),
    ("I want maximum cashback on all my purchases",                  None,            "cashback",          None),
    ("Card for international travel with no forex markup",           "travel",        "travel_points",     None),
    ("Best fuel surcharge waiver credit card",                       "fuel",          "fuel_rewards",      None),
    ("Card with grocery supermarket rewards",                        "groceries",     "cashback",          None),
    ("Premium lifestyle card with golf and concierge",               "luxury",        "premium_points",    None),
    ("Card for online shopping and OTT subscriptions",               "online_spends", "shopping_rewards",  None),
    ("I want rewards for eating out at restaurants",                 "dining",        "dining_rewards",    None),
    ("Business travel card with high points on flights",             "travel",        "travel_points",     None),
    ("Affordable card with good rewards, fee under 1000 rupees",     None,            None,                1000),
]

TOP_K = 5   # evaluate at top-5


def dcg_at_k(hits: list[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    hits = hits[:k]
    return sum(h / np.log2(i + 2) for i, h in enumerate(hits))


def ndcg_at_k(hits: list[int], k: int) -> float:
    """Normalised DCG at k (ideal = 1.0 in position 0)."""
    ideal = dcg_at_k(sorted(hits, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(hits, k) / ideal


def evaluate_query(
    query: str,
    expected_category: str | None,
    expected_reward: str | None,
    max_fee: float | None,
    k: int = TOP_K,
) -> dict:
    results = recommend(
        goal_text=query,
        preferred_category=expected_category or "all",
        max_annual_fee=max_fee,
        top_k=k,
    )

    cats    = results["primary_usage_category"].astype(str).str.lower().tolist()
    rewards = results["reward_type"].astype(str).str.lower().tolist()

    # binary relevance: 1 if category OR reward_type matches expectation
    hits = []
    for cat, rew in zip(cats, rewards):
        match = False
        if expected_category and expected_category.lower() in cat:
            match = True
        if expected_reward and expected_reward.lower() in rew:
            match = True
        hits.append(1 if match else 0)

    hit_at_3 = int(any(hits[:3]))
    hit_at_k = int(any(hits[:k]))

    # MRR: 1/rank of first hit
    rr = 0.0
    for rank, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / rank
            break

    ndcg = ndcg_at_k(hits, k)

    return {
        "query":    query,
        "hit@3":    hit_at_3,
        f"hit@{k}": hit_at_k,
        "mrr":      round(rr, 4),
        f"ndcg@{k}": round(ndcg, 4),
        "top_results": results["product_name"].tolist()[:3],
    }


def run_eval() -> None:
    print(f"\n{'='*70}")
    print(f"  FiReco Evaluation  |  {len(GOLDEN_QUERIES)} queries  |  top-{TOP_K}")
    print(f"{'='*70}\n")

    rows = []
    for query, cat, rew, fee in GOLDEN_QUERIES:
        result = evaluate_query(query, cat, rew, fee)
        rows.append(result)
        status = "✓" if result["hit@3"] else "✗"
        print(f"  {status}  {query[:55]:<55}  MRR={result['mrr']:.2f}  NDCG={result[f'ndcg@{TOP_K}']:.2f}")

    df = pd.DataFrame(rows)
    hit3   = df["hit@3"].mean()
    hit5   = df[f"hit@{TOP_K}"].mean()
    mrr    = df["mrr"].mean()
    ndcg   = df[f"ndcg@{TOP_K}"].mean()

    print(f"\n{'─'*70}")
    print(f"  Hit@3      : {hit3:.1%}")
    print(f"  Hit@{TOP_K}      : {hit5:.1%}")
    print(f"  Mean MRR   : {mrr:.4f}")
    print(f"  Mean NDCG@{TOP_K}: {ndcg:.4f}")
    print(f"{'─'*70}\n")

    summary = {
        "model":     "sentence-transformers/all-MiniLM-L6-v2",
        "n_queries": len(GOLDEN_QUERIES),
        "top_k":     TOP_K,
        "hit@3":     round(hit3, 4),
        f"hit@{TOP_K}":  round(hit5, 4),
        "mean_mrr":  round(mrr, 4),
        f"mean_ndcg@{TOP_K}": round(ndcg, 4),
        "per_query": rows,
    }

    out = Path(__file__).parent / "eval_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  Results saved to: {out}\n")


if __name__ == "__main__":
    run_eval()