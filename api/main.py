from typing import Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from recommender.ranker import recommend
from recommender.explain import generate_explanation
from ai.assistant import summarize_recommendations

app = FastAPI(title="FiReco AI Assistant")


class RecommendationRequest(BaseModel):
    goal: str = Field(..., min_length=1)
    preferred_category: str = "all"
    max_annual_fee: Optional[float] = None
    top_k: int = 5
    use_ai_summary: bool = True
    clarifications: Optional[Dict[str, str]] = None


@app.get("/")
def home():
    return {"message": "FiReco AI Assistant API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def get_recommendation(payload: RecommendationRequest):

    results = recommend(
        goal_text=payload.goal,
        preferred_category=payload.preferred_category,
        max_annual_fee=payload.max_annual_fee,
        top_k=payload.top_k,
    )

    recommendations = []

    for idx, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        reason = generate_explanation(
            row=row,
            goal_text=payload.goal,
            preferred_category=payload.preferred_category,
            max_annual_fee=payload.max_annual_fee,
        )

        recommendations.append(
            {
                "rank": idx,
                "product_name": row.get("product_name"),
                "reward_type": row.get("reward_type"),
                "speciality": row.get("speciality"),
                "points_accumulation_rate": row.get(
                    "points_accumulation_rate"
                ),
                "primary_usage_category": row.get(
                    "primary_usage_category"
                ),
                "bank_name": row.get("bank_name"),
                "annual_fee": float(
                    row.get("annual_fee", 0) or 0
                ),
                "score": float(
                    row.get("score", 0) or 0
                ),
                "reason": reason,
            }
        )

    ai_summary = ""

    if payload.use_ai_summary:
        ai_summary = summarize_recommendations(
            payload.goal,
            recommendations
        )

    return {
        "recommendations": recommendations,
        "ai_summary": ai_summary,
    }