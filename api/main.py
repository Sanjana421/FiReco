from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from recommender.ranker import recommend
from recommender.explain import generate_explanation
from ai.assistant import summarize_recommendations

app = FastAPI(title="Financial Product Recommendation API")


class RecommendationRequest(BaseModel):
    goal: str = Field(..., min_length=1, description="User goal text")
    preferred_category: Optional[str] = Field(default="all", description="e.g. cashback, travel, shopping")
    max_annual_fee: Optional[float] = Field(default=None, ge=0)
    top_k: int = Field(default=5, ge=1, le=10)
    use_ai_summary: bool = True


@app.get("/")
def home():
    return {"message": "Financial Recommendation API is running"}


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
    for _, row in results.iterrows():
        reason = generate_explanation(
            row=row,
            goal_text=payload.goal,
            preferred_category=payload.preferred_category,
            max_annual_fee=payload.max_annual_fee,
        )

        recommendations.append(
            {
                "product_name": row.get("product_name"),
                "reward_type": row.get("reward_type"),
                "speciality": row.get("speciality"),
                "points_accumulation_rate": row.get("points_accumulation_rate"),
                "primary_usage_category": row.get("primary_usage_category"),
                "bank_name": row.get("bank_name"),
                "annual_fee": float(row.get("annual_fee", 0) or 0),
                "score": float(row.get("score", 0) or 0),
                "reason": reason,
            }
        )

    ai_summary = summarize_recommendations(payload.goal, recommendations) if payload.use_ai_summary else ""

    return {
        "query": payload.model_dump(),
        "recommendations": recommendations,
        "ai_summary": ai_summary,
    }