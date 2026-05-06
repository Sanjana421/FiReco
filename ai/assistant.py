"""
assistant.py  –  LLM reasoning layer using OpenAI GPT
Upgraded from: GPT summary-only wrapper
Upgraded to:   Structured JSON output with typed recommendations

Key improvements for interviews:
  - GPT returns typed JSON (Pydantic-validated), not free text
  - The LLM now reasons about WHY a card fits, not just describes it
  - Falls back gracefully when no API key is present
"""

import json
import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

# ── Pydantic schema for LLM output ────────────────────────────────────────
class CardInsight(BaseModel):
    product_name: str
    primary_reason: str           = Field(description="One sentence: why this card fits the user goal")
    best_for:       str           = Field(description="The single strongest use case")
    watch_out_for:  str           = Field(description="One tradeoff or caveat to be aware of")
    fee_verdict:    str           = Field(description="Is the annual fee worth it for this user? Brief.")

class RecommendationSummary(BaseModel):
    headline:    str              = Field(description="One punchy sentence summarising the top pick")
    cards:       list[CardInsight]
    overall_tip: str              = Field(description="One actionable tip for the user based on their goal")


# ── client ────────────────────────────────────────────────────────────────
_api_key = os.getenv("OPENAI_API_KEY")
_client  = OpenAI(api_key=_api_key) if _api_key else None

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # fast + cheap; set OPENAI_MODEL=gpt-4o for prod


# ── fallback (no API key) ──────────────────────────────────────────────────
def _fallback(goal_text: str, recommendations: list[dict[str, Any]]) -> dict:
    if not recommendations:
        return {"headline": "No recommendations found.", "cards": [], "overall_tip": ""}
    top = recommendations[0]
    return {
        "headline": f"Top pick: {top.get('product_name')} — {top.get('reason', 'strong match')}.",
        "cards": [
            {
                "product_name":  r.get("product_name"),
                "primary_reason": r.get("reason", ""),
                "best_for":       r.get("primary_usage_category", ""),
                "watch_out_for":  f"Annual fee ₹{r.get('annual_fee', 0):.0f}",
                "fee_verdict":    "Check if fee is waived on minimum spend.",
            }
            for r in recommendations[:3]
        ],
        "overall_tip": "Compare annual fees against your expected reward earnings.",
    }


# ── main function ──────────────────────────────────────────────────────────
def summarize_recommendations(
    goal_text: str,
    recommendations: list[dict[str, Any]],
) -> dict:
    """
    Ask Claude to reason about the recommendations and return structured JSON.
    Returns a dict matching RecommendationSummary schema.
    Falls back gracefully if Anthropic key is absent.
    """
    if _client is None or not recommendations:
        return _fallback(goal_text, recommendations)

    compact = "\n".join(
        f"{i}. {r['product_name']} | category={r.get('primary_usage_category')} "
        f"| reward={r.get('reward_type')} | fee=₹{r.get('annual_fee', 0):.0f} "
        f"| score={r.get('score', 0):.3f}"
        for i, r in enumerate(recommendations[:5], 1)
    )

    prompt = f"""You are a financial product advisor helping an Indian user choose a credit card.

User goal: {goal_text}

Top candidates from the recommendation engine:
{compact}

Return ONLY valid JSON matching this schema exactly — no markdown, no preamble:
{{
  "headline": "<one punchy sentence about the best pick>",
  "cards": [
    {{
      "product_name": "<name>",
      "primary_reason": "<why it fits the user goal in one sentence>",
      "best_for": "<single strongest use case>",
      "watch_out_for": "<one tradeoff or caveat>",
      "fee_verdict": "<is the annual fee worth it for this user?>"
    }}
  ],
  "overall_tip": "<one actionable tip for this user>"
}}

Include insights for the top 3 cards only."""

    try:
        message = _client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.choices[0].message.content.strip()

        # strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        parsed = RecommendationSummary(**json.loads(raw))
        return parsed.model_dump()

    except Exception as e:
        print(f"[assistant] LLM call failed ({e}), using fallback")
        return _fallback(goal_text, recommendations)