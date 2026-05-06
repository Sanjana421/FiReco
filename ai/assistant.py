import os
from typing import List, Dict, Any

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _extract_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text.strip()

    pieces = []
    try:
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                piece = getattr(content, "text", None)
                if piece:
                    pieces.append(piece)
    except Exception:
        pass

    return "\n".join(pieces).strip()


def fallback_summary(goal_text: str, recommendations: List[Dict[str, Any]]) -> str:
    if not recommendations:
        return "No recommendations were found for that input."

    top = recommendations[0]
    top_name = top.get("product_name", "the top product")
    top_reason = top.get("reason", "a strong match")
    return f"Top match: {top_name}. It is recommended because it {top_reason}."


def summarize_recommendations(goal_text: str, recommendations: List[Dict[str, Any]]) -> str:
    if client is None:
        return fallback_summary(goal_text, recommendations)

    try:
        compact_lines = []
        for i, rec in enumerate(recommendations[:5], start=1):
            compact_lines.append(
                f"{i}. {rec.get('product_name')} | score={rec.get('score')} | reason={rec.get('reason')}"
            )

        prompt = f"""
You are a professional financial product assistant.

User goal:
{goal_text}

Top recommendations:
{chr(10).join(compact_lines)}

Generate:
1) a short personalized summary,
2) why the top choice is best,
3) one short caution to consider.

Keep it concise, helpful, and non-technical.
"""

        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )

        text = _extract_text(response)
        return text if text else fallback_summary(goal_text, recommendations)

    except Exception:
        return fallback_summary(goal_text, recommendations)