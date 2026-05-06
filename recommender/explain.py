def generate_explanation(
    row,
    goal_text,
    preferred_category=None,
    max_annual_fee=None
):
    reasons = []

    goal = (goal_text or "").lower()

    desc = str(row.get("description", "")).lower()
    tags = str(row.get("tags", "")).lower()
    category = str(
        row.get("primary_usage_category", "")
    ).lower()

    if "cashback" in goal and (
        "cashback" in desc or "cashback" in tags
    ):
        reasons.append(
            "excellent cashback benefits"
        )

    if "travel" in goal and (
        "travel" in desc or "travel" in tags
    ):
        reasons.append(
            "strong travel rewards and benefits"
        )

    if "shopping" in goal and (
        "shopping" in desc or "shopping" in tags
    ):
        reasons.append(
            "great for shopping purchases"
        )

    if "student" in goal and (
        "student" in desc or "student" in tags
    ):
        reasons.append(
            "student-friendly features"
        )

    if "business" in goal and (
        "business" in desc or
        "business" in tags or
        "business" in category
    ):
        reasons.append(
            "well suited for business spending"
        )

    try:
        fee = float(row.get("annual_fee", 0) or 0)

        if fee == 0:
            reasons.append("no annual fee")

        elif (
            max_annual_fee is not None and
            fee <= float(max_annual_fee)
        ):
            reasons.append(
                f"fits your budget with ₹{fee:.0f} annual fee"
            )

    except Exception:
        pass

    if not reasons:
        reasons.append(
            "strong semantic match for your preferences"
        )

    return ", ".join(dict.fromkeys(reasons))