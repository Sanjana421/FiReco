def generate_explanation(row, goal_text: str, preferred_category: str | None = None, max_annual_fee: float | None = None) -> str:
    reasons = []
    goal = (goal_text or "").lower()
    desc = str(row.get("description", "")).lower()
    tags = str(row.get("tags", "")).lower()
    category = str(row.get("primary_usage_category", "")).lower()

    if "cashback" in goal and ("cashback" in desc or "cashback" in tags):
        reasons.append("matches cashback preference")

    if "travel" in goal and ("travel" in desc or "travel" in tags):
        reasons.append("matches travel preference")

    if "shopping" in goal and ("shopping" in desc or "shopping" in tags):
        reasons.append("matches shopping preference")

    if "student" in goal and ("student" in desc or "student" in tags):
        reasons.append("student-friendly option")

    if preferred_category and preferred_category.lower() != "all":
        if preferred_category.lower() in category:
            reasons.append(f"fits {preferred_category.lower()} category")

    try:
        fee = float(row.get("annual_fee", 0) or 0)
        if fee == 0:
            reasons.append("no annual fee")
        elif max_annual_fee is not None and fee <= float(max_annual_fee):
            reasons.append(f"within your annual fee budget (₹{fee:.0f})")
    except Exception:
        pass

    if not reasons:
        reasons.append("strong overall text match")

    return ", ".join(dict.fromkeys(reasons))