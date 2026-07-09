from typing import Any


def generate_insight_cards(
    business_health: dict,
    business_drivers: dict,
    business_risks: dict,
    business_opportunities: dict,
) -> list[dict[str, Any]]:

    cards = []

    # -----------------------------
    # Business Health
    # -----------------------------
    cards.append({
        "type": "health",
        "title": "Business Health",
        "icon": "activity",
        "priority": business_health.get("confidence", "Medium"),
        "headline": business_health.get("overall_health", "Unknown"),
        "description": (
            business_health.get("strengths", ["No summary available."])[0]
            if business_health.get("strengths")
            else "No summary available."
        ),
    })

    # -----------------------------
    # Key Drivers
    # -----------------------------
    for driver in business_drivers.get("major_drivers", [])[:2]:
        cards.append({
            "type": "driver",
            "title": "Key Business Driver",
            "icon": "trending-up",
            "priority": driver.get("strength", "Medium").title(),
            "headline": driver.get("driver"),
            "description": driver.get("business_meaning"),
        })

    # -----------------------------
    # Risks
    # -----------------------------
    for risk in business_risks.get("business_risks", [])[:2]:
        cards.append({
            "type": "risk",
            "title": "Business Risk",
            "icon": "alert-triangle",
            "priority": risk.get("severity", "Medium").title(),
            "headline": risk.get("risk"),
            "description": risk.get("evidence"),
        })

    # -----------------------------
    # Opportunities
    # -----------------------------
    for opportunity in business_opportunities.get(
        "business_opportunities", []
    )[:2]:
        cards.append({
            "type": "opportunity",
            "title": "Growth Opportunity",
            "icon": "sparkles",
            "priority": opportunity.get(
                "potential_impact",
                "Medium",
            ).title(),
            "headline": opportunity.get("opportunity"),
            "description": opportunity.get("evidence"),
        })

    # -----------------------------
    # Remove duplicate headlines
    # -----------------------------
    seen = set()
    unique_cards = []

    for card in cards:
        headline = card.get("headline")

        if headline in seen:
            continue

        seen.add(headline)
        unique_cards.append(card)
    
    priority_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1
    }

    type_order = {
        "risk": 5,
        "driver": 4,
        "opportunity": 3,
        "health": 2,
        "action": 1
    }

    unique_cards = sorted(
        unique_cards,
        key=lambda x: (
            priority_order.get(x.get("priority", "Medium"), 2),
            type_order.get(x.get("type"), 0),
        ),
        reverse=True,
    )

    return unique_cards[:8]