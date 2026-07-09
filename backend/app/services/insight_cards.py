import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_priority_score(priority: str) -> int:
    priority = str(priority).title()

    if priority == "High":
        return 95

    if priority == "Medium":
        return 75

    if priority == "Low":
        return 55

    return 70


def normalize_priority(value: str | None) -> str:
    value = str(value or "Medium").title()

    if value in ["High", "Medium", "Low"]:
        return value

    return "Medium"


def rewrite_insight_cards_with_ai(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not GEMINI_API_KEY or not cards:
        return cards

    prompt = f"""
You are PulseIQ, an executive business analyst.

Rewrite the insight cards below so they sound polished, clear, and executive-friendly.

Return ONLY valid JSON as a list of cards.

Rules:
- Do not invent new facts.
- Do not change the meaning.
- Do not change type, priority, score, category, or icon.
- Only improve title, headline, and description.
- Keep each description concise.
- Use professional business language.
- Keep the cards useful for executives and managers.

Cards:
{json.dumps(cards, default=str)}
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        content = response.text or ""
        content = content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "", 1).replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        start = content.find("[")
        end = content.rfind("]")

        if start == -1 or end == -1:
            return cards

        rewritten_cards = json.loads(content[start:end + 1])

        if not isinstance(rewritten_cards, list):
            return cards

        return validate_rewritten_cards(original_cards=cards, rewritten_cards=rewritten_cards)

    except Exception as e:
        print(f"Insight card AI rewrite error: {e}")
        return cards


def validate_rewritten_cards(
    original_cards: list[dict[str, Any]],
    rewritten_cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    validated_cards = []

    for index, original_card in enumerate(original_cards):
        if index >= len(rewritten_cards):
            validated_cards.append(original_card)
            continue

        rewritten_card = rewritten_cards[index]

        if not isinstance(rewritten_card, dict):
            validated_cards.append(original_card)
            continue

        validated_cards.append({
            "type": original_card.get("type"),
            "icon": original_card.get("icon"),
            "priority": original_card.get("priority"),
            "score": original_card.get("score"),
            "category": original_card.get("category"),
            "title": rewritten_card.get("title") or original_card.get("title"),
            "headline": rewritten_card.get("headline") or original_card.get("headline"),
            "description": rewritten_card.get("description") or original_card.get("description"),
        })

    return validated_cards


def generate_insight_cards(
    business_health: dict,
    business_drivers: dict,
    business_risks: dict,
    business_opportunities: dict,
) -> list[dict[str, Any]]:
    cards = []

    # -----------------------------
    # Business Health Card
    # -----------------------------
    health_priority = normalize_priority(business_health.get("confidence", "Medium"))

    health_description = (
        business_health.get("strengths", ["No summary available."])[0]
        if business_health.get("strengths")
        else "No summary available."
    )

    cards.append({
        "type": "health",
        "title": "Business Health",
        "icon": "activity",
        "priority": health_priority,
        "score": get_priority_score(health_priority),
        "category": "Health",
        "headline": business_health.get("overall_health", "Unknown"),
        "description": health_description,
    })

    # -----------------------------
    # Driver Cards
    # -----------------------------
    for driver in business_drivers.get("major_drivers", [])[:3]:
        priority = normalize_priority(driver.get("strength", "Medium"))

        cards.append({
            "type": "driver",
            "title": "Key Business Driver",
            "icon": "trending-up",
            "priority": priority,
            "score": get_priority_score(priority),
            "category": "Driver",
            "headline": driver.get("driver"),
            "description": driver.get("business_meaning") or driver.get("evidence"),
        })

    # -----------------------------
    # Risk Cards
    # -----------------------------
    for risk in business_risks.get("business_risks", [])[:3]:
        priority = normalize_priority(risk.get("severity", "Medium"))

        cards.append({
            "type": "risk",
            "title": "Business Risk",
            "icon": "alert-triangle",
            "priority": priority,
            "score": get_priority_score(priority),
            "category": "Risk",
            "headline": risk.get("risk"),
            "description": risk.get("evidence"),
        })

    # -----------------------------
    # Opportunity Cards
    # -----------------------------
    for opportunity in business_opportunities.get("business_opportunities", [])[:3]:
        priority = normalize_priority(opportunity.get("potential_impact", "Medium"))

        cards.append({
            "type": "opportunity",
            "title": "Growth Opportunity",
            "icon": "sparkles",
            "priority": priority,
            "score": get_priority_score(priority),
            "category": "Opportunity",
            "headline": opportunity.get("opportunity"),
            "description": opportunity.get("evidence"),
        })

    # -----------------------------
    # Remove invalid / duplicate cards
    # -----------------------------
    unique_cards = []
    seen = set()

    for card in cards:
        headline = card.get("headline")
        description = card.get("description")

        if not headline:
            continue

        card_key = f"{card.get('type')}::{headline}::{description}"

        if card_key in seen:
            continue

        seen.add(card_key)
        unique_cards.append(card)

    # -----------------------------
    # Sort by importance
    # -----------------------------
    priority_order = {
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }

    type_order = {
        "risk": 5,
        "driver": 4,
        "opportunity": 3,
        "health": 2,
    }

    unique_cards = sorted(
        unique_cards,
        key=lambda x: (
            priority_order.get(x.get("priority", "Medium"), 2),
            type_order.get(x.get("type"), 0),
            x.get("score", 0),
        ),
        reverse=True,
    )

    return rewrite_insight_cards_with_ai(unique_cards[:8])