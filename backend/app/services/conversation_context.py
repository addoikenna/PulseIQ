from typing import Any


QUESTION_INTENTS = {
    "risk": [
        "risk",
        "issue",
        "problem",
        "concern",
        "challenge",
        "danger",
        "weakness",
    ],
    "opportunity": [
        "opportunity",
        "growth",
        "improve",
        "increase",
        "expand",
        "optimize",
    ],
    "driver": [
        "driver",
        "factor",
        "cause",
        "reason",
        "because",
        "why",
    ],
    "dashboard": [
        "dashboard",
        "chart",
        "graph",
        "visual",
        "kpi",
        "metric",
    ],
    "quality": [
        "quality",
        "missing",
        "duplicate",
        "clean",
        "reliable",
        "trust",
    ],
    "statistics": [
        "distribution",
        "outlier",
        "correlation",
        "average",
        "median",
        "variance",
        "statistics",
    ],
    "summary": [
        "summary",
        "overview",
        "overall",
        "business",
        "performance",
    ],
}

def detect_question_intent(question: str) -> str:

    question = question.lower()

    scores = {}

    for intent, keywords in QUESTION_INTENTS.items():

        score = 0

        for keyword in keywords:
            if keyword in question:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    if scores[best_intent] == 0:
        return "general"

    return best_intent


def build_conversation_context(
    analysis: dict[str, Any],
    question: str,
) -> dict[str, Any]:

    intent = detect_question_intent(question)

    context = {
        "intent": intent,
        "filename": analysis.get("filename"),
        "overview": analysis.get("overview"),
    }

    if intent == "risk":

        context["business_health"] = analysis.get("business_health")
        context["business_risks"] = analysis.get("business_risks")
        context["executive_analysis"] = analysis.get("report", {}).get(
            "executive_analysis"
        )

    elif intent == "opportunity":

        context["business_opportunities"] = analysis.get(
            "business_opportunities"
        )
        context["business_drivers"] = analysis.get(
            "business_drivers"
        )

        context["executive_analysis"] = analysis.get("report", {}).get(
            "executive_analysis"
        )

    elif intent == "driver":

        context["business_drivers"] = analysis.get(
            "business_drivers"
        )

        context["business_metrics"] = analysis.get(
            "business_metrics"
        )

    elif intent == "dashboard":

        context["dashboard"] = analysis.get("dashboard")

        context["insight_cards"] = analysis.get(
            "insight_cards"
        )

    elif intent == "quality":

        context["data_quality_summary"] = analysis.get(
            "data_quality_summary"
        )

        context["business_health"] = analysis.get(
            "business_health"
        )

    elif intent == "statistics":

        context["statistical_profile"] = analysis.get(
            "statistical_profile"
        )

        context["business_metrics"] = analysis.get(
            "business_metrics"
        )

    else:

        context["business_health"] = analysis.get(
            "business_health"
        )

        context["business_drivers"] = analysis.get(
            "business_drivers"
        )

        context["business_risks"] = analysis.get(
            "business_risks"
        )

        context["business_opportunities"] = analysis.get(
            "business_opportunities"
        )

        context["insight_cards"] = analysis.get(
            "insight_cards"
        )

        context["executive_analysis"] = analysis.get(
            "report",
            {},
        ).get("executive_analysis")

    return context

