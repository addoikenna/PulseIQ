import json
import os
from typing import Any

import requests
from dotenv import load_dotenv
from google import genai

from app.services.supabase_client import get_supabase_client


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY_MODEL = os.getenv(
    "OPENROUTER_PRIMARY_MODEL",
    "deepseek/deepseek-chat-v3.1",
)
OPENROUTER_FALLBACK_MODEL = os.getenv(
    "OPENROUTER_FALLBACK_MODEL",
    "deepseek/deepseek-v3.1-terminus",
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

ANALYSIS_PAYLOAD_COLUMN = "analysis_json"


def parse_llm_json(content: str) -> dict[str, Any]:
    if not content:
        raise ValueError("LLM returned empty content.")

    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")

    return json.loads(content[start:end + 1])


def normalize_analysis_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        parsed = json.loads(value)

        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Saved analysis does not contain valid analysis JSON.")


def get_owned_analysis(
    analysis_id: str,
    user_id: str,
) -> dict[str, Any]:
    supabase = get_supabase_client()

    response = (
        supabase.table("analyses")
        .select(f"id, user_id, filename, {ANALYSIS_PAYLOAD_COLUMN}")
        .eq("id", analysis_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        raise ValueError("Analysis not found or access denied.")

    return response.data[0]


def build_shared_context(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    dashboard = analysis.get("dashboard", {})
    report = analysis.get("report", {})

    return {
        "filename": analysis.get("filename"),
        "overview": analysis.get("overview"),
        "column_profile": analysis.get("column_profile"),
        "business_metrics": analysis.get("business_metrics"),
        "statistical_profile": analysis.get("statistical_profile"),
        "data_quality_summary": analysis.get("data_quality_summary"),
        "business_health": analysis.get("business_health"),
        "business_drivers": analysis.get("business_drivers"),
        "business_risks": analysis.get("business_risks"),
        "business_opportunities": analysis.get("business_opportunities"),
        "insight_cards": analysis.get("insight_cards"),
        "chart_summary": analysis.get("chart_summary"),
        "executive_analysis": report.get("executive_analysis"),
        "filters": dashboard.get("filters"),
    }


def find_kpi(
    analysis: dict[str, Any],
    kpi_index: int,
) -> dict[str, Any]:
    kpis = analysis.get("dashboard", {}).get("kpis", [])

    if not isinstance(kpis, list):
        raise ValueError("Saved analysis does not contain a valid KPI list.")

    if kpi_index >= len(kpis):
        raise ValueError("KPI index is outside the available KPI range.")

    kpi = kpis[kpi_index]

    if not isinstance(kpi, dict):
        raise ValueError("Selected KPI is invalid.")

    return kpi


def find_chart(
    analysis: dict[str, Any],
    chart_index: int,
) -> dict[str, Any]:
    dashboard = analysis.get("dashboard", {})

    charts = dashboard.get("chart_recommendations")

    if not isinstance(charts, list) or not charts:
        charts = dashboard.get("charts", [])

    if not isinstance(charts, list):
        raise ValueError("Saved analysis does not contain a valid chart list.")

    if chart_index >= len(charts):
        raise ValueError("Chart index is outside the available chart range.")

    chart = charts[chart_index]

    if not isinstance(chart, dict):
        raise ValueError("Selected chart is invalid.")

    return chart


def build_kpi_prompt(
    kpi: dict[str, Any],
    context: dict[str, Any],
) -> str:
    return f"""
You are PulseIQ, a grounded senior business intelligence analyst.

Explain the selected KPI using only the supplied KPI definition and saved
analysis context.

Do not invent figures, causes, comparisons, benchmarks, or trends.
Do not perform a new calculation.
Do not expose identifiers or contact details.
Translate technical details into clear business language.

Return ONLY valid JSON with exactly this structure:

{{
  "title": "...",
  "value": "...",
  "explanation": "...",
  "calculation": "...",
  "business_interpretation": "...",
  "cautions": [
    "..."
  ],
  "supporting_evidence": [
    {{
      "source": "...",
      "detail": "..."
    }}
  ],
  "suggested_question": "...",
  "confidence": "high | medium | low"
}}

Guidance:
- explanation: what the KPI measures.
- calculation: explain the aggregation and source column in plain language.
- business_interpretation: explain what the current value suggests.
- cautions: mention limitations, quality issues, or interpretation risks.
- supporting_evidence: include only evidence present in the saved context.
- If the context is insufficient, say so and use low confidence.

Selected KPI:
{json.dumps(kpi, default=str)}

Saved analysis context:
{json.dumps(context, default=str)}
"""


def build_chart_prompt(
    chart: dict[str, Any],
    context: dict[str, Any],
) -> str:
    return f"""
You are PulseIQ, a grounded senior business intelligence analyst.

Explain the selected chart using only the supplied chart definition and saved
analysis context.

Do not invent values, trends, causes, comparisons, or conclusions.
Do not claim to see data points that are not contained in the supplied context.
Do not expose identifiers or contact details.
Translate chart design and analytical meaning into clear business language.

Return ONLY valid JSON with exactly this structure:

{{
  "title": "...",
  "chart_type": "...",
  "explanation": "...",
  "chart_logic": "...",
  "business_interpretation": "...",
  "cautions": [
    "..."
  ],
  "supporting_evidence": [
    {{
      "source": "...",
      "detail": "..."
    }}
  ],
  "suggested_question": "...",
  "confidence": "high | medium | low"
}}

Guidance:
- explanation: describe what the chart is intended to show.
- chart_logic: explain the axes, metric, aggregation, and time grain.
- business_interpretation: explain the supported business meaning.
- cautions: mention filters, aggregation limitations, data quality, or weak evidence.
- supporting_evidence: include only evidence present in the saved context.
- If the context does not contain the plotted results, explain the chart purpose
  without inventing the pattern and use medium or low confidence.

Selected chart:
{json.dumps(chart, default=str)}

Saved analysis context:
{json.dumps(context, default=str)}
"""


def call_gemini(prompt: str) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return parse_llm_json(response.text or "")


def call_openrouter(
    model: str,
    prompt: str,
) -> dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not configured.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://pulseiq-dsxe.onrender.com",
        "X-Title": "PulseIQ",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a grounded business intelligence analyst. "
                    "Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=45,
    )

    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"].get("content", "")

    return parse_llm_json(content)


def generate_explanation(
    prompt: str,
) -> dict[str, Any]:
    try:
        result = call_gemini(prompt)

        return {
            **result,
            "model_used": GEMINI_MODEL,
            "provider": "gemini",
        }

    except Exception as error:
        print(f"Gemini explanation error: {error}")

    for model in [
        OPENROUTER_PRIMARY_MODEL,
        OPENROUTER_FALLBACK_MODEL,
    ]:
        try:
            result = call_openrouter(model, prompt)

            return {
                **result,
                "model_used": model,
                "provider": "openrouter",
            }

        except Exception as error:
            print(f"OpenRouter explanation error for {model}: {error}")

    raise ValueError("Unable to generate a reliable explanation.")


def normalize_confidence(value: Any) -> str:
    confidence = str(value or "low").lower()

    if confidence not in {"high", "medium", "low"}:
        return "low"

    return confidence


def normalize_evidence(
    value: Any,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    evidence = []

    for item in value[:5]:
        if not isinstance(item, dict):
            continue

        source = item.get("source")
        detail = item.get("detail")

        if source and detail:
            evidence.append({
                "source": str(source),
                "detail": str(detail),
            })

    return evidence


def explain_kpi(
    analysis_id: str,
    kpi_index: int,
    user_id: str,
) -> dict[str, Any]:
    analysis_record = get_owned_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
    )

    analysis = normalize_analysis_payload(
        analysis_record.get(ANALYSIS_PAYLOAD_COLUMN)
    )

    kpi = find_kpi(
        analysis=analysis,
        kpi_index=kpi_index,
    )

    context = build_shared_context(analysis)

    result = generate_explanation(
        build_kpi_prompt(
            kpi=kpi,
            context=context,
        )
    )

    return {
        "status": "success",
        "item_type": "kpi",
        "title": str(
            result.get("title")
            or kpi.get("label")
            or "KPI Explanation"
        ),
        "value": kpi.get("value"),
        "explanation": str(result.get("explanation", "")),
        "calculation": str(result.get("calculation", "")),
        "business_interpretation": str(
            result.get("business_interpretation", "")
        ),
        "cautions": [
            str(item)
            for item in result.get("cautions", [])
            if item
        ][:5],
        "supporting_evidence": normalize_evidence(
            result.get("supporting_evidence")
        ),
        "suggested_question": result.get("suggested_question"),
        "confidence": normalize_confidence(
            result.get("confidence")
        ),
        "model_used": result.get("model_used"),
        "provider": result.get("provider"),
    }


def explain_chart(
    analysis_id: str,
    chart_index: int,
    user_id: str,
) -> dict[str, Any]:
    analysis_record = get_owned_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
    )

    analysis = normalize_analysis_payload(
        analysis_record.get(ANALYSIS_PAYLOAD_COLUMN)
    )

    chart = find_chart(
        analysis=analysis,
        chart_index=chart_index,
    )

    context = build_shared_context(analysis)

    result = generate_explanation(
        build_chart_prompt(
            chart=chart,
            context=context,
        )
    )

    return {
        "status": "success",
        "item_type": "chart",
        "title": str(
            result.get("title")
            or chart.get("title")
            or "Chart Explanation"
        ),
        "chart_type": (
            result.get("chart_type")
            or chart.get("chart_type")
        ),
        "explanation": str(result.get("explanation", "")),
        "chart_logic": str(result.get("chart_logic", "")),
        "business_interpretation": str(
            result.get("business_interpretation", "")
        ),
        "cautions": [
            str(item)
            for item in result.get("cautions", [])
            if item
        ][:5],
        "supporting_evidence": normalize_evidence(
            result.get("supporting_evidence")
        ),
        "suggested_question": result.get("suggested_question"),
        "confidence": normalize_confidence(
            result.get("confidence")
        ),
        "model_used": result.get("model_used"),
        "provider": result.get("provider"),
    }