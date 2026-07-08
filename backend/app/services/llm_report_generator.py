import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

from google import genai


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


DEFAULT_EXECUTIVE_ANALYSIS = {
    "business_situation": [
        "AI executive analysis is currently unavailable. Rule-based insights are shown instead."
    ],
    "key_business_drivers": [],
    "critical_risks": [],
    "growth_opportunities": [],
    "strategic_priorities": [],
    "ninety_day_action_plan": [],
    "confidence_assessment": {
        "level": "low",
        "reason": "AI executive analysis was unavailable."
    },
    "provider": "fallback",
    "model_used": None,
}


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
        raise ValueError(f"No JSON object found in model response: {content}")

    json_text = content[start:end + 1]

    return json.loads(json_text)


def build_report_prompt(summary: dict[str, Any]) -> str:
    compact_summary = {
        "overview": {
            "rows": summary.get("rows"),
            "columns": summary.get("columns"),
            "data_quality_score": summary.get("data_quality_score"),
            "total_missing_values": summary.get("total_missing_values"),
            "duplicate_rows": summary.get("duplicate_rows"),
        },
        "column_profile": summary.get("column_profile"),
        "business_metrics": summary.get("business_metrics"),
        "statistical_profile": summary.get("statistical_profile"),
        "chart_summary": summary.get("chart_summary"),
        "kpis": summary.get("kpis"),
        "missing_values": summary.get("missing_values"),
        "insights": summary.get("insights"),
        "data_quality_summary": summary.get("data_quality_summary"),
        "business_health": summary.get("business_health"),
        "business_drivers": summary.get("business_drivers"),
        "business_risks": summary.get("business_risks"),
        "business_opportunities": summary.get("business_opportunities"),
        "preview": summary.get("preview"),
    }

    return f"""
    You are PulseIQ, a senior Business Intelligence Consultant and Executive Advisor.

    Your audience is:
    - CEOs
    - Founders
    - Business Managers
    - Department Heads
    - Investors

    Analyze the information provided and generate an executive-level business report.

    Focus on:
    - Business performance
    - Trends and patterns
    - Operational opportunities
    - Strategic risks
    - Growth opportunities
    - Resource allocation insights
    - Areas requiring management attention
    - Distribution patterns
    - Outliers
    - Variability
    - Strong relationships between metrics

    Do NOT focus on:
    - Number of rows
    - Number of columns
    - Technical dataset structure
    - Programming concepts

    Only mention data quality concerns if they materially affect decision-making.

    Return ONLY valid JSON.

    The JSON must contain exactly:

    {{
        "business_situation": [
            "..."
        ],
        "key_business_drivers": [
            {{
            "driver": "...",
            "why_it_matters": "...",
            "evidence": "..."
            }}
        ],
        "critical_risks": [
            {{
            "risk": "...",
            "severity": "High | Medium | Low",
            "business_impact": "...",
            "evidence": "..."
            }}
        ],
        "growth_opportunities": [
            {{
            "opportunity": "...",
            "expected_impact": "High | Medium | Low",
            "evidence": "..."
            }}
        ],
        "strategic_priorities": [
            {{
            "priority": "High | Medium | Low",
            "recommendation": "...",
            "expected_impact": "High | Medium | Low",
            "evidence": "...",
            "suggested_owner": "...",
            "timeline": "Immediate | 30 Days | 90 Days | Long-term"
            }}
        ],
        "ninety_day_action_plan": [
            {{
            "timeline": "Immediate | 30 Days | 60 Days | 90 Days",
            "action": "...",
            "owner": "...",
            "success_measure": "..."
            }}
        ],
        "confidence_assessment": {{
            "level": "High | Medium | Low",
            "reason": "..."
        }}
    }}

    Rules:
    - Do not invent facts not supported by the business context.
    - Do not recalculate metrics.
    - Use the backend-generated business_health, business_drivers, business_risks, and business_opportunities as the main evidence base.
    - Every strategic priority must include evidence.
    - Every risk must include business impact.
    - Every action must have an owner and success measure.
    - Avoid technical dataset language unless it affects decision-making.
    - Write like a senior consultant preparing an executive briefing.

    Guidelines:

    EXECUTIVE SUMMARY
    - 3 to 5 concise bullets.
    - Summarize overall business performance.
    - Highlight major observations.

    KEY FINDINGS
    - Most important business observations.
    - Mention dominant categories, top metrics, trends, and patterns.

    OPPORTUNITIES
    - Revenue growth opportunities.
    - Efficiency improvements.
    - Performance improvement opportunities.
    - Areas showing strong performance.

    RISKS
    - Underperformance.
    - Data quality concerns impacting decisions.
    - Concentration risks.
    - Operational concerns.

    RECOMMENDATIONS
    - Actionable business recommendations.
    - Management-level suggestions.
    - Strategic next steps.

    NEXT ACTIONS
    - Immediate actions stakeholders should take.

    CONFIDENCE LEVEL
    - High = strong business evidence available.
    - Medium = moderate evidence available.
    - Low = limited evidence available.

    Business Context:
    {json.dumps(compact_summary, default=str)}

    Use the statistical profile to identify:

    - unusually high or low variability
    - significant outliers
    - skewed distributions
    - meaningful correlations
    - possible operational anomalies

    Do not describe statistics for their own sake.
    Translate every statistical observation into a business implication.

    For example:

    Instead of:
    "The salary column has a standard deviation of 25,000."

    Say:
    "Salary variation is unusually high, suggesting considerable differences in role seniority or compensation structure."

    Instead of:
    "There are 14 outliers."

    Say:
    "A small number of unusually large transactions may warrant review to determine whether they represent strategic accounts or exceptional events."
    """


def call_openrouter(model: str, prompt: str) -> dict[str, Any]:
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
                "content": "You are a careful executive data analyst. Return only valid JSON, no markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": 0.2,
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

    # print("OpenRouter raw response:", content)

    return parse_llm_json(content)


def call_gemini(prompt: str) -> dict[str, Any]:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    content = response.text or ""

    return parse_llm_json(content)


def generate_llm_executive_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    prompt = build_report_prompt(summary)

    try:
        report = call_gemini(prompt)

        return {
            "business_situation": report.get("business_situation", []),
            "key_business_drivers": report.get("key_business_drivers", []),
            "critical_risks": report.get("critical_risks", []),
            "growth_opportunities": report.get("growth_opportunities", []),
            "strategic_priorities": report.get("strategic_priorities", []),
            "ninety_day_action_plan": report.get("ninety_day_action_plan", []),
            "confidence_assessment": report.get(
                "confidence_assessment",
                {
                    "level": "medium",
                    "reason": "Confidence was not explicitly provided by the model."
                }
            ),
            "model_used": GEMINI_MODEL,
            "provider": "gemini",
        }

    except Exception as e:
        print(f"Gemini executive analysis error: {e}")

    for model in [OPENROUTER_PRIMARY_MODEL, OPENROUTER_FALLBACK_MODEL]:
        try:
            report = call_openrouter(model, prompt)

            return {
                "business_situation": report.get("business_situation", []),
                "key_business_drivers": report.get("key_business_drivers", []),
                "critical_risks": report.get("critical_risks", []),
                "growth_opportunities": report.get("growth_opportunities", []),
                "strategic_priorities": report.get("strategic_priorities", []),
                "ninety_day_action_plan": report.get("ninety_day_action_plan", []),
                "confidence_assessment": report.get(
                    "confidence_assessment",
                    {
                        "level": "medium",
                        "reason": "Confidence was not explicitly provided by the model."
                    }
                ),
                "model_used": model,
                "provider": "openrouter",
            }

        except Exception as e:
            print(f"OpenRouter error for model {model}: {e}")
            continue

    return DEFAULT_EXECUTIVE_ANALYSIS