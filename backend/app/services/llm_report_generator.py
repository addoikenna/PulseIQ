import json
import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

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
    "executive_summary": [
        "AI executive analysis is currently unavailable. Rule-based insights are shown instead."
    ],
    "key_findings": [],
    "opportunities": [],
    "risks": [],
    "recommendations": [],
    "next_actions": [],
}


def parse_llm_json(content: str) -> dict[str, Any]:
    if not content:
        raise ValueError("OpenRouter returned empty content.")

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
        "kpis": summary.get("kpis"),
        "missing_values": summary.get("missing_values"),
        "insights": summary.get("insights"),
        "preview": summary.get("preview"),
    }

    return f"""
You are PulseIQ, an executive data analyst.

Analyze the dataset summary below and produce a concise business report.

Return ONLY valid JSON. Do not use markdown. Do not wrap the response in code fences.

The JSON must use exactly these keys:
{{
  "executive_summary": ["..."],
  "key_findings": ["..."],
  "opportunities": ["..."],
  "risks": ["..."],
  "recommendations": ["..."],
  "next_actions": ["..."]
}}

Rules:
- Do not invent facts not supported by the data summary.
- Keep language clear, professional, and business-friendly.
- Mention uncertainty where data is limited.
- Avoid technical jargon.
- Keep each bullet concise.
- Each section should contain 2 to 5 bullet strings where possible.

Dataset summary:
{json.dumps(compact_summary, default=str)}
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

    print("OpenRouter raw response:", content)

    return parse_llm_json(content)


def generate_llm_executive_analysis(summary: dict[str, Any]) -> dict[str, Any]:
    prompt = build_report_prompt(summary)

    for model in [OPENROUTER_PRIMARY_MODEL, OPENROUTER_FALLBACK_MODEL]:
        try:
            report = call_openrouter(model, prompt)

            return {
                "executive_summary": report.get("executive_summary", []),
                "key_findings": report.get("key_findings", []),
                "opportunities": report.get("opportunities", []),
                "risks": report.get("risks", []),
                "recommendations": report.get("recommendations", []),
                "next_actions": report.get("next_actions", []),
                "model_used": model,
            }

        except Exception as e:
            print(f"OpenRouter error for model {model}: {e}")
            continue

    return DEFAULT_EXECUTIVE_ANALYSIS