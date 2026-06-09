import json
import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_PRIMARY_MODEL = os.getenv("OPENROUTER_PRIMARY_MODEL", "deepseek/deepseek-chat-v3.1")
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-v3.1-terminus")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


DEFAULT_DASHBOARD_PLAN = {
    "kpi_plan": [],
    "chart_plan": [],
    "filter_plan": []
}


def parse_llm_json(content: str) -> dict[str, Any]:
    if not content:
        raise ValueError("Model returned empty content.")

    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).replace("```", "").strip()
    elif content.startswith("```"):
        content = content.replace("```", "").strip()

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model response: {content}")

    return json.loads(content[start:end + 1])


def build_dashboard_prompt(summary: dict[str, Any]) -> str:
    compact_summary = {
        "overview": {
            "rows": summary.get("rows"),
            "columns": summary.get("columns"),
            "data_quality_score": summary.get("data_quality_score"),
        },
        "column_profile": summary.get("column_profile"),
        "summary_statistics": summary.get("summary_statistics"),
        "preview": summary.get("preview"),
    }

    return f"""
You are PulseIQ, an expert BI dashboard designer.

Create a dashboard plan for the dataset summary below.

Return ONLY valid JSON. No markdown. No code fences.

The JSON must use exactly these keys:
{{
  "kpi_plan": [
    {{
      "label": "...",
      "type": "sum | average | maximum | minimum | count | top_category",
      "column": "...",
      "reason": "..."
    }}
  ],
  "chart_plan": [
    {{
      "title": "...",
      "chart_type": "line | bar | pie | histogram | scatter",
      "x_axis": "...",
      "y_axis": "...",
      "aggregation": "sum | average | count | none",
      "time_grain": "day | month | quarter | year | none",
      "reason": "..."
    }}
  ],
  "filter_plan": [
    {{
      "label": "...",
      "column": "...",
      "type": "select | date_range",
      "reason": "..."
    }}
  ]
}}

Rules:
- Only use columns that exist in the dataset.
- Do not use identifier columns as KPIs or chart axes unless counting records.
- Prefer business KPIs over technical metrics.
- Pick at most 6 KPIs.
- Pick at most 6 charts.
- Use line charts only when there is a date column and numeric metric.
- Use bar charts for category vs numeric metric or category counts.
- Use pie charts only for categorical share with 2 to 6 categories.
- Use histogram for numeric distribution.
- Use scatter only when there are at least two meaningful numeric metrics.
- Use categorical columns and date columns as filters.
- Do not use numeric columns as filters.

Choose aggregation based on metric meaning:
- Use average for bounded metrics, performance metrics, scores, ratings, grades, percentages, rates, ratios, indexes, and normalized values.
- Use sum for additive metrics such as money, quantities, units, costs, revenue, expenses, totals, and counts already represented as numeric measures.
- Use count for record counts by category.
- Never sum or average identifier columns.
- For counts of entities, count rows or non-null records, not ID values.

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
                "content": "You are a careful BI dashboard planner. Return only valid JSON.",
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

    return parse_llm_json(content)


def infer_safe_aggregation(column: str, requested_aggregation: str | None = None) -> str:
    column_lower = column.lower()

    average_keywords = [
        "score", "rating", "grade", "gpa", "cgpa", "percentage",
        "percent", "rate", "ratio", "index", "margin", "average",
        "avg", "satisfaction", "performance"
    ]

    sum_keywords = [
        "sales", "revenue", "profit", "amount", "cost", "quantity",
        "qty", "total", "spend", "budget", "expense", "income",
        "units", "orders", "budget"
    ]

    if any(keyword in column_lower for keyword in average_keywords):
        return "average"

    if any(keyword in column_lower for keyword in sum_keywords):
        return "sum"

    if requested_aggregation in ["sum", "average", "count", "maximum", "minimum", "none"]:
        return requested_aggregation

    return "sum"


def validate_dashboard_plan(plan: dict[str, Any], column_profile: dict) -> dict[str, Any]:
    all_columns = set(
        column_profile.get("date_columns", [])
        + column_profile.get("numeric_columns", [])
        + column_profile.get("categorical_columns", [])
        + column_profile.get("text_columns", [])
        + column_profile.get("id_columns", [])
        + column_profile.get("boolean_columns", [])
    )

    numeric_columns = set(column_profile.get("numeric_columns", []))
    categorical_columns = set(column_profile.get("categorical_columns", []))
    date_columns = set(column_profile.get("date_columns", []))
    id_columns = set(column_profile.get("id_columns", []))

    valid_kpis = []
    for kpi in plan.get("kpi_plan", []):
        column = kpi.get("column")
        kpi_type = kpi.get("type")

        if column not in all_columns:
            continue

        if column in id_columns and kpi_type != "count":
            continue

        if kpi_type in ["sum", "average", "maximum", "minimum"] and column not in numeric_columns:
            continue

        if kpi_type == "top_category" and column not in categorical_columns:
            continue

        if kpi_type in ["sum", "average", "maximum", "minimum"]:
            safe_aggregation = infer_safe_aggregation(column, kpi_type)

            if safe_aggregation == "average" and kpi_type == "sum":
                kpi["type"] = "average"
                if kpi.get("label"):
                    kpi["label"] = kpi["label"].replace("Total", "Average")

        valid_kpis.append(kpi)

    valid_charts = []
    for chart in plan.get("chart_plan", []):
        chart_type = chart.get("chart_type")
        x_axis = chart.get("x_axis")
        y_axis = chart.get("y_axis")

        if chart_type not in ["line", "bar", "pie", "histogram", "scatter"]:
            continue

        if x_axis and x_axis != "none" and x_axis not in all_columns:
            continue

        if y_axis and y_axis != "count" and y_axis != "none" and y_axis not in all_columns:
            continue

        if chart_type == "line":
            if x_axis not in date_columns or y_axis not in numeric_columns:
                continue

        if chart_type == "bar":
            if x_axis not in categorical_columns and x_axis not in date_columns:
                continue

        if chart_type == "pie":
            if x_axis not in categorical_columns:
                continue

        if chart_type == "histogram":
            if x_axis not in numeric_columns:
                continue

        if chart_type == "scatter":
            if x_axis not in numeric_columns or y_axis not in numeric_columns:
                continue
        
        if y_axis and y_axis not in ["count", "none"]:
            chart["aggregation"] = infer_safe_aggregation(
                y_axis,
                chart.get("aggregation")
            )

        valid_charts.append(chart)

    valid_filters = []
    for filter_item in plan.get("filter_plan", []):
        column = filter_item.get("column")
        filter_type = filter_item.get("type")

        if column not in all_columns:
            continue

        if filter_type == "select" and column in categorical_columns:
            valid_filters.append(filter_item)

        if filter_type == "date_range" and column in date_columns:
            valid_filters.append(filter_item)

    return {
        "kpi_plan": valid_kpis[:6],
        "chart_plan": valid_charts[:6],
        "filter_plan": valid_filters[:6],
    }


def generate_llm_dashboard_plan(summary: dict[str, Any], column_profile: dict) -> dict[str, Any]:
    prompt = build_dashboard_prompt(summary)

    for model in [OPENROUTER_PRIMARY_MODEL, OPENROUTER_FALLBACK_MODEL]:
        try:
            plan = call_openrouter(model, prompt)
            validated_plan = validate_dashboard_plan(plan, column_profile)
            validated_plan["model_used"] = model
            return validated_plan

        except Exception as e:
            print(f"OpenRouter dashboard planner error for model {model}: {e}")
            continue

    return DEFAULT_DASHBOARD_PLAN