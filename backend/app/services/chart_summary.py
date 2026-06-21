from typing import Any


def generate_chart_summary(
    chart_recommendations: list[dict[str, Any]],
    dashboard_plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    chart_summary = []

    plan_lookup = {}

    if dashboard_plan:
        for chart in dashboard_plan.get("chart_plan", []):
            title = chart.get("title")
            if title:
                plan_lookup[title] = chart

    for chart in chart_recommendations:
        title = chart.get("title")
        chart_type = chart.get("chart_type")

        plan_item = plan_lookup.get(title, {})

        chart_summary.append({
            "title": title,
            "chart_type": chart_type,
            "x_axis": chart.get("x_axis") or plan_item.get("x_axis"),
            "y_axis": chart.get("y_axis") or plan_item.get("y_axis"),
            "aggregation": chart.get("aggregation") or plan_item.get("aggregation"),
            "time_grain": chart.get("time_grain") or plan_item.get("time_grain"),
            "business_question": infer_business_question(
                chart_type=chart_type,
                x_axis=chart.get("x_axis") or plan_item.get("x_axis"),
                y_axis=chart.get("y_axis") or plan_item.get("y_axis"),
                aggregation=chart.get("aggregation") or plan_item.get("aggregation"),
            ),
        })

    return chart_summary


def infer_business_question(
    chart_type: str | None,
    x_axis: str | None,
    y_axis: str | None,
    aggregation: str | None,
) -> str:
    if chart_type == "line":
        return f"How does {aggregation or 'the metric'} {y_axis} change over time?"

    if chart_type == "bar":
        if y_axis == "count":
            return f"How are records distributed across {x_axis}?"
        return f"How does {aggregation or 'total'} {y_axis} compare across {x_axis}?"

    if chart_type == "pie":
        return f"What share of the total is represented by each {x_axis} group?"

    if chart_type == "histogram":
        return f"What is the distribution pattern of {x_axis}?"

    if chart_type == "scatter":
        return f"What relationship exists between {x_axis} and {y_axis}?"

    return "What business pattern does this chart reveal?"