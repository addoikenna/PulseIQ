def format_analysis_response(summary: dict, filename: str) -> dict:
    return {
        "status": "success",
        "message": "Dataset analyzed successfully.",
        "filename": filename,

        "overview": {
            "rows": summary.get("rows"),
            "columns": summary.get("columns"),
            "data_quality_score": summary.get("data_quality_score"),
            "total_missing_values": summary.get("total_missing_values"),
            "duplicate_rows": summary.get("duplicate_rows"),
        },

        "column_profile": summary.get("column_profile"),

        "dashboard": {
            "kpis": summary.get("kpis"),
            "charts": summary.get("charts"),
            "filters": summary.get("filters"),
            "chart_recommendations": summary.get("chart_recommendations"),
        },

        "report": {
            "executive_analysis": summary.get("executive_analysis"),

            "insights": summary.get("insights"),

            "data_quality": {
                "missing_values": summary.get("missing_values"),
                "cleaning_report": summary.get("cleaning_report"),
            },

            "recommendations": [
                "Review missing values before making business decisions.",
                "Investigate duplicate rows to avoid skewed analysis.",
                "Use dashboard filters to explore trends and segments.",
                "Monitor top-performing categories and metrics regularly."
            ]
        },

        "preview": summary.get("preview"),
        "data": summary.get("data"),
    }