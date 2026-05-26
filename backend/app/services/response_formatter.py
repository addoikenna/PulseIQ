def format_analysis_response(filename: str, summary: dict) -> dict:
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
        "columns": {
            "names": summary.get("column_names"),
            "data_types": summary.get("data_types"),
            "numeric_columns": summary.get("numeric_columns"),
            "text_columns": summary.get("text_columns"),
            "possible_date_columns": summary.get("possible_date_columns"),
        },
        "data_quality": {
            "missing_values": summary.get("missing_values"),
            "cleaning_report": summary.get("cleaning_report"),
        },
        "insights": summary.get("insights"),
        "chart_recommendations": summary.get("chart_recommendations"),
        "charts": summary.get("charts"),
        "preview": summary.get("preview"),
    }