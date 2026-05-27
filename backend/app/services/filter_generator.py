import pandas as pd


def generate_filters(df: pd.DataFrame) -> list:
    filters = []

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    text_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()

    # Detect date columns
    date_columns = []
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) >= 0.7:
            date_columns.append(col)

    # Categorical dropdown filters
    for col in text_columns[:5]:
        unique_values = df[col].dropna().unique().tolist()

        if 2 <= len(unique_values) <= 30:
            filters.append({
                "label": col.replace("_", " ").title(),
                "column": col,
                "type": "select",
                "options": [str(value) for value in unique_values],
                "description": f"Filter dashboard by {col}."
            })

    # Date range filters
    for col in date_columns[:2]:
        converted = pd.to_datetime(df[col], errors="coerce")

        filters.append({
            "label": col.replace("_", " ").title(),
            "column": col,
            "type": "date_range",
            "min": str(converted.min().date()) if converted.notna().any() else None,
            "max": str(converted.max().date()) if converted.notna().any() else None,
            "description": f"Filter dashboard by {col} date range."
        })

    # Numeric range filters
    for col in numeric_columns[:5]:
        filters.append({
            "label": col.replace("_", " ").title(),
            "column": col,
            "type": "number_range",
            "min": float(df[col].min(skipna=True)) if df[col].notna().any() else None,
            "max": float(df[col].max(skipna=True)) if df[col].notna().any() else None,
            "description": f"Filter dashboard by {col} range."
        })

    return filters