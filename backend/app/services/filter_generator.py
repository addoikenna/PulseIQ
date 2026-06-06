import pandas as pd


def generate_filters(df: pd.DataFrame, column_profile: dict) -> list:
    filters = []

    categorical_columns = column_profile.get("categorical_columns", [])
    date_columns = column_profile.get("date_columns", [])

    # Categorical dropdown filters
    for col in categorical_columns[:5]:
        unique_values = df[col].dropna().unique().tolist()

        if 2 <= len(unique_values) <= 50:
            filters.append({
                "label": col.replace("_", " ").title(),
                "column": col,
                "type": "select",
                "options": [str(value) for value in unique_values],
                "description": f"Filter dashboard by {col}."
            })

    # Date range filters
    for col in date_columns[:1]:
        converted = pd.to_datetime(df[col], errors="coerce")

        filters.append({
            "label": col.replace("_", " ").title(),
            "column": col,
            "type": "date_range",
            "min": str(converted.min().date()) if converted.notna().any() else None,
            "max": str(converted.max().date()) if converted.notna().any() else None,
            "description": f"Filter dashboard by {col} date range."
        })

    return filters