import pandas as pd


def generate_business_metrics(
    df: pd.DataFrame,
    column_profile: dict,
) -> dict:
    metrics = {
        "numeric_metrics": [],
        "category_metrics": [],
    }

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])

    blocked_columns = set(
        column_profile.get("id_columns", [])
        + column_profile.get("contact_columns", [])
    )

    # Numeric business metrics
    for col in numeric_columns:
        if col in blocked_columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if series.empty:
            continue

        metrics["numeric_metrics"].append({
            "column": col,
            "sum": round(float(series.sum()), 2),
            "average": round(float(series.mean()), 2),
            "maximum": round(float(series.max()), 2),
            "minimum": round(float(series.min()), 2),
            "median": round(float(series.median()), 2),
        })

    # Category business metrics
    for col in categorical_columns:
        if col in blocked_columns:
            continue

        value_counts = df[col].value_counts(dropna=True)

        if value_counts.empty:
            continue

        metrics["category_metrics"].append({
            "column": col,
            "top_value": str(value_counts.index[0]),
            "top_value_count": int(value_counts.iloc[0]),
            "unique_values": int(df[col].nunique(dropna=True)),
        })

    return metrics