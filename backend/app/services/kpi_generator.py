import pandas as pd


def generate_kpis(
    df: pd.DataFrame,
    column_profile: dict,
    kpi_plan: list | None = None
) -> list:
    if kpi_plan:
        planned_kpis = generate_kpis_from_plan(df, kpi_plan, column_profile)

        if planned_kpis:
            return planned_kpis

    return generate_rule_based_kpis(df, column_profile)


def generate_kpis_from_plan(
    df: pd.DataFrame,
    kpi_plan: list,
    column_profile: dict
) -> list:
    kpis = []

    blocked_columns = set(column_profile.get("id_columns", [])) | set(column_profile.get("contact_columns", []))

    for item in kpi_plan:
        label = item.get("label")
        kpi_type = item.get("type")
        column = item.get("column")
        reason = item.get("reason", "")

        if not label or not kpi_type or not column:
            continue

        if column not in df.columns:
            continue

        if column in blocked_columns:
            continue

        series = df[column]

        try:
            if kpi_type == "sum":
                value = float(series.sum(skipna=True))

            elif kpi_type == "average":
                value = float(series.mean(skipna=True))

            elif kpi_type == "maximum":
                value = float(series.max(skipna=True))

            elif kpi_type == "minimum":
                value = float(series.min(skipna=True))

            elif kpi_type == "count":
                value = int(series.count())

            elif kpi_type == "top_category":
                if series.dropna().empty:
                    continue
                value = str(series.value_counts(dropna=True).idxmax())

            else:
                continue

            kpis.append({
                "label": label,
                "value": value,
                "type": kpi_type,
                "source_column": column,
                "description": reason or f"{label} calculated from {column}."
            })

        except Exception:
            continue

    return kpis


def generate_rule_based_kpis(df: pd.DataFrame, column_profile: dict) -> list:
    kpis = []

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])
    performance_columns = set(column_profile.get("performance_metric_columns", []))
    blocked_columns = set(column_profile.get("id_columns", [])) | set(column_profile.get("contact_columns", []))

    for col in numeric_columns[:5]:
        if col in blocked_columns:
            continue

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue

        clean_col_name = col.replace("_", " ").title()

        if col in performance_columns:
            kpis.append({
                "label": f"Average {clean_col_name}",
                "value": float(df[col].mean(skipna=True)),
                "type": "average",
                "source_column": col,
                "description": f"Average value of {col}."
            })

            kpis.append({
                "label": f"Maximum {clean_col_name}",
                "value": float(df[col].max(skipna=True)),
                "type": "maximum",
                "source_column": col,
                "description": f"Highest value recorded in {col}."
            })

            kpis.append({
                "label": f"Minimum {clean_col_name}",
                "value": float(df[col].min(skipna=True)),
                "type": "minimum",
                "source_column": col,
                "description": f"Lowest value recorded in {col}."
            })

        else:
            kpis.append({
                "label": f"Total {clean_col_name}",
                "value": float(df[col].sum(skipna=True)),
                "type": "sum",
                "source_column": col,
                "description": f"Sum of all values in {col}."
            })

            kpis.append({
                "label": f"Average {clean_col_name}",
                "value": float(df[col].mean(skipna=True)),
                "type": "average",
                "source_column": col,
                "description": f"Average value of {col}."
            })

            kpis.append({
                "label": f"Maximum {clean_col_name}",
                "value": float(df[col].max(skipna=True)),
                "type": "maximum",
                "source_column": col,
                "description": f"Highest value recorded in {col}."
            })

    for col in categorical_columns[:3]:
        if col in blocked_columns:
            continue

        if df[col].nunique(dropna=True) > 0:
            top_value = df[col].value_counts(dropna=True).idxmax()
            top_count = int(df[col].value_counts(dropna=True).max())

            kpis.append({
                "label": f"Top {col.replace('_', ' ').title()}",
                "value": str(top_value),
                "type": "top_category",
                "source_column": col,
                "description": f"Most frequent value in {col}, appearing {top_count} time(s)."
            })

    return kpis