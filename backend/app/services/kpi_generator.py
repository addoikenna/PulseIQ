import pandas as pd


def generate_kpis(df: pd.DataFrame) -> list:
    kpis = []

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Business-style numeric KPIs
    for col in numeric_columns[:5]:
        clean_col_name = col.replace("_", " ").title()

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

    # Category KPIs
    for col in text_columns[:3]:
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