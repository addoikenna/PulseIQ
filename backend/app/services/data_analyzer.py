import pandas as pd

from app.services.chart_generator import generate_chart_recommendations, generate_plotly_charts
from app.services.kpi_generator import generate_kpis
from app.services.filter_generator import generate_filters
from app.services.column_profiler import profile_columns
from app.services.insight_generator import generate_basic_insights
from app.services.llm_report_generator import generate_llm_executive_analysis


def analyze_dataframe(df: pd.DataFrame) -> dict:
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.replace(["N/A", "NA", "null", "NULL", "None", "none", "?"], pd.NA)

    rows, columns = df.shape

    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    column_profile = profile_columns(df)

    numeric_columns = column_profile.get("numeric_columns", [])
    text_columns = column_profile.get("categorical_columns", []) + column_profile.get("text_columns", [])
    possible_date_columns = column_profile.get("date_columns", [])

    summary_statistics = df.describe(include="all").fillna("").to_dict()

    data_quality_score = 100

    if total_missing > 0:
        data_quality_score -= min(
            30,
            int((total_missing / max(rows * columns, 1)) * 100),
        )

    if duplicate_rows > 0:
        data_quality_score -= min(
            20,
            int((duplicate_rows / max(rows, 1)) * 100),
        )

    data_quality_score = max(data_quality_score, 0)

    kpis = generate_kpis(df, column_profile)
    filters = generate_filters(df, column_profile)
    chart_recommendations = generate_chart_recommendations(df, column_profile)
    charts = generate_plotly_charts(df, column_profile)

    insights = generate_basic_insights(
        rows=rows,
        columns=columns,
        total_missing=total_missing,
        duplicate_rows=duplicate_rows,
        column_profile=column_profile,
        data_quality_score=data_quality_score,
    )

    preview = df.head(5).fillna("").to_dict(orient="records")

    summary_for_llm = {
        "rows": rows,
        "columns": columns,
        "data_quality_score": data_quality_score,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "column_profile": column_profile,
        "kpis": kpis,
        "missing_values": missing_values,
        "insights": insights,
        "preview": preview,
    }

    executive_analysis = generate_llm_executive_analysis(summary_for_llm)

    return {
        "rows": rows,
        "columns": columns,
        "kpis": kpis,
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_values,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "text_columns": text_columns,
        "possible_date_columns": possible_date_columns,
        "summary_statistics": summary_statistics,
        "data_quality_score": data_quality_score,
        "preview": preview,
        "insights": insights,
        "chart_recommendations": chart_recommendations,
        "charts": charts,
        "filters": filters,
        "column_profile": column_profile,
        "executive_analysis": executive_analysis,
    }