import pandas as pd

from app.services.chart_generator import generate_chart_recommendations, generate_plotly_charts
from app.services.kpi_generator import generate_kpis
from app.services.filter_generator import generate_filters
from app.services.column_profiler import profile_columns
from app.services.insight_generator import generate_basic_insights
from app.services.llm_report_generator import generate_llm_executive_analysis
from app.services.llm_dashboard_planner import generate_llm_dashboard_plan
from app.services.business_metrics import generate_business_metrics
from app.services.chart_summary import generate_chart_summary
from app.services.data_quality_summary import generate_data_quality_summary
from app.services.statistical_profile import generate_statistical_profile

MAX_FRONTEND_ROWS = 10000
CHART_SAMPLE_ROWS = 10000

def analyze_dataframe(df: pd.DataFrame) -> dict:
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.replace(["N/A", "NA", "null", "NULL", "None", "none", "?"], pd.NA)

    rows, columns = df.shape

    is_large_dataset = rows > MAX_FRONTEND_ROWS

    chart_df = (
        df.sample(n=CHART_SAMPLE_ROWS, random_state=42)
        if rows > CHART_SAMPLE_ROWS
        else df
    )

    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    column_profile = profile_columns(df)

    business_metrics = generate_business_metrics(
        df,
        column_profile,
    )

    statistical_profile = generate_statistical_profile(
        df,
        column_profile,
    )

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

    data_quality_summary = generate_data_quality_summary(
        rows=rows,
        columns=columns,
        total_missing_values=total_missing,
        duplicate_rows=duplicate_rows,
        data_quality_score=data_quality_score,
    )

    dashboard_plan_summary = {
        "rows": rows,
        "columns": columns,
        "data_quality_score": data_quality_score,
        "column_profile": column_profile,
        "summary_statistics": summary_statistics,
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    }

    dashboard_plan = generate_llm_dashboard_plan(
        summary=dashboard_plan_summary,
        column_profile=column_profile,
    )

    kpis = generate_kpis(
        df=df,
        column_profile=column_profile,
        kpi_plan=dashboard_plan.get("kpi_plan"),
    )

    filters = generate_filters(df, column_profile)

    chart_recommendations = generate_chart_recommendations(
        df=chart_df,
        column_profile=column_profile,
        chart_plan=dashboard_plan.get("chart_plan"),
    )

    charts = generate_plotly_charts(
        df=chart_df,
        column_profile=column_profile,
        chart_plan=dashboard_plan.get("chart_plan"),
    )

    chart_summary = generate_chart_summary(
        chart_recommendations=chart_recommendations,
        dashboard_plan=dashboard_plan,
    )

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
        "data_quality_summary": data_quality_summary,
        "total_missing_values": total_missing,
        "duplicate_rows": duplicate_rows,
        "column_profile": column_profile,
        "business_metrics": business_metrics,
        "statistical_profile": statistical_profile,
        "kpis": kpis,
        "missing_values": missing_values,
        "insights": insights,
        "preview": preview,
        "chart_summary": chart_summary,
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
        "data_quality_summary": data_quality_summary,
        "preview": preview,
        "data": df.fillna("").to_dict(orient="records") if not is_large_dataset else [],
        "processing": {
            "is_large_dataset": is_large_dataset,
            "row_level_data_returned": not is_large_dataset,
            "max_frontend_rows": MAX_FRONTEND_ROWS,
            "chart_sample_rows": CHART_SAMPLE_ROWS,
            "message": (
                "Large dataset detected. Dashboard charts were generated from a sample, and row-level frontend filtering is disabled."
                if is_large_dataset
                else "Full row-level data returned for interactive dashboard filtering."
            ),
        },
        "insights": insights,
        "chart_recommendations": chart_recommendations,
        "charts": charts,
        "chart_summary": chart_summary,
        "filters": filters,
        "column_profile": column_profile,
        "business_metrics": business_metrics,
        "statistical_profile": statistical_profile,
        "executive_analysis": executive_analysis,
        "dashboard_plan": dashboard_plan,
    }