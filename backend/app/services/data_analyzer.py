import pandas as pd

from app.services.chart_generator import generate_chart_recommendations
from app.services.chart_generator import generate_chart_recommendations, generate_plotly_charts
from app.services.kpi_generator import generate_kpis


def analyze_dataframe(df: pd.DataFrame) -> dict:
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.replace(["N/A", "NA", "null", "NULL", "None", "none", "?"], pd.NA)

    rows, columns = df.shape

    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    text_columns = df.select_dtypes(include=["object"]).columns.tolist()

    possible_date_columns = []
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) >= 0.7:
            possible_date_columns.append(col)

    summary_statistics = df.describe(include="all").fillna("").to_dict()

    data_quality_score = 100

    if total_missing > 0:
        data_quality_score -= min(30, int((total_missing / max(rows * columns, 1)) * 100))

    if duplicate_rows > 0:
        data_quality_score -= min(20, int((duplicate_rows / max(rows, 1)) * 100))

    data_quality_score = max(data_quality_score, 0)

    chart_recommendations = generate_chart_recommendations(df)

    charts = generate_plotly_charts(df)

    insights = generate_basic_insights(
        rows=rows,
        columns=columns,
        total_missing=total_missing,
        duplicate_rows=duplicate_rows,
        numeric_columns=numeric_columns,
        text_columns=text_columns,
        possible_date_columns=possible_date_columns,
        data_quality_score=data_quality_score,
    )

    kpis = generate_kpis(df)

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
        "preview": df.head(5).fillna("").to_dict(orient="records"),
        "insights": insights,
        "chart_recommendations": chart_recommendations,
        "charts": charts,
    }


def generate_basic_insights(
    rows: int,
    columns: int,
    total_missing: int,
    duplicate_rows: int,
    numeric_columns: list,
    text_columns: list,
    possible_date_columns: list,
    data_quality_score: int,
) -> list:
    insights = []

    insights.append(
        f"The dataset contains {rows} rows and {columns} columns."
    )

    if data_quality_score >= 80:
        insights.append(
            f"The data quality score is {data_quality_score}/100, which suggests the dataset is in good condition for analysis."
        )
    elif data_quality_score >= 50:
        insights.append(
            f"The data quality score is {data_quality_score}/100. The dataset can be analyzed, but it needs some cleaning."
        )
    else:
        insights.append(
            f"The data quality score is {data_quality_score}/100. The dataset needs serious cleaning before reliable analysis."
        )

    if total_missing > 0:
        insights.append(
            f"There are {total_missing} missing values in the dataset. These should be reviewed before analysis."
        )
    else:
        insights.append(
            "There are no missing values detected in the dataset."
        )

    if duplicate_rows > 0:
        insights.append(
            f"There are {duplicate_rows} duplicate rows. You may need to remove them to avoid misleading results."
        )
    else:
        insights.append(
            "No duplicate rows were detected."
        )

    if numeric_columns:
        insights.append(
            f"The dataset has {len(numeric_columns)} numeric column(s), which can be used for statistical analysis and charts."
        )

    if text_columns:
        insights.append(
            f"The dataset has {len(text_columns)} text/categorical column(s), which can be used for grouping and comparisons."
        )

    if possible_date_columns:
        insights.append(
            f"Possible date column(s) detected: {', '.join(possible_date_columns)}. These can support trend analysis."
        )

    return insights