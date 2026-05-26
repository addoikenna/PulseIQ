import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_chart_recommendations(df: pd.DataFrame) -> list:
    charts = []

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    date_columns = []
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) >= 0.7:
            date_columns.append(col)

    for col in numeric_columns[:3]:
        charts.append({
            "chart_type": "histogram",
            "title": f"Distribution of {col}",
            "x_axis": col,
            "y_axis": "count",
            "description": f"Shows how values in {col} are distributed."
        })

    for cat_col in categorical_columns[:3]:
        unique_count = df[cat_col].nunique(dropna=True)

        charts.append({
            "chart_type": "bar",
            "title": f"Count by {cat_col}",
            "x_axis": cat_col,
            "y_axis": "count",
            "description": f"Compares the number of records across {cat_col}."
        })

        if 2 <= unique_count <= 6:
            charts.append({
                "chart_type": "pie",
                "title": f"Share by {cat_col}",
                "category": cat_col,
                "value": "count",
                "description": f"Shows the percentage share of each {cat_col} group."
            })

    if date_columns and numeric_columns:
        date_col = date_columns[0]

        for num_col in numeric_columns[:2]:
            charts.append({
                "chart_type": "line",
                "title": f"{num_col} Trend Over Time",
                "x_axis": date_col,
                "y_axis": num_col,
                "description": f"Shows how {num_col} changes over time."
            })

    if len(numeric_columns) >= 2:
        charts.append({
            "chart_type": "scatter",
            "title": f"Relationship between {numeric_columns[0]} and {numeric_columns[1]}",
            "x_axis": numeric_columns[0],
            "y_axis": numeric_columns[1],
            "description": f"Shows the relationship between {numeric_columns[0]} and {numeric_columns[1]}."
        })

        charts.append({
            "chart_type": "heatmap",
            "title": "Correlation Heatmap",
            "columns": numeric_columns,
            "description": "Shows relationships between numeric columns."
        })

    return charts


def generate_plotly_charts(df: pd.DataFrame) -> list:
    charts = []

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    date_columns = []
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().sum() / max(len(df), 1) >= 0.7:
            date_columns.append(col)

    for col in numeric_columns[:3]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append({
            "chart_type": "histogram",
            "title": f"Distribution of {col}",
            "figure": pio.to_json(fig)
        })

    for cat_col in categorical_columns[:3]:
        counts = df[cat_col].value_counts().reset_index()
        counts.columns = [cat_col, "count"]

        fig = px.bar(counts, x=cat_col, y="count", title=f"Count by {cat_col}")
        charts.append({
            "chart_type": "bar",
            "title": f"Count by {cat_col}",
            "figure": pio.to_json(fig)
        })

        if 2 <= df[cat_col].nunique(dropna=True) <= 6:
            fig = px.pie(counts, names=cat_col, values="count", title=f"Share by {cat_col}")
            charts.append({
                "chart_type": "pie",
                "title": f"Share by {cat_col}",
                "figure": pio.to_json(fig)
            })

    if date_columns and numeric_columns:
        date_col = date_columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        for num_col in numeric_columns[:2]:
            trend = df.groupby(date_col)[num_col].sum().reset_index()
            fig = px.line(trend, x=date_col, y=num_col, title=f"{num_col} Trend Over Time")
            charts.append({
                "chart_type": "line",
                "title": f"{num_col} Trend Over Time",
                "figure": pio.to_json(fig)
            })

    if len(numeric_columns) >= 2:
        fig = px.scatter(
            df,
            x=numeric_columns[0],
            y=numeric_columns[1],
            title=f"Relationship between {numeric_columns[0]} and {numeric_columns[1]}"
        )
        charts.append({
            "chart_type": "scatter",
            "title": f"Relationship between {numeric_columns[0]} and {numeric_columns[1]}",
            "figure": pio.to_json(fig)
        })

    return charts