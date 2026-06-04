import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_chart_recommendations(df: pd.DataFrame, column_profile: dict) -> list:
    charts = []

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])
    date_columns = column_profile.get("date_columns", [])

    # Numeric distribution charts
    for col in numeric_columns[:3]:
        charts.append({
            "chart_type": "histogram",
            "title": f"Distribution of {col}",
            "x_axis": col,
            "y_axis": "count",
            "description": f"Shows how values in {col} are distributed."
        })

    # Category count charts
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

    # Date trend charts
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

    # Relationship chart
    if len(numeric_columns) >= 2:
        charts.append({
            "chart_type": "scatter",
            "title": f"Relationship between {numeric_columns[0]} and {numeric_columns[1]}",
            "x_axis": numeric_columns[0],
            "y_axis": numeric_columns[1],
            "description": f"Shows the relationship between {numeric_columns[0]} and {numeric_columns[1]}."
        })

    return charts


def generate_plotly_charts(df: pd.DataFrame, column_profile: dict) -> list:
    charts = []

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])
    date_columns = column_profile.get("date_columns", [])

    # Numeric histograms
    for col in numeric_columns[:3]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append({
            "chart_type": "histogram",
            "title": f"Distribution of {col}",
            "figure": pio.to_json(fig)
        })

    # Categorical charts
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

    # Date trend charts
    if date_columns and numeric_columns:
        date_col = date_columns[0]

        chart_df = df.copy()
        chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
        chart_df = chart_df.dropna(subset=[date_col])

        for num_col in numeric_columns[:2]:
            trend = chart_df.groupby(date_col)[num_col].sum().reset_index()
            fig = px.line(trend, x=date_col, y=num_col, title=f"{num_col} Trend Over Time")
            charts.append({
                "chart_type": "line",
                "title": f"{num_col} Trend Over Time",
                "figure": pio.to_json(fig)
            })

    # Scatter chart
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