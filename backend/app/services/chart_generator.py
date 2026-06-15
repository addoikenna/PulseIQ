import pandas as pd
import plotly.express as px
import plotly.io as pio


def generate_chart_recommendations(
    df: pd.DataFrame,
    column_profile: dict,
    chart_plan: list | None = None
) -> list:
    if chart_plan:
        recommendations = []

        for chart in chart_plan:
            recommendations.append({
                "chart_type": chart.get("chart_type"),
                "title": chart.get("title"),
                "x_axis": chart.get("x_axis"),
                "y_axis": chart.get("y_axis"),
                "aggregation": chart.get("aggregation"),
                "time_grain": chart.get("time_grain"),
                "description": chart.get("reason", "")
            })

        if recommendations:
            return recommendations

    return generate_rule_based_chart_recommendations(df, column_profile)


def generate_plotly_charts(
    df: pd.DataFrame,
    column_profile: dict,
    chart_plan: list | None = None
) -> list:
    if chart_plan:
        planned_charts = generate_charts_from_plan(df, chart_plan)

        if planned_charts:
            return planned_charts

    return generate_rule_based_plotly_charts(df, column_profile)


def generate_charts_from_plan(df: pd.DataFrame, chart_plan: list) -> list:
    charts = []

    for chart in chart_plan:
        chart_type = chart.get("chart_type")
        title = chart.get("title")
        x_axis = chart.get("x_axis")
        y_axis = chart.get("y_axis")
        # aggregation = chart.get("aggregation", "sum")
        aggregation = chart.get("aggregation") or "sum"
        time_grain = chart.get("time_grain", "month")

        if not chart_type or not title:
            continue

        try:
            if chart_type == "histogram":
                if x_axis not in df.columns:
                    continue

                fig = px.histogram(df, x=x_axis, title=title)

            elif chart_type == "bar":
                if x_axis not in df.columns:
                    continue

                if y_axis == "count" or y_axis == "none" or not y_axis:
                    chart_df = df[x_axis].value_counts().reset_index()
                    chart_df.columns = [x_axis, "count"]
                    fig = px.bar(chart_df, x=x_axis, y="count", title=title)

                else:
                    if y_axis not in df.columns:
                        continue

                    chart_df = aggregate_category_metric(
                        df=df,
                        category_column=x_axis,
                        metric_column=y_axis,
                        aggregation=aggregation,
                    )

                    fig = px.bar(chart_df, x=x_axis, y=y_axis, title=title)

            elif chart_type == "pie":
                if x_axis not in df.columns:
                    continue

                chart_df = df[x_axis].value_counts().reset_index()
                chart_df.columns = [x_axis, "count"]

                fig = px.pie(chart_df, names=x_axis, values="count", title=title)

            elif chart_type == "line":
                if x_axis not in df.columns or y_axis not in df.columns:
                    continue

                chart_df = aggregate_time_metric(
                    df=df,
                    date_column=x_axis,
                    metric_column=y_axis,
                    aggregation=aggregation,
                    time_grain=time_grain,
                )

                fig = px.line(chart_df, x=x_axis, y=y_axis, title=title)

            elif chart_type == "scatter":
                if x_axis not in df.columns or y_axis not in df.columns:
                    continue

                fig = px.scatter(df, x=x_axis, y=y_axis, title=title)

            else:
                continue

            charts.append({
                "chart_type": chart_type,
                "title": title,
                "figure": pio.to_json(fig),
            })

        except Exception:
            continue

    return charts


def aggregate_category_metric(
    df: pd.DataFrame,
    category_column: str,
    metric_column: str,
    aggregation: str = "sum",
) -> pd.DataFrame:
    if aggregation == "average":
        grouped = df.groupby(category_column, dropna=False)[metric_column].mean().reset_index()
    elif aggregation == "maximum":
        grouped = df.groupby(category_column, dropna=False)[metric_column].max().reset_index()
    elif aggregation == "minimum":
        grouped = df.groupby(category_column, dropna=False)[metric_column].min().reset_index()
    elif aggregation == "count":
        grouped = df.groupby(category_column, dropna=False)[metric_column].count().reset_index()
    else:
        grouped = df.groupby(category_column, dropna=False)[metric_column].sum().reset_index()

    return grouped.sort_values(by=metric_column, ascending=False).head(20)


def aggregate_time_metric(
    df: pd.DataFrame,
    date_column: str,
    metric_column: str,
    aggregation: str = "sum",
    time_grain: str = "month",
) -> pd.DataFrame:
    chart_df = df.copy()
    chart_df[date_column] = pd.to_datetime(chart_df[date_column], errors="coerce")
    chart_df = chart_df.dropna(subset=[date_column])

    if time_grain == "day":
        chart_df["_period"] = chart_df[date_column].dt.to_period("D").dt.to_timestamp()
    elif time_grain == "quarter":
        chart_df["_period"] = chart_df[date_column].dt.to_period("Q").dt.to_timestamp()
    elif time_grain == "year":
        chart_df["_period"] = chart_df[date_column].dt.to_period("Y").dt.to_timestamp()
    else:
        chart_df["_period"] = chart_df[date_column].dt.to_period("M").dt.to_timestamp()

    if aggregation == "average":
        grouped = chart_df.groupby("_period")[metric_column].mean().reset_index()
    elif aggregation == "maximum":
        grouped = chart_df.groupby("_period")[metric_column].max().reset_index()
    elif aggregation == "minimum":
        grouped = chart_df.groupby("_period")[metric_column].min().reset_index()
    elif aggregation == "count":
        grouped = chart_df.groupby("_period")[metric_column].count().reset_index()
    else:
        grouped = chart_df.groupby("_period")[metric_column].sum().reset_index()

    grouped = grouped.rename(columns={"_period": date_column})
    return grouped.sort_values(by=date_column)


def generate_rule_based_chart_recommendations(df: pd.DataFrame, column_profile: dict) -> list:
    charts = []

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])
    date_columns = column_profile.get("date_columns", [])

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

    return charts


def generate_rule_based_plotly_charts(df: pd.DataFrame, column_profile: dict) -> list:
    charts = []

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])
    date_columns = column_profile.get("date_columns", [])

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

        for num_col in numeric_columns[:2]:
            trend = aggregate_time_metric(
                df=df,
                date_column=date_col,
                metric_column=num_col,
                aggregation=(
                    "average"
                    if num_col in column_profile.get("performance_metric_columns", [])
                    else "sum"
                ),
                time_grain="month",
            )

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