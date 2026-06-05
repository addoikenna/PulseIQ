def generate_basic_insights(
    rows: int,
    columns: int,
    total_missing: int,
    duplicate_rows: int,
    column_profile: dict,
    data_quality_score: int,
) -> list:
    insights = []

    numeric_columns = column_profile.get("numeric_columns", [])
    categorical_columns = column_profile.get("categorical_columns", [])
    date_columns = column_profile.get("date_columns", [])
    id_columns = column_profile.get("id_columns", [])

    insights.append(f"The dataset contains {rows} rows and {columns} columns.")

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
        insights.append("There are no missing values detected in the dataset.")

    if duplicate_rows > 0:
        insights.append(
            f"There are {duplicate_rows} duplicate rows. You may need to remove them to avoid misleading results."
        )
    else:
        insights.append("No duplicate rows were detected.")

    if numeric_columns:
        insights.append(
            f"Detected {len(numeric_columns)} numeric metric column(s): {', '.join(numeric_columns)}."
        )

    if categorical_columns:
        insights.append(
            f"Detected {len(categorical_columns)} categorical column(s): {', '.join(categorical_columns)}."
        )

    if date_columns:
        insights.append(
            f"Detected date column(s): {', '.join(date_columns)}. These can support trend analysis."
        )

    if id_columns:
        insights.append(
            f"Detected possible identifier column(s): {', '.join(id_columns)}. These are usually excluded from KPI and chart calculations."
        )

    return insights