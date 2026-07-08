def detect_business_drivers(
    rows: int,
    business_metrics: dict,
    statistical_profile: dict,
) -> dict:
    drivers = []

    # Category concentration drivers
    for item in business_metrics.get("category_metrics", []):
        column = item.get("column")
        top_value = item.get("top_value")
        top_count = item.get("top_value_count", 0)

        share = round((top_count / max(rows, 1)) * 100, 2)

        if share >= 30:
            drivers.append({
                "driver_type": "category_concentration",
                "driver": f"{top_value} in {column}",
                "evidence": f"{top_value} represents {share}% of records in {column}.",
                "business_meaning": "This segment has strong influence on overall performance or resource allocation.",
                "strength": "high" if share >= 50 else "medium",
            })

    # Numeric scale drivers
    for item in business_metrics.get("numeric_metrics", []):
        column = item.get("column")
        total = item.get("sum")
        average = item.get("average")
        maximum = item.get("maximum")

        if total is not None:
            drivers.append({
                "driver_type": "numeric_scale",
                "driver": column,
                "evidence": f"{column} has total value {total}, average {average}, and maximum {maximum}.",
                "business_meaning": "This metric is a major measurable contributor in the dataset.",
                "strength": "medium",
            })

    # Variability drivers
    for item in statistical_profile.get("numeric_profiles", []):
        column = item.get("column")
        std_dev = item.get("standard_deviation", 0)
        mean = item.get("mean", 0)
        outlier_percentage = item.get("outlier_percentage", 0)
        distribution_shape = item.get("distribution_shape")

        if mean and std_dev / abs(mean) >= 0.4:
            drivers.append({
                "driver_type": "high_variability",
                "driver": column,
                "evidence": f"{column} has high variability relative to its average.",
                "business_meaning": "Performance or values may differ meaningfully across records or segments.",
                "strength": "high" if std_dev / abs(mean) >= 0.7 else "medium",
            })

        if outlier_percentage >= 5:
            drivers.append({
                "driver_type": "outlier_influence",
                "driver": column,
                "evidence": f"{outlier_percentage}% of values in {column} are statistical outliers.",
                "business_meaning": "A small set of unusual records may be influencing overall results.",
                "strength": "high" if outlier_percentage >= 10 else "medium",
            })

        if distribution_shape in ["right-skewed", "left-skewed"]:
            drivers.append({
                "driver_type": "distribution_shape",
                "driver": column,
                "evidence": f"{column} distribution is {distribution_shape}.",
                "business_meaning": "The average may not fully represent typical performance.",
                "strength": "medium",
            })

    # Correlation drivers
    for item in statistical_profile.get("correlations", []):
        strength = item.get("strength")
        correlation = item.get("correlation")

        if strength in ["strong", "very strong"]:
            drivers.append({
                "driver_type": "metric_relationship",
                "driver": f"{item.get('column_a')} and {item.get('column_b')}",
                "evidence": f"Correlation between {item.get('column_a')} and {item.get('column_b')} is {correlation}.",
                "business_meaning": "These metrics may move together and should be analyzed jointly.",
                "strength": strength,
            })

    return {
        "major_drivers": drivers[:10]
    }