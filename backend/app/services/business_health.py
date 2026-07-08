def generate_business_health(
    rows: int,
    columns: int,
    data_quality_summary: dict,
    statistical_profile: dict,
    business_metrics: dict,
) -> dict:
    strengths = []
    concerns = []

    quality_score = data_quality_summary.get("quality_score", 0)
    missing_percentage = data_quality_summary.get("missing_percentage", 0)
    duplicate_percentage = data_quality_summary.get("duplicate_percentage", 0)

    if quality_score >= 90:
        strengths.append("Data quality is strong enough to support reliable decision-making.")
    elif quality_score >= 70:
        concerns.append("Data quality is moderate and should be reviewed before major decisions.")
    else:
        concerns.append("Data quality is weak and may reduce confidence in the analysis.")

    if missing_percentage > 5:
        concerns.append(
            f"Missing values affect {missing_percentage}% of the dataset, which may limit analysis reliability."
        )

    if duplicate_percentage > 2:
        concerns.append(
            f"Duplicate records represent {duplicate_percentage}% of the dataset and may distort results."
        )

    numeric_profiles = statistical_profile.get("numeric_profiles", [])

    high_outlier_columns = [
        item for item in numeric_profiles
        if item.get("outlier_percentage", 0) >= 5
    ]

    if high_outlier_columns:
        concerns.append(
            "Some key numeric metrics contain notable outliers that may require management review."
        )
    elif numeric_profiles:
        strengths.append("Numeric metrics appear reasonably stable with limited extreme outliers.")

    category_metrics = business_metrics.get("category_metrics", [])

    concentrated_categories = [
        item for item in category_metrics
        if item.get("top_value_count", 0) / max(rows, 1) >= 0.4
    ]

    if concentrated_categories:
        concerns.append(
            "One or more categories show concentration, which may indicate dependency or imbalance."
        )

    if rows >= 100 and columns >= 3:
        strengths.append("Dataset has enough structure to support meaningful business analysis.")
    else:
        concerns.append("Dataset is relatively small, so insights should be interpreted cautiously.")

    if len(strengths) >= 2 and len(concerns) <= 1:
        overall_health = "Good"
        confidence = "High"
    elif len(concerns) <= 3:
        overall_health = "Moderate"
        confidence = "Medium"
    else:
        overall_health = "Needs attention"
        confidence = "Low"

    return {
        "overall_health": overall_health,
        "strengths": strengths,
        "concerns": concerns,
        "confidence": confidence,
    }