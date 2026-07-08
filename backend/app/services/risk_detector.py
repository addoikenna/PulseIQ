def detect_business_risks(
    rows: int,
    data_quality_summary: dict,
    business_health: dict,
    business_drivers: dict,
    statistical_profile: dict,
) -> dict:
    risks = []

    quality_score = data_quality_summary.get("quality_score", 0)
    missing_percentage = data_quality_summary.get("missing_percentage", 0)
    duplicate_percentage = data_quality_summary.get("duplicate_percentage", 0)

    if quality_score < 70:
        risks.append({
            "risk_type": "data_quality",
            "risk": "Low data quality may reduce confidence in business decisions.",
            "evidence": f"Data quality score is {quality_score}.",
            "severity": "high",
        })

    elif quality_score < 90:
        risks.append({
            "risk_type": "data_quality",
            "risk": "Moderate data quality may require review before major decisions.",
            "evidence": f"Data quality score is {quality_score}.",
            "severity": "medium",
        })

    if missing_percentage > 5:
        risks.append({
            "risk_type": "missing_data",
            "risk": "Missing data may limit the reliability of some insights.",
            "evidence": f"{missing_percentage}% of dataset cells are missing.",
            "severity": "high" if missing_percentage >= 15 else "medium",
        })

    if duplicate_percentage > 2:
        risks.append({
            "risk_type": "duplicate_data",
            "risk": "Duplicate records may distort totals, averages, and trends.",
            "evidence": f"{duplicate_percentage}% of records are duplicates.",
            "severity": "high" if duplicate_percentage >= 10 else "medium",
        })

    for driver in business_drivers.get("major_drivers", []):
        if driver.get("driver_type") == "category_concentration":
            risks.append({
                "risk_type": "concentration_risk",
                "risk": "Performance or activity may be overly dependent on one segment.",
                "evidence": driver.get("evidence"),
                "severity": "high" if driver.get("strength") == "high" else "medium",
            })

        if driver.get("driver_type") == "outlier_influence":
            risks.append({
                "risk_type": "outlier_risk",
                "risk": "Unusual records may be influencing overall results.",
                "evidence": driver.get("evidence"),
                "severity": "high" if driver.get("strength") == "high" else "medium",
            })

        if driver.get("driver_type") == "high_variability":
            risks.append({
                "risk_type": "variability_risk",
                "risk": "High variability may indicate inconsistent performance, pricing, compensation, or operations.",
                "evidence": driver.get("evidence"),
                "severity": "medium",
            })

    for profile in statistical_profile.get("numeric_profiles", []):
        column = profile.get("column")
        distribution_shape = profile.get("distribution_shape")
        outlier_percentage = profile.get("outlier_percentage", 0)

        if distribution_shape in ["right-skewed", "left-skewed"]:
            risks.append({
                "risk_type": "skewed_distribution",
                "risk": f"{column} may not be well represented by the average alone.",
                "evidence": f"{column} distribution is {distribution_shape}.",
                "severity": "medium",
            })

        if outlier_percentage >= 10:
            risks.append({
                "risk_type": "extreme_outliers",
                "risk": f"{column} contains a high proportion of outliers.",
                "evidence": f"{outlier_percentage}% of values in {column} are outliers.",
                "severity": "high",
            })

    if business_health.get("confidence") == "Low":
        risks.append({
            "risk_type": "low_confidence",
            "risk": "The overall confidence level is low, so findings should be treated cautiously.",
            "evidence": "Business health engine classified confidence as Low.",
            "severity": "high",
        })

    return {
        "business_risks": risks[:10]
    }