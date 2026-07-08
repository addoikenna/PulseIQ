def detect_business_opportunities(
    rows: int,
    business_health: dict,
    business_drivers: dict,
    business_risks: dict,
    business_metrics: dict,
    statistical_profile: dict,
) -> dict:
    opportunities = []

    if business_health.get("overall_health") == "Good":
        opportunities.append({
            "opportunity_type": "scale_strength",
            "opportunity": "Build on the current healthy business position.",
            "evidence": "Business health is classified as Good.",
            "potential_impact": "medium",
        })

    for driver in business_drivers.get("major_drivers", []):
        driver_type = driver.get("driver_type")

        if driver_type == "category_concentration":
            opportunities.append({
                "opportunity_type": "segment_focus",
                "opportunity": "Focus strategy on the dominant segment while monitoring dependency risk.",
                "evidence": driver.get("evidence"),
                "potential_impact": "high" if driver.get("strength") == "high" else "medium",
            })

        if driver_type == "numeric_scale":
            opportunities.append({
                "opportunity_type": "metric_optimization",
                "opportunity": f"Optimize performance around {driver.get('driver')}.",
                "evidence": driver.get("evidence"),
                "potential_impact": "medium",
            })

        if driver_type == "metric_relationship":
            opportunities.append({
                "opportunity_type": "relationship_leverage",
                "opportunity": "Use related metrics together for better planning and forecasting.",
                "evidence": driver.get("evidence"),
                "potential_impact": "medium",
            })

    for risk in business_risks.get("business_risks", []):
        risk_type = risk.get("risk_type")

        if risk_type in ["data_quality", "missing_data", "duplicate_data"]:
            opportunities.append({
                "opportunity_type": "data_governance",
                "opportunity": "Improve data governance to increase confidence in future analysis.",
                "evidence": risk.get("evidence"),
                "potential_impact": "medium",
            })

        if risk_type == "concentration_risk":
            opportunities.append({
                "opportunity_type": "diversification",
                "opportunity": "Reduce dependency on one dominant segment by growing alternative segments.",
                "evidence": risk.get("evidence"),
                "potential_impact": "high",
            })

        if risk_type in ["outlier_risk", "extreme_outliers"]:
            opportunities.append({
                "opportunity_type": "outlier_review",
                "opportunity": "Review unusual records to identify exceptional performers, errors, or one-off events.",
                "evidence": risk.get("evidence"),
                "potential_impact": "medium",
            })

        if risk_type == "variability_risk":
            opportunities.append({
                "opportunity_type": "standardization",
                "opportunity": "Investigate variability and standardize processes, pricing, compensation, or operations where needed.",
                "evidence": risk.get("evidence"),
                "potential_impact": "medium",
            })

    for category in business_metrics.get("category_metrics", []):
        unique_values = category.get("unique_values", 0)

        if unique_values >= 3:
            opportunities.append({
                "opportunity_type": "segment_comparison",
                "opportunity": f"Compare performance across {category.get('column')} segments to identify high-performing groups.",
                "evidence": f"{category.get('column')} contains {unique_values} unique segments.",
                "potential_impact": "medium",
            })

    for profile in statistical_profile.get("numeric_profiles", []):
        column = profile.get("column")
        distribution_shape = profile.get("distribution_shape")

        if distribution_shape in ["right-skewed", "left-skewed"]:
            opportunities.append({
                "opportunity_type": "distribution_analysis",
                "opportunity": f"Segment {column} further to understand what is driving the uneven distribution.",
                "evidence": f"{column} distribution is {distribution_shape}.",
                "potential_impact": "medium",
            })

    return {
        "business_opportunities": opportunities[:10]
    }