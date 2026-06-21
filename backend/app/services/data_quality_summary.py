def generate_data_quality_summary(
    rows: int,
    columns: int,
    total_missing_values: int,
    duplicate_rows: int,
    data_quality_score: int,
) -> dict:

    total_cells = max(rows * columns, 1)

    missing_percentage = round(
        (total_missing_values / total_cells) * 100,
        2,
    )

    duplicate_percentage = round(
        (duplicate_rows / max(rows, 1)) * 100,
        2,
    )

    if data_quality_score >= 90:
        quality_assessment = "High"

    elif data_quality_score >= 70:
        quality_assessment = "Medium"

    else:
        quality_assessment = "Low"

    return {
        "quality_score": data_quality_score,
        "quality_assessment": quality_assessment,
        "missing_percentage": missing_percentage,
        "duplicate_percentage": duplicate_percentage,
    }