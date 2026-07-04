import pandas as pd


def generate_statistical_profile(
    df: pd.DataFrame,
    column_profile: dict,
) -> dict:
    numeric_columns = column_profile.get("numeric_columns", [])

    blocked_columns = set(
        column_profile.get("id_columns", [])
        + column_profile.get("contact_columns", [])
    )

    numeric_profiles = []
    correlations = []

    for col in numeric_columns:
        if col in blocked_columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce").dropna()

        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        outliers = series[
            (series < lower_bound) | (series > upper_bound)
        ]

        mean = float(series.mean())
        median = float(series.median())
        std_dev = float(series.std()) if len(series) > 1 else 0.0

        numeric_profiles.append({
            "column": col,
            "count": int(series.count()),
            "mean": round(mean, 2),
            "median": round(median, 2),
            "standard_deviation": round(std_dev, 2),
            "minimum": round(float(series.min()), 2),
            "maximum": round(float(series.max()), 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(float(iqr), 2),
            "outlier_count": int(outliers.count()),
            "outlier_percentage": round((outliers.count() / max(series.count(), 1)) * 100, 2),
            "distribution_shape": infer_distribution_shape(mean, median, std_dev),
        })

    valid_numeric_columns = [
        col for col in numeric_columns
        if col not in blocked_columns
    ]

    if len(valid_numeric_columns) >= 2:
        corr_matrix = df[valid_numeric_columns].corr(numeric_only=True)

        for i, col_a in enumerate(valid_numeric_columns):
            for col_b in valid_numeric_columns[i + 1:]:
                corr_value = corr_matrix.loc[col_a, col_b]

                if pd.isna(corr_value):
                    continue

                corr_record = {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": round(float(corr_value), 3),
                    "strength": infer_correlation_strength(float(corr_value)),
                }

                correlations.append(corr_record)

                correlations = sorted(
                    correlations,
                    key=lambda x: abs(x["correlation"]),
                    reverse=True,
                )[:10]

                numeric_profiles = sorted(
                    numeric_profiles,
                    key=lambda x: (
                        x["outlier_percentage"],
                        x["standard_deviation"]
                    ),
                    reverse=True,
                )[:12]

    return {
        "numeric_profiles": numeric_profiles,
        "correlations": correlations,
    }


def infer_distribution_shape(mean: float, median: float, std_dev: float) -> str:
    if std_dev == 0:
        return "constant"

    difference_ratio = abs(mean - median) / std_dev

    if difference_ratio < 0.1:
        return "roughly symmetric"

    if mean > median:
        return "right-skewed"

    return "left-skewed"


def infer_correlation_strength(correlation: float) -> str:
    absolute_corr = abs(correlation)

    if absolute_corr >= 0.8:
        return "very strong"

    if absolute_corr >= 0.6:
        return "strong"

    if absolute_corr >= 0.4:
        return "moderate"

    if absolute_corr >= 0.2:
        return "weak"

    return "very weak"