from typing import Any

import pandas as pd


MINIMUM_ROWS_CLASSIFICATION = 100
MINIMUM_ROWS_REGRESSION = 100
RECOMMENDED_ROWS = 300

MAX_TARGET_MISSING_PERCENTAGE = 30.0
MAX_FEATURE_MISSING_PERCENTAGE = 70.0
MAX_CLASSIFICATION_CLASSES = 20
HIGH_CARDINALITY_PERCENTAGE = 80.0


IDENTIFIER_KEYWORDS = {
    "id",
    "uuid",
    "identifier",
    "reference",
    "ref",
    "invoice_number",
    "invoice_no",
    "account_number",
    "transaction_number",
    "student_number",
    "customer_number",
}

PERSONAL_INFORMATION_KEYWORDS = {
    "email",
    "phone",
    "mobile",
    "address",
    "full_name",
    "firstname",
    "first_name",
    "lastname",
    "last_name",
    "contact",
}


def calculate_missing_percentage(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0

    return round(
        float(series.isna().mean() * 100),
        2,
    )


def is_constant_column(series: pd.Series) -> bool:
    return series.nunique(dropna=True) <= 1


def is_high_cardinality(
    series: pd.Series,
    total_rows: int,
) -> bool:
    if total_rows == 0:
        return False

    unique_percentage = (
        series.nunique(dropna=True) / total_rows
    ) * 100

    return unique_percentage >= HIGH_CARDINALITY_PERCENTAGE


def normalize_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def looks_like_identifier(
    column: str,
    series: pd.Series,
    total_rows: int,
) -> bool:
    normalized = normalize_column_name(column)

    if normalized in IDENTIFIER_KEYWORDS:
        return True

    if normalized.endswith("_id"):
        return True

    if normalized.endswith("_uuid"):
        return True

    if normalized.startswith("id_"):
        return True

    if total_rows > 0:
        uniqueness_ratio = (
            series.nunique(dropna=True) / total_rows
        )

        identifier_terms = (
            "id",
            "number",
            "reference",
            "code",
            "serial",
            "key",
        )

        if (
            uniqueness_ratio >= 0.95
            and any(term in normalized for term in identifier_terms)
        ):
            return True

    return False


def looks_like_personal_information(column: str) -> bool:
    normalized = normalize_column_name(column)

    return any(
        keyword in normalized
        for keyword in PERSONAL_INFORMATION_KEYWORDS
    )


def is_boolean_like(series: pd.Series) -> bool:
    values = {
        str(value).strip().lower()
        for value in series.dropna().unique()
    }

    known_binary_values = {
        "yes",
        "no",
        "true",
        "false",
        "1",
        "0",
        "y",
        "n",
        "pass",
        "fail",
        "passed",
        "failed",
        "active",
        "inactive",
        "churned",
        "retained",
        "delayed",
        "on time",
        "on_time",
    }

    return (
        len(values) == 2
        and values.issubset(known_binary_values)
    )


def infer_problem_type(
    series: pd.Series,
) -> str:
    clean_series = series.dropna()

    if clean_series.empty:
        return "unsupported"

    unique_values = clean_series.nunique()

    if unique_values == 2:
        return "binary_classification"

    if (
        not pd.api.types.is_numeric_dtype(clean_series)
        and 3 <= unique_values <= MAX_CLASSIFICATION_CLASSES
    ):
        return "multiclass_classification"

    if (
        pd.api.types.is_numeric_dtype(clean_series)
        and unique_values > MAX_CLASSIFICATION_CLASSES
    ):
        return "regression"

    if (
        pd.api.types.is_numeric_dtype(clean_series)
        and 3 <= unique_values <= MAX_CLASSIFICATION_CLASSES
    ):
        return "multiclass_classification"

    return "unsupported"


def build_class_distribution(
    series: pd.Series,
) -> dict[str, int] | None:
    clean_series = series.dropna()

    unique_values = clean_series.nunique()

    if unique_values < 2:
        return None

    if unique_values > MAX_CLASSIFICATION_CLASSES:
        return None

    counts = clean_series.value_counts(dropna=True)

    return {
        str(index): int(value)
        for index, value in counts.items()
    }


def detect_class_imbalance(
    class_distribution: dict[str, int] | None,
) -> bool:
    if not class_distribution:
        return False

    counts = list(class_distribution.values())

    if len(counts) < 2:
        return False

    largest_class = max(counts)
    smallest_class = min(counts)

    if smallest_class == 0:
        return True

    return largest_class / smallest_class >= 3


def assess_target_candidate(
    column: str,
    series: pd.Series,
    total_rows: int,
) -> dict[str, Any] | None:
    missing_percentage = calculate_missing_percentage(series)
    unique_values = int(series.nunique(dropna=True))

    if unique_values < 2:
        return None

    if missing_percentage > MAX_TARGET_MISSING_PERCENTAGE:
        return None

    if looks_like_identifier(
        column=column,
        series=series,
        total_rows=total_rows,
    ):
        return None

    if looks_like_personal_information(column):
        return None

    problem_type = infer_problem_type(series)

    if problem_type == "unsupported":
        return None

    cautions = []
    class_distribution = None

    if problem_type in {
        "binary_classification",
        "multiclass_classification",
    }:
        class_distribution = build_class_distribution(series)

        if detect_class_imbalance(class_distribution):
            cautions.append(
                "The target classes appear imbalanced. "
                "Model evaluation should prioritize precision, recall, "
                "F1 score, and class-level performance."
            )

        if total_rows < MINIMUM_ROWS_CLASSIFICATION:
            cautions.append(
                f"The dataset contains fewer than "
                f"{MINIMUM_ROWS_CLASSIFICATION} rows recommended "
                "for classification."
            )

        if unique_values == 2:
            reason = (
                "The column contains two meaningful outcome values "
                "and is suitable for binary classification."
            )
        else:
            reason = (
                f"The column contains {unique_values} outcome groups "
                "and is suitable for multiclass classification."
            )

    else:
        if total_rows < MINIMUM_ROWS_REGRESSION:
            cautions.append(
                f"The dataset contains fewer than "
                f"{MINIMUM_ROWS_REGRESSION} rows recommended "
                "for regression."
            )

        reason = (
            "The column is numeric and contains enough distinct "
            "values to be considered a regression target."
        )

    if missing_percentage > 10:
        cautions.append(
            f"The target contains {missing_percentage}% missing values."
        )

    if total_rows >= RECOMMENDED_ROWS and not cautions:
        confidence = "high"
    elif total_rows >= 100:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "column": column,
        "problem_type": problem_type,
        "confidence": confidence,
        "reason": reason,
        "unique_values": unique_values,
        "missing_percentage": missing_percentage,
        "class_distribution": class_distribution,
        "cautions": cautions,
    }


def assess_excluded_column(
    column: str,
    series: pd.Series,
    total_rows: int,
) -> dict[str, str] | None:
    missing_percentage = calculate_missing_percentage(series)

    if looks_like_personal_information(column):
        return {
            "column": column,
            "reason": (
                "The column appears to contain personal or contact "
                "information and should not be used as a predictor."
            ),
            "category": "personal_information",
        }

    if looks_like_identifier(
        column=column,
        series=series,
        total_rows=total_rows,
    ):
        return {
            "column": column,
            "reason": (
                "The column appears to be a record identifier and "
                "is unlikely to provide meaningful predictive value."
            ),
            "category": "identifier",
        }

    if is_constant_column(series):
        return {
            "column": column,
            "reason": (
                "The column contains one or no meaningful values "
                "and cannot contribute to a predictive model."
            ),
            "category": "constant",
        }

    if missing_percentage > MAX_FEATURE_MISSING_PERCENTAGE:
        return {
            "column": column,
            "reason": (
                f"The column contains {missing_percentage}% missing "
                "values and is not reliable enough for modelling."
            ),
            "category": "mostly_missing",
        }

    if is_high_cardinality(
        series=series,
        total_rows=total_rows,
    ):
        if not pd.api.types.is_numeric_dtype(series):
            return {
                "column": column,
                "reason": (
                    "The column contains an unusually high number of "
                    "unique values and may behave like an identifier."
                ),
                "category": "high_cardinality",
            }

    return None


def calculate_readiness_score(
    total_rows: int,
    usable_features: int,
    missing_percentage: float,
    duplicate_percentage: float,
    candidate_target_count: int,
) -> float:
    score = 100.0

    if total_rows < 50:
        score -= 45
    elif total_rows < 100:
        score -= 30
    elif total_rows < RECOMMENDED_ROWS:
        score -= 10

    if usable_features < 2:
        score -= 40
    elif usable_features < 4:
        score -= 15

    if missing_percentage >= 50:
        score -= 35
    elif missing_percentage >= 25:
        score -= 20
    elif missing_percentage >= 10:
        score -= 10
    elif missing_percentage >= 5:
        score -= 5

    if duplicate_percentage >= 20:
        score -= 20
    elif duplicate_percentage >= 10:
        score -= 10
    elif duplicate_percentage >= 2:
        score -= 5

    if candidate_target_count == 0:
        score -= 40

    return round(
        max(0.0, min(100.0, score)),
        2,
    )


def determine_readiness_status(
    readiness_score: float,
    total_rows: int,
    usable_features: int,
    candidate_target_count: int,
) -> str:
    if (
        total_rows < 50
        or usable_features < 2
        or candidate_target_count == 0
        or readiness_score < 50
    ):
        return "not_ready"

    if readiness_score < 80:
        return "ready_with_cautions"

    return "ready"


def rank_target_candidate(
    candidate: dict[str, Any],
) -> tuple[int, int, float]:
    confidence_order = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    problem_type_order = {
        "binary_classification": 3,
        "regression": 2,
        "multiclass_classification": 1,
        "unsupported": 0,
    }

    return (
        confidence_order.get(candidate.get("confidence"), 0),
        problem_type_order.get(candidate.get("problem_type"), 0),
        -candidate.get("missing_percentage", 0),
    )


def assess_predictive_readiness(
    df: pd.DataFrame,
    analysis_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    if df is None or df.empty:
        raise ValueError(
            "The dataset is empty and cannot be assessed for prediction."
        )

    total_rows = int(len(df))
    total_columns = int(len(df.columns))

    duplicate_rows = int(df.duplicated().sum())

    total_cells = max(total_rows * total_columns, 1)
    total_missing_values = int(df.isna().sum().sum())

    missing_percentage = round(
        (total_missing_values / total_cells) * 100,
        2,
    )

    duplicate_percentage = round(
        (duplicate_rows / max(total_rows, 1)) * 100,
        2,
    )

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    date_columns = [
        column
        for column in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[column])
    ]

    categorical_columns = [
        column
        for column in df.columns
        if (
            column not in numeric_columns
            and column not in date_columns
        )
    ]

    candidate_targets = []
    excluded_columns = []
    usable_feature_columns = []

    for column in df.columns:
        series = df[column]

        excluded = assess_excluded_column(
            column=column,
            series=series,
            total_rows=total_rows,
        )

        if excluded:
            excluded_columns.append(excluded)
            continue

        usable_feature_columns.append(column)

        target_candidate = assess_target_candidate(
            column=column,
            series=series,
            total_rows=total_rows,
        )

        if target_candidate:
            candidate_targets.append(target_candidate)

    candidate_targets = sorted(
        candidate_targets,
        key=rank_target_candidate,
        reverse=True,
    )

    usable_features = len(usable_feature_columns)

    usable_rows = int(
        len(
            df.dropna(
                how="all",
                subset=usable_feature_columns,
            )
        )
        if usable_feature_columns
        else 0
    )

    warnings = []
    recommendations = []

    if total_rows < MINIMUM_ROWS_CLASSIFICATION:
        warnings.append(
            "The dataset contains fewer than 100 rows. "
            "Any predictive model would have limited reliability."
        )

    elif total_rows < RECOMMENDED_ROWS:
        warnings.append(
            "The dataset is usable for exploratory modelling, "
            "but more rows would improve model reliability."
        )

    if duplicate_rows > 0:
        warnings.append(
            f"The dataset contains {duplicate_rows} duplicate row(s)."
        )

        recommendations.append(
            "Review duplicate rows before model training to prevent "
            "repeated records from influencing validation results."
        )

    if missing_percentage > 10:
        warnings.append(
            f"Missing values represent {missing_percentage}% "
            "of the dataset."
        )

        recommendations.append(
            "Review columns with substantial missing values and confirm "
            "the appropriate imputation strategy."
        )

    elif missing_percentage > 0:
        warnings.append(
            "Some predictor or target values are missing."
        )

    high_missing_columns = [
        column
        for column in df.columns
        if calculate_missing_percentage(df[column]) >
        MAX_FEATURE_MISSING_PERCENTAGE
    ]

    if high_missing_columns:
        recommendations.append(
            "Exclude or carefully review columns where most values "
            "are missing."
        )

    if not candidate_targets:
        warnings.append(
            "No suitable classification or regression target was detected."
        )

        recommendations.append(
            "Confirm that the dataset contains a meaningful outcome "
            "column with sufficient non-missing values."
        )

    if usable_features < 2:
        warnings.append(
            "Too few usable predictor columns remain after exclusions."
        )

        recommendations.append(
            "Provide additional meaningful predictor columns before "
            "training a model."
        )

    if excluded_columns:
        recommendations.append(
            "Review excluded identifier, personal-information, constant, "
            "and high-cardinality columns before training."
        )

    if candidate_targets:
        recommendations.append(
            "Confirm the selected target and remove any columns that "
            "would only become known after the target outcome occurs."
        )

    readiness_score = calculate_readiness_score(
        total_rows=total_rows,
        usable_features=usable_features,
        missing_percentage=missing_percentage,
        duplicate_percentage=duplicate_percentage,
        candidate_target_count=len(candidate_targets),
    )

    readiness_status = determine_readiness_status(
        readiness_score=readiness_score,
        total_rows=total_rows,
        usable_features=usable_features,
        candidate_target_count=len(candidate_targets),
    )

    recommended_target = None
    recommended_problem_type = None

    if candidate_targets:
        recommended_target = candidate_targets[0]["column"]
        recommended_problem_type = candidate_targets[0][
            "problem_type"
        ]

    return {
        "status": "success",
        "readiness_status": readiness_status,
        "analysis_id": analysis_id,
        "filename": filename,
        "dataset_summary": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "usable_rows": usable_rows,
            "usable_features": usable_features,
            "duplicate_rows": duplicate_rows,
            "numeric_columns": len(numeric_columns),
            "categorical_columns": len(categorical_columns),
            "date_columns": len(date_columns),
            "total_missing_values": total_missing_values,
            "missing_percentage": missing_percentage,
        },
        "candidate_targets": candidate_targets,
        "excluded_columns": excluded_columns,
        "recommended_target": recommended_target,
        "recommended_problem_type": recommended_problem_type,
        "warnings": list(dict.fromkeys(warnings)),
        "recommendations": list(dict.fromkeys(recommendations)),
        "readiness_score": readiness_score,
        "minimum_requirements": {
            "minimum_rows_classification": (
                MINIMUM_ROWS_CLASSIFICATION
            ),
            "minimum_rows_regression": MINIMUM_ROWS_REGRESSION,
            "recommended_rows": RECOMMENDED_ROWS,
            "maximum_target_missing_percentage": (
                MAX_TARGET_MISSING_PERCENTAGE
            ),
            "maximum_feature_missing_percentage": (
                MAX_FEATURE_MISSING_PERCENTAGE
            ),
            "maximum_classification_classes": (
                MAX_CLASSIFICATION_CLASSES
            ),
        },
    }