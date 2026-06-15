import pandas as pd


def profile_columns(df: pd.DataFrame) -> dict:
    profile = {
        "date_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "id_columns": [],
        "contact_columns": [],
        "text_columns": [],
        "boolean_columns": [],
        "financial_columns": [],
        "performance_metric_columns": [],
        "count_metric_columns": [],
        "unknown_columns": [],
    }

    row_count = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        unique_count = non_null.nunique()
        unique_ratio = unique_count / row_count
        col_lower = str(col).lower().strip()

        if is_contact_column(col_lower):
            profile["contact_columns"].append(col)
            profile["id_columns"].append(col)
            continue

        if pd.api.types.is_bool_dtype(series):
            profile["boolean_columns"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            profile["date_columns"].append(col)
            continue

        if looks_like_date(series):
            profile["date_columns"].append(col)
            continue

        if is_identifier_column(col_lower, unique_ratio, unique_count):
            profile["id_columns"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            if is_financial_metric(col_lower):
                profile["financial_columns"].append(col)
                profile["numeric_columns"].append(col)
            elif is_performance_metric(col_lower, series):
                profile["performance_metric_columns"].append(col)
                profile["numeric_columns"].append(col)
            elif is_count_metric(col_lower):
                profile["count_metric_columns"].append(col)
                profile["numeric_columns"].append(col)
            else:
                profile["numeric_columns"].append(col)

            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if 2 <= unique_count <= 50:
                profile["categorical_columns"].append(col)
            elif unique_ratio > 0.9 and unique_count > 10:
                profile["id_columns"].append(col)
            else:
                profile["text_columns"].append(col)

            continue

        profile["unknown_columns"].append(col)

    return profile


def is_identifier_column(col_lower: str, unique_ratio: float, unique_count: int) -> bool:
    id_keywords = [
        "id", "uuid", "serial", "reference", "ref", "code",
        "number", "no", "account", "invoice", "transaction",
        "student", "employee", "customer", "user", "member",
    ]

    metric_exceptions = [
        "score", "rating", "grade", "gpa", "cgpa", "sales",
        "revenue", "profit", "amount", "cost", "price",
        "quantity", "qty", "salary", "income", "expense",
        "age", "rate", "percent", "percentage", "margin",
    ]

    if any(word in col_lower for word in metric_exceptions):
        return False

    if any(word in col_lower for word in id_keywords):
        return True

    if unique_ratio > 0.98 and unique_count > 50:
        return True

    return False


def is_contact_column(col_lower: str) -> bool:
    contact_keywords = [
        "phone", "mobile", "telephone", "tel", "contact",
        "email", "mail", "address", "whatsapp",
    ]

    return any(word in col_lower for word in contact_keywords)


def looks_like_date(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False

    converted = pd.to_datetime(series, errors="coerce")
    return converted.notna().sum() / max(len(series), 1) >= 0.7


def is_financial_metric(col_lower: str) -> bool:
    financial_keywords = [
        "sales", "revenue", "profit", "income", "expense",
        "cost", "price", "amount", "salary", "wage",
        "budget", "spend", "payment", "balance", "rent",
    ]

    return any(word in col_lower for word in financial_keywords)


def is_performance_metric(col_lower: str, series: pd.Series) -> bool:
    performance_keywords = [
        "score", "rating", "grade", "gpa", "cgpa", "rate",
        "percent", "percentage", "margin", "ratio", "index",
        "satisfaction", "performance",
    ]

    if any(word in col_lower for word in performance_keywords):
        return True

    numeric = pd.to_numeric(series, errors="coerce").dropna()

    if numeric.empty:
        return False

    min_value = numeric.min()
    max_value = numeric.max()

    # Bounded numeric metrics are usually averages, scores, ratings, rates, or indexes.
    if 0 <= min_value and max_value <= 5:
        return True

    if 0 <= min_value and max_value <= 10:
        return True

    if 0 <= min_value and max_value <= 100 and "count" not in col_lower:
        return True

    return False


def is_count_metric(col_lower: str) -> bool:
    count_keywords = [
        "count", "total_students", "total_users", "total_customers",
        "total_orders", "num_", "number_of", "volume", "units",
        "quantity", "qty",
    ]

    return any(word in col_lower for word in count_keywords)