import re

import pandas as pd


# Only parse columns with date-like names; sampling avoids full-column parsing.
_DATE_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:date|datetime|timestamp|dob|birthday)(?:$|_)|"
    r"(?:^|_)(?:created|updated|modified|enrolled|registered)_at$"
)
_DATE_VALUE_PATTERN = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}|"
    r"\d{1,2}[-/ ]+[A-Za-z]{3,9}[-/ ,]+\d{4}|"
    r"[A-Za-z]{3,9}[-/ ]+\d{1,2},?[-/ ]+\d{4})"
    r"(?:[ T].*)?$"
)


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

        if _DATE_NAME_PATTERN.search(col_lower) and looks_like_date(series):
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
    """Check a small sample for recognizable dates, without altering the column.

    We deliberately avoid converting every text column and never interpret bare
    numbers (such as student IDs or GPAs) as epoch-based timestamps.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    sample = series.dropna().astype(str).str.strip()
    sample = sample[~sample.str.lower().isin({"", "-", "unknown", "null", "none", "n/a", "na"})]
    if sample.empty:
        return False

    sample = sample.iloc[:64]
    candidates = sample.map(lambda value: bool(_DATE_VALUE_PATTERN.fullmatch(value)))
    if candidates.mean() < 0.7:
        return False

    # Pandas >=2 supports mixed formats directly, including YYYY-MM-DD and
    # 17-May-2014 in the same column, without the dateutil inference warning.
    try:
        parsed = pd.to_datetime(sample[candidates], format="mixed", errors="coerce")
    except ValueError:
        # Compatibility fallback for pandas versions predating format="mixed".
        # Parse common explicit formats rather than triggering format inference.
        parsed = pd.Series(pd.NaT, index=sample[candidates].index, dtype="datetime64[ns]")
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
                    "%m/%d/%Y", "%d-%b-%Y", "%d-%B-%Y"):
            current = pd.to_datetime(sample[candidates], format=fmt, errors="coerce")
            parsed = parsed.fillna(current)

    return int(parsed.notna().sum()) / len(sample) >= 0.7


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