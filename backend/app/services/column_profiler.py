import pandas as pd


def profile_columns(df: pd.DataFrame) -> dict:
    profile = {
        "date_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "id_columns": [],
        "text_columns": [],
        "boolean_columns": [],
        "unknown_columns": [],
    }

    row_count = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        unique_count = non_null.nunique()
        unique_ratio = unique_count / row_count

        if pd.api.types.is_bool_dtype(series):
            profile["boolean_columns"].append(col)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            profile["date_columns"].append(col)
            continue

        if pd.api.types.is_numeric_dtype(series):
            col_lower = col.lower()

            if (
                "id" in col_lower
                or unique_ratio > 0.9
                and unique_count > 10
            ):
                profile["id_columns"].append(col)
            else:
                profile["numeric_columns"].append(col)

            continue

        # Try date detection for object/string columns
        converted_date = pd.to_datetime(series, errors="coerce")
        date_ratio = converted_date.notna().sum() / row_count

        if date_ratio >= 0.7:
            profile["date_columns"].append(col)
            continue

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if 2 <= unique_count <= 30:
                profile["categorical_columns"].append(col)
            elif unique_ratio > 0.9 and unique_count > 10:
                profile["id_columns"].append(col)
            else:
                profile["text_columns"].append(col)

            continue

        profile["unknown_columns"].append(col)

    return profile