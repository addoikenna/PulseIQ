import pandas as pd


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cleaning_report = {
        "columns_renamed": {},
        "missing_markers_converted": 0,
        "text_values_trimmed": 0,
        "numeric_columns_converted": [],
        "date_columns_detected": [],
        "duplicates_detected": 0,
        "warnings": [],
    }

    cleaned_df = df.copy()

    # Clean column names
    original_columns = list(cleaned_df.columns)
    new_columns = []

    for col in original_columns:
        new_col = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        new_columns.append(new_col)

        if col != new_col:
            cleaning_report["columns_renamed"][col] = new_col

    cleaned_df.columns = new_columns

    # Trim text values
    text_columns = cleaned_df.select_dtypes(include=["object"]).columns

    for col in text_columns:
        before = cleaned_df[col].astype(str)
        after = before.str.strip()

        trimmed_count = int((before != after).sum())
        cleaning_report["text_values_trimmed"] += trimmed_count

        cleaned_df[col] = after

    # Convert common missing markers to missing values
    before_missing = int(cleaned_df.isna().sum().sum())

    missing_markers = ["", "N/A", "NA", "null", "NULL", "None", "none", "?"]
    cleaned_df = cleaned_df.replace(missing_markers, pd.NA)

    after_missing = int(cleaned_df.isna().sum().sum())
    cleaning_report["missing_markers_converted"] = max(after_missing - before_missing, 0)

    # Try converting numeric-looking columns
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            converted = pd.to_numeric(cleaned_df[col], errors="coerce")

            non_missing_original = cleaned_df[col].notna().sum()
            successful_conversions = converted.notna().sum()

            if non_missing_original > 0 and successful_conversions / non_missing_original >= 0.8:
                cleaned_df[col] = converted
                cleaning_report["numeric_columns_converted"].append(col)

    # Detect possible date columns
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == "object":
            converted = pd.to_datetime(cleaned_df[col], errors="coerce")
            non_missing_original = cleaned_df[col].notna().sum()
            successful_conversions = converted.notna().sum()

            if non_missing_original > 0 and successful_conversions / non_missing_original >= 0.7:
                cleaning_report["date_columns_detected"].append(col)

    # Detect duplicates
    duplicate_rows = int(cleaned_df.duplicated().sum())
    cleaning_report["duplicates_detected"] = duplicate_rows

    if duplicate_rows > 0:
        cleaning_report["warnings"].append(
            f"{duplicate_rows} duplicate row(s) were detected. They were not removed automatically."
        )

    missing_values = cleaned_df.isna().sum()
    columns_with_missing = missing_values[missing_values > 0]

    for col, count in columns_with_missing.items():
        cleaning_report["warnings"].append(
            f"Column '{col}' has {int(count)} missing value(s) and may need review."
        )

    return cleaned_df, cleaning_report