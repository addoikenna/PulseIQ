"""PulseIQ Data Cleaning Studio - deterministic, non-destructive cleaning engine.

Every operation works on a copy. Statistical imputation here is for exploratory
analysis only: ML preprocessing must fit imputers inside the training split.
"""
from __future__ import annotations

from typing import Any, Literal
import math
import re
import pandas as pd
from pydantic import BaseModel, Field, model_validator

Operation = Literal[
    "map_values", "trim", "case", "replace_missing", "drop_duplicates",
    "impute_missing", "convert_type", "rename_column", "drop_column",
    "replace_values", "parse_date", "flag_outliers", "clip_range",
]


class CleaningRule(BaseModel):
    operation: Operation
    column: str | None = None
    mapping: dict[str, str | float | int | bool | None] | None = None
    case_style: Literal["lower", "upper", "title"] | None = None
    missing_value: str | float | int | bool | None = None
    duplicate_columns: list[str] | None = None
    duplicate_keep: Literal["first", "last"] = "first"
    strategy: Literal["mean", "median", "mode"] | None = None
    target_type: Literal["numeric", "integer", "text", "datetime", "boolean"] | None = None
    date_format: str | None = None
    dayfirst: bool = False
    new_column: str | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    iqr_multiplier: float = Field(default=1.5, gt=0, le=10)

    @model_validator(mode="after")
    def check_rule(self):
        if self.operation != "drop_duplicates" and not self.column:
            raise ValueError(f"{self.operation} requires a column")
        if self.operation in {"map_values", "replace_values"} and not self.mapping:
            raise ValueError("A non-empty mapping is required")
        if self.operation == "case" and not self.case_style:
            raise ValueError("case_style is required")
        if self.operation == "replace_missing" and self.missing_value is None:
            raise ValueError("An explicit replacement value is required")
        if self.operation == "impute_missing" and not self.strategy:
            raise ValueError("impute_missing requires strategy: mean, median, or mode")
        if self.operation == "convert_type" and not self.target_type:
            raise ValueError("convert_type requires target_type")
        if self.operation in {"rename_column", "flag_outliers"} and not self.new_column:
            raise ValueError(f"{self.operation} requires new_column")
        if self.operation == "clip_range":
            if self.lower_bound is None and self.upper_bound is None:
                raise ValueError("clip_range requires lower_bound and/or upper_bound")
            if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
                raise ValueError("lower_bound must not exceed upper_bound")
        return self


def json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float,)) and not math.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat() if pd.notna(value) else None
    if hasattr(value, "item"):
        value = value.item()
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return value


def profile(df: pd.DataFrame, sample_size: int = 8) -> dict[str, Any]:
    columns = []
    for col in df.columns:
        series = df[col]
        freq = series.astype("string").fillna("<NULL>").value_counts(dropna=False).head(20)
        columns.append({
            "name": str(col), "data_type": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "top_values": [{"value": str(v), "count": int(n)} for v, n in freq.items()],
        })
    sample = [{str(k): json_value(v) for k, v in row.items()}
              for row in df.head(sample_size).to_dict("records")]
    return {"rows": len(df), "columns_count": len(df.columns),
            "duplicate_rows": int(df.duplicated().sum()), "columns": columns, "sample": sample}


def _text_column(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _numeric_column(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _changed_count(old: pd.Series, new: pd.Series) -> int:
    # String conversion distinguishes ordinary numeric changes and is stable for nulls.
    a = old.astype("string")
    b = new.astype("string")
    return int((a.fillna("\0") != b.fillna("\0")).sum())


def _cast_mapping_value(value: Any, source: pd.Series) -> Any:
    # Text mappings may change the dtype intentionally; numeric mappings preserve
    # the existing dtype when possible and may promote when values require it.
    if value is None:
        return None
    if _numeric_column(source) and isinstance(value, str):
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value
    return value


def _strict_convert(series: pd.Series, target: str, date_format: str | None, dayfirst: bool) -> pd.Series:
    if target == "text":
        return series.astype("string")
    if target == "datetime":
        converted = pd.to_datetime(series, errors="coerce", format=date_format, dayfirst=dayfirst)
    elif target in {"numeric", "integer"}:
        converted = pd.to_numeric(series, errors="coerce")
    elif target == "boolean":
        def one(v: Any):
            if pd.isna(v):
                return pd.NA
            if isinstance(v, bool):
                return v
            token = str(v).strip().lower()
            if token in {"true", "yes", "y", "1"}:
                return True
            if token in {"false", "no", "n", "0"}:
                return False
            return pd.NA
        converted = series.map(one).astype("boolean")
    else:
        raise ValueError("Unsupported target type")
    lost = int((series.notna() & converted.isna()).sum())
    if lost:
        raise ValueError(f"Conversion would introduce {lost} new missing values; fix invalid entries first")
    if target == "integer":
        numeric = converted.dropna()
        if not ((numeric % 1) == 0).all():
            raise ValueError("Integer conversion would discard fractional values")
        converted = converted.astype("Int64")
    return converted


def apply_rules(df: pd.DataFrame, rules: list[CleaningRule]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    result = df.copy(deep=True)
    events: list[dict[str, Any]] = []
    for rule in rules:
        before_rows = len(result)
        before_cols = len(result.columns)
        if rule.operation == "drop_duplicates":
            keys = rule.duplicate_columns or None
            if keys and any(c not in result.columns for c in keys):
                raise ValueError("Unknown duplicate key column")
            result = result.drop_duplicates(subset=keys, keep=rule.duplicate_keep).reset_index(drop=True)
            count = before_rows - len(result)
        else:
            col = rule.column
            assert col is not None
            if col not in result.columns:
                raise ValueError(f"Unknown column: {col}")
            series = result[col]
            count = 0
            if rule.operation == "drop_column":
                if len(result.columns) == 1:
                    raise ValueError("Cannot remove the final column")
                result = result.drop(columns=[col])
                count = before_rows
            elif rule.operation == "rename_column":
                new_name = rule.new_column.strip() if rule.new_column else ""
                if not new_name or (new_name in result.columns and new_name != col):
                    raise ValueError("New column name is empty or already exists")
                result = result.rename(columns={col: new_name})
                count = before_rows if new_name != col else 0
            else:
                if rule.operation in {"trim", "case"} and not _text_column(series):
                    raise ValueError(f"{rule.operation} requires a text column: {col}")
                if rule.operation in {"impute_missing", "clip_range", "flag_outliers"} and rule.operation != "impute_missing" and not _numeric_column(series):
                    raise ValueError(f"{rule.operation} requires a numeric column: {col}")
                if rule.operation in {"map_values", "replace_values"}:
                    # Exact matching of textual representations supports both
                    # numeric category codes and text labels.
                    mapping = rule.mapping or {}
                    string_keys = {str(k): _cast_mapping_value(v, series) for k, v in mapping.items()}
                    mask = series.notna() & series.astype("string").isin(string_keys.keys())
                    new = series.astype("object").copy()
                    new.loc[mask] = series.loc[mask].astype("string").map(string_keys).to_numpy()
                    new = new.infer_objects()
                elif rule.operation == "trim":
                    new = series.astype("string").str.strip()
                elif rule.operation == "case":
                    accessor = series.astype("string").str
                    new = {"lower": accessor.lower, "upper": accessor.upper,
                           "title": accessor.title}[rule.case_style]()
                elif rule.operation == "replace_missing":
                    new = series.fillna(rule.missing_value)
                elif rule.operation == "impute_missing":
                    if rule.strategy in {"mean", "median"} and not _numeric_column(series):
                        raise ValueError("Mean/median imputation requires a numeric column")
                    observed = series.dropna()
                    if observed.empty:
                        raise ValueError("Cannot impute a column containing only missing values")
                    replacement = {"mean": observed.mean, "median": observed.median,
                                   "mode": lambda: observed.mode(dropna=True).iloc[0]}[rule.strategy]()
                    new = series.fillna(replacement)
                elif rule.operation in {"convert_type", "parse_date"}:
                    new = _strict_convert(series, "datetime" if rule.operation == "parse_date" else rule.target_type,
                                          rule.date_format, rule.dayfirst)
                elif rule.operation == "clip_range":
                    new = series.clip(lower=rule.lower_bound, upper=rule.upper_bound)
                elif rule.operation == "flag_outliers":
                    new_name = rule.new_column.strip() if rule.new_column else ""
                    if not new_name or new_name in result.columns:
                        raise ValueError("Outlier flag column name is empty or already exists")
                    numeric = series.dropna()
                    if numeric.empty:
                        raise ValueError("Cannot calculate outliers from an empty numeric column")
                    q1, q3 = numeric.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower, upper = q1 - rule.iqr_multiplier * iqr, q3 + rule.iqr_multiplier * iqr
                    flag = (series < lower) | (series > upper)
                    result[new_name] = flag.fillna(False).astype(bool)
                    count = int(flag.sum())
                    events.append({"operation": rule.operation, "column": col, "affected_rows": count,
                                   "rows_before": before_rows, "rows_after": len(result),
                                   "columns_before": before_cols, "columns_after": len(result.columns),
                                   "details": {"flag_column": new_name, "lower_fence": float(lower),
                                               "upper_fence": float(upper)}})
                    continue
                else:
                    raise ValueError(f"Unsupported operation: {rule.operation}")
                count = _changed_count(series, new)
                result[col] = new
        events.append({"operation": rule.operation, "column": rule.column,
                       "affected_rows": count, "rows_before": before_rows,
                       "rows_after": len(result), "columns_before": before_cols,
                       "columns_after": len(result.columns)})
    return result, events
