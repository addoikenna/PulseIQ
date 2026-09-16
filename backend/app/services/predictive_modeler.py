from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20

MINIMUM_TRAINING_ROWS = 50
MAX_CATEGORICAL_CARDINALITY = 100


class PredictiveModelingError(Exception):
    """Raised when predictive model training cannot be completed."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:
    if target_column not in dataframe.columns:
        raise PredictiveModelingError(
            f"Target column '{target_column}' does not exist in the dataset."
        )

    target = dataframe[target_column]

    non_null_target = target.dropna()

    if non_null_target.empty:
        raise PredictiveModelingError(
            f"Target column '{target_column}' contains no usable values."
        )

    if non_null_target.nunique() < 2:
        raise PredictiveModelingError(
            f"Target column '{target_column}' must contain at least two "
            "different values."
        )


def validate_training_size(
    dataframe: pd.DataFrame,
    target_column: str,
) -> None:
    usable_rows = dataframe[target_column].notna().sum()

    if usable_rows < MINIMUM_TRAINING_ROWS:
        raise PredictiveModelingError(
            "There are not enough rows with a valid target value to train "
            "a reliable model."
        )


# ---------------------------------------------------------------------------
# Problem type
# ---------------------------------------------------------------------------


def determine_problem_type(
    target: pd.Series,
) -> str:
    """
    Infer the likely prediction problem type when the user has not
    explicitly supplied one.

    Explicit problem-type selection should take priority during training.
    """

    target = target.dropna()

    if target.empty:
        raise PredictiveModelingError(
            "The selected target contains no usable values."
        )

    unique_count = int(target.nunique())

    # Boolean
    if pd.api.types.is_bool_dtype(target):
        return "binary_classification"

    # Text/string/category
    if (
        pd.api.types.is_object_dtype(target)
        or pd.api.types.is_string_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
    ):
        if unique_count == 2:
            return "binary_classification"

        if unique_count <= 20:
            return "multiclass_classification"

        raise PredictiveModelingError(
            "The selected text target contains too many unique values "
            "to be treated as a classification target."
        )

    # Numeric
    if pd.api.types.is_numeric_dtype(target):
        if unique_count == 2:
            return "binary_classification"

        values = target.astype(float)

        # Numeric categorical targets are usually compact codes,
        # e.g. 1-5 ratings or 0-10 classes.
        if (
            unique_count <= 20
            and np.all(np.isclose(values, np.round(values)))
        ):
            numeric_range = float(
                values.max() - values.min()
            )

            compact_range_threshold = max(
                20.0,
                unique_count * 5.0,
            )

            if numeric_range <= compact_range_threshold:
                return "multiclass_classification"

        # Otherwise treat numeric measurements as regression.
        return "regression"

    raise PredictiveModelingError(
        "PulseIQ could not determine the prediction problem type "
        "for the selected target."
    )


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def select_features(
    dataframe: pd.DataFrame,
    target_column: str,
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select the feature columns used for modelling.

    Target is always removed.
    Explicit exclusions are respected.
    """

    excluded_features = excluded_features or []

    if selected_features:
        missing_features = [
            column
            for column in selected_features
            if column not in dataframe.columns
        ]

        if missing_features:
            raise PredictiveModelingError(
                "The following selected features do not exist in the dataset: "
                + ", ".join(missing_features)
            )

        feature_columns = [
            column
            for column in selected_features
            if column != target_column
            and column not in excluded_features
        ]

    else:
        feature_columns = [
            column
            for column in dataframe.columns
            if column != target_column
            and column not in excluded_features
        ]

    if not feature_columns:
        raise PredictiveModelingError(
            "No usable feature columns were selected for modelling."
        )

    return dataframe[feature_columns].copy(), feature_columns


def remove_unsupported_features(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove columns that cannot safely enter the current V1 modelling
    pipeline.

    Datetime columns are excluded for now rather than silently converting
    them into arbitrary numeric values.
    """

    removed_columns: List[str] = []

    for column in list(features.columns):
        series = features[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            removed_columns.append(column)
            continue

        if series.dropna().nunique() <= 1:
            removed_columns.append(column)
            continue

        if (
            not pd.api.types.is_numeric_dtype(series)
            and series.dropna().nunique() > MAX_CATEGORICAL_CARDINALITY
        ):
            removed_columns.append(column)

    if removed_columns:
        features = features.drop(columns=removed_columns)

    if features.empty:
        raise PredictiveModelingError(
            "No usable features remain after removing unsupported columns."
        )

    return features, removed_columns


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def build_preprocessor(
    features: pd.DataFrame,
) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in features.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    if not transformers:
        raise PredictiveModelingError(
            "No numeric or categorical features are available for modelling."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def get_classification_models() -> Dict[str, BaseEstimator]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def get_regression_models() -> Dict[str, BaseEstimator]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def calculate_classification_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> Dict[str, Optional[float]]:
    predictions = pipeline.predict(X_test)

    if problem_type == "binary_classification":
        classes = list(pipeline.classes_)

        if len(classes) != 2:
            raise PredictiveModelingError(
                "Binary classification requires exactly two target classes."
            )

        # scikit-learn orders the classes consistently.
        # The second class is also the class represented by
        # predict_proba(... )[:, 1].
        positive_class = classes[1]

        metrics: Dict[str, Optional[float]] = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    average="binary",
                    pos_label=positive_class,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    average="binary",
                    pos_label=positive_class,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="binary",
                    pos_label=positive_class,
                    zero_division=0,
                )
            ),
            "roc_auc": None,
        }

        if hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(X_test)

                if probabilities.shape[1] == 2:
                    positive_probabilities = probabilities[:, 1]

                    binary_y_test = (
                        y_test == positive_class
                    ).astype(int)

                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            binary_y_test,
                            positive_probabilities,
                        )
                    )

            except (ValueError, AttributeError):
                metrics["roc_auc"] = None

        return metrics

    # Multiclass classification
    return {
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "roc_auc": None,
    }

def calculate_regression_metrics(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    return {
        "mae": float(
            mean_absolute_error(
                y_test,
                predictions,
            )
        ),
        "rmse": float(np.sqrt(mse)),
        "r2": float(
            r2_score(
                y_test,
                predictions,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def extract_feature_importance(
    pipeline: Pipeline,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance_values = model.feature_importances_

        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_)

            if coefficients.ndim == 1:
                importance_values = np.abs(coefficients)

            else:
                importance_values = np.mean(
                    np.abs(coefficients),
                    axis=0,
                )

        else:
            return []

        importance = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance_values,
            }
        )

        importance = importance.sort_values(
            "importance",
            ascending=False,
        ).head(limit)

        return [
            {
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
            }
            for _, row in importance.iterrows()
        ]

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Train individual models
# ---------------------------------------------------------------------------


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
) -> Tuple[
    List[Dict[str, Any]],
    str,
    Pipeline,
]:
    if problem_type in {
        "binary_classification",
        "multiclass_classification",
    }:
        models = get_classification_models()

    elif problem_type == "regression":
        models = get_regression_models()

    else:
        raise PredictiveModelingError(
            f"Unsupported problem type: {problem_type}"
        )

    results: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}

    for model_name, model in models.items():
        try:
            preprocessor = build_preprocessor(X_train)

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "model",
                        model,
                    ),
                ]
            )

            pipeline.fit(
                X_train,
                y_train,
            )

            if problem_type in {
                "binary_classification",
                "multiclass_classification",
            }:
                metrics = calculate_classification_metrics(
                    pipeline=pipeline,
                    X_test=X_test,
                    y_test=y_test,
                    problem_type=problem_type,
                )

                selection_score = metrics["f1"]

            else:
                metrics = calculate_regression_metrics(
                    pipeline=pipeline,
                    X_test=X_test,
                    y_test=y_test,
                )

                # Lower RMSE is better.
                selection_score = -metrics["rmse"]

            results.append(
                {
                    "model_name": model_name,
                    "metrics": metrics,
                    "selection_score": float(
                        selection_score
                    ),
                }
            )

            trained_pipelines[model_name] = pipeline

        except Exception as error:
            results.append(
                {
                    "model_name": model_name,
                    "metrics": {},
                    "selection_score": None,
                    "error": str(error),
                }
            )

    successful_results = [
        result
        for result in results
        if result.get("selection_score") is not None
    ]

    if not successful_results:
        raise PredictiveModelingError(
            "None of the candidate models could be trained successfully."
        )

    best_result = max(
        successful_results,
        key=lambda result: result["selection_score"],
    )

    best_model_name = best_result["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    return (
        results,
        best_model_name,
        best_pipeline,
    )


# ---------------------------------------------------------------------------
# Main modelling engine
# ---------------------------------------------------------------------------


def train_predictive_models(
    dataframe: pd.DataFrame,
    target_column: str,
    problem_type: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Train and compare predictive models for a selected target.

    This function does not persist the trained model yet. Model persistence
    will be introduced separately after the training workflow is stable.
    """

    if dataframe is None or dataframe.empty:
        raise PredictiveModelingError(
            "The stored dataset is empty."
        )

    validate_target(
        dataframe=dataframe,
        target_column=target_column,
    )

    validate_training_size(
        dataframe=dataframe,
        target_column=target_column,
    )

    modelling_df = dataframe.loc[
        dataframe[target_column].notna()
    ].copy()

    y = modelling_df[target_column]

    #problem_type = determine_problem_type(y)

    allowed_problem_types = {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }

    if problem_type is not None:
        if problem_type not in allowed_problem_types:
            raise PredictiveModelingError(
                f"Unsupported prediction problem type: {problem_type}"
            )

        resolved_problem_type = problem_type

    else:
        resolved_problem_type = determine_problem_type(y)

    X, selected_feature_names = select_features(
        dataframe=modelling_df,
        target_column=target_column,
        selected_features=selected_features,
        excluded_features=excluded_features,
    )

    X, automatically_removed_features = remove_unsupported_features(
        X
    )

    selected_feature_names = X.columns.tolist()

    stratify = None

    if problem_type in {
        "binary_classification",
        "multiclass_classification",
    }:
        class_counts = y.value_counts()

        if (
            len(class_counts) >= 2
            and class_counts.min() >= 2
        ):
            stratify = y

    try:
        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )

    except ValueError as error:
        raise PredictiveModelingError(
            f"PulseIQ could not create a valid train/test split: {error}"
        ) from error

    (
        model_results,
        best_model_name,
        best_pipeline,
    ) = train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        problem_type=resolved_problem_type,
    )

    feature_importance = extract_feature_importance(
        best_pipeline
    )

    best_result = next(
        result
        for result in model_results
        if result["model_name"] == best_model_name
    )

    warnings: List[str] = []

    if automatically_removed_features:
        warnings.append(
            "Some columns were automatically excluded because they were "
            "constant, high-cardinality categorical fields, or unsupported "
            "datetime fields."
        )

    if stratify is None and problem_type in {
        "binary_classification",
        "multiclass_classification",
    }:
        warnings.append(
            "The dataset could not be safely stratified during the "
            "train/test split because one or more classes had too few rows."
        )

    return {
        "status": "success",
        "problem_type": resolved_problem_type,
        "target_column": target_column,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "selected_features": selected_feature_names,
        "automatically_removed_features": automatically_removed_features,
        "models": model_results,
        "best_model": best_model_name,
        "best_model_metrics": best_result["metrics"],
        "feature_importance": feature_importance,
        "warnings": warnings,
    }