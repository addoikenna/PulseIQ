"""PulseIQ predictive modelling

Training-only cross-validation selects a model; the held-out test set is used
once for final evaluation. No trained models or datasets are persisted here.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Configuration and modelling limits
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
MINIMUM_TRAINING_ROWS = 50
MAX_CATEGORICAL_CARDINALITY = 100
MAX_CLASSIFICATION_CLASSES = 20
MAX_CV_FOLDS = 5
SUPPORTED_PROBLEM_TYPES = {"binary_classification", "multiclass_classification", "regression"}

# Conservative, name-based exclusions. These are not a substitute for domain review.
IDENTIFIER_NAMES = {"id", "uuid", "guid", "employee_id", "student_id", "user_id", "customer_id", "record_id", "transaction_id", "matric_no"}
SENSITIVE_NAMES = {"email", "email_address", "phone", "phone_number", "first_name", "last_name", "full_name", "fullname", "address", "date_of_birth", "dob", "ssn", "national_id", "passport_number"}



# ---------------------------------------------------------------------------
# Custom modelling exception
# ---------------------------------------------------------------------------

class PredictiveModelingError(Exception):
    """Invalid modelling request or a dataset unsuitable for modelling."""



# ---------------------------------------------------------------------------
# Feature-name safety checks
# ---------------------------------------------------------------------------

def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _is_identifier(name: str) -> bool:
    normalized = _normalized_name(name)
    return normalized in IDENTIFIER_NAMES or normalized.endswith(("_id", "_uuid", "_guid"))


def _is_sensitive(name: str) -> bool:
    normalized = _normalized_name(name)
    return normalized in SENSITIVE_NAMES or normalized.endswith(("_email", "_phone"))



# ---------------------------------------------------------------------------
# Target and request validation
# ---------------------------------------------------------------------------

def validate_target(dataframe: pd.DataFrame, target_column: str) -> None:
    if target_column not in dataframe.columns:
        raise PredictiveModelingError(f"Target column '{target_column}' does not exist in the dataset.")
    values = dataframe[target_column].dropna()
    if values.empty or values.nunique() < 2:
        raise PredictiveModelingError("The selected target needs at least two distinct non-null values.")


def validate_training_size(dataframe: pd.DataFrame, target_column: str) -> None:
    if int(dataframe[target_column].notna().sum()) < MINIMUM_TRAINING_ROWS:
        raise PredictiveModelingError(f"At least {MINIMUM_TRAINING_ROWS} rows with a valid target are required.")


def validate_test_size(test_size: float) -> None:
    if not np.isfinite(test_size) or not 0.1 <= test_size <= 0.4:
        raise PredictiveModelingError("test_size must be between 0.1 and 0.4.")



# ---------------------------------------------------------------------------
# Problem-type inference
# ---------------------------------------------------------------------------

def determine_problem_type(target: pd.Series) -> str:
    target = target.dropna()
    if target.empty:
        raise PredictiveModelingError("The target contains no usable values.")
    count = int(target.nunique())
    if pd.api.types.is_bool_dtype(target):
        return "binary_classification"
    if (pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target)
            or isinstance(target.dtype, pd.CategoricalDtype)):
        if count == 2:
            return "binary_classification"
        if 3 <= count <= MAX_CLASSIFICATION_CLASSES:
            return "multiclass_classification"
        raise PredictiveModelingError("Text target has too many classes; choose another target.")
    if pd.api.types.is_numeric_dtype(target):
        if count == 2:
            return "binary_classification"
        values = pd.to_numeric(target, errors="coerce").to_numpy(dtype=float)
        if (count <= MAX_CLASSIFICATION_CLASSES and np.isfinite(values).all()
                and np.all(np.isclose(values, np.round(values)))):
            if float(values.max() - values.min()) <= max(20.0, count * 5.0):
                return "multiclass_classification"
        return "regression"
    raise PredictiveModelingError("Cannot infer a prediction type for this target.")


def validate_problem_type(problem_type: str, target: pd.Series) -> None:
    if problem_type not in SUPPORTED_PROBLEM_TYPES:
        raise PredictiveModelingError(f"Unsupported prediction problem type: {problem_type}")
    values = target.dropna()
    count = int(values.nunique())
    if problem_type == "binary_classification" and count != 2:
        raise PredictiveModelingError("Binary classification requires exactly two classes.")
    if problem_type == "multiclass_classification" and not 3 <= count <= MAX_CLASSIFICATION_CLASSES:
        raise PredictiveModelingError(f"Multiclass classification requires 3–{MAX_CLASSIFICATION_CLASSES} classes.")
    if problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            raise PredictiveModelingError("Regression requires a numeric, non-boolean target.")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise PredictiveModelingError("Regression target contains infinite values.")



# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(
    dataframe: pd.DataFrame,
    target_column: str,
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    exclusions = set(excluded_features or [])
    if selected_features is not None:
        missing = [c for c in selected_features if c not in dataframe.columns]
        if missing:
            raise PredictiveModelingError("Selected features not found: " + ", ".join(missing))
        candidates = selected_features
    else:
        candidates = list(dataframe.columns)
    columns = list(dict.fromkeys(c for c in candidates if c != target_column and c not in exclusions))
    if not columns:
        raise PredictiveModelingError("No feature columns remain after selection.")
    return dataframe[columns].copy(), columns



# ---------------------------------------------------------------------------
# Feature sanitization and exclusions
# ---------------------------------------------------------------------------

def remove_unsupported_features(features: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    removed = []
    for column in features.columns:
        series = features[column]
        non_null = series.dropna()
        if (_is_identifier(column) or _is_sensitive(column)
                or pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_timedelta64_dtype(series)
                or pd.api.types.is_complex_dtype(series)
                or non_null.nunique() <= 1
                or (not pd.api.types.is_numeric_dtype(series)
                    and non_null.nunique() > MAX_CATEGORICAL_CARDINALITY)):
            removed.append(column)
    usable = features.drop(columns=removed).copy()
    if usable.shape[1] == 0:
        raise PredictiveModelingError("No usable features remain after safety and type exclusions.")
    # Convert pandas extension/string dtypes to ordinary object for sklearn compatibility.
    for column in usable.columns:
        series = usable[column]
        if pd.api.types.is_numeric_dtype(series):
            usable[column] = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        else:
            usable[column] = series.astype(object).where(series.notna(), np.nan)
    return usable, removed



# ---------------------------------------------------------------------------
# Leakage-safe preprocessing pipelines
# ---------------------------------------------------------------------------

def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise PredictiveModelingError("No usable numeric or categorical features.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)



# ---------------------------------------------------------------------------
# Classification candidate models
# ---------------------------------------------------------------------------

def get_classification_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }



# ---------------------------------------------------------------------------
# Regression candidate models
# ---------------------------------------------------------------------------

def get_regression_models(random_state: int = RANDOM_STATE) -> Dict[str, BaseEstimator]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=random_state),
    }



# ---------------------------------------------------------------------------
# Classification performance metrics
# ---------------------------------------------------------------------------

def _classification_metrics(y_true: pd.Series, predictions: np.ndarray, problem_type: str, probabilities=None, classes=None) -> Dict[str, Any]:
    """Compute consistent overall and class-level metrics using training-derived labels."""
    binary = problem_type == "binary_classification"
    labels = list(classes) if classes is not None else sorted(pd.unique(y_true), key=str)
    positive = labels[1] if binary else None
    options = {"average": "binary" if binary else "weighted", "zero_division": 0}
    if binary:
        options["pos_label"] = positive
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, **options)),
        "recall": float(recall_score(y_true, predictions, **options)),
        "f1": float(f1_score(y_true, predictions, **options)),
        "macro_f1": float(f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=labels, average="weighted", zero_division=0)),
        "roc_auc": None,
    }
    if binary and probabilities is not None and len(pd.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score((y_true == positive).astype(int), probabilities[:, 1]))
        except ValueError:
            pass
    metrics["class_labels"] = [str(label) for label in labels]
    metrics["confusion_matrix"] = confusion_matrix(y_true, predictions, labels=labels).astype(int).tolist()
    metrics["class_performance"] = [
        {
            "class_label": str(label),
            "support": int((np.asarray(y_true) == label).sum()),
            "precision": float(precision_score(y_true, predictions, labels=[label], average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, predictions, labels=[label], average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, predictions, labels=[label], average="macro", zero_division=0)),
        }
        for label in labels
    ]
    return metrics


def calculate_classification_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> Dict[str, Optional[float]]:
    probabilities = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
    return _classification_metrics(y_test, pipeline.predict(X_test), problem_type, probabilities, pipeline.classes_)



# ---------------------------------------------------------------------------
# Regression performance metrics
# ---------------------------------------------------------------------------

def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, predictions)
    return {"mae": float(mean_absolute_error(y_true, predictions)), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(y_true, predictions))}


def calculate_regression_metrics(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    return _regression_metrics(y_test, pipeline.predict(X_test))



# ---------------------------------------------------------------------------
# Feature importance and interpretation
# ---------------------------------------------------------------------------

def extract_feature_importance(pipeline: Pipeline, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        names = preprocessor.get_feature_names_out()
        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_)
        elif hasattr(model, "coef_"):
            coefficients = np.asarray(model.coef_)
            values = np.abs(coefficients) if coefficients.ndim == 1 else np.mean(np.abs(coefficients), axis=0)
        else:
            return []
        if len(names) != len(values):
            return []
        order = np.argsort(-values)[:limit]
        return [{"feature": str(names[i]), "importance": float(values[i]), "rank": rank}
                for rank, i in enumerate(order, start=1)]
    except (AttributeError, KeyError, ValueError, TypeError):
        return []



# ---------------------------------------------------------------------------
# Cross-validation configuration and scoring
# ---------------------------------------------------------------------------

def _cv_setup(y_train: pd.Series, problem_type: str, random_state: int):
    if problem_type == "regression":
        folds = min(MAX_CV_FOLDS, len(y_train))
        return KFold(n_splits=folds, shuffle=True, random_state=random_state), folds
    counts = y_train.value_counts()
    if counts.min() < 2:
        raise PredictiveModelingError("Each target class needs at least two training examples for stratified cross-validation.")
    folds = min(MAX_CV_FOLDS, int(counts.min()))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state), folds


def _scorer(problem_type: str, y_train: pd.Series):
    if problem_type == "regression":
        return "neg_root_mean_squared_error"
    if problem_type == "multiclass_classification":
        def weighted_f1(estimator, X, y):
            return float(f1_score(y, estimator.predict(X), average="weighted", zero_division=0))
        return weighted_f1
    positive_class = sorted(y_train.unique(), key=lambda value: str(value))[1]
    def binary_f1(estimator, X, y):
        return float(f1_score(y, estimator.predict(X), average="binary", pos_label=positive_class, zero_division=0))
    return binary_f1



# ---------------------------------------------------------------------------
# Pipeline construction and model comparison
# ---------------------------------------------------------------------------

def _build_pipeline(X_train: pd.DataFrame, estimator: BaseEstimator) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor(X_train)), ("model", clone(estimator))])


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str,
    random_state: int = RANDOM_STATE,
) -> Tuple[List[Dict[str, Any]], str, Pipeline]:
    """Select by CV on training data; fit winner; evaluate test elsewhere.

    X_test/y_test are accepted for compatibility with the original helper but
    are deliberately not inspected during model selection.
    """
    models = (get_regression_models(random_state) if problem_type == "regression"
              else get_classification_models(random_state))
    cv, folds = _cv_setup(y_train, problem_type, random_state)
    scorer = _scorer(problem_type, y_train)
    results: List[Dict[str, Any]] = []
    for name, estimator in models.items():
        try:
            pipeline = _build_pipeline(X_train, estimator)
            scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scorer, error_score="raise", n_jobs=1)["test_score"]
            mean_score = float(np.mean(scores))
            results.append({
                "model_name": name,
                "metrics": {},  # Held-out test metrics belong only to the winner.
                "cv_mean_score": mean_score if problem_type != "regression" else -mean_score,
                "cv_std_score": float(np.std(scores)),
                "cv_folds": folds,
                "selection_score": mean_score,
            })
        except Exception as error:
            results.append({"model_name": name, "metrics": {}, "selection_score": None, "error": str(error)})
    successful = [item for item in results if item["selection_score"] is not None and np.isfinite(item["selection_score"])]
    if not successful:
        errors = " | ".join(f"{item['model_name']}: {item.get('error', 'unknown')}" for item in results)
        raise PredictiveModelingError("No candidate model passed cross-validation: " + errors)
    winner = max(successful, key=lambda item: item["selection_score"])
    winner_name = winner["model_name"]
    best_pipeline = _build_pipeline(X_train, models[winner_name])
    try:
        best_pipeline.fit(X_train, y_train)
    except Exception as error:
        raise PredictiveModelingError(f"Selected model could not fit on the training data: {error}") from error
    return results, winner_name, best_pipeline



# ---------------------------------------------------------------------------
# Statistical baseline models
# ---------------------------------------------------------------------------

def _baselines(y_train: pd.Series, y_test: pd.Series, problem_type: str) -> Dict[str, Any]:
    if problem_type == "regression":
        mean = DummyRegressor(strategy="mean").fit(np.zeros((len(y_train), 1)), y_train)
        median = DummyRegressor(strategy="median").fit(np.zeros((len(y_train), 1)), y_train)
        mean_metrics = _regression_metrics(y_test, mean.predict(np.zeros((len(y_test), 1))))
        median_metrics = _regression_metrics(y_test, median.predict(np.zeros((len(y_test), 1))))
        return {"primary": "training_mean", "mean": {"prediction": float(mean.constant_[0, 0]), "metrics": mean_metrics},
                "median": {"prediction": float(median.constant_[0, 0]), "metrics": median_metrics}}
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    predicted = dummy.predict(np.zeros((len(y_test), 1)))
    metrics = _classification_metrics(y_test, predicted, problem_type, classes=dummy.classes_)
    majority = dummy.classes_[np.argmax(dummy.class_prior_)]
    return {"primary": "majority_class", "majority_class": str(majority), "metrics": metrics}



# ---------------------------------------------------------------------------
# Baseline improvement calculations
# ---------------------------------------------------------------------------

def _percentage_improvement(baseline: float, model: float, lower_is_better: bool) -> Optional[float]:
    if baseline == 0 or not np.isfinite(baseline) or not np.isfinite(model):
        return None
    return float(100 * ((baseline - model) if lower_is_better else (model - baseline)) / abs(baseline))


def _baseline_comparison(baseline: Dict[str, Any], metrics: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    if problem_type == "regression":
        mean = baseline["mean"]["metrics"]
        median = baseline["median"]["metrics"]
        return {
            "primary_metric": "rmse",
            "baseline_metric": mean["rmse"],
            "model_metric": metrics["rmse"],
            "improvement_percentage": _percentage_improvement(mean["rmse"], metrics["rmse"], True),
            "mae_improvement_vs_median_percentage": _percentage_improvement(median["mae"], metrics["mae"], True),
            "beats_primary_baseline": bool(metrics["rmse"] < mean["rmse"]),
            "rmse_absolute_improvement": float(mean["rmse"] - metrics["rmse"]),
        }
    base = baseline["metrics"]
    return {
        "primary_metric": "f1", "baseline_metric": base["f1"], "model_metric": metrics["f1"],
        "improvement_percentage": _percentage_improvement(base["f1"], metrics["f1"], False),
        "beats_primary_baseline": bool(metrics["f1"] > base["f1"]),
        "accuracy_improvement_percentage_points": float(100 * (metrics["accuracy"] - base["accuracy"])),
        "f1_improvement_percentage_points": float(100 * (metrics["f1"] - base["f1"])),
        "macro_f1_improvement_percentage_points": float(100 * (metrics["macro_f1"] - base["macro_f1"])),
        "beats_accuracy_baseline": bool(metrics["accuracy"] > base["accuracy"]),
        "beats_f1_baseline": bool(metrics["f1"] > base["f1"]),
    }


# ---------------------------------------------------------------------------
# Model reliability assessment (heuristic, not a validation guarantee)
# ---------------------------------------------------------------------------

def assess_model_reliability(metrics: Dict[str, Any], comparison: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    """Contextual, conservative signal for UI; never a universal quality verdict."""
    reasons = []
    if problem_type == "regression":
        improvement = comparison.get("improvement_percentage")
        r2 = metrics["r2"]
        if not comparison["beats_primary_baseline"] or r2 <= 0:
            level = "weak"
            reasons.append("Test RMSE does not beat the mean baseline or test R² is non-positive.")
        elif r2 >= 0.7 and improvement is not None and improvement >= 10:
            level = "strong"
            reasons.append("Test R² is high and RMSE meaningfully improves on the mean baseline.")
        else:
            level = "moderate"
            reasons.append("Test RMSE beats the mean baseline, but predictive strength needs contextual review.")
    else:
        f1 = metrics["f1"]
        macro = metrics["macro_f1"]
        if not comparison["beats_f1_baseline"] or f1 < 0.5 or macro < 0.4:
            level = "weak"
            reasons.append("F1 or macro F1 is low, or the model does not beat the F1 baseline.")
        elif f1 >= 0.8 and macro >= 0.7 and comparison["beats_accuracy_baseline"]:
            level = "strong"
            reasons.append("F1 and macro F1 are high and accuracy beats the majority baseline.")
        else:
            level = "moderate"
            reasons.append("F1 improves on baseline, but class-level errors and deployment context still matter.")
        if not comparison["beats_accuracy_baseline"]:
            reasons.append("Accuracy does not exceed the majority-class baseline.")
    return {
        "level": level,
        "reasons": reasons,
        "note": "Heuristic assessment on one held-out split; not calibrated confidence or proof of future performance.",
    }



# ---------------------------------------------------------------------------
# Main predictive modelling engine
# ---------------------------------------------------------------------------

def train_predictive_models(
    dataframe: pd.DataFrame,
    target_column: str,
    problem_type: Optional[str] = None,
    selected_features: Optional[List[str]] = None,
    excluded_features: Optional[List[str]] = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Train V2 models without persisting them. Maintains the V1 API signature."""
    if dataframe is None or dataframe.empty:
        raise PredictiveModelingError("The stored dataset is empty.")
    if not dataframe.columns.is_unique:
        raise PredictiveModelingError("Duplicate column names must be resolved before training.")
    validate_target(dataframe, target_column)
    validate_training_size(dataframe, target_column)
    validate_test_size(test_size)
    modelling_df = dataframe.loc[dataframe[target_column].notna()].copy()
    y = modelling_df[target_column]
    resolved_problem_type = problem_type if problem_type is not None else determine_problem_type(y)
    validate_problem_type(resolved_problem_type, y)
    if resolved_problem_type != "regression":
        # Object/string extension labels are normalized for consistent sklearn behavior.
        if y.map(type).nunique() > 1:
            raise PredictiveModelingError("Classification target mixes label types. Clean the target first.")
        y = y.copy()
    X, _ = select_features(modelling_df, target_column, selected_features, excluded_features)
    X, automatically_removed_features = remove_unsupported_features(X)
    warnings: List[str] = []
    if automatically_removed_features:
        warnings.append("Excluded unsupported, identifier, or potentially sensitive columns: " + ", ".join(automatically_removed_features) + ".")
    if resolved_problem_type != "regression":
        counts = y.value_counts()
        if counts.min() < 2:
            raise PredictiveModelingError("Each class needs at least two rows for a valid stratified split.")
        n_test = int(np.ceil(len(y) * test_size))
        n_train = len(y) - n_test
        if min(n_test, n_train) < len(counts):
            raise PredictiveModelingError("Train/test split is too small to represent every class.")
        stratify = y
        if counts.max() / counts.min() >= 3:
            warnings.append("The target is imbalanced. Review F1, precision and recall rather than accuracy alone.")
    else:
        stratify = None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
    except ValueError as error:
        raise PredictiveModelingError(f"Unable to split the dataset: {error}") from error
    model_results, best_name, best_pipeline = train_models(
        X_train, X_test, y_train, y_test, resolved_problem_type, random_state
    )
    if resolved_problem_type == "regression":
        best_metrics = calculate_regression_metrics(best_pipeline, X_test, y_test)
    else:
        best_metrics = calculate_classification_metrics(best_pipeline, X_test, y_test, resolved_problem_type)
    baseline = _baselines(y_train, y_test, resolved_problem_type)
    comparison = _baseline_comparison(baseline, best_metrics, resolved_problem_type)
    reliability = assess_model_reliability(best_metrics, comparison, resolved_problem_type)
    if not comparison["beats_primary_baseline"]:
        warnings.append("The selected model does not outperform the simple training-derived baseline on the held-out test set. Predictions may not be reliable.")
    if resolved_problem_type != "regression" and not comparison["beats_accuracy_baseline"]:
        warnings.append("Model accuracy does not exceed the majority-class baseline, even if F1 does.")
    if resolved_problem_type == "regression" and best_metrics["r2"] < 0:
        warnings.append("Negative test R²: the model performs worse than predicting the test-set mean. Consider more informative features.")
    if resolved_problem_type == "regression" and best_metrics["r2"] < 0.1:
        warnings.append("The model explains very little variation in the target. Avoid using it for consequential decisions.")
    if resolved_problem_type != "regression" and best_metrics["f1"] < 0.5:
        warnings.append("Low test F1 score: classification performance is weak and should be reviewed before use.")
    failed = [item["model_name"] for item in model_results if item.get("error")]
    if failed:
        warnings.append("Some candidate models failed cross-validation: " + ", ".join(failed) + ".")
    warnings.append("Review potential target leakage and whether each feature is available at prediction time; column-name checks cannot detect all leakage.")
    importance = extract_feature_importance(best_pipeline)
    estimator = best_pipeline.named_steps["model"]
    importance_note = (
        "Absolute linear coefficients on transformed features; not causal effects or directly comparable to tree importances."
        if hasattr(estimator, "coef_") else
        "Model-based tree impurity importances; not causal effects and may favor some features."
    )
    return {
        "status": "success", "problem_type": resolved_problem_type, "target_column": target_column,
        "training_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "test_size": float(test_size), "random_state": int(random_state),
        "selected_features": X.columns.tolist(),
        "automatically_removed_features": automatically_removed_features,
        "models": model_results, "best_model": best_name,
        "selection_metric": "rmse" if resolved_problem_type == "regression" else "f1",
        "selection_method": "training_only_cross_validation",
        "best_model_metrics": best_metrics,
        "baseline": baseline, "baseline_comparison": comparison,
        "model_reliability": reliability,
        "feature_importance": importance, "feature_importance_note": importance_note,
        "warnings": warnings,
    }
