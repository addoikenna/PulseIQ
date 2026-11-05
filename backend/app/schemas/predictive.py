from typing import Any, Literal

from pydantic import BaseModel, Field


ReadinessStatus = Literal[
    "ready",
    "ready_with_cautions",
    "not_ready",
]

PredictionProblemType = Literal[
    "binary_classification",
    "multiclass_classification",
    "regression",
    "unsupported",
]


class PredictiveReadinessRequest(BaseModel):
    analysis_id: str = Field(
        ...,
        description="ID of the saved analysis to assess.",
    )


class TargetCandidate(BaseModel):
    column: str

    problem_type: PredictionProblemType

    confidence: Literal[
        "high",
        "medium",
        "low",
    ]

    reason: str

    unique_values: int | None = None
    missing_percentage: float = 0.0

    class_distribution: dict[str, int] | None = None

    cautions: list[str] = Field(default_factory=list)


class ExcludedPredictiveColumn(BaseModel):
    column: str
    reason: str

    category: Literal[
        "identifier",
        "personal_information",
        "constant",
        "high_cardinality",
        "mostly_missing",
        "unsupported",
        "possible_leakage",
        "other",
    ]


class PredictiveDatasetSummary(BaseModel):
    total_rows: int
    total_columns: int

    usable_rows: int
    usable_features: int

    duplicate_rows: int = 0

    numeric_columns: int = 0
    categorical_columns: int = 0
    date_columns: int = 0

    total_missing_values: int = 0
    missing_percentage: float = 0.0


class PredictiveReadinessResponse(BaseModel):
    status: Literal["success"] = "success"

    readiness_status: ReadinessStatus

    analysis_id: str
    filename: str | None = None

    dataset_summary: PredictiveDatasetSummary

    candidate_targets: list[TargetCandidate] = Field(
        default_factory=list
    )

    excluded_columns: list[ExcludedPredictiveColumn] = Field(
        default_factory=list
    )

    recommended_target: str | None = None
    recommended_problem_type: PredictionProblemType | None = None

    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    minimum_requirements: dict[str, Any] = Field(
        default_factory=dict
    )


class PredictiveFeatureSelection(BaseModel):
    included_features: list[str] = Field(default_factory=list)
    excluded_features: list[str] = Field(default_factory=list)


class TrainPredictionRequest(BaseModel):
    analysis_id: str = Field(
        ...,
        description="ID of the saved analysis whose dataset will be used.",
    )

    target_column: str = Field(
        ...,
        description="Column the user wants PulseIQ to predict.",
    )

    problem_type: PredictionProblemType | None = Field(
        default=None,
        description=(
            "Optional prediction problem type. "
            "If omitted, PulseIQ will infer it from the target."
        ),
    )

    feature_selection: PredictiveFeatureSelection | None = None

    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
    )

    random_state: int = 42

    # V2.2: omit model_id for a new saved model; provide it to add a version.
    model_id: str | None = None
    model_name: str | None = Field(default=None, max_length=150)
    save_model: bool = True


# ---------------------------------------------------------------------------
# V2.1 training response models
# ---------------------------------------------------------------------------

class ModelComparisonResult(BaseModel):
    model_name: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    cv_mean_score: float | None = None
    cv_std_score: float | None = None
    cv_folds: int | None = None
    selection_score: float | None = None
    error: str | None = None


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    rank: int


class ClassPerformanceItem(BaseModel):
    class_label: str
    support: int
    precision: float
    recall: float
    f1: float


class ClassificationMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    weighted_f1: float
    roc_auc: float | None = None
    class_labels: list[str]
    confusion_matrix: list[list[int]]
    class_performance: list[ClassPerformanceItem]


class RegressionMetrics(BaseModel):
    mae: float
    rmse: float
    r2: float


class RegressionBaselineItem(BaseModel):
    prediction: float
    metrics: RegressionMetrics


class RegressionBaseline(BaseModel):
    primary: Literal["training_mean"]
    mean: RegressionBaselineItem
    median: RegressionBaselineItem


class ClassificationBaseline(BaseModel):
    primary: Literal["majority_class"]
    majority_class: str
    metrics: ClassificationMetrics


class BaselineComparison(BaseModel):
    primary_metric: Literal["rmse", "f1"]
    baseline_metric: float
    model_metric: float
    improvement_percentage: float | None = None
    beats_primary_baseline: bool
    accuracy_improvement_percentage_points: float | None = None
    f1_improvement_percentage_points: float | None = None
    macro_f1_improvement_percentage_points: float | None = None
    beats_accuracy_baseline: bool | None = None
    beats_f1_baseline: bool | None = None
    mae_improvement_vs_median_percentage: float | None = None
    rmse_absolute_improvement: float | None = None


class ModelReliability(BaseModel):
    level: Literal["weak", "moderate", "strong"]
    reasons: list[str] = Field(default_factory=list)
    note: str


class PredictionTrainingResponse(BaseModel):
    status: Literal["success"] = "success"
    analysis_id: str
    filename: str | None = None
    problem_type: Literal["binary_classification", "multiclass_classification", "regression"]
    target_column: str
    training_rows: int
    test_rows: int
    test_size: float
    random_state: int
    selected_features: list[str]
    automatically_removed_features: list[str] = Field(default_factory=list)
    models: list[ModelComparisonResult]
    best_model: str
    selection_metric: Literal["rmse", "f1"]
    selection_method: str
    best_model_metrics: ClassificationMetrics | RegressionMetrics
    baseline: ClassificationBaseline | RegressionBaseline
    baseline_comparison: BaselineComparison
    model_reliability: ModelReliability
    feature_importance: list[FeatureImportanceItem] = Field(default_factory=list)
    feature_importance_note: str
    warnings: list[str] = Field(default_factory=list)
    model_id: str | None = None
    version_id: str | None = None
    version_number: int | None = None
    saved: bool = False


# V2.2 read endpoints use flexible dictionaries because registry rows include JSONB.
class SavedModelsResponse(BaseModel):
    models: list[dict[str, Any]] = Field(default_factory=list)


class SavedModelVersionsResponse(BaseModel):
    model_id: str
    versions: list[dict[str, Any]] = Field(default_factory=list)
