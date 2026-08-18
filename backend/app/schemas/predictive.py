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
    analysis_id: str

    target_column: str

    problem_type: PredictionProblemType | None = Field(
        default=None,
        description=(
            "Optional override. PulseIQ will infer the problem type "
            "when this is omitted."
        ),
    )

    feature_selection: PredictiveFeatureSelection | None = None

    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.4,
    )

    random_state: int = 42


class ModelMetric(BaseModel):
    name: str
    value: float
    display_value: str | None = None


class ModelComparisonResult(BaseModel):
    model_name: str
    metrics: list[ModelMetric]

    training_rows: int
    testing_rows: int

    selected: bool = False


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float
    rank: int


class PredictionTrainingResponse(BaseModel):
    status: Literal["success"] = "success"

    prediction_id: str
    analysis_id: str

    target_column: str
    problem_type: PredictionProblemType

    selected_model: str
    selection_metric: str

    models_compared: list[ModelComparisonResult]

    feature_importance: list[FeatureImportanceItem] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(default_factory=list)

    business_summary: str | None = None
    model_confidence: Literal[
        "high",
        "medium",
        "low",
    ]