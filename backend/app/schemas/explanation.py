from typing import Literal

from pydantic import BaseModel, Field


class ExplainKPIRequest(BaseModel):
    analysis_id: str = Field(
        ...,
        description="ID of the saved analysis containing the KPI.",
    )

    kpi_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of the KPI in dashboard.kpis.",
    )


class ExplainChartRequest(BaseModel):
    analysis_id: str = Field(
        ...,
        description="ID of the saved analysis containing the chart.",
    )

    chart_index: int = Field(
        ...,
        ge=0,
        description="Zero-based position of the chart in dashboard.charts.",
    )


class ExplanationEvidence(BaseModel):
    source: str
    detail: str


class KPIExplanationResponse(BaseModel):
    status: Literal["success"] = "success"
    item_type: Literal["kpi"] = "kpi"

    title: str
    value: str | int | float | None = None

    explanation: str
    calculation: str
    business_interpretation: str

    cautions: list[str] = Field(default_factory=list)
    supporting_evidence: list[ExplanationEvidence] = Field(
        default_factory=list
    )

    suggested_question: str | None = None

    confidence: Literal[
        "high",
        "medium",
        "low",
    ]

    model_used: str | None = None
    provider: str | None = None


class ChartExplanationResponse(BaseModel):
    status: Literal["success"] = "success"
    item_type: Literal["chart"] = "chart"

    title: str
    chart_type: str | None = None

    explanation: str
    chart_logic: str
    business_interpretation: str

    cautions: list[str] = Field(default_factory=list)
    supporting_evidence: list[ExplanationEvidence] = Field(
        default_factory=list
    )

    suggested_question: str | None = None

    confidence: Literal[
        "high",
        "medium",
        "low",
    ]

    model_used: str | None = None
    provider: str | None = None