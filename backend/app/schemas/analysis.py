from pydantic import BaseModel
from typing import Any


class AnalysisResponse(BaseModel):
    status: str
    message: str
    filename: str

    overview: dict[str, Any]
    data_quality_summary: dict[str, Any] | None = None
    business_health: dict[str, Any] | None = None
    business_drivers: dict[str, Any] | None = None
    business_risks: dict[str, Any] | None = None
    business_opportunities: dict[str, Any] | None = None
    column_profile: dict[str, Any] | None = None
    dashboard: dict[str, Any]
    report: dict[str, Any]

    preview: list[dict[str, Any]]
    data: list[dict[str, Any]] | None = None
    processing: dict[str, Any] | None = None