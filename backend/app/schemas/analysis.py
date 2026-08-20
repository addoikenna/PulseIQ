from pydantic import BaseModel
from typing import Any


class DatasetStorageMetadata(BaseModel):
    dataset_storage_path: str
    dataset_format: str
    dataset_size_bytes: int
    dataset_row_count: int
    dataset_column_count: int
    dataset_stored_at: str
    dataset_expires_at: str | None = None
    prediction_ready: bool


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
    insight_cards: list[dict[str, Any]] | None = None
    column_profile: dict[str, Any] | None = None
    dashboard: dict[str, Any]
    report: dict[str, Any]

    preview: list[dict[str, Any]]
    data: list[dict[str, Any]] | None = None
    processing: dict[str, Any] | None = None

    analysis_id: str | None = None
    dataset_metadata: DatasetStorageMetadata | None = None