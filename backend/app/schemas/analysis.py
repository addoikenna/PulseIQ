from pydantic import BaseModel
from typing import Any


class AnalysisResponse(BaseModel):
    status: str
    message: str
    filename: str

    overview: dict[str, Any]
    column_profile: dict[str, Any] | None = None
    dashboard: dict[str, Any]
    report: dict[str, Any]

    preview: list[dict[str, Any]]
    data: list[dict[str, Any]] | None = None
    processing: dict[str, Any] | None = None