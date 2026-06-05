from pydantic import BaseModel
from typing import Any


class AnalysisResponse(BaseModel):
    status: str
    message: str
    filename: str

    overview: dict[str, Any]
    dashboard: dict[str, Any]
    report: dict[str, Any]

    preview: list[dict[str, Any]]