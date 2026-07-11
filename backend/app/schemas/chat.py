from typing import Literal

from pydantic import BaseModel, Field


class ChatAskRequest(BaseModel):
    analysis_id: str = Field(
        ...,
        description="ID of the analysis being queried.",
    )

    session_id: str | None = Field(
        default=None,
        description="Existing chat session. Leave empty to create a new chat.",
    )

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question about the selected analysis.",
    )


class ChatEvidence(BaseModel):
    source: str
    detail: str


class ChatAskResponse(BaseModel):
    status: Literal["success"] = "success"

    session_id: str
    session_title: str

    answer: str

    answer_type: Literal[
        "analysis_context",
        "unsupported",
    ]

    confidence: Literal[
        "high",
        "medium",
        "low",
    ]

    evidence: list[ChatEvidence] = []
    suggested_questions: list[str] = []

    model_used: str | None = None
    provider: str | None = None