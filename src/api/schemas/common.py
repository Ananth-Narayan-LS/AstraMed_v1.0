from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str = Field(
        ...,
        description="Stable machine-readable error code.",
    )
    message: str = Field(
        ...,
        description="Human-readable explanation.",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional diagnostic information.",
    )


class ErrorResponse(BaseModel):
    request_id: str
    status: str = Field(default="error", pattern="^error$")
    error: ErrorDetail