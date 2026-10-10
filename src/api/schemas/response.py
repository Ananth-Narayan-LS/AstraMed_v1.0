from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_name: str
    version: str


class PredictionResult(BaseModel):
    predicted_class: Literal["pneumonia", "normal"]
    pneumonia_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)



class ExplanationResult(BaseModel):
    method: Literal["gradcam", "attention"]

    original_image_base64: str
    heatmap_base64: str
    overlay_base64: str | None = None

    image_mime_type: Literal["image/png"] = "image/png"
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ModelMetadata(BaseModel):
    name: str
    version: str


class TimingMetadata(BaseModel):
    total_ms: float = Field(ge=0.0)


class PredictionWithExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["success"] = "success"
    prediction: PredictionResult
    explanation: ExplanationResult
    model: ModelMetadata
    timing: TimingMetadata


class PredictionResponse(PredictionWithExplanationResponse):
    """Backward-compatible response schema."""


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["success"] = "success"
    prediction: PredictionResult
    model: ModelMetadata
    timing: TimingMetadata


class UploadValidationResponse(BaseModel):
    status: Literal["success"] = "success"
    request_id: str
    message: str
    image_format: Literal["PNG", "JPEG", "DICOM"]
    size_bytes: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)