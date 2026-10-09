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
    heatmap_base64: str | None = None
    overlay_base64: str | None = None
    image_mime_type: Literal["image/png"] = "image/png"


class ModelMetadata(BaseModel):
    name: str
    version: str


class TimingMetadata(BaseModel):
    total_ms: float = Field(ge=0.0)


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["success"] = "success"
    prediction: PredictionResult
    explanation: ExplanationResult
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


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Literal["success"] = "success"
    prediction: PredictionResult
    model: ModelMetadata
    timing: TimingMetadata
