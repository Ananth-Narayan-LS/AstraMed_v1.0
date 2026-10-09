
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: Literal["resnet50", "vit_b16"] = Field(
        default="resnet50",
        description="Model architecture used for inference.",
    )

    explanation_method: Literal["gradcam", "attention"] = Field(
        default="gradcam",
        description="Explanation method requested by the client.",
    )

    include_overlay: bool = Field(
        default=True,
        description="Whether to return an explanation overlay.",
    )
