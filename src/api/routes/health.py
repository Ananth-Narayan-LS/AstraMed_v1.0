import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas.response import HealthResponse
from src.api.services.prediction_service import (
    PredictionServiceError,
    check_model_ready,
)

logger = logging.getLogger(__name__)


router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check API health",
)
def health_check():
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
    )


@router.get(
    "/ready",
    summary="Check that the configured model checkpoint is available",
    responses={503: {"description": "The model checkpoint is not available."}},
)
def readiness_check():
    """
    Readiness (not liveness): 200 when the default model's checkpoint file exists,
    503 otherwise. It does not load the model and never reveals file-system paths.
    """
    model = settings.default_model
    try:
        check_model_ready(model, load=False)
    except PredictionServiceError as exc:
        logger.warning("Readiness check failed: model=%s code=%s", model, exc.code)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "model": model, "reason": exc.code},
        )
    return {"status": "ready", "model": model}
