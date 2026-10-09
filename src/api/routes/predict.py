
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas.common import ErrorResponse
from src.api.schemas.response import InferenceResponse
from src.api.services.image_validation import (
    UploadValidationError,
    validate_uploaded_image,
)
from src.api.services.prediction_service import (
    PredictionServiceError,
    predict_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


def error_response(
    request_id: str,
    code: str,
    message: str,
    status_code: int,
) -> JSONResponse:
    body = ErrorResponse(
        request_id=request_id,
        error={
            "code": code,
            "message": message,
        },
    )

    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(body.model_dump()),
    )


@router.post(
    "/predict",
    summary="Predict pneumonia from a chest X-ray",
    description=(
        "Validates an uploaded PNG, JPEG, or DICOM image and runs "
        "the local ResNet-50 checkpoint. Research prototype; "
        "not for clinical diagnosis."
    ),
    response_model=InferenceResponse,
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid or corrupted image.",
        },
        413: {
            "model": ErrorResponse,
            "description": "Upload exceeds the configured size limit.",
        },
        415: {
            "model": ErrorResponse,
            "description": "Unsupported image format.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Inference failed.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Model checkpoint unavailable.",
        },
        422: {
            "description": "Invalid request fields.",
        },
    },
)
async def predict(
    image: UploadFile = File(
        ...,
        description="PNG, JPEG, or DICOM chest X-ray.",
    ),
):
    request_id = str(uuid4())
    start_time = perf_counter()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        # Enforce the limit before loading an unbounded upload.
        content = await image.read(max_bytes + 1)

        if len(content) > max_bytes:
            return error_response(
                request_id,
                "FILE_TOO_LARGE",
                f"File exceeds the {settings.max_upload_mb} MB limit.",
                413,
            )

        validated = validate_uploaded_image(
            content=content,
            filename=image.filename or "uploaded-image",
            max_bytes=max_bytes,
        )

        result = predict_image(
            content=content,
            image_format=validated.image_format,
        )

        elapsed_ms = (perf_counter() - start_time) * 1000

        logger.info(
            "Inference complete: request_id=%s model=%s "
            "probability=%.4f elapsed_ms=%.2f",
            request_id,
            result.model_name,
            result.pneumonia_probability,
            elapsed_ms,
        )

        return InferenceResponse(
            request_id=request_id,
            status="success",
            prediction={
                "predicted_class": result.predicted_class,
                "pneumonia_probability": result.pneumonia_probability,
                "threshold": result.threshold,
            },
            model={
                "name": result.model_name,
                "version": result.model_version,
            },
            timing={"total_ms": elapsed_ms},
        )

    except UploadValidationError as exc:
        return error_response(
            request_id,
            exc.code,
            exc.message,
            exc.status_code,
        )

    except PredictionServiceError as exc:
        logger.warning(
            "Prediction service error: request_id=%s code=%s",
            request_id,
            exc.code,
        )
        return error_response(
            request_id,
            exc.code,
            exc.message,
            exc.status_code,
        )

    except Exception:
        logger.exception(
            "Unexpected prediction request failure: request_id=%s",
            request_id,
        )
        return error_response(
            request_id,
            "PREDICTION_REQUEST_FAILED",
            "The prediction request could not be completed.",
            500,
        )

    finally:
        await image.close()
