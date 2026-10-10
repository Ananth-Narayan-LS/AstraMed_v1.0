import logging
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from src.api.config import settings
from src.api.schemas.common import ErrorResponse
from src.api.schemas.response import PredictionWithExplanationResponse

from src.api.services.image_validation import (
    UploadValidationError,
    validate_uploaded_image,
)
from src.api.services.prediction_service import (
    PredictionServiceError,
    predict_with_context,
)
from src.api.services.explanation_service import (
    ExplanationServiceError,
    compute_heatmap,
    render_explanation,
)
from src.api.services.explanation_compatibility import (
    validate_explanation_compatibility,
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
    summary="Predict pneumonia and generate an explanation",
    description=(
        "Validates a PNG, JPEG, or DICOM image and returns a local "
        "model prediction with a Grad-CAM or ViT attention heatmap. "
        "Research prototype; not for clinical diagnosis."
    ),
    response_model=PredictionWithExplanationResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict(
    image: UploadFile = File(...),
    model_name: str = Form(default="resnet50"),
    explanation_method: str = Form(default="gradcam"),
    include_overlay: bool = Form(default=True),
):
    request_id = str(uuid4())
    start_time = perf_counter()
    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        # 1. Read the upload with a maximum-size guard.
        content = await image.read(max_bytes + 1)

        if len(content) > max_bytes:
            return error_response(
                request_id,
                "FILE_TOO_LARGE",
                f"File exceeds the {settings.max_upload_mb} MB limit.",
                413,
            )

        # 2. Validate the uploaded file before inference.
        validated = validate_uploaded_image(
            content=content,
            filename=image.filename or "uploaded-image",
            max_bytes=max_bytes,
        )

        # 3. Check model/explanation compatibility.
        validate_explanation_compatibility(
            model_name,
            explanation_method,
        )

        # 4. Predict once and retain the model and input tensor.
        result, model, model_input, original_grayscale = (
            predict_with_context(
                content=content,
                image_format=validated.image_format,
                model_name=model_name,
            )
        )

        # 5. Generate the architecture-specific explanation.
        heatmap = compute_heatmap(
            model=model,
            model_input=model_input,
            model_name=model_name,
            explanation_method=explanation_method,
        )

        # 6. Render the heatmap and optional overlay.
        images = render_explanation(
            heatmap=heatmap,
            original_grayscale=original_grayscale,
            include_overlay=include_overlay,
        )

        elapsed_ms = (perf_counter() - start_time) * 1000

        logger.info(
            "Prediction with explanation complete: "
            "request_id=%s model=%s method=%s probability=%.4f "
            "elapsed_ms=%.2f",
            request_id,
            result.model_name,
            explanation_method,
            result.pneumonia_probability,
            elapsed_ms,
        )

        # 7. Return prediction AND explanation in the response.
        return PredictionWithExplanationResponse(
            request_id=request_id,
            status="success",
            prediction={
                "predicted_class": result.predicted_class,
                "pneumonia_probability": result.pneumonia_probability,
                "threshold": result.threshold,
            },
            
            explanation={
                "method": explanation_method.strip().lower(),
                "original_image_base64": images["original_image_base64"],
                "heatmap_base64": images["heatmap_base64"],
                "overlay_base64": images["overlay_base64"],
                "image_mime_type": "image/png",
                "width": images["width"],
                "height": images["height"],
            },

            model={
                "name": result.model_name,
                "version": result.model_version,
            },
            timing={
                "total_ms": elapsed_ms,
            },
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
            "Prediction error: request_id=%s code=%s",
            request_id,
            exc.code,
        )
        return error_response(
            request_id,
            exc.code,
            exc.message,
            exc.status_code,
        )

    except ExplanationServiceError as exc:
        logger.warning(
            "Explanation error: request_id=%s code=%s",
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
            "The prediction and explanation request could not be completed.",
            500,
        )

    finally:
        await image.close()