from src.api.services.explanation_service import ExplanationServiceError


SUPPORTED_METHODS = {
    "resnet50": {"gradcam"},
    "vit_b16": {"attention"},
}


def validate_explanation_compatibility(
    model_name: str,
    explanation_method: str,
) -> None:
    """Reject explanation methods incompatible with the selected model."""

    model_name = model_name.strip().lower()
    explanation_method = explanation_method.strip().lower()

    if model_name not in SUPPORTED_METHODS:
        raise ExplanationServiceError(
            code="UNSUPPORTED_MODEL",
            message=f"Unsupported model: {model_name}.",
            status_code=422,
        )

    allowed_methods = SUPPORTED_METHODS[model_name]

    if explanation_method not in allowed_methods:
        raise ExplanationServiceError(
            code="INCOMPATIBLE_EXPLANATION_METHOD",
            message=(
                f"{explanation_method!r} is not supported for "
                f"{model_name!r}. Allowed method(s): "
                f"{', '.join(sorted(allowed_methods))}."
            ),
            status_code=422,
        )