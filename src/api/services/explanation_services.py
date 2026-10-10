import numpy as np
from PIL import Image
from io import BytesIO
import base64


class ExplanationServiceError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32).squeeze()

    if heatmap.ndim != 2:
        raise ExplanationServiceError(
            "INVALID_HEATMAP",
            "Expected a 2D heatmap.",
        )

    if not np.isfinite(heatmap).all():
        raise ExplanationServiceError(
            "INVALID_HEATMAP",
            "Heatmap contains invalid values.",
        )

    low, high = float(heatmap.min()), float(heatmap.max())

    if high > low:
        heatmap = (heatmap - low) / (high - low)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap


def _png_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_explanation(
    heatmap: np.ndarray,
    original_grayscale: np.ndarray,
    include_overlay: bool = True,
) -> dict:
    heatmap = _normalize_heatmap(heatmap)

    original = np.asarray(original_grayscale, dtype=np.float32).squeeze()

    if original.ndim != 2 or not np.isfinite(original).all():
        raise ExplanationServiceError(
            "INVALID_SOURCE_IMAGE",
            "Expected a valid grayscale image.",
            400,
        )

    original = original - original.min()
    if original.max() > 0:
        original = original / original.max()

    original_image = Image.fromarray(
        (original * 255).astype(np.uint8)
    ).convert("RGB")

    heatmap_image = Image.fromarray(
        (heatmap * 255).astype(np.uint8)
    ).resize(original_image.size, Image.Resampling.BILINEAR)

    # Apply a color map for easier visual interpretation.
    colored_heatmap = heatmap_image.convert("RGB")
    overlay = Image.blend(original_image, colored_heatmap, alpha=0.4)

    return {
        "heatmap_base64": _png_base64(heatmap_image),
        "overlay_base64": (
            _png_base64(overlay) if include_overlay else None
        ),
    }


def compute_heatmap(
    model,
    model_input,
    model_name: str,
    explanation_method: str,
) -> np.ndarray:
    """Delegate to the existing architecture-specific explainability code."""
    from src.explainability.grad_cam import GradCAM
    from src.explainability.vit_attention import (
        extract_vit_attention,
        cls_attention_map,
    )

    if model_name == "resnet50" and explanation_method == "gradcam":
        extractor = GradCAM(model, model.layer4[-1])
        try:
            heatmap, _ = extractor.generate(
                model_input,
                target="positive",
            )
            return _normalize_heatmap(heatmap)
        finally:
            extractor.close()

    if model_name == "vit_b16" and explanation_method == "attention":
        result = extract_vit_attention(model, model_input)
        return _normalize_heatmap(
            cls_attention_map(
                result["last_cls_attention"],
                image_size=224,
            )
        )

    raise ExplanationServiceError(
        "INCOMPATIBLE_EXPLANATION_METHOD",
        f"{explanation_method!r} is incompatible with {model_name!r}.",
        422,
    )