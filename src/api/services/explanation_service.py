import logging
from io import BytesIO
import base64

import numpy as np
import torch
from PIL import Image, ImageOps

from src.explainability.grad_cam import GradCAM
from src.explainability.vit_attention import (
    ViTAttentionCapture,
    cls_attention_map,
)


logger = logging.getLogger(__name__)


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


def _normalize_heatmap(heatmap) -> np.ndarray:
    """Convert a heatmap tensor or array to a normalized 2D NumPy array."""

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()

    heatmap = np.asarray(heatmap, dtype=np.float32).squeeze()

    if heatmap.ndim != 2:
        raise ExplanationServiceError(
            "INVALID_HEATMAP",
            f"Expected a 2D heatmap; received shape {heatmap.shape}.",
        )

    if not np.isfinite(heatmap).all():
        raise ExplanationServiceError(
            "INVALID_HEATMAP",
            "The heatmap contains non-finite values.",
        )

    minimum = float(heatmap.min())
    maximum = float(heatmap.max())

    if maximum > minimum:
        heatmap = (heatmap - minimum) / (maximum - minimum)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.float32)

    return heatmap


def _encode_png(image: Image.Image) -> str:
    """Encode a PIL image as a Base64 PNG string."""

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def compute_heatmap(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    model_name: str,
    explanation_method: str,
) -> np.ndarray:
    """
    Generate a heatmap using the existing architecture-specific code.

    ResNet-50 -> Grad-CAM on model.layer4[-1]
    ViT-B/16  -> final transformer block's CLS attention
    """

    model_name = model_name.strip().lower()
    explanation_method = explanation_method.strip().lower()

    # Check the pairing before running an explanation.
    supported = {
        "resnet50": {"gradcam"},
        "vit_b16": {"attention"},
    }

    if model_name not in supported:
        raise ExplanationServiceError(
            "UNSUPPORTED_MODEL",
            f"Unsupported explanation model: {model_name}.",
            422,
        )

    if explanation_method not in supported[model_name]:
        raise ExplanationServiceError(
            "INCOMPATIBLE_EXPLANATION_METHOD",
            (
                f"{explanation_method!r} is incompatible with "
                f"{model_name!r}."
            ),
            422,
        )

    if (
        model_input.ndim != 4
        or tuple(model_input.shape) != (1, 3, 224, 224)
    ):
        raise ExplanationServiceError(
            "INVALID_MODEL_INPUT_SHAPE",
            "Expected model input shape [1, 3, 224, 224].",
        )

    try:
        model.eval()

        if model_name == "resnet50":
            # This is the target layer used by the existing ResNet model.
            extractor = GradCAM(
                model=model,
                target_layer=model.layer4[-1],
            )

            try:
                heatmap, _probability = extractor.generate(
                    model_input,
                    target="positive",
                )
            finally:
                extractor.close()

            return _normalize_heatmap(heatmap)

        # Existing ViT implementation captures attention from
        # torchvision-style nn.MultiheadAttention modules.
        with ViTAttentionCapture(model) as capture:
            with torch.no_grad():
                model(model_input)

            attention_maps = capture.get_attention_maps()

            if not attention_maps:
                raise ExplanationServiceError(
                    "ATTENTION_NOT_CAPTURED",
                    "No ViT attention maps were captured.",
                )

            # The final transformer block's attention.
            final_attention = attention_maps[-1]

            heatmap = cls_attention_map(
                final_attention,
                image_size=224,
            )

        return _normalize_heatmap(heatmap)

    except ExplanationServiceError:
        raise

    except Exception as exc:
        logger.exception(
            "Heatmap generation failed for model=%s method=%s.",
            model_name,
            explanation_method,
        )
        raise ExplanationServiceError(
            "HEATMAP_GENERATION_FAILED",
            (
                f"Could not generate {explanation_method} "
                f"for {model_name}."
            ),
            500,
        ) from exc



def render_explanation(
    heatmap,
    original_grayscale,
    include_overlay=True,
):
    """
    Package an existing heatmap into visualization images.

    Returns Base64-encoded PNGs and the image dimensions.
    Does not run the model or generate a new heatmap.
    """
    heatmap_array = _normalize_heatmap(heatmap)

    original_array = np.asarray(
        original_grayscale,
        dtype=np.float32,
    )

    if original_array.ndim != 2:
        raise ExplanationServiceError(
            "INVALID_ORIGINAL_IMAGE",
            "Original image must be a 2D grayscale image.",
            422,
        )

    if original_array.size == 0 or not np.isfinite(original_array).all():
        raise ExplanationServiceError(
            "INVALID_ORIGINAL_IMAGE",
            "Original image is empty or contains invalid values.",
            422,
        )

    # Convert the original X-ray into a displayable RGB image.
    image_min = float(original_array.min())
    image_max = float(original_array.max())

    if image_max > image_min:
        display_array = (
            (original_array - image_min)
            / (image_max - image_min)
            * 255
        ).astype(np.uint8)
    else:
        display_array = np.zeros(
            original_array.shape,
            dtype=np.uint8,
        )

    original_image = Image.fromarray(display_array).convert("RGB")

    width, height = original_image.size

    # Resize the existing heatmap to match the original X-ray.
    heatmap_uint8 = (
        np.clip(heatmap_array, 0.0, 1.0) * 255
    ).astype(np.uint8)

    heatmap_gray = Image.fromarray(
        heatmap_uint8,
        mode="L",
    ).resize((width, height), Image.Resampling.BILINEAR)

    # Create a colored visualization of the existing heatmap.
    heatmap_color = ImageOps.colorize(
        heatmap_gray,
        black="blue",
        white="red",
    )

    # Blend the heatmap with the original X-ray.
    overlay = None

    if include_overlay:
        overlay = Image.blend(
            original_image,
            heatmap_color,
            alpha=0.4,
        )

    # Package the images and their shared dimensions.
    return {
        "original_image_base64": _encode_png(original_image),
        "heatmap_base64": _encode_png(heatmap_color),
        "overlay_base64": (
            _encode_png(overlay)
            if overlay is not None
            else None
        ),
        "width": width,
        "height": height,
    }
