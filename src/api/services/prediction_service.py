import logging
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image

from src.data.advanced_input import AdvancedInputAdapter
from src.data.preprocessing import preprocess_image
from src.models.model_factory import build_model
from src.utils.config import PROJECT_ROOT


logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = PROJECT_ROOT / "outputs" / "experiments"

CHECKPOINT_PATHS = {
    "resnet50": (
        EXPERIMENTS_DIR
        / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
        / "checkpoints"
        / "best_model.pth"
    ),
}

PREDICTION_THRESHOLD = 0.5

SUPPORTED_MODELS = {"resnet50", "vit_b16"}


class PredictionServiceError(Exception):
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


@dataclass(frozen=True)
class PredictionOutput:
    pneumonia_probability: float
    predicted_class: str
    threshold: float
    model_name: str
    model_version: str


def _resolve_checkpoint(model_name: str) -> Path:
    """Resolve a local checkpoint without substituting pretrained weights."""
    if model_name == "resnet50":
        checkpoint_path = CHECKPOINT_PATHS["resnet50"]

    elif model_name == "vit_b16":
        matches = sorted(
            EXPERIMENTS_DIR.glob(
                "vit_b16*/checkpoints/best_model.pth"
            )
        )

        if not matches:
            raise PredictionServiceError(
                "CHECKPOINT_NOT_FOUND",
                "No local ViT-B/16 best_model.pth checkpoint was found "
                "under outputs/experiments/vit_b16*/checkpoints/.",
                503,
            )

        if len(matches) > 1:
            raise PredictionServiceError(
                "AMBIGUOUS_CHECKPOINT",
                "Multiple ViT-B/16 checkpoints were found. Configure "
                "the intended experiment checkpoint explicitly.",
                500,
            )

        checkpoint_path = matches[0]

    else:
        raise PredictionServiceError(
            "UNSUPPORTED_MODEL",
            f"Model {model_name!r} is not supported.",
            422,
        )

    if not checkpoint_path.is_file():
        raise PredictionServiceError(
            "CHECKPOINT_NOT_FOUND",
            f"The local checkpoint for {model_name} was not found: "
            f"{checkpoint_path}",
            503,
        )

    return checkpoint_path


@lru_cache(maxsize=2)
def _load_model_bundle(model_name: str):
    """Load and cache a model, its adapter, device, and checkpoint version."""
    model_name = model_name.lower().strip()

    if model_name not in SUPPORTED_MODELS:
        raise PredictionServiceError(
            "UNSUPPORTED_MODEL",
            f"Model {model_name!r} is not supported.",
            422,
        )

    checkpoint_path = _resolve_checkpoint(model_name)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        if (
            not isinstance(checkpoint, dict)
            or "model_state_dict" not in checkpoint
        ):
            raise PredictionServiceError(
                "INVALID_CHECKPOINT",
                "The checkpoint has no model_state_dict.",
                500,
            )

        # Build the requested architecture without downloading pretrained weights.
        model = build_model(model_name, pretrained=False)

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        model.to(device)
        model.eval()

        adapter = AdvancedInputAdapter(
            image_size=224,
            use_imagenet_normalization=True,
        )
        adapter.to(device)
        adapter.eval()

        epoch = checkpoint.get("epoch", "unknown")
        model_version = f"checkpoint-epoch-{epoch}"

        logger.info(
            "Loaded model=%s device=%s epoch=%s checkpoint=%s",
            model_name,
            device,
            epoch,
            checkpoint_path,
        )

        return model, adapter, device, model_version

    except PredictionServiceError:
        raise

    except Exception as exc:
        logger.exception(
            "Could not load the %s prediction checkpoint.",
            model_name,
        )
        raise PredictionServiceError(
            "MODEL_LOAD_FAILED",
            f"The local {model_name} checkpoint could not be loaded. "
            "Check that its architecture matches the model factory.",
            503,
        ) from exc


def _decode_uploaded_image(
    content: bytes,
    image_format: str,
) -> np.ndarray:
    """Decode an already-validated upload into a 2D grayscale array."""
    try:
        if image_format == "DICOM":
            dataset = pydicom.dcmread(
                BytesIO(content),
                force=False,
            )
            image = dataset.pixel_array.astype(np.float32)

        elif image_format in {"PNG", "JPEG"}:
            with Image.open(BytesIO(content)) as source:
                image = np.asarray(source.convert("L")).copy()

        else:
            raise PredictionServiceError(
                "UNSUPPORTED_IMAGE_FORMAT",
                "The image format is not supported for inference.",
                415,
            )

        if image.ndim != 2:
            raise PredictionServiceError(
                "INVALID_IMAGE_DIMENSIONS",
                "Inference requires a single grayscale image.",
                400,
            )

        if image.size == 0 or not np.isfinite(image).all():
            raise PredictionServiceError(
                "INVALID_IMAGE_PIXELS",
                "The image contains invalid pixel data.",
                400,
            )

        return image

    except PredictionServiceError:
        raise

    except Exception as exc:
        logger.warning("Uploaded image could not be decoded.")
        raise PredictionServiceError(
            "IMAGE_DECODE_FAILED",
            "The image pixels could not be decoded.",
            400,
        ) from exc


def _prepare_model_input(
    image: np.ndarray,
    adapter: AdvancedInputAdapter,
    device: torch.device,
):
    """
    Return the original grayscale image and model input.

    Model input shape: [1, 3, 224, 224].
    """
    processed = preprocess_image(
        image,
        target_size=(512, 512),
    )

    input_tensor = torch.from_numpy(
        np.asarray(processed, dtype=np.float32)
    ).float().unsqueeze(0).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        model_input = adapter(input_tensor)

    if tuple(model_input.shape) != (1, 3, 224, 224):
        raise PredictionServiceError(
            "INVALID_MODEL_INPUT_SHAPE",
            f"Expected model input [1, 3, 224, 224], got "
            f"{tuple(model_input.shape)}.",
            500,
        )

    return image.copy(), model_input


def predict_with_context(
    content: bytes,
    image_format: str,
    model_name: str = "resnet50",
):
    """
    Run inference and return context for explanation generation.

    Returns:
        (PredictionOutput, model, model_input, original_grayscale)

    The model_input is the same tensor used for the prediction.
    """
    model_name = model_name.lower().strip()

    model, adapter, device, model_version = _load_model_bundle(
        model_name
    )

    image = _decode_uploaded_image(
        content=content,
        image_format=image_format,
    )

    original_grayscale, model_input = _prepare_model_input(
        image=image,
        adapter=adapter,
        device=device,
    )

    try:
        with torch.inference_mode():
            logits = model(model_input)

            # Both supported classifiers must return logits.
            probability = torch.sigmoid(logits).reshape(-1)[0].item()

    except Exception as exc:
        logger.exception(
            "Local %s inference failed.",
            model_name,
        )
        raise PredictionServiceError(
            "INFERENCE_FAILED",
            "The model could not complete inference.",
            500,
        ) from exc

    predicted_class = (
        "pneumonia"
        if probability >= PREDICTION_THRESHOLD
        else "normal"
    )

    result = PredictionOutput(
        pneumonia_probability=float(probability),
        predicted_class=predicted_class,
        threshold=PREDICTION_THRESHOLD,
        model_name=model_name,
        model_version=model_version,
    )

    return result, model, model_input, original_grayscale


def predict_image(
    content: bytes,
    image_format: str,
) -> PredictionOutput:
    """Backward-compatible M3 inference interface."""
    result, _, _, _ = predict_with_context(
        content=content,
        image_format=image_format,
        model_name="resnet50",
    )
    return result