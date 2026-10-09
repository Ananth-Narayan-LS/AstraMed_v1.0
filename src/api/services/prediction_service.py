import logging
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import pydicom
import torch
from PIL import Image, UnidentifiedImageError

from src.data.advanced_input import AdvancedInputAdapter
from src.data.preprocessing import preprocess_image
from src.models.model_factory import build_model
from src.utils.config import PROJECT_ROOT


logger = logging.getLogger(__name__)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "resnet50_pretrained_img224_bs4_lr0.0001_ep2_seed42"
    / "checkpoints"
    / "best_model.pth"
)

PREDICTION_THRESHOLD = 0.5


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


@lru_cache(maxsize=1)
def _load_model_bundle():
    """Load the verified local checkpoint once per process."""

    if not CHECKPOINT_PATH.is_file():
        raise PredictionServiceError(
            "CHECKPOINT_NOT_FOUND",
            "The configured ResNet-50 checkpoint was not found.",
            503,
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )

        if "model_state_dict" not in checkpoint:
            raise PredictionServiceError(
                "INVALID_CHECKPOINT",
                "The checkpoint has no model_state_dict.",
                500,
            )

        model = build_model(
            "resnet50",
            pretrained=False,
        )

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

        logger.info(
            "Loaded ResNet-50 checkpoint: device=%s epoch=%s",
            device,
            epoch,
        )

        return model, adapter, device, str(epoch)

    except PredictionServiceError:
        raise

    except Exception as exc:
        logger.exception("Could not load the prediction checkpoint.")
        raise PredictionServiceError(
            "MODEL_LOAD_FAILED",
            "The local prediction model could not be loaded.",
            503,
        ) from exc


def _decode_uploaded_image(
    content: bytes,
    image_format: str,
) -> np.ndarray:
    """
    Decode an already validated upload into a 2D grayscale array.

    The array is passed through AstraMed's existing preprocessing.
    """

    try:
        if image_format == "DICOM":
            dataset = pydicom.dcmread(
                BytesIO(content),
                force=False,
            )
            image = dataset.pixel_array.astype(np.float32)

        elif image_format in {"PNG", "JPEG"}:
            with Image.open(BytesIO(content)) as source:
                image = np.asarray(
                    source.convert("L")
                ).copy()

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


def predict_image(
    content: bytes,
    image_format: str,
) -> PredictionOutput:
    """Run local ResNet-50 pneumonia inference."""

    model, adapter, device, epoch = _load_model_bundle()

    image = _decode_uploaded_image(
        content=content,
        image_format=image_format,
    )

    # Existing AstraMed preprocessing: grayscale -> 512x512 -> [0, 1].
    processed = preprocess_image(
        image,
        target_size=(512, 512),
    )

    # Convert [H, W] NumPy array to [B, C, H, W].
    input_tensor = torch.from_numpy(
        processed
    ).float().unsqueeze(0).unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # Existing adapter: [1, 1, 512, 512] -> [1, 3, 224, 224].
    input_tensor = adapter(input_tensor)

    try:
        with torch.inference_mode():
            logits = model(input_tensor)

            # This model returns logits, not probabilities.
            probability = torch.sigmoid(logits).reshape(-1)[0].item()

    except Exception as exc:
        logger.exception("Local model inference failed.")
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

    return PredictionOutput(
        pneumonia_probability=float(probability),
        predicted_class=predicted_class,
        threshold=PREDICTION_THRESHOLD,
        model_name="resnet50",
        model_version=f"checkpoint-epoch-{epoch}",
    )
