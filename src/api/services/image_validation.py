
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError
import pydicom
from pydicom.errors import InvalidDicomError

from src.api.config import MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class ValidatedImage:
    filename: str
    image_format: str
    content_type: str
    size_bytes: int
    width: int | None = None
    height: int | None = None


class UploadValidationError(Exception):
    def __init__(self, code: str, message: str, status_code: int):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _validate_dimensions(width: int, height: int) -> None:
    """Reject invalid dimensions and images that exceed the pixel limit."""
    if width <= 0 or height <= 0:
        raise UploadValidationError(
            "INVALID_IMAGE_DIMENSIONS",
            "The image dimensions are invalid.",
            400,
        )

    if width * height > MAX_IMAGE_PIXELS:
        raise UploadValidationError(
            "IMAGE_TOO_LARGE_TO_DECODE",
            "The image dimensions exceed the configured pixel limit.",
            413,
        )


def validate_uploaded_image(
    content: bytes,
    filename: str,
    max_bytes: int,
) -> ValidatedImage:
    """
    Validate PNG, JPEG, or DICOM content.

    The caller should enforce the byte limit while reading the upload.
    This function also checks the supplied content length independently.
    """
    if not content:
        raise UploadValidationError(
            "EMPTY_FILE",
            "The uploaded file is empty.",
            400,
        )

    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")

    if len(content) > max_bytes:
        raise UploadValidationError(
            "FILE_TOO_LARGE",
            f"The uploaded file exceeds the {max_bytes}-byte limit.",
            413,
        )

    # First inspect the actual content as a PNG or JPEG.
    try:
        with Image.open(BytesIO(content)) as image:
            detected_format = image.format

            if detected_format not in {"PNG", "JPEG"}:
                raise UploadValidationError(
                    "UNSUPPORTED_IMAGE_FORMAT",
                    "Only PNG, JPEG, and DICOM files are supported.",
                    415,
                )

            width, height = image.size
            _validate_dimensions(width, height)

            # Verify the encoded file without retaining a decoded image.
            image.verify()

        return ValidatedImage(
            filename=filename,
            image_format=detected_format,
            content_type=(
                "image/png" if detected_format == "PNG"
                else "image/jpeg"
            ),
            size_bytes=len(content),
            width=width,
            height=height,
        )

    except UploadValidationError:
        raise

    except Image.DecompressionBombError as exc:
        raise UploadValidationError(
            "IMAGE_TOO_LARGE_TO_DECODE",
            "The image dimensions exceed the safe decoding limit.",
            413,
        ) from exc

    except Image.DecompressionBombWarning as exc:
        # Pillow can warn about very large images before decoding.
        raise UploadValidationError(
            "IMAGE_TOO_LARGE_TO_DECODE",
            "The image dimensions exceed the safe decoding limit.",
            413,
        ) from exc

    except (UnidentifiedImageError, OSError, SyntaxError):
        # It was not a readable PNG/JPEG; try DICOM parsing.
        pass

    # DICOM is parsed separately from raster images.
    try:
        dataset = pydicom.dcmread(
            BytesIO(content),
            force=False,
        )
    except (InvalidDicomError, OSError, ValueError, EOFError) as exc:
        raise UploadValidationError(
            "INVALID_IMAGE_FILE",
            "The file is not a valid supported PNG, JPEG, or DICOM image.",
            400,
        ) from exc

    # Confirm this is an image DICOM with pixel data.
    if not hasattr(dataset, "Rows") or not hasattr(dataset, "Columns"):
        raise UploadValidationError(
            "INVALID_DICOM_IMAGE",
            "The DICOM file is missing image dimensions.",
            400,
        )

    if "PixelData" not in dataset:
        raise UploadValidationError(
            "MISSING_PIXEL_DATA",
            "The DICOM file does not contain image pixel data.",
            400,
        )

    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UploadValidationError(
            "INVALID_IMAGE_DIMENSIONS",
            "The DICOM image dimensions are invalid.",
            400,
        ) from exc

    _validate_dimensions(columns, rows)

    return ValidatedImage(
        filename=filename,
        image_format="DICOM",
        content_type="application/dicom",
        size_bytes=len(content),
        width=columns,
        height=rows,
    )
