
from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError
import pydicom
from pydicom.errors import InvalidDicomError


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


def validate_uploaded_image(
    content: bytes,
    filename: str,
    max_bytes: int,
) -> ValidatedImage:
    """
    Validate uploaded PNG, JPEG, or DICOM content.

    The caller must enforce the byte limit while reading the upload.
    This function performs content-level validation.
    """
    if not content:
        raise UploadValidationError(
            "EMPTY_FILE",
            "The uploaded file is empty.",
            400,
        )

    if len(content) > max_bytes:
        raise UploadValidationError(
            "FILE_TOO_LARGE",
            f"The uploaded file exceeds the {max_bytes}-byte limit.",
            413,
        )

    # First try ordinary raster images using their actual file content.
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

            if width <= 0 or height <= 0:
                raise UploadValidationError(
                    "INVALID_IMAGE_DIMENSIONS",
                    "The image dimensions are invalid.",
                    400,
                )

            # Verify the encoded image is structurally valid.
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
            400,
        ) from exc

    except (UnidentifiedImageError, OSError, SyntaxError):
        # Not a valid PNG/JPEG; try parsing it as DICOM below.
        pass

    # DICOM is not a Pillow image. Parse it independently.
    try:
        dataset = pydicom.dcmread(
            BytesIO(content),
            force=False,
            stop_before_pixels=False,
        )
    except (InvalidDicomError, OSError, ValueError) as exc:
        raise UploadValidationError(
            "INVALID_IMAGE_FILE",
            "The file is not a valid supported PNG, JPEG, or DICOM image.",
            400,
        ) from exc

    # Check that this looks like an image DICOM, not just any DICOM object.
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

    rows = int(dataset.Rows)
    columns = int(dataset.Columns)

    if rows <= 0 or columns <= 0:
        raise UploadValidationError(
            "INVALID_IMAGE_DIMENSIONS",
            "The DICOM image dimensions are invalid.",
            400,
        )

    return ValidatedImage(
        filename=filename,
        image_format="DICOM",
        content_type="application/dicom",
        size_bytes=len(content),
        width=columns,
        height=rows,
    )
