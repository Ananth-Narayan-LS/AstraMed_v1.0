
"""Central configuration for the AstraMed API."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from the repository root.
# Existing system environment variables take precedence.
load_dotenv(PROJECT_ROOT / ".env", override=False)


SUPPORTED_MODELS = frozenset({"resnet50", "vit_b16"})
VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


def _read_positive_int(name: str, default: int) -> int:
    """Read an environment variable that must be a positive integer."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive integer."
        ) from exc

    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return value


def _read_probability(name: str, default: float) -> float:
    """Read a numeric value between zero and one, inclusive."""
    raw_value = os.getenv(name, str(default))

    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a number between 0 and 1."
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")

    return value


def _read_log_level() -> str:
    """Read and validate the configured logging level."""
    value = os.getenv("ASTRAMED_LOG_LEVEL", "INFO").strip().upper()

    if value not in VALID_LOG_LEVELS:
        raise ValueError(
            "ASTRAMED_LOG_LEVEL must be DEBUG, INFO, WARNING, "
            "ERROR, or CRITICAL."
        )

    return value


@dataclass(frozen=True)
class APISettings:
    """Validated settings used by the AstraMed API."""

    app_name: str = "AstraMed API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"

    max_upload_mb: int = 20
    max_image_pixels: int = 25_000_000

    default_model: str = "resnet50"
    prediction_threshold: float = 0.5

    environment: str = "development"

    @classmethod
    def from_environment(cls) -> "APISettings":
        """Build validated settings from environment variables."""
        app_name = os.getenv(
            "ASTRAMED_APP_NAME",
            os.getenv(
                "ASTRAMED_API_TITLE",
                "AstraMed API",
            ),
        ).strip()

        app_version = os.getenv(
            "ASTRAMED_APP_VERSION",
            os.getenv("ASTRAMED_API_VERSION", "1.0.0"),
        ).strip()

        host = os.getenv("ASTRAMED_HOST", "127.0.0.1").strip()

        port = _read_positive_int("ASTRAMED_PORT", 8000)

        log_level = _read_log_level()

        max_upload_mb = _read_positive_int(
            "ASTRAMED_MAX_UPLOAD_MB",
            20,
        )

        max_image_pixels = _read_positive_int(
            "ASTRAMED_MAX_IMAGE_PIXELS",
            25_000_000,
        )

        default_model = os.getenv(
            "ASTRAMED_DEFAULT_MODEL",
            "resnet50",
        ).strip().lower()

        if default_model not in SUPPORTED_MODELS:
            raise ValueError(
                "ASTRAMED_DEFAULT_MODEL must be one of: "
                "resnet50, vit_b16."
            )

        prediction_threshold = _read_probability(
            "ASTRAMED_PREDICTION_THRESHOLD",
            0.5,
        )

        environment = os.getenv(
            "ASTRAMED_API_ENV",
            "development",
        ).strip().lower()

        if not app_name:
            raise ValueError("ASTRAMED_APP_NAME cannot be empty.")

        if not app_version:
            raise ValueError("ASTRAMED_APP_VERSION cannot be empty.")

        if not host:
            raise ValueError("ASTRAMED_HOST cannot be empty.")

        if environment not in {
            "development",
            "testing",
            "production",
        }:
            raise ValueError(
                "ASTRAMED_API_ENV must be development, testing, "
                "or production."
            )

        return cls(
            app_name=app_name,
            app_version=app_version,
            host=host,
            port=port,
            log_level=log_level,
            max_upload_mb=max_upload_mb,
            max_image_pixels=max_image_pixels,
            default_model=default_model,
            prediction_threshold=prediction_threshold,
            environment=environment,
        )


# Create one validated settings object for the application.
settings = APISettings.from_environment()


# Compatibility constants for existing modules.
# These preserve imports used by the current API code.
API_TITLE = settings.app_name
API_VERSION = settings.app_version
API_ENV = settings.environment
LOG_LEVEL = settings.log_level

MAX_UPLOAD_MB = settings.max_upload_mb
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_IMAGE_PIXELS = settings.max_image_pixels

DEFAULT_MODEL = settings.default_model
PREDICTION_THRESHOLD = settings.prediction_threshold
