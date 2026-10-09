
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class APISettings:
    app_name: str = "AstraMed API"
    app_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    max_upload_mb: int = 20

    @classmethod
    def from_environment(cls):
        return cls(
            app_name=os.getenv("ASTRAMED_APP_NAME", "AstraMed API"),
            app_version=os.getenv("ASTRAMED_APP_VERSION", "1.0.0"),
            host=os.getenv("ASTRAMED_HOST", "127.0.0.1"),
            port=int(os.getenv("ASTRAMED_PORT", "8000")),
            log_level=os.getenv("ASTRAMED_LOG_LEVEL", "INFO").upper(),
            max_upload_mb=int(os.getenv("ASTRAMED_MAX_UPLOAD_MB", "20")),
        )


settings = APISettings.from_environment()
