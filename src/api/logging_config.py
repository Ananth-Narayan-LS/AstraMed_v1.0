import logging

from src.api.config import settings


def configure_logging():
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    logging.getLogger("uvicorn.access").setLevel(level)
