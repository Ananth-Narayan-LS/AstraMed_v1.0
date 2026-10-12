import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.config import settings
from src.api.logging_config import configure_logging
from src.api.routes import health, predict

from src.api.middleware import RequestIDMiddleware

configure_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    """Optional fail-fast model preload (ASTRAMED_PRELOAD_MODEL=true)."""
    if settings.preload_model:
        from src.api.services.prediction_service import (
            PredictionServiceError,
            check_model_ready,
        )

        try:
            check_model_ready(settings.default_model, load=True)
        except PredictionServiceError as exc:
            logger.error(
                "Startup aborted: model=%s code=%s. Mount the checkpoint or set "
                "ASTRAMED_PRELOAD_MODEL=false.",
                settings.default_model,
                exc.code,
            )
            raise RuntimeError(
                f"Model {settings.default_model!r} is unavailable ({exc.code})."
            ) from None
        logger.info("Model %s preloaded.", settings.default_model)
    yield



app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AstraMed backend API for chest X-ray pneumonia prediction "
        "and model explanations. Research prototype; not for "
        "clinical diagnosis."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)

app.include_router(
    health.router,
    prefix=settings.api_prefix,
)

app.include_router(
    predict.router,
    prefix=settings.api_prefix,
)


@app.get("/", tags=["Health"], include_in_schema=False)
def root():
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
