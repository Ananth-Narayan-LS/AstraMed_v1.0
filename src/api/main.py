from fastapi import FastAPI

from src.api.config import settings
from src.api.logging_config import configure_logging
from src.api.routes import health, predict


configure_logging()

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
)

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
