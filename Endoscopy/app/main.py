from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.model import build_transform, load_model
from app.routes.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    logger.info("Loading model from %s …", settings.checkpoint_path)
    model, meta = load_model(
        checkpoint_path=settings.checkpoint_path,
        num_classes=settings.num_classes,
        device=settings.device,
        hf_repo_id=settings.hf_repo_id,
        hf_filename=settings.hf_filename,
    )
    logger.info(
        "Model ready — epoch=%d  %s=%.4f  device=%s",
        meta["epoch"],
        meta["metric"],
        meta["value"],
        settings.device,
    )

    app.state.model = model
    app.state.transform = build_transform(settings.image_size)
    app.state.checkpoint_meta = meta
    app.state.settings = settings
    app.state.device = settings.device

    yield

    del app.state.model
    logger.info("Model unloaded.")


app = FastAPI(
    title="DINOv2 Classifier API",
    description=(
        "Dual-stream DINOv2 ViT-B/14 image classifier. "
        "POST an image to **/predict** to get class probabilities."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(predict_router, prefix="/api/v1", tags=["inference"])


@app.get("/", tags=["ops"], summary="Space root — API info")
async def root() -> dict:
    """Hugging Face and browsers hit `/` by default; without this route they would see 404."""
    return {
        "service": "VCE capsule endoscopy classifier API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /api/v1/predict (multipart image upload)",
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.get("/health", tags=["ops"], summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["ops"], summary="Readiness probe")
async def ready(request: Request) -> dict:
    if not hasattr(request.app.state, "model"):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "model not loaded"},
        )
    return {"status": "ready"}
