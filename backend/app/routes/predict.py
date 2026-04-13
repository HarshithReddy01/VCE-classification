from __future__ import annotations

import io
from typing import Annotated

import torch
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.core.config import CLASS_NAMES
from app.schemas.prediction import ModelInfo, PredictionResponse

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"}
_MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def _read_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is not a recognised image format.",
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a capsule endoscopy image",
    description=(
        "Upload a single image (JPEG / PNG / WebP / BMP / TIFF). "
        "Returns the predicted Kvasir-Capsule finding, its softmax confidence, "
        "and the full probability distribution across all 13 classes."
    ),
)
async def predict(
    request: Request,
    file: Annotated[UploadFile, File(description="Capsule endoscopy frame to classify")],
) -> PredictionResponse:
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Accepted: {', '.join(sorted(_ALLOWED_CONTENT_TYPES))}",
        )

    raw = await file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    image = _read_image(raw)

    model = request.app.state.model
    transform = request.app.state.transform
    device: str = request.app.state.device

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=-1).squeeze(0)

    class_id = int(probs.argmax().item())

    return PredictionResponse(
        class_id=class_id,
        class_name=CLASS_NAMES[class_id],
        confidence=float(probs[class_id].item()),
        probabilities=probs.tolist(),
    )


@router.get("/model-info", response_model=ModelInfo, summary="Checkpoint and model metadata")
async def model_info(request: Request) -> ModelInfo:
    meta = request.app.state.checkpoint_meta
    settings = request.app.state.settings
    return ModelInfo(
        architecture="DINOv2 ViT-B/14 + dual-stream classifier (LADL-Net)",
        num_classes=settings.num_classes,
        class_names=CLASS_NAMES,
        checkpoint_epoch=meta["epoch"],
        checkpoint_metric=meta["metric"],
        checkpoint_value=meta["value"],
        device=settings.device,
    )
