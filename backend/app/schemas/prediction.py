from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictionResponse(BaseModel):
    class_id: int = Field(..., ge=0)
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: list[float]

    @field_validator("probabilities")
    @classmethod
    def probabilities_sum_to_one(cls, v: list[float]) -> list[float]:
        if not (0.99 <= sum(v) <= 1.01):
            raise ValueError(f"Probabilities must sum to ~1.0, got {sum(v):.4f}")
        return v


class ModelInfo(BaseModel):
    architecture: str
    num_classes: int
    class_names: list[str]
    checkpoint_epoch: int
    checkpoint_metric: str
    checkpoint_value: float
    device: str
