from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

CLASS_NAMES: list[str] = [
    "ampulla_of_vater",
    "angiectasia",
    "blood_fresh",
    "blood_hematin",
    "erosion",
    "erythema",
    "foreign_body",
    "ileocecal_valve",
    "lymphangiectasia",
    "normal_clean_mucosa",
    "pylorus",
    "reduced_mucosal_view",
    "ulcer",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    checkpoint_path: Path = Path("/app/checkpoints/best_mcc.pt")
    hf_repo_id: str = ""
    hf_filename: str = "best_mcc.pt"
    num_classes: int = len(CLASS_NAMES)
    image_size: int = 518
    device: str = "cpu"


@lru_cache
def get_settings() -> Settings:
    return Settings()
