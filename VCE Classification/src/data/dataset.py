from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.transforms import DualViewTransform


def infer_video_id(filename: str) -> str:
    stem = Path(filename).stem
    return "_".join(stem.split("_")[:-1]) if "_" in stem else stem


def infer_frame_number(filename: str) -> int:
    stem = Path(filename).stem
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return -1


def resolve_bbox(row: pd.Series) -> list[float] | None:
    corners = [
        ("x1", "y1", "x2", "y2"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("xmin", "ymin", "xmax", "ymax"),
    ]
    for cols in corners:
        if all(col in row.index for col in cols):
            values = [row[c] for c in cols]
            if any(pd.isna(v) for v in values):
                return None
            return [float(v) for v in values]
    wh_cols = [("bbox_x", "bbox_y", "bbox_w", "bbox_h"), ("x", "y", "w", "h")]
    for cols in wh_cols:
        if all(col in row.index for col in cols):
            x, y, w, h = [row[c] for c in cols]
            if any(pd.isna(v) for v in (x, y, w, h)):
                return None
            return [float(x), float(y), float(x + w), float(y + h)]
    return None


def scan_class_folders(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        for image_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
            rows.append(
                {
                    "filename": str(image_path.relative_to(dataset_root)),
                    "label": class_dir.name,
                    "video_id": infer_video_id(image_path.name),
                    "frame_number": infer_frame_number(image_path.name),
                    "x1": np.nan,
                    "y1": np.nan,
                    "x2": np.nan,
                    "y2": np.nan,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No images found under {dataset_root}.")
    return pd.DataFrame(rows)


class KvasirCapsuleDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        metadata_path: str | Path | None,
        split_csv: str | Path | None,
        classes: list[str] | None,
        use_filtered_classes: bool,
        transform_cfg: dict[str, Any],
        split: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.transform = DualViewTransform(transform_cfg, split=split)
        self.split = split
        self.samples = self._load_dataframe(metadata_path=metadata_path, split_csv=split_csv)
        if use_filtered_classes and classes:
            self.samples = self.samples[self.samples["label"].isin(classes)].reset_index(drop=True)
        self.classes = list(classes) if classes else sorted(self.samples["label"].astype(str).unique().tolist())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = self.samples[self.samples["label"].isin(self.classes)].reset_index(drop=True)
        self.samples["target"] = self.samples["label"].map(self.class_to_idx)
        self.class_counts = self.samples["target"].value_counts().sort_index().reindex(range(len(self.classes)), fill_value=0)

    def _load_dataframe(self, metadata_path: str | Path | None, split_csv: str | Path | None) -> pd.DataFrame:
        if split_csv and Path(split_csv).exists():
            df = pd.read_csv(split_csv)
        elif metadata_path and Path(metadata_path).exists():
            df = pd.read_csv(metadata_path)
        else:
            warnings.warn("metadata.csv not found; falling back to directory scan without bounding boxes.", stacklevel=2)
            df = scan_class_folders(self.dataset_root)
        if "filename" not in df.columns:
            raise ValueError("Metadata must contain a `filename` column.")
        if "label" not in df.columns:
            raise ValueError("Metadata must contain a `label` column.")
        if "video_id" not in df.columns:
            df["video_id"] = df["filename"].astype(str).map(infer_video_id)
        if "frame_number" not in df.columns:
            df["frame_number"] = df["filename"].astype(str).map(infer_frame_number)
        return df

    def build_sampler(self, sampler_cfg: dict[str, Any]) -> WeightedRandomSampler | None:
        if not sampler_cfg.get("class_balanced", False):
            return None
        counts = self.class_counts.to_numpy(dtype=np.float32)
        beta = float(sampler_cfg.get("beta", 0.999))
        effective_num = 1.0 - np.power(beta, np.maximum(counts, 1.0))
        weights_per_class = (1.0 - beta) / np.clip(effective_num, 1e-8, None)
        sample_weights = weights_per_class[self.samples["target"].to_numpy()]
        return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image_path = self.dataset_root / str(row["filename"])
        if not image_path.exists():
            fallback = self.dataset_root / Path(str(row["filename"])).name
            if fallback.exists():
                image_path = fallback
            else:
                raise FileNotFoundError(f"Missing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        transformed = self.transform(image=image, bbox=resolve_bbox(row))
        transformed.update(
            {
                "target": int(row["target"]),
                "label": str(row["label"]),
                "filename": str(row["filename"]),
                "video_id": str(row["video_id"]),
                "frame_number": int(row["frame_number"]),
            }
        )
        return transformed
