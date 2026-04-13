from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torchvision.transforms import InterpolationMode


BBox = list[float] | tuple[float, float, float, float] | None


@dataclass
class TransformState:
    angle: float = 0.0
    flipped: bool = False


def sanitize_bbox(bbox: BBox, width: int, height: int) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 1.0), float(width))
    y2 = min(max(y2, 1.0), float(height))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [x1, y1, x2, y2]


def rotate_point(x: float, y: float, cx: float, cy: float, angle_deg: float) -> tuple[float, float]:
    angle = math.radians(angle_deg)
    tx, ty = x - cx, y - cy
    rx = tx * math.cos(angle) - ty * math.sin(angle)
    ry = tx * math.sin(angle) + ty * math.cos(angle)
    return rx + cx, ry + cy


def rotate_bbox(bbox: list[float] | None, width: int, height: int, angle_deg: float) -> list[float] | None:
    if bbox is None or abs(angle_deg) < 1e-6:
        return bbox
    x1, y1, x2, y2 = bbox
    cx, cy = width / 2.0, height / 2.0
    corners = [
        rotate_point(x1, y1, cx, cy, angle_deg),
        rotate_point(x1, y2, cx, cy, angle_deg),
        rotate_point(x2, y1, cx, cy, angle_deg),
        rotate_point(x2, y2, cx, cy, angle_deg),
    ]
    xs, ys = zip(*corners)
    return sanitize_bbox([min(xs), min(ys), max(xs), max(ys)], width=width, height=height)


def flip_bbox_horizontal(bbox: list[float] | None, width: int, height: int) -> list[float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return sanitize_bbox([width - x2, y1, width - x1, y2], width=width, height=height)


def expand_bbox(bbox: list[float] | None, width: int, height: int, expand_ratio: float) -> list[int]:
    if bbox is None:
        return [0, 0, width, height]
    x1, y1, x2, y2 = bbox
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    bw = max(x2 - x1, 2.0) * expand_ratio
    bh = max(y2 - y1, 2.0) * expand_ratio
    crop = [
        max(int(round(cx - bw / 2.0)), 0),
        max(int(round(cy - bh / 2.0)), 0),
        min(int(round(cx + bw / 2.0)), width),
        min(int(round(cy + bh / 2.0)), height),
    ]
    if crop[2] - crop[0] < 4 or crop[3] - crop[1] < 4:
        return [0, 0, width, height]
    return crop


def bbox_to_prior(
    bbox: list[float] | None,
    width: int,
    height: int,
    sigma_scale: float = 0.35,
    background: float = 0.15,
    normalize: bool = True,
) -> torch.Tensor:
    if bbox is None:
        return torch.full((1, height, width), background, dtype=torch.float32)
    x1, y1, x2, y2 = bbox
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32), torch.arange(width, dtype=torch.float32), indexing="ij")
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    sx, sy = max((x2 - x1) * sigma_scale, 2.0), max((y2 - y1) * sigma_scale, 2.0)
    gaussian = torch.exp(-(((xx - cx) ** 2) / (2 * sx**2) + ((yy - cy) ** 2) / (2 * sy**2)))
    rect = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
    prior = background + (1.0 - background) * torch.maximum(gaussian, rect)
    if normalize:
        prior = prior / prior.max().clamp_min(1e-6)
    return prior.unsqueeze(0)


class DualViewTransform:
    def __init__(self, cfg: dict[str, Any], split: str) -> None:
        self.cfg = cfg
        self.split = split
        self.split_cfg = cfg["train"] if split == "train" else cfg["eval"]
        self.image_size = int(cfg["image_size"])
        self.local_image_size = int(cfg.get("local_image_size", self.image_size))
        self.crop_expand_ratio = float(cfg["crop_expand_ratio"])
        self.normalize_mean = cfg["normalize"]["mean"]
        self.normalize_std = cfg["normalize"]["std"]

    def _sample_state(self) -> TransformState:
        if self.split != "train":
            return TransformState()
        return TransformState(
            angle=random.uniform(-self.split_cfg["rotation_deg"], self.split_cfg["rotation_deg"]),
            flipped=random.random() < self.split_cfg["hflip_prob"],
        )

    def _apply_geom(self, image: Image.Image, bbox: list[float] | None, state: TransformState) -> tuple[Image.Image, list[float] | None]:
        width, height = image.size
        if abs(state.angle) > 1e-6:
            image = TF.rotate(image, state.angle, interpolation=InterpolationMode.BILINEAR)
            bbox = rotate_bbox(bbox, width=width, height=height, angle_deg=state.angle)
        if state.flipped:
            image = TF.hflip(image)
            bbox = flip_bbox_horizontal(bbox, width=image.size[0], height=image.size[1])
        return image, sanitize_bbox(bbox, width=image.size[0], height=image.size[1])

    def _apply_appearance(self, image: Image.Image) -> Image.Image:
        if self.split != "train":
            return image
        cj = self.split_cfg["color_jitter"]
        image = TF.adjust_brightness(image, random.uniform(max(0.0, 1 - cj["brightness"]), 1 + cj["brightness"]))
        image = TF.adjust_contrast(image, random.uniform(max(0.0, 1 - cj["contrast"]), 1 + cj["contrast"]))
        image = TF.adjust_saturation(image, random.uniform(max(0.0, 1 - cj["saturation"]), 1 + cj["saturation"]))
        image = TF.adjust_hue(image, random.uniform(-cj["hue"], cj["hue"]))
        if random.random() < self.split_cfg["blur_prob"]:
            image = image.filter(ImageFilter.GaussianBlur(radius=self.split_cfg["blur_kernel"] / 3.0))
        return image

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        tensor = TF.to_tensor(image)
        noise_std = float(self.split_cfg.get("noise_std", 0.0))
        if self.split == "train" and noise_std > 0:
            tensor = torch.clamp(tensor + torch.randn_like(tensor) * noise_std, 0.0, 1.0)
        return TF.normalize(tensor, self.normalize_mean, self.normalize_std)

    def __call__(self, image: Image.Image, bbox: BBox) -> dict[str, Any]:
        bbox = sanitize_bbox(bbox, width=image.size[0], height=image.size[1])
        state = self._sample_state()
        image, bbox = self._apply_geom(image, bbox, state)
        aug_image = self._apply_appearance(image)

        width, height = aug_image.size
        prior = bbox_to_prior(
            bbox=bbox,
            width=width,
            height=height,
            sigma_scale=self.cfg["attention"]["sigma_scale"],
            background=self.cfg["attention"]["background"],
            normalize=self.cfg["attention"].get("normalize", True),
        )
        crop_box = expand_bbox(bbox, width=width, height=height, expand_ratio=self.crop_expand_ratio)
        local_image = aug_image.crop(tuple(crop_box))
        global_resized = TF.resize(aug_image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        local_resized = TF.resize(local_image, [self.local_image_size, self.local_image_size], interpolation=InterpolationMode.BILINEAR)
        prior_resized = TF.resize(prior, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)

        return {
            "global_image": self._to_tensor(global_resized),
            "local_image": self._to_tensor(local_resized),
            "attention_prior": prior_resized.float(),
            "bbox": torch.tensor(bbox if bbox is not None else [-1, -1, -1, -1], dtype=torch.float32),
            "crop_box": torch.tensor(crop_box, dtype=torch.float32),
            "has_bbox": float(bbox is not None),
        }
