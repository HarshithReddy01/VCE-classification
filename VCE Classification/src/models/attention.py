from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BBoxSpatialAttention(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.mode = cfg.get("mode", "additive")
        self.alpha = nn.Parameter(torch.tensor(float(cfg.get("alpha", 0.8))), requires_grad=cfg.get("learnable_scale", True))
        self.temperature = float(cfg.get("temperature", 1.0))
        self.proj = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, feature_map: torch.Tensor, attention_prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        resized_prior = F.interpolate(attention_prior, size=feature_map.shape[-2:], mode="bilinear", align_corners=False)
        refined_prior = torch.sigmoid(self.proj(resized_prior) / max(self.temperature, 1e-6))
        if self.mode == "multiplicative":
            attended = feature_map * (1.0 + self.alpha * refined_prior)
        else:
            attended = feature_map + self.alpha * refined_prior * feature_map
        return attended, refined_prior


class AttentionAlignmentRegularizer(nn.Module):
    def forward(self, learned_attention: torch.Tensor, target_prior: torch.Tensor, has_bbox: torch.Tensor) -> torch.Tensor:
        target = F.interpolate(target_prior, size=learned_attention.shape[-2:], mode="bilinear", align_corners=False)
        mask = has_bbox.view(-1, 1, 1, 1)
        loss = F.smooth_l1_loss(learned_attention, target, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
