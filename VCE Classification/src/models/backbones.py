from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import timm
import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    feature_map: torch.Tensor
    pooled: torch.Tensor


class TIMMBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
            dynamic_img_size=True,
        )
        self.feature_dim = self._infer_dim()

    def _infer_dim(self) -> int:
        if hasattr(self.model, "num_features"):
            return int(self.model.num_features)
        if hasattr(self.model, "feature_info"):
            channels = self.model.feature_info.channels()
            if channels:
                return int(channels[-1])
        raise AttributeError("Unable to infer backbone feature dimension.")

    def _vit_to_map(self, tokens: torch.Tensor) -> torch.Tensor:
        patch_tokens = tokens[:, 1:, :] if tokens.shape[1] > 1 and int((tokens.shape[1] - 1) ** 0.5) ** 2 == tokens.shape[1] - 1 else tokens
        batch, num_tokens, channels = patch_tokens.shape
        grid = int(num_tokens**0.5)
        if grid * grid != num_tokens:
            raise ValueError(f"Cannot reshape {num_tokens} tokens into a square feature map.")
        return patch_tokens.transpose(1, 2).reshape(batch, channels, grid, grid)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        features = self.model.forward_features(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if features.ndim == 4:
            feature_map = features
            pooled = feature_map.mean(dim=(2, 3))
        elif features.ndim == 3:
            feature_map = self._vit_to_map(features)
            pooled = feature_map.mean(dim=(2, 3))
        else:
            raise ValueError(f"Unsupported backbone output shape: {features.shape}")
        return BackboneOutput(feature_map=feature_map, pooled=pooled)


def build_backbone(backbone_cfg: dict[str, Any], fallback_cfg: dict[str, Any] | None = None) -> TIMMBackbone:
    name = backbone_cfg.get("name", "dinov2")
    model_name = backbone_cfg.get("model_name")
    aliases = {
        "dinov2": model_name or "vit_base_patch14_dinov2.lvd142m",
        "mamba_vision": model_name or "mambaout_small.in1k",
    }
    resolved_name = aliases.get(name, model_name or name)
    try:
        return TIMMBackbone(
            resolved_name,
            pretrained=bool(backbone_cfg.get("pretrained", True)),
            drop_path_rate=float(backbone_cfg.get("drop_path_rate", 0.0)),
        )
    except Exception as exc:
        if fallback_cfg is None:
            raise RuntimeError(f"Failed to create backbone `{resolved_name}`.") from exc
        return TIMMBackbone(
            fallback_cfg.get("model_name", fallback_cfg.get("name", "resnet50")),
            pretrained=bool(fallback_cfg.get("pretrained", True)),
        )
