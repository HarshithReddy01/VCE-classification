from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.attention import AttentionAlignmentRegularizer, BBoxSpatialAttention
from src.models.backbones import build_backbone
from src.models.fusion import GatedFusion
from src.models.losses import SupervisedContrastiveLoss, ViewConsistencyLoss


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, use_bn: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.extend([nn.GELU(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LADLNet(nn.Module):
    def __init__(self, num_classes: int, class_names: list[str], model_cfg: dict[str, Any], loss_cfg: dict[str, Any]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.class_names = class_names
        self.mode = model_cfg.get("mode", "dual_view")
        self.use_bbox_attention = bool(model_cfg.get("use_bbox_attention", True))
        self.use_aux_heads = bool(model_cfg.get("use_aux_heads", True))

        self.backbone = build_backbone(model_cfg["backbone"], fallback_cfg=model_cfg.get("fallback_backbone"))
        feature_dim = self.backbone.feature_dim
        emb_dim = int(model_cfg["embedding_dim"])
        proj_dim = int(model_cfg["projection_dim"])
        dropout = float(model_cfg.get("dropout", 0.0))

        self.global_proj = ProjectionHead(feature_dim, emb_dim, use_bn=bool(model_cfg.get("use_projection_bn", True)), dropout=dropout)
        self.local_proj = ProjectionHead(feature_dim, emb_dim, use_bn=bool(model_cfg.get("use_projection_bn", True)), dropout=dropout)
        self.fusion = GatedFusion(feature_dim=emb_dim, hidden_dim=int(model_cfg["fusion_hidden_dim"]), dropout=dropout)
        self.attention = BBoxSpatialAttention(model_cfg["bbox_attention"])
        self.attn_regularizer = AttentionAlignmentRegularizer()
        self.embedding_head = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(proj_dim, num_classes)
        self.global_classifier = nn.Linear(emb_dim, num_classes)
        self.local_classifier = nn.Linear(emb_dim, num_classes)

        self.supcon = SupervisedContrastiveLoss(temperature=float(loss_cfg.get("supcon_temperature", 0.1)))
        self.view_consistency = ViewConsistencyLoss(loss_type=loss_cfg.get("consistency_loss", "mse"))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        global_view = self.backbone(batch["global_image"])
        local_view = self.backbone(batch["local_image"])

        if self.use_bbox_attention:
            global_map, learned_attention = self.attention(global_view.feature_map, batch["attention_prior"])
            global_pooled = global_map.mean(dim=(2, 3))
        else:
            global_pooled = global_view.pooled
            learned_attention = batch["attention_prior"]

        local_pooled = local_view.pooled
        global_emb = self.global_proj(global_pooled)
        local_emb = self.local_proj(local_pooled)

        if self.mode == "global_only":
            fused_emb, gate = global_emb, torch.zeros_like(global_emb)
        elif self.mode == "local_only":
            fused_emb, gate = local_emb, torch.ones_like(local_emb)
        else:
            fused_emb, gate = self.fusion(global_emb, local_emb)

        fused_proj = self.embedding_head(fused_emb)
        return {
            "logits": self.classifier(fused_proj),
            "fused_embedding": fused_proj,
            "global_embedding": global_emb,
            "local_embedding": local_emb,
            "global_logits": self.global_classifier(global_emb),
            "local_logits": self.local_classifier(local_emb),
            "learned_attention": learned_attention,
            "gate": gate,
        }
