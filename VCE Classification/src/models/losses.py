from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, class_counts: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("class_counts", class_counts.float().clamp_min(1.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.class_counts.log().unsqueeze(0), target)


class LogitAdjustedCrossEntropy(nn.Module):
    def __init__(self, class_counts: torch.Tensor, tau: float = 1.0) -> None:
        super().__init__()
        priors = class_counts.float().clamp_min(1.0)
        priors = priors / priors.sum()
        self.register_buffer("adjustment", tau * priors.log())

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adjustment.unsqueeze(0), target)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", weight=self.weight)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        logits = embeddings @ embeddings.t() / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        labels = labels.view(-1, 1)
        mask = (labels == labels.t()).float()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return -mean_log_prob_pos.mean()


class ViewConsistencyLoss(nn.Module):
    def __init__(self, loss_type: str = "mse") -> None:
        super().__init__()
        self.loss_type = loss_type

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor, has_bbox: torch.Tensor, missing_weight: float = 0.25) -> torch.Tensor:
        raw = 1.0 - F.cosine_similarity(lhs, rhs, dim=1) if self.loss_type == "cosine" else F.mse_loss(lhs, rhs, reduction="none").mean(dim=1)
        weights = has_bbox.float() + (1.0 - has_bbox.float()) * missing_weight
        return (raw * weights).mean()


def compute_class_weights(class_counts: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    inv = class_counts.float().clamp_min(1.0).pow(-power)
    return inv / inv.sum() * len(inv)


def build_primary_loss(loss_cfg: dict[str, Any], class_counts: torch.Tensor) -> nn.Module:
    primary = loss_cfg.get("primary", "balanced_softmax")
    if primary == "balanced_softmax":
        return BalancedSoftmaxLoss(class_counts)
    if primary == "weighted_ce":
        return nn.CrossEntropyLoss(weight=compute_class_weights(class_counts, power=float(loss_cfg.get("weighted_ce_power", 0.5))))
    if primary == "focal":
        return FocalLoss(
            gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            weight=compute_class_weights(class_counts, power=float(loss_cfg.get("weighted_ce_power", 0.5))),
        )
    if primary == "ce":
        return nn.CrossEntropyLoss(label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    if primary == "logit_adjusted":
        return LogitAdjustedCrossEntropy(class_counts=class_counts, tau=float(loss_cfg.get("logit_adjustment_tau", 1.0)))
    raise ValueError(f"Unsupported primary loss: {primary}")
