from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.gate(torch.cat([global_feat, local_feat], dim=1)))
        fused = alpha * local_feat + (1.0 - alpha) * global_feat
        return fused, alpha
