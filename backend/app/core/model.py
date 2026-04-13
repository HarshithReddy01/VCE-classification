from __future__ import annotations

import math
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torchvision import transforms


class ViTBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch14_dinov2",
            pretrained=False,
            num_classes=0,
            img_size=518,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)  # (B, 1+N, D)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(()))
        self.proj = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, N, _ = patch_tokens.shape
        H = W = int(math.isqrt(N))
        scores = patch_tokens.norm(dim=-1).view(B, 1, H, W)
        attn = self.proj(scores).view(B, N)
        return torch.softmax(self.alpha * attn, dim=-1)


class FusionGate(nn.Module):
    def __init__(self, in_dim: int = 1024, out_dim: int = 512) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        return self.gate(torch.cat([global_feat, local_feat], dim=-1))


class DINOv2Classifier(nn.Module):
    def __init__(self, num_classes: int = 13) -> None:
        super().__init__()
        self.backbone = ViTBackbone()
        self.global_proj = ProjectionHead(768, 512)
        self.local_proj = ProjectionHead(768, 512)
        self.attention = SpatialAttention()
        self.fusion = FusionGate(1024, 512)
        self.embedding_head = nn.Sequential(nn.Linear(512, 256))  # Sequential required: key is embedding_head.0.*
        self.classifier = nn.Linear(256, num_classes)
        self.global_classifier = nn.Linear(512, num_classes)  # aux — loaded for state-dict compat, unused at inference
        self.local_classifier = nn.Linear(512, num_classes)   # aux — loaded for state-dict compat, unused at inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(x)
        cls, patches = tokens[:, 0], tokens[:, 1:]

        global_feat = self.global_proj(cls)

        attn_weights = self.attention(patches)
        attended = (patches * attn_weights.unsqueeze(-1)).sum(1)
        local_feat = self.local_proj(attended)

        fused = self.fusion(global_feat, local_feat)
        return self.classifier(self.embedding_head(fused))


def build_transform(image_size: int = 518) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _resolve_checkpoint(path: Path, hf_repo_id: str, hf_filename: str) -> Path:
    if path.exists():
        return path
    if hf_repo_id:
        from huggingface_hub import hf_hub_download
        return Path(hf_hub_download(repo_id=hf_repo_id, filename=hf_filename))
    raise FileNotFoundError(
        f"Checkpoint not found at '{path}'. Set APP_HF_REPO_ID to download from HuggingFace Hub."
    )


def load_model(
    checkpoint_path: Path,
    num_classes: int,
    device: str,
    hf_repo_id: str = "",
    hf_filename: str = "best_mcc.pt",
) -> tuple[DINOv2Classifier, dict]:
    resolved = _resolve_checkpoint(checkpoint_path, hf_repo_id, hf_filename)
    ckpt = torch.load(resolved, map_location=device, weights_only=False)

    model = DINOv2Classifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    meta = {
        "epoch": ckpt.get("epoch", -1),
        "metric": ckpt.get("metric", ""),
        "value": ckpt.get("value", 0.0),
    }
    return model, meta
