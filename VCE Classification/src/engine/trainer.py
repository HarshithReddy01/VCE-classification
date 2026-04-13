from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from src.engine.evaluator import Evaluator
from src.models.losses import build_primary_loss
from src.utils.io import ensure_dir, save_checkpoint, save_json
from src.utils.visualization import plot_training_curves


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        class_names: list[str],
        output_dir: Path,
        logger: Any,
    ) -> None:
        self.cfg = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.logger = logger

        self.device = torch.device(self.cfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(self.cfg["runtime"].get("amp", True) and self.device.type == "cuda"))
        class_counts = torch.tensor(train_loader.dataset.class_counts.to_numpy(), dtype=torch.float32)
        self.primary_loss = build_primary_loss(self.cfg["loss"], class_counts).to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.evaluator = Evaluator(self.class_names, self.output_dir / "evaluation", logger, self.cfg["metrics"])
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.history = {"train_loss": [], "val_macro_f1": [], "val_mcc": []}

    def _build_optimizer(self) -> torch.optim.Optimizer:
        base_lr = float(self.cfg["optimizer"]["lr"])
        backbone_lr = base_lr * float(self.cfg["optimizer"].get("backbone_lr_scale", 1.0))
        return torch.optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": backbone_lr},
                {"params": [p for n, p in self.model.named_parameters() if not n.startswith("backbone.")], "lr": base_lr},
            ],
            lr=base_lr,
            weight_decay=float(self.cfg["optimizer"]["weight_decay"]),
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        epochs = int(self.cfg["scheduler"]["epochs"])
        min_lr = float(self.cfg["scheduler"].get("min_lr", 1e-6))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)
        warmup_epochs = int(self.cfg["scheduler"].get("warmup_epochs", 0))
        if warmup_epochs <= 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=warmup_epochs)
        return torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_epochs])

    def _compute_losses(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        target = batch["target"].to(self.device, non_blocking=True)
        has_bbox = batch["has_bbox"].to(self.device, non_blocking=True)
        loss_cfg = self.cfg["loss"]

        cls_loss = self.primary_loss(outputs["logits"], target)
        supcon_loss = self.model.supcon(outputs["fused_embedding"], target) if loss_cfg["lambda_supcon"] > 0 else outputs["logits"].new_tensor(0.0)
        if loss_cfg.get("consistency_on", "embeddings") == "logits":
            lhs, rhs = outputs["global_logits"], outputs["local_logits"]
        else:
            lhs, rhs = outputs["global_embedding"], outputs["local_embedding"]
        view_loss = self.model.view_consistency(
            lhs,
            rhs,
            has_bbox=has_bbox,
            missing_weight=float(loss_cfg.get("consistency_weight_missing_bbox", 0.25)),
        ) if loss_cfg["lambda_view"] > 0 else outputs["logits"].new_tensor(0.0)
        attn_loss = self.model.attn_regularizer(
            outputs["learned_attention"],
            batch["attention_prior"].to(self.device, non_blocking=True),
            has_bbox=has_bbox,
        ) if loss_cfg["lambda_attn"] > 0 and self.cfg["model"]["attention_alignment"]["enabled"] else outputs["logits"].new_tensor(0.0)
        aux_loss = outputs["logits"].new_tensor(0.0)
        if self.cfg["model"].get("use_aux_heads", True) and loss_cfg.get("lambda_aux", 0.0) > 0:
            aux_loss = 0.5 * (self.primary_loss(outputs["global_logits"], target) + self.primary_loss(outputs["local_logits"], target))
        total = (
            loss_cfg["lambda_cls"] * cls_loss
            + loss_cfg["lambda_supcon"] * supcon_loss
            + loss_cfg["lambda_view"] * view_loss
            + loss_cfg["lambda_attn"] * attn_loss
            + loss_cfg.get("lambda_aux", 0.0) * aux_loss
        )
        return total, {
            "loss": float(total.detach().cpu()),
            "cls": float(cls_loss.detach().cpu()),
            "supcon": float(supcon_loss.detach().cpu()),
            "view": float(view_loss.detach().cpu()),
            "attn": float(attn_loss.detach().cpu()),
            "aux": float(aux_loss.detach().cpu()),
        }

    def fit(self) -> None:
        best_macro_f1, best_mcc = -1.0, -1.0
        epochs = int(self.cfg["scheduler"]["epochs"])
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
            for step, batch in enumerate(progress, start=1):
                self.optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                    outputs = self.model(
                        {
                            "global_image": batch["global_image"].to(self.device, non_blocking=True),
                            "local_image": batch["local_image"].to(self.device, non_blocking=True),
                            "attention_prior": batch["attention_prior"].to(self.device, non_blocking=True),
                            "has_bbox": batch["has_bbox"].to(self.device, non_blocking=True),
                        }
                    )
                    loss, stats = self._compute_losses(outputs, batch)
                self.scaler.scale(loss).backward()
                if self.cfg["runtime"].get("grad_clip_norm", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["runtime"]["grad_clip_norm"]))
                self.scaler.step(self.optimizer)
                self.scaler.update()
                running_loss += stats["loss"]
                if step % int(self.cfg["runtime"].get("log_interval", 20)) == 0:
                    progress.set_postfix(loss=f"{stats['loss']:.4f}", cls=f"{stats['cls']:.4f}", view=f"{stats['view']:.4f}")

            self.scheduler.step()
            epoch_loss = running_loss / max(len(self.train_loader), 1)
            self.history["train_loss"].append(epoch_loss)
            self.logger.info("Epoch %d | train_loss=%.4f", epoch, epoch_loss)

            val_metrics = self.evaluator.evaluate(self.model, self.val_loader, self.device, split_name=f"val_epoch_{epoch:03d}")
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_mcc"].append(val_metrics["mcc"])

            if val_metrics["macro_f1"] >= best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                save_checkpoint(
                    {
                        "model": self.model.state_dict(),
                        "epoch": epoch,
                        "metric": "macro_f1",
                        "value": best_macro_f1,
                        "class_names": self.class_names,
                        "num_classes": len(self.class_names),
                    },
                    self.checkpoint_dir / "best_macro_f1.pt",
                )
            if val_metrics["mcc"] >= best_mcc:
                best_mcc = val_metrics["mcc"]
                save_checkpoint(
                    {
                        "model": self.model.state_dict(),
                        "epoch": epoch,
                        "metric": "mcc",
                        "value": best_mcc,
                        "class_names": self.class_names,
                        "num_classes": len(self.class_names),
                    },
                    self.checkpoint_dir / "best_mcc.pt",
                )

            plot_training_curves(self.history, self.output_dir / "training_curves.png")

        save_json(self.history, self.output_dir / "history.json")
        state = torch.load(self.checkpoint_dir / "best_macro_f1.pt", map_location="cpu")
        self.model.load_state_dict(state["model"], strict=False)
        test_metrics = self.evaluator.evaluate(self.model, self.test_loader, self.device, split_name="test_best_macro_f1")
        self.logger.info("Final test | macro_f1=%.4f | mcc=%.4f", test_metrics["macro_f1"], test_metrics["mcc"])
