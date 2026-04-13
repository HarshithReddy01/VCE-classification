from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from src.engine.metrics import bootstrap_metric_ci, compute_classification_metrics
from src.utils.io import ensure_dir, save_json
from src.utils.visualization import plot_confusion_matrix, plot_embeddings, plot_multiclass_curves


class Evaluator:
    def __init__(self, class_names: list[str], output_dir: Path, logger: Any, metrics_cfg: dict[str, Any]) -> None:
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.metrics_cfg = metrics_cfg
        ensure_dir(self.output_dir)

    def evaluate(self, model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, split_name: str) -> dict[str, Any]:
        model.eval()
        all_logits, all_targets, all_embeddings = [], [], []
        records: list[dict[str, Any]] = []

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
                inputs = {
                    "global_image": batch["global_image"].to(device, non_blocking=True),
                    "local_image": batch["local_image"].to(device, non_blocking=True),
                    "attention_prior": batch["attention_prior"].to(device, non_blocking=True),
                    "has_bbox": batch["has_bbox"].to(device, non_blocking=True),
                }
                outputs = model(inputs)
                logits = outputs["logits"].detach().cpu()
                preds = logits.argmax(dim=1)
                probs = logits.softmax(dim=1)
                all_logits.append(logits)
                all_targets.append(batch["target"].cpu())
                all_embeddings.append(outputs["fused_embedding"].detach().cpu())

                for i in range(logits.size(0)):
                    records.append(
                        {
                            "filename": batch["filename"][i],
                            "video_id": batch["video_id"][i],
                            "frame_number": int(batch["frame_number"][i]),
                            "target": int(batch["target"][i]),
                            "prediction": int(preds[i]),
                            "target_name": self.class_names[int(batch["target"][i])],
                            "prediction_name": self.class_names[int(preds[i])],
                            "has_bbox": float(batch["has_bbox"][i]),
                            "probability": float(probs[i, preds[i]]),
                            "logits": logits[i].tolist(),
                        }
                    )

        logits_np = torch.cat(all_logits).numpy()
        y_true = torch.cat(all_targets).numpy()
        embeddings = torch.cat(all_embeddings).numpy()
        y_pred = logits_np.argmax(axis=1)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            class_counts=np.bincount(y_true, minlength=len(self.class_names)),
            tail_percentile=float(self.metrics_cfg.get("tail_percentile", 0.4)),
        )
        metrics["macro_f1_ci"] = bootstrap_metric_ci(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            metric_key="macro_f1",
            num_samples=int(self.metrics_cfg.get("bootstrap_samples", 0)),
        )
        metrics["mcc_ci"] = bootstrap_metric_ci(
            y_true=y_true,
            y_pred=y_pred,
            class_names=self.class_names,
            metric_key="mcc",
            num_samples=int(self.metrics_cfg.get("bootstrap_samples", 0)),
        )

        probs_np = torch.tensor(logits_np).softmax(dim=1).numpy()
        pred_df = pd.DataFrame.from_records(records)
        pred_df.to_csv(self.output_dir / f"{split_name}_predictions.csv", index=False)
        pd.DataFrame(logits_np, columns=[f"logit_{name}" for name in self.class_names]).to_csv(
            self.output_dir / f"{split_name}_logits.csv",
            index=False,
        )
        per_class_df = metrics.pop("per_class")
        per_class_df.to_csv(self.output_dir / f"{split_name}_per_class.csv", index=False)
        plot_confusion_matrix(np.array(metrics["confusion_matrix"]), self.class_names, self.output_dir / f"{split_name}_confusion_matrix.png")

        if self.metrics_cfg.get("save_embeddings", True):
            plot_embeddings(embeddings, y_true, self.class_names, self.output_dir / f"{split_name}_embeddings.png")

        if self.metrics_cfg.get("save_roc_pr", True):
            plot_multiclass_curves(y_true, probs_np, self.class_names, self.output_dir, split_name)
            one_hot = np.eye(len(self.class_names))[y_true]
            try:
                valid_cols = [idx for idx in range(one_hot.shape[1]) if 0 < one_hot[:, idx].sum() < len(one_hot[:, idx])]
                if valid_cols:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        metrics["macro_roc_auc_ovr"] = float(
                            roc_auc_score(one_hot[:, valid_cols], probs_np[:, valid_cols], average="macro", multi_class="ovr")
                        )
                        metrics["macro_pr_auc_ovr"] = float(
                            average_precision_score(one_hot[:, valid_cols], probs_np[:, valid_cols], average="macro")
                        )
                else:
                    metrics["macro_roc_auc_ovr"] = 0.0
                    metrics["macro_pr_auc_ovr"] = 0.0
            except ValueError:
                metrics["macro_roc_auc_ovr"] = 0.0
                metrics["macro_pr_auc_ovr"] = 0.0

        save_json({k: v for k, v in metrics.items() if k != "confusion_matrix"}, self.output_dir / f"{split_name}_metrics.json")
        self.logger.info("%s | macro_f1=%.4f | mcc=%.4f", split_name, metrics["macro_f1"], metrics["mcc"])
        return metrics
