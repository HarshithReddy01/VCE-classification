from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    class_counts: np.ndarray | None = None,
    tail_percentile: float = 0.4,
) -> dict[str, Any]:
    labels = np.arange(len(class_names))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
        balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred))
    metrics = {
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=np.maximum(support, 1))),
        "mcc": mcc,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": pd.DataFrame(
            {
                "class_name": class_names,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        ),
    }
    if class_counts is not None and len(class_counts) == len(class_names):
        threshold = np.quantile(class_counts, tail_percentile)
        head_mask, tail_mask = class_counts > threshold, class_counts <= threshold
        metrics["head_recall"] = float(recall[head_mask].mean()) if head_mask.any() else 0.0
        metrics["tail_recall"] = float(recall[tail_mask].mean()) if tail_mask.any() else 0.0
    else:
        metrics["head_recall"] = 0.0
        metrics["tail_recall"] = 0.0
    return metrics


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    metric_key: str,
    num_samples: int,
    seed: int = 42,
) -> dict[str, float]:
    if num_samples <= 0:
        base = compute_classification_metrics(y_true, y_pred, class_names)[metric_key]
        return {"mean": float(base), "lower": float(base), "upper": float(base)}
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    n = len(y_true)
    for _ in range(num_samples):
        idx = rng.integers(0, n, size=n)
        metrics = compute_classification_metrics(y_true[idx], y_pred[idx], class_names)
        scores.append(float(metrics[metric_key]))
    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(scores, 2.5)),
        "upper": float(np.percentile(scores, 97.5)),
    }
