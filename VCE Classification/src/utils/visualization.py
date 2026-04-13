from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import label_binarize

from src.utils.io import ensure_dir


def plot_confusion_matrix(confusion: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion, cmap="mako", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    plt.figure(figsize=(8, 5))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    reduced_input = PCA(n_components=min(50, embeddings.shape[1])).fit_transform(embeddings) if embeddings.shape[1] > 50 else embeddings
    perplexity = min(30, max(5, len(embeddings) // 20))
    reduced = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(reduced_input)
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(class_names):
        mask = labels == idx
        if mask.any():
            plt.scatter(reduced[mask, 0], reduced[mask, 1], s=12, alpha=0.7, label=name)
    plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_multiclass_curves(y_true: np.ndarray, probs: np.ndarray, class_names: list[str], output_dir: str | Path, split_name: str) -> None:
    output_dir = ensure_dir(output_dir)
    one_hot = label_binarize(y_true, classes=np.arange(len(class_names)))
    for idx, class_name in enumerate(class_names):
        positives = int(one_hot[:, idx].sum())
        negatives = int(len(one_hot[:, idx]) - positives)
        if positives == 0 or negatives == 0:
            continue
        plt.figure(figsize=(6, 5))
        RocCurveDisplay.from_predictions(one_hot[:, idx], probs[:, idx], name=class_name)
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_roc_{class_name}.png", dpi=180)
        plt.close()

        plt.figure(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(one_hot[:, idx], probs[:, idx], name=class_name)
        plt.tight_layout()
        plt.savefig(output_dir / f"{split_name}_pr_{class_name}.png", dpi=180)
        plt.close()
