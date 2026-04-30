from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history: list[dict[str, float]], output_path: str | Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy + L2")
    axes[0].set_title("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="Train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    threshold = matrix.max() * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image, [2, 98])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def plot_first_layer_filters(
    first_weight: np.ndarray,
    image_shape: tuple[int, int, int],
    output_path: str | Path,
    max_filters: int = 36,
) -> None:
    num_filters = min(max_filters, first_weight.shape[1])
    cols = min(6, num_filters)
    rows = int(np.ceil(num_filters / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.7), dpi=150)
    axes = np.asarray(axes).reshape(-1)

    norms = np.linalg.norm(first_weight, axis=0)
    filter_ids = np.argsort(norms)[::-1][:num_filters]
    for axis, filter_id in zip(axes, filter_ids):
        image = first_weight[:, filter_id].reshape(image_shape)
        axis.imshow(_normalize_image(image))
        axis.set_title(f"h{filter_id}", fontsize=8)
        axis.axis("off")

    for axis in axes[num_filters:]:
        axis.axis("off")
    fig.suptitle("First Hidden Layer Weights")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_class_linked_filters(
    first_weight: np.ndarray,
    second_weight: np.ndarray,
    image_shape: tuple[int, int, int],
    class_names: list[str],
    output_path: str | Path,
    filters_per_class: int = 4,
) -> None:
    if second_weight.ndim != 2 or second_weight.shape[1] != len(class_names):
        return

    rows = len(class_names)
    cols = filters_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.3), dpi=150)
    for class_id, class_name in enumerate(class_names):
        top_hidden = np.argsort(second_weight[:, class_id])[::-1][:filters_per_class]
        for col, hidden_id in enumerate(top_hidden):
            axis = axes[class_id, col]
            image = first_weight[:, hidden_id].reshape(image_shape)
            axis.imshow(_normalize_image(image))
            axis.axis("off")
            if col == 0:
                axis.set_ylabel(class_name, fontsize=7)
            axis.set_title(f"h{hidden_id}", fontsize=7)
    fig.suptitle("First-Layer Filters Most Connected to Each Class")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_misclassified(
    x_raw: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    max_images: int = 16,
) -> None:
    wrong = np.flatnonzero(y_true != y_pred)[:max_images]
    if len(wrong) == 0:
        return

    cols = min(4, len(wrong))
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.3), dpi=150)
    axes = np.asarray(axes).reshape(-1)
    for axis, idx in zip(axes, wrong):
        axis.imshow(x_raw[idx])
        axis.set_title(f"T: {class_names[int(y_true[idx])]}\nP: {class_names[int(y_pred[idx])]}", fontsize=7)
        axis.axis("off")
    for axis in axes[len(wrong):]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
