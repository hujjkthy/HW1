from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from mlp_eurosat.data import build_dataset
from mlp_eurosat.metrics import accuracy, confusion_matrix
from mlp_eurosat.model import MLP
from mlp_eurosat.visualize import (
    plot_confusion_matrix,
    plot_first_layer_filters,
    plot_history,
    plot_misclassified,
)


def parse_hidden_dims(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def evaluate_loss_acc(model: MLP, x: np.ndarray, y: np.ndarray, weight_decay: float, batch_size: int) -> tuple[float, float]:
    losses: list[float] = []
    preds: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        batch_x = x[start : start + batch_size]
        batch_y = y[start : start + batch_size]
        result, _, _ = model.loss_and_gradients(batch_x, batch_y, weight_decay=weight_decay)
        losses.append(result.loss * batch_x.shape[0])
        preds.append(result.probs.argmax(axis=1))
    return float(np.sum(losses) / x.shape[0]), accuracy(y, np.concatenate(preds))


def global_grad_norm(grad_weights: list[np.ndarray], grad_biases: list[np.ndarray]) -> float:
    total = 0.0
    for grad in grad_weights + grad_biases:
        total += float(np.sum(grad.astype(np.float64) * grad.astype(np.float64)))
    return float(np.sqrt(total))


def clip_gradients(
    grad_weights: list[np.ndarray],
    grad_biases: list[np.ndarray],
    max_norm: float | None,
) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    norm = global_grad_norm(grad_weights, grad_biases)
    if max_norm is None or max_norm <= 0 or norm <= max_norm:
        return grad_weights, grad_biases, norm
    scale = max_norm / (norm + 1e-12)
    clipped_weights = [grad * scale for grad in grad_weights]
    clipped_biases = [grad * scale for grad in grad_biases]
    return clipped_weights, clipped_biases, norm


def train_model(args: argparse.Namespace) -> dict[str, float | str]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        args.data_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_per_class=args.max_per_class,
    )

    model = MLP(
        input_dim=dataset.x_train.shape[1],
        hidden_dims=parse_hidden_dims(args.hidden_dim),
        output_dim=len(dataset.class_names),
        activation=args.activation,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float]] = []
    best_state = model.copy_state()
    best_val_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        lr = args.lr * (args.lr_decay ** (epoch - 1))
        indices = rng.permutation(dataset.x_train.shape[0])
        train_losses: list[float] = []
        grad_norms: list[float] = []

        for start in range(0, len(indices), args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            result, grad_weights, grad_biases = model.loss_and_gradients(
                dataset.x_train[batch_idx],
                dataset.y_train[batch_idx],
                weight_decay=args.weight_decay,
            )
            if not np.isfinite(result.loss):
                raise FloatingPointError(
                    "Non-finite training loss detected before the parameter update. "
                    "Try a smaller --lr, e.g. 0.005, or keep --grad-clip enabled."
                )
            grad_weights, grad_biases, grad_norm = clip_gradients(grad_weights, grad_biases, args.grad_clip)
            grad_norms.append(grad_norm)
            model.update(grad_weights, grad_biases, lr=lr)
            if not all(np.all(np.isfinite(w)) for w in model.weights):
                raise FloatingPointError(
                    "Model weights became non-finite after an update. "
                    "Try a smaller --lr, e.g. 0.005, or a smaller --grad-clip."
                )
            train_losses.append(result.loss * len(batch_idx))

        train_loss, train_acc = evaluate_loss_acc(model, dataset.x_train, dataset.y_train, args.weight_decay, args.eval_batch_size)
        val_loss, val_acc = evaluate_loss_acc(model, dataset.x_val, dataset.y_val, args.weight_decay, args.eval_batch_size)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise FloatingPointError(
                "Non-finite evaluation loss detected. "
                "Use a smaller --lr, e.g. 0.005, and keep --grad-clip enabled."
            )

        row = {
            "epoch": float(epoch),
            "lr": float(lr),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "grad_norm": float(np.mean(grad_norms)),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs} lr={lr:.5f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"grad_norm={row['grad_norm']:.4f}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = model.copy_state()
            model.save(output_dir / "best_model.npz", dataset.class_names, dataset.mean, dataset.std, dataset.image_shape)

    model.load_state(best_state)
    test_pred = model.predict(dataset.x_test, batch_size=args.eval_batch_size)
    test_acc = accuracy(dataset.y_test, test_pred)
    cm = confusion_matrix(dataset.y_test, test_pred, len(dataset.class_names))

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "grad_norm"])
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "class_names": dataset.class_names,
        "confusion_matrix": cm.tolist(),
        "config": vars(args),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    plot_history(history, output_dir / "training_curves.png")
    plot_confusion_matrix(cm, dataset.class_names, output_dir / "confusion_matrix.png")
    plot_misclassified(dataset.x_test_raw, dataset.y_test, test_pred, dataset.class_names, output_dir / "misclassified_examples.png")
    plot_first_layer_filters(model.weights[0], dataset.image_shape, output_dir / "first_layer_filters.png")

    print(f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} test_acc={test_acc:.4f}", flush=True)
    return {"best_val_acc": best_val_acc, "test_acc": test_acc, "best_epoch": best_epoch, "output_dir": str(output_dir)}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a NumPy-only MLP on EuroSAT_RGB.")
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs/run")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", default="512,512", help="Comma-separated hidden sizes, e.g. 512,512.")
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Global gradient norm clipping. Set <=0 to disable.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser


if __name__ == "__main__":
    train_model(build_argparser().parse_args())
