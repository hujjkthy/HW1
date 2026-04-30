from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlp_eurosat.data import apply_saved_normalization, load_images, stratified_split
from mlp_eurosat.metrics import accuracy, confusion_matrix
from mlp_eurosat.model import MLP
from mlp_eurosat.visualize import plot_confusion_matrix, plot_misclassified


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved NumPy MLP model on EuroSAT_RGB.")
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, metadata = MLP.load(args.model)
    x_raw, y, class_names = load_images(args.data_dir, image_size=args.image_size, max_per_class=args.max_per_class)
    if class_names != metadata["class_names"]:
        raise ValueError("Class folder names do not match the saved model metadata")

    _, _, test_idx = stratified_split(y, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    x_test = apply_saved_normalization(x_raw[test_idx], metadata["mean"], metadata["std"])
    y_test = y[test_idx]

    y_pred = model.predict(x_test, batch_size=args.batch_size)
    test_acc = accuracy(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, len(class_names))

    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    plot_misclassified(x_raw[test_idx], y_test, y_pred, class_names, output_dir / "misclassified_examples.png")

    metrics = {
        "test_acc": test_acc,
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
