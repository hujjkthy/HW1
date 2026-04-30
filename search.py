from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

from train import build_argparser, train_model


def split_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter search for the NumPy EuroSAT MLP.")
    parser.add_argument("--data-dir", default="EuroSAT_RGB")
    parser.add_argument("--output-dir", default="outputs/search")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", default="128,256")
    parser.add_argument("--activations", default="relu,tanh")
    parser.add_argument("--lrs", default="0.01,0.005")
    parser.add_argument("--lr-decays", default="0.95")
    parser.add_argument("--weight-decays", default="0,0.0001")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    base_parser = build_argparser()
    combinations = itertools.product(
        split_values(args.hidden_dims),
        split_values(args.activations),
        split_values(args.lrs),
        split_values(args.lr_decays),
        split_values(args.weight_decays),
    )

    for run_id, (hidden_dim, activation, lr, lr_decay, weight_decay) in enumerate(combinations, start=1):
        run_args = base_parser.parse_args([])
        run_args.data_dir = args.data_dir
        run_args.output_dir = str(output_dir / f"run_{run_id:03d}")
        run_args.epochs = args.epochs
        run_args.batch_size = args.batch_size
        run_args.hidden_dim = hidden_dim
        run_args.activation = activation
        run_args.lr = float(lr)
        run_args.lr_decay = float(lr_decay)
        run_args.weight_decay = float(weight_decay)
        run_args.max_per_class = args.max_per_class
        run_args.seed = args.seed
        result = train_model(run_args)
        result.update(
            {
                "hidden_dim": hidden_dim,
                "activation": activation,
                "lr": lr,
                "lr_decay": lr_decay,
                "weight_decay": weight_decay,
            }
        )
        results.append(result)

    with (output_dir / "search_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["hidden_dim", "activation", "lr", "lr_decay", "weight_decay", "best_epoch", "best_val_acc", "test_acc", "output_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
