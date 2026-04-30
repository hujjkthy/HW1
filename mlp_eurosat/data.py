from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Dataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_test_raw: np.ndarray
    class_names: list[str]
    mean: np.ndarray
    std: np.ndarray
    image_shape: tuple[int, int, int]


def load_images(
    data_dir: str | Path,
    image_size: int = 64,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    class_dirs = sorted([p for p in data_path.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class folders found in {data_path}")

    images: list[np.ndarray] = []
    labels: list[int] = []
    class_names = [p.name for p in class_dirs]

    for class_id, class_dir in enumerate(class_dirs):
        files = sorted(class_dir.glob("*.jpg"))
        if max_per_class is not None:
            files = files[:max_per_class]
        if not files:
            raise ValueError(f"No .jpg files found in {class_dir}")

        for image_path in files:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if image.size != (image_size, image_size):
                    image = image.resize((image_size, image_size), Image.BILINEAR)
                images.append(np.asarray(image, dtype=np.uint8))
                labels.append(class_id)

    x = np.stack(images, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, class_names


def stratified_split(
    y: np.ndarray,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1")

    rng = np.random.default_rng(seed)
    train_idx: list[np.ndarray] = []
    val_idx: list[np.ndarray] = []
    test_idx: list[np.ndarray] = []

    for class_id in np.unique(y):
        idx = np.flatnonzero(y == class_id)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test : n_test + n_val])
        train_idx.append(idx[n_test + n_val :])

    train = np.concatenate(train_idx)
    val = np.concatenate(val_idx)
    test = np.concatenate(test_idx)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _flatten_float(x: np.ndarray) -> np.ndarray:
    flat_dim = int(np.prod(x.shape[1:]))
    return x.reshape((x.shape[0], flat_dim)).astype(np.float32) / 255.0


def build_dataset(
    data_dir: str | Path,
    image_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    max_per_class: int | None = None,
) -> Dataset:
    x_raw, y, class_names = load_images(data_dir, image_size=image_size, max_per_class=max_per_class)
    train_idx, val_idx, test_idx = stratified_split(y, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    x_train = _flatten_float(x_raw[train_idx])
    x_val = _flatten_float(x_raw[val_idx])
    x_test = _flatten_float(x_raw[test_idx])

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return Dataset(
        x_train=x_train,
        y_train=y[train_idx],
        x_val=x_val,
        y_val=y[val_idx],
        x_test=x_test,
        y_test=y[test_idx],
        x_test_raw=x_raw[test_idx],
        class_names=class_names,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        image_shape=(image_size, image_size, 3),
    )


def apply_saved_normalization(x_raw: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = _flatten_float(x_raw)
    return (x - mean) / std
