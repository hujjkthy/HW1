from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BatchResult:
    loss: float
    data_loss: float
    l2_loss: float
    probs: np.ndarray


class MLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        seed: int = 42,
    ) -> None:
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer")
        if activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("activation must be one of: relu, sigmoid, tanh")

        self.input_dim = int(input_dim)
        self.hidden_dims = [int(v) for v in hidden_dims]
        self.output_dim = int(output_dim)
        self.activation = activation
        self.rng = np.random.default_rng(seed)

        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(layer_dims[:-1], layer_dims[1:]):
            if activation == "relu":
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights.append((self.rng.normal(0.0, scale, size=(fan_in, fan_out))).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))
        return np.tanh(z)

    def _activation_grad(self, activated: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (activated > 0).astype(np.float32)
        if self.activation == "sigmoid":
            return activated * (1.0 - activated)
        return 1.0 - activated * activated

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [x]
        out = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            out = self._activate(out @ weight + bias)
            activations.append(out)
        logits = out @ self.weights[-1] + self.biases[-1]
        activations.append(logits)
        return logits, activations

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def loss_and_gradients(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weight_decay: float = 0.0,
    ) -> tuple[BatchResult, list[np.ndarray], list[np.ndarray]]:
        logits, activations = self.forward(x)
        probs = self.softmax(logits)
        batch_size = x.shape[0]

        correct_log_probs = -np.log(probs[np.arange(batch_size), y] + 1e-12)
        data_loss = float(correct_log_probs.mean())
        l2_loss = 0.5 * weight_decay * sum(float(np.sum(w * w)) for w in self.weights)
        loss = data_loss + l2_loss

        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), y] -= 1.0
        grad_logits /= batch_size

        grad_weights: list[np.ndarray] = [np.empty_like(w) for w in self.weights]
        grad_biases: list[np.ndarray] = [np.empty_like(b) for b in self.biases]
        grad = grad_logits

        for layer in reversed(range(len(self.weights))):
            prev_activation = activations[layer]
            grad_weights[layer] = prev_activation.T @ grad + weight_decay * self.weights[layer]
            grad_biases[layer] = grad.sum(axis=0, keepdims=True)
            if layer > 0:
                grad = (grad @ self.weights[layer].T) * self._activation_grad(activations[layer])

        result = BatchResult(loss=loss, data_loss=data_loss, l2_loss=l2_loss, probs=probs)
        return result, grad_weights, grad_biases

    def update(self, grad_weights: list[np.ndarray], grad_biases: list[np.ndarray], lr: float) -> None:
        for idx in range(len(self.weights)):
            self.weights[idx] -= lr * grad_weights[idx].astype(np.float32)
            self.biases[idx] -= lr * grad_biases[idx].astype(np.float32)

    def predict_proba(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        probs: list[np.ndarray] = []
        for start in range(0, x.shape[0], batch_size):
            logits, _ = self.forward(x[start : start + batch_size])
            probs.append(self.softmax(logits))
        return np.vstack(probs)

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        return self.predict_proba(x, batch_size=batch_size).argmax(axis=1)

    def copy_state(self) -> dict[str, list[np.ndarray]]:
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def load_state(self, state: dict[str, list[np.ndarray]]) -> None:
        self.weights = [w.copy() for w in state["weights"]]
        self.biases = [b.copy() for b in state["biases"]]

    def save(
        self,
        path: str | Path,
        class_names: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        image_shape: tuple[int, int, int],
    ) -> None:
        payload: dict[str, np.ndarray] = {
            "input_dim": np.asarray(self.input_dim),
            "hidden_dims": np.asarray(self.hidden_dims, dtype=np.int64),
            "output_dim": np.asarray(self.output_dim),
            "activation": np.asarray(self.activation),
            "class_names": np.asarray(class_names),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "image_shape": np.asarray(image_shape, dtype=np.int64),
        }
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            payload[f"W{idx}"] = weight
            payload[f"b{idx}"] = bias
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> tuple["MLP", dict[str, np.ndarray | list[str] | tuple[int, int, int]]]:
        data = np.load(path, allow_pickle=False)
        hidden_dims = data["hidden_dims"].astype(int).tolist()
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dims=hidden_dims,
            output_dim=int(data["output_dim"]),
            activation=str(data["activation"]),
            seed=0,
        )
        model.weights = [data[f"W{idx}"].astype(np.float32) for idx in range(len(hidden_dims) + 1)]
        model.biases = [data[f"b{idx}"].astype(np.float32) for idx in range(len(hidden_dims) + 1)]
        metadata = {
            "class_names": [str(v) for v in data["class_names"].tolist()],
            "mean": data["mean"].astype(np.float32),
            "std": data["std"].astype(np.float32),
            "image_shape": tuple(int(v) for v in data["image_shape"].tolist()),
        }
        return model, metadata
