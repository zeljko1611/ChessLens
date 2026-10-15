"""Square classifier: a CNN exported to ONNX and run with onnxruntime (no PyTorch needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

# CNN input normalisation (must match training/train_cnn.py).
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.25, 0.25, 0.25], np.float32)


class SquareClassifier(Protocol):
    name: str

    def predict_proba(self, crops: np.ndarray) -> np.ndarray: ...


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def to_cnn_input(crops: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 BGR -> (N, 3, H, W) float32 RGB, normalised."""
    x = crops[..., ::-1].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


class OnnxClassifier:
    def __init__(self, path: str | Path):
        import onnxruntime as ort

        self.name = f"cnn:{Path(path).name}"
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_proba(self, crops: np.ndarray) -> np.ndarray:
        logits = self.session.run(None, {self.input_name: to_cnn_input(crops)})[0]
        return softmax(logits.astype(np.float32))


def load_classifier(model_dir: str | Path) -> SquareClassifier:
    path = Path(model_dir) / "square_cnn.onnx"
    if not path.exists():
        raise FileNotFoundError(f"Model {path} not found (train it with training/train_cnn.py).")
    return OnnxClassifier(path)
