"""Square classifier: hand-crafted features (HOG + colour statistics) and an MLP.

predict_proba(crops) returns (N, 13) probabilities in the order of CLASSES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .position import CLASSES


class SquareClassifier(Protocol):
    name: str

    def predict_proba(self, crops: np.ndarray) -> np.ndarray: ...


def hog(gray: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients (HOG), independent of the OpenCV version.

    gray: (N, H, W) float32. Returns (N, D): per-cell histograms over 8x8 cells,
    normalised in 2x2-cell blocks (L2-Hys, as in the original paper).
    """
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    gy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0  # unsigned gradients
    pos = ang / (180.0 / bins)
    lo = np.floor(pos).astype(int) % bins
    hi = (lo + 1) % bins
    w_hi = pos - np.floor(pos)
    n, h, w = gray.shape
    ch, cw = h // cell, w // cell
    hist = np.zeros((n, ch, cw, bins), np.float32)
    cy = np.minimum(np.arange(h) // cell, ch - 1)
    cx = np.minimum(np.arange(w) // cell, cw - 1)
    ii = np.broadcast_to(np.arange(n)[:, None, None], gray.shape)
    yy = np.broadcast_to(cy[None, :, None], gray.shape)
    xx = np.broadcast_to(cx[None, None, :], gray.shape)
    np.add.at(hist, (ii, yy, xx, lo), mag * (1 - w_hi))
    np.add.at(hist, (ii, yy, xx, hi), mag * w_hi)
    blocks = np.concatenate([hist[:, :-1, :-1], hist[:, 1:, :-1], hist[:, :-1, 1:], hist[:, 1:, 1:]], axis=-1)
    blocks = blocks / (np.linalg.norm(blocks, axis=-1, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / (np.linalg.norm(blocks, axis=-1, keepdims=True) + 1e-6)
    return blocks.reshape(n, -1)


def handcrafted_features(crops: np.ndarray, batch: int = 2048) -> np.ndarray:
    """HOG (piece shape) + colour statistics (white/black piece, empty square)."""
    if len(crops) > batch:  # in chunks, to keep memory bounded
        return np.concatenate([handcrafted_features(crops[i : i + batch], batch) for i in range(0, len(crops), batch)])
    grays = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops]).astype(np.float32)
    hogs = hog(grays)
    feats = []
    for crop, gray, h_feat in zip(crops, grays.astype(np.uint8), hogs):
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        h = crop.shape[0]
        regions = [lab[h // 2 :], lab[h // 4 : 3 * h // 4], lab[: h // 2]]  # bottom = base of the piece
        color = []
        for reg in regions:
            L = reg[..., 0]
            color += [L.mean(), L.std(), np.percentile(L, 10), np.percentile(L, 90), reg[..., 1].mean(), reg[..., 2].mean()]
        edges = cv2.Canny(gray, 50, 150)
        color += [edges[h // 2 :].mean(), edges[: h // 2].mean()]
        feats.append(np.concatenate([h_feat, np.array(color, np.float32) / 100.0]))
    return np.stack(feats).astype(np.float32)


class SklearnClassifier:
    def __init__(self, path: str | Path):
        import joblib

        self.name = f"baseline:{Path(path).name}"
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = list(bundle["classes"])  # class indices the model knows

    def predict_proba(self, crops: np.ndarray) -> np.ndarray:
        p = self.model.predict_proba(handcrafted_features(crops))
        out = np.zeros((len(crops), len(CLASSES)), np.float32)
        out[:, self.classes] = p
        return out


def load_classifier(model_dir: str | Path) -> SquareClassifier:
    path = Path(model_dir) / "square_baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model {path} not found (train it with training/train_baseline.py).")
    return SklearnClassifier(path)
