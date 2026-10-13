"""Baseline model: HOG features + an MLP (scikit-learn).

A reference point for later models that trains on a CPU without PyTorch.

    python training/train_baseline.py --train data/train.npz --out models/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.classifier import handcrafted_features  # noqa: E402
from chessvision.position import CLASSES  # noqa: E402


def split_by_group(group: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by image, not by crop (otherwise validation would be too easy)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(group)
    rng.shuffle(ids)
    is_val = np.isin(group, ids[: int(len(ids) * val_frac)])
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    args = ap.parse_args()

    d = np.load(args.train)
    X, y, group = d["X"], d["y"], d["group"]
    tr, va = split_by_group(group, args.val_frac, seed=0)

    t0 = time.time()
    print(f"Computing features for {len(tr)} + {len(va)} crops...", flush=True)
    F_tr, F_va = handcrafted_features(X[tr]), handcrafted_features(X[va])
    print(f"  done in {time.time() - t0:.0f} s, dimension {F_tr.shape[1]}", flush=True)

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=tuple(args.hidden), alpha=1e-3, batch_size=256, max_iter=40,
                      early_stopping=True, n_iter_no_change=5, random_state=0),
    )
    t0 = time.time()
    model.fit(F_tr, y[tr])
    print(f"Training: {time.time() - t0:.0f} s", flush=True)

    pred = model.predict(F_va)
    print(f"Square accuracy (validation): {(pred == y[va]).mean() * 100:.2f}%")
    per_class = {CLASSES[c]: float((pred[y[va] == c] == c).mean()) for c in np.unique(y[va])}
    print("Per class:", {k: f"{v * 100:.1f}%" for k, v in per_class.items()})

    args.out.mkdir(parents=True, exist_ok=True)
    clf = model.steps[-1][1]
    joblib.dump({"model": model, "classes": clf.classes_.tolist()}, args.out / "square_baseline.joblib", compress=3)
    print(f"Saved: {args.out / 'square_baseline.joblib'}")


if __name__ == "__main__":
    main()
