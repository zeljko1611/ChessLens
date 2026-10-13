"""Turns labeled photos into a dataset of square crops for training.

The input is a folder with `labels.jsonl` (the format of data/synth.py or
data/chesscog.py): for each image a FEN and the four playing-area corners
(a8, h8, h1, a1). The output is an .npz with:
  X      uint8 (N, 64, 32, 3)  crops
  y      int64 (N,)            class index (see chessvision.position.CLASSES)
  group  int64 (N,)            image index (so splits never mix crops of one image)

Corners are deliberately jittered (--jitter) so the model also works when the
board detection is not perfect.

    python training/make_dataset.py --data data/generated/train_a data/generated/train_b --out data/train.npz --jitter 0.06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.position import CLASS_INDEX, fen_to_board  # noqa: E402
from chessvision.squares import CROP_H, CROP_W, crop_squares  # noqa: E402


def jitter_corners(corners: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Move each corner randomly by up to `amount` square sizes."""
    square = np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]) / 8
    return corners + rng.uniform(-amount, amount, corners.shape) * square


def build(folders: list[Path], jitter: float, seed: int, limit: int, repeat: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    jobs = []
    for folder in folders:
        rows = [json.loads(line) for line in open(folder / "labels.jsonl", encoding="utf-8")]
        rows = [r for r in rows if r.get("corners") is not None]
        if limit:
            rows = rows[:limit]
        jobs += [(folder, row) for row in rows for _ in range(repeat)]
    # Preallocate the arrays: a list of crops plus np.stack would double the memory.
    X = np.zeros((len(jobs) * 64, CROP_H, CROP_W, 3), np.uint8)
    y = np.zeros(len(jobs) * 64, np.int64)
    group = np.zeros(len(jobs) * 64, np.int64)
    n, gid, img_key = 0, -1, None
    img = None
    for folder, row in jobs:
        key = (folder, row["image"])
        if key != img_key:  # repeats of one image share a group (validation never mixes them)
            img = cv2.imread(str(folder / row["image"]))
            img_key = key
            gid += 1
        if img is None:
            continue
        corners = np.array(row["corners"], np.float32)
        if jitter > 0:
            corners = jitter_corners(corners, jitter, rng)
        crops = crop_squares(img, corners)  # grid = board because the corners are a8, h8, h1, a1
        board = fen_to_board(row["fen"])
        X[n : n + 64] = crops.reshape(64, CROP_H, CROP_W, 3)
        y[n : n + 64] = [CLASS_INDEX[board[r][c]] for r in range(8) for c in range(8)]
        group[n : n + 64] = gid
        n += 64
        if (n // 64) % 200 == 0:
            print(f"{n // 64} images", flush=True)
    return {"X": X[:n], "y": y[:n], "group": group[:n]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jitter", type=float, default=0.06)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="maximum images per folder (0 = all)")
    ap.add_argument("--repeat", type=int, default=1, help="how many times to use each image (each time with different corner jitter)")
    args = ap.parse_args()
    data = build(args.data, args.jitter, args.seed, args.limit, args.repeat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data)
    counts = np.bincount(data["y"], minlength=13)
    print(f"Saved {len(data['y'])} crops to {args.out}; per class: {counts.tolist()}")


if __name__ == "__main__":
    main()
