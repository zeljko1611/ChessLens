"""Conversion of the chesscog dataset (realistic 3D renders, Wölflein & Arandjelović 2021) to our format.

Dataset: https://osf.io/xf3ka/ (CC BY 4.0). After unpacking, every image
NNNN.png has an NNNN.json next to it with the fields:
  fen      piece placement
  corners  the four playing-area corners in pixels (not in a8, h8, h1, a1 order)
  pieces   [{piece, square, box: [x, y, w, h]}]

The corner order is recovered from the piece boxes: of the 8 possible orders
(4 rotations x 2 directions), we pick the one where the bottom of each piece
box falls into the square given in its label.

    python data/chesscog.py --src ~/datasets/chesscog/ex/train --out data/chesscog_train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

GRID = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32)  # a8, h8, h1, a1 in grid coordinates


def candidate_orders(corners: np.ndarray) -> list[np.ndarray]:
    orders = []
    for seq in (corners, corners[::-1]):
        for k in range(4):
            orders.append(np.roll(seq, -k, axis=0))
    return orders


def order_corners_by_pieces(corners: np.ndarray, pieces: list[dict]) -> tuple[np.ndarray | None, float]:
    """The corner order (a8, h8, h1, a1) that best explains the piece positions."""
    if not pieces:
        return None, 0.0
    # The box bottom is in front of the square centre (the piece stands on it); lift the point slightly.
    pts = np.array([[p["box"][0] + p["box"][2] / 2, p["box"][1] + p["box"][3] - 0.12 * p["box"][2]] for p in pieces], np.float32)
    want = np.array([[ord(p["square"][0]) - ord("a"), 8 - int(p["square"][1])] for p in pieces])
    best, best_score = None, -1.0
    for cand in candidate_orders(corners):
        H = cv2.getPerspectiveTransform(cand.astype(np.float32), GRID)
        g = cv2.perspectiveTransform(pts.reshape(1, -1, 2), H)[0]
        score = float((np.floor(g).astype(int) == want).all(axis=1).mean())
        if score > best_score:
            best, best_score = cand, score
    return best, best_score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-score", type=float, default=0.7, help="minimum share of pieces that fall into their own square")
    args = ap.parse_args()

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    kept = skipped = 0
    with open(args.out / "labels.jsonl", "w", encoding="utf-8") as f:
        for jpath in sorted(args.src.glob("*.json")):
            img_path = jpath.with_suffix(".png")
            if not img_path.exists():
                continue
            ann = json.loads(jpath.read_text(encoding="utf-8"))
            corners, score = order_corners_by_pieces(np.array(ann["corners"], np.float32), ann.get("pieces", []))
            if corners is None or score < args.min_score:
                skipped += 1
                continue
            dst = args.out / "images" / img_path.name
            if not dst.exists():
                dst.symlink_to(img_path.resolve())
            row = {"image": f"images/{dst.name}", "fen": ann["fen"].split()[0], "corners": corners.round(2).tolist(),
                   "source": "chesscog", "corner_score": round(score, 3)}
            f.write(json.dumps(row) + "\n")
            kept += 1
    print(f"Kept {kept} images, skipped {skipped}.")


if __name__ == "__main__":
    main()
