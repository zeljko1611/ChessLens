"""Board detection accuracy: mean corner error in square sizes (below 0.25 counts as a success).

    python training/eval_detection.py --data data/synth_test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.board import detect_board, order_corners  # noqa: E402


def corner_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Error in units of one square; the corner order does not have to match."""
    gt = order_corners(gt)
    pred = order_corners(pred)
    square = np.mean([np.linalg.norm(gt[i] - gt[(i + 1) % 4]) for i in range(4)]) / 8
    errs = [np.linalg.norm(np.roll(pred, k, axis=0) - gt, axis=1).mean() for k in range(4)]
    return float(min(errs) / square)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--show-failures", type=Path, help="folder for images of failed detections")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.data / "labels.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    errors, times, failed = [], [], []
    for row in rows:
        img = cv2.imread(str(args.data / row["image"]))
        t0 = time.perf_counter()
        det = detect_board(img)
        times.append(time.perf_counter() - t0)
        err = corner_error(det.corners, np.array(row["corners"], np.float32)) if det else float("inf")
        errors.append(err)
        if err >= 0.25:
            failed.append((row["image"], err))
            if args.show_failures:
                args.show_failures.mkdir(parents=True, exist_ok=True)
                vis = img.copy()
                if det is not None:
                    cv2.polylines(vis, [det.corners.astype(np.int32)], True, (0, 0, 255), 3)
                cv2.polylines(vis, [np.array(row["corners"], np.int32)], True, (0, 255, 0), 2)
                cv2.imwrite(str(args.show_failures / Path(row["image"]).name), vis)

    e = np.array(errors)
    ok = e < 0.25
    print(f"Images: {len(e)}")
    print(f"Successful detections (error < 0.25 squares): {ok.mean() * 100:.1f}%")
    print(f"Median error (successful): {np.median(e[ok]) if ok.any() else float('nan'):.3f} squares")
    print(f"Mean time: {np.mean(times) * 1000:.0f} ms")
    for name, err in failed[:10]:
        print(f"  failed: {name} (error {err:.2f})")


if __name__ == "__main__":
    main()
