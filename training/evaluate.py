"""Evaluation of the WHOLE system on labeled photos.

Unlike per-crop accuracy (train_cnn.py), this runs the same code as the app:
detection -> crops -> model -> orientation -> repair.

Metrics:
  - detection: share of images where the board is found with error < 0.25 squares
  - square accuracy: correct squares out of 64 (on images where the board was found)
  - perfect boards: share of images with all 64 squares correct (the strictest metric)
  - at most 1 error: share of images with at most one wrong square
  - orientation: whether the board is rotated correctly (a1 in the right place)

    python training/evaluate.py --data data/generated/test --models models/ --json results/test.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_detection import corner_error  # noqa: E402

from chessvision.classifier import load_classifier  # noqa: E402
from chessvision.pipeline import BoardNotFound, recognize  # noqa: E402
from chessvision.position import fen_to_board  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", type=Path, help="save the results as JSON")
    ap.add_argument("--errors-dir", type=Path, help="save visualisations of boards with errors")
    args = ap.parse_args()

    clf = load_classifier(args.models)
    rows = [json.loads(line) for line in open(args.data / "labels.jsonl", encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    n = len(rows)
    detected = 0
    square_correct, square_total = 0, 0
    perfect, at_most_one, orient_ok = 0, 0, 0
    with_oracle_orientation = 0
    times = []
    for row in rows:
        img = cv2.imread(str(args.data / row["image"]))
        truth = fen_to_board(row["fen"])
        t0 = time.perf_counter()
        try:
            res = recognize(img, clf)
        except BoardNotFound:
            continue
        times.append(time.perf_counter() - t0)
        gt_corners = np.array(row["corners"], np.float32) if row.get("corners") else None
        if gt_corners is not None and corner_error(np.array(res.corners), gt_corners) >= 0.25:
            continue  # the board was not found accurately
        detected += 1

        wrong = sum(res.board[r][c] != truth[r][c] for r in range(8) for c in range(8))
        square_correct += 64 - wrong
        square_total += 64
        perfect += wrong == 0
        at_most_one += wrong <= 1

        if gt_corners is not None:
            square = np.linalg.norm(gt_corners[0] - gt_corners[1]) / 8
            a8_ok = np.linalg.norm(np.array(res.corners[0]) - gt_corners[0]) < 0.5 * square
            orient_ok += bool(a8_ok)
            if not a8_ok:
                # How many boards would be correct with the right orientation (impact of the heuristic).
                for k in range(4):
                    alt = recognize(img, clf, corners=res.corners, orientation=k)
                    if np.linalg.norm(np.array(alt.corners[0]) - gt_corners[0]) < 0.5 * square:
                        with_oracle_orientation += alt.board == truth
                        break

        if wrong and args.errors_dir:
            args.errors_dir.mkdir(parents=True, exist_ok=True)
            vis = img.copy()
            for r in range(8):
                for c in range(8):
                    name = "abcdefgh"[c] + str(8 - r)
                    x, y = map(int, res.square_centers[name])
                    pred, true = res.board[r][c], truth[r][c]
                    if pred != true:
                        cv2.putText(vis, f"{pred}/{true}", (x - 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(str(args.errors_dir / Path(row["image"]).name), vis)

    results = {
        "images": n,
        "model": clf.name,
        "detection_%": round(100 * detected / n, 2),
        "square_accuracy_%": round(100 * square_correct / max(square_total, 1), 2),
        "perfect_boards_%": round(100 * perfect / n, 2),
        "at_most_one_error_%": round(100 * at_most_one / n, 2),
        "orientation_%": round(100 * orient_ok / max(detected, 1), 2),
        "perfect_boards_with_known_orientation_%": round(100 * (perfect + with_oracle_orientation) / n, 2),
        "mean_time_ms": round(1000 * float(np.mean(times)) if times else 0, 1),
    }
    for k, v in results.items():
        print(f"{k:45s} {v}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
