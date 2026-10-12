"""Import of photos whose position (FEN) is known but whose board corners are not.

Example: samryan18/chess-dataset (500 real photos, MIT licence), where the FEN
is written in the file name ("1B2b3-Kp6-8-8-2k5-8-8-8.JPG").

    git clone --depth 1 https://github.com/samryan18/chess-dataset
    python data/import_fen_photos.py --src chess-dataset/labeled_originals --out data/real/samryan18

For each image:
  1. our detector finds the 8x8 grid (no corner labels needed);
  2. of the 4 possible orientations, the one where the OCCUPIED squares from the
     FEN best match the image is chosen. Occupancy is the difference between a
     square and its "empty" colour (light/dark), which does not depend on the
     model recognising pieces; the rule that h1 is a light square also applies;
  3. the image is kept only if the match is convincing (--min-agree) and the
     second-best orientation is clearly worse, since bad labels would hurt training.

The output uses the format of data/synth.py (labels.jsonl + images/), so
training/make_dataset.py reads it as is.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.board import detect_board  # noqa: E402
from chessvision.position import EMPTY, board_to_fen, fen_to_board, validate  # noqa: E402
from chessvision.squares import rotate_grid  # noqa: E402

MAX_SIDE = 1600
CELL = 32


def fen_from_name(name: str) -> str | None:
    """'rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR_2.JPG' -> FEN (suffix _2 = repeated photo)."""
    import re

    stem = re.sub(r"_\d+$", "", Path(name).stem).replace("-", "/")
    try:
        board = fen_to_board(stem)
    except ValueError:
        return None
    return board_to_fen(board)


def occupancy_scores(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """(8, 8) measure of "how occupied a square is", in grid coordinates.

    For each square: the median absolute pixel difference from the typical colour
    of empty squares of the same parity. A piece (of either colour) changes the
    centre of a square; an empty square does not.
    """
    size = 8 * CELL
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    lab = cv2.cvtColor(cv2.warpPerspective(img, H, (size, size), flags=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = CELL // 5
    cells = lab.reshape(8, CELL, 8, CELL, 3)[:, m:-m, :, m:-m].transpose(0, 2, 1, 3, 4).reshape(8, 8, -1, 3)
    med = np.median(cells, axis=2)  # typical colour of each square
    parity = (np.add.outer(np.arange(8), np.arange(8)) % 2).astype(bool)
    out = np.zeros((8, 8), np.float32)
    for p in (False, True):
        ref = np.median(med[parity == p], axis=0)  # most squares are empty
        diff = np.abs(cells[parity == p] - ref).sum(-1)  # (n, pixels)
        out[parity == p] = np.percentile(diff, 75, axis=1)
    return out, med[..., 0]


def best_orientation(occ: np.ndarray, light: np.ndarray, board: list[list[str]]) -> tuple[int, float, float]:
    """Return (k, agreement of the best, agreement of the second best)."""
    truth = np.array([[p != EMPTY for p in row] for row in board])
    n_occ = int(truth.sum())
    results = []
    for k in range(4):
        o = rotate_grid(occ, k)
        lt = rotate_grid(light, k)
        # Threshold: the n_occ most occupied squares are declared occupied.
        thr = np.sort(o.ravel())[-n_occ] if n_occ else np.inf
        pred = o >= thr
        agree = float((pred == truth).mean())
        light_sq = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0  # a8 and h1 are light
        color_ok = np.median(lt[light_sq]) > np.median(lt[~light_sq])
        results.append((agree + (0.05 if color_ok else -0.05), k, agree))
    results.sort(reverse=True)
    return results[0][1], results[0][2], results[1][2]


def fixed_agreement(occ: np.ndarray, board: list[list[str]]) -> float:
    truth = np.array([[p != EMPTY for p in row] for row in board])
    n_occ = int(truth.sum())
    thr = np.sort(occ.ravel())[-n_occ] if n_occ else np.inf
    return float(((occ >= thr) == truth).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder of images whose name is a FEN")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-agree", type=float, default=0.80, help="minimum share of squares whose occupancy agrees with the FEN")
    ap.add_argument("--min-margin", type=float, default=0.08, help="how much worse the second-best orientation must be")
    ap.add_argument("--source-name", default=None)
    ap.add_argument("--a8-top-left", action="store_true",
                    help="the photographer mostly stood on the same side with a8 top left (like samryan18); used "
                         "when occupancy cannot decide the orientation (symmetric positions)")
    ap.add_argument("--debug-dir", type=Path, help="save visualisations of rejected images")
    args = ap.parse_args()

    files = sorted(p for p in args.src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    kept, reasons = 0, {"fen": 0, "illegal": 0, "no_board": 0, "low_agreement": 0}
    with open(args.out / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, path in enumerate(files):
            fen = fen_from_name(path.name)
            if fen is None:
                reasons["fen"] += 1
                continue
            board = fen_to_board(fen)
            if not validate(board).ok:
                reasons["illegal"] += 1
                continue
            img = cv2.imread(str(path))
            scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            det = detect_board(img)
            if det is None:
                reasons["no_board"] += 1
                continue
            occ, light = occupancy_scores(img, det.corners)
            k, agree, second = best_orientation(occ, light, board)
            if agree - second < args.min_margin and args.a8_top_left:
                # Symmetric position (e.g. the initial one): occupancy cannot tell k from k+2.
                # Then the "a8 top left" assumption (rotation 0) applies if it is at least as good.
                agree_tl = fixed_agreement(occ, board)
                if agree_tl >= agree - 0.02:
                    k, agree, second = 0, agree_tl, -1.0
            if agree < args.min_agree or agree - second < args.min_margin:
                reasons["low_agreement"] += 1
                if args.debug_dir:
                    args.debug_dir.mkdir(parents=True, exist_ok=True)
                    vis = img.copy()
                    cv2.polylines(vis, [det.corners.astype(np.int32)], True, (0, 0, 255), 3)
                    cv2.putText(vis, f"{agree:.2f}/{second:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.imwrite(str(args.debug_dir / f"{path.stem}.jpg"), vis)
                continue
            # Corners in the order a8, h8, h1, a1: the grid corner belonging to square a8, etc.
            idx = rotate_grid(np.arange(64).reshape(8, 8), k)
            grid_corner = {0: 0, 7: 1, 63: 2, 56: 3}  # corner squares of the grid -> grid corner
            corners = [det.corners[grid_corner[int(idx[r, c])]].tolist() for r, c in ((0, 0), (0, 7), (7, 7), (7, 0))]
            name = f"images/{path.stem}.jpg"
            cv2.imwrite(str(args.out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            row = {"image": name, "fen": fen, "corners": np.round(corners, 2).tolist(), "source": args.source_name or args.src.name,
                   "orientation_agreement": round(agree, 3)}
            f.write(json.dumps(row) + "\n")
            kept += 1
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(files)}  kept {kept}", flush=True)
    print(f"Kept {kept} of {len(files)}; rejected: {reasons}")


if __name__ == "__main__":
    main()
