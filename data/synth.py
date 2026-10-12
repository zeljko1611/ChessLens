"""Generator of synthetic chessboard photos.

The board is drawn flat (top-down) and placed into the scene with a homography,
as if photographed at an angle with a phone. Pieces are NOT warped with the
board; they are drawn upright in image space, with their base at the square
centre and a size that depends on the distance from the camera. This gives the
key effect of real photos: pieces stick up out of their square and hide the
squares behind them.

For each image the generator stores:
  - the FEN piece placement,
  - the four playing-area corners in the image, in the order a8, h8, h1, a1,
  - camera_rot: which side the camera is on (0 = White is closest to the camera).

Usage:
    python data/synth.py --out data/synth_train --count 3000 --seed 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.position import (  # noqa: E402
    EMPTY,
    STARTING_BOARD_FEN,
    board_to_fen,
    fen_to_board,
    validate,
)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    # macOS / Windows (when running the generator locally)
    "/System/Library/Fonts/Apple Symbols.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
]

# Filled glyphs (the black set); we colour them ourselves for both sides.
GLYPHS = {"k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f"}
# Piece height relative to the square width (the king is the tallest).
HEIGHT = {"k": 1.55, "q": 1.45, "b": 1.25, "n": 1.1, "r": 1.0, "p": 0.85}

PALETTES = [  # (light squares, dark squares, frame)
    ((238, 238, 210), (118, 150, 86), (60, 70, 50)),  # green tournament
    ((240, 217, 181), (181, 136, 99), (90, 60, 40)),  # wood
    ((222, 196, 160), (140, 95, 60), (60, 40, 25)),  # darker wood
    ((230, 230, 230), (120, 120, 120), (40, 40, 40)),  # grey
    ((235, 225, 205), (75, 60, 50), (30, 25, 20)),  # walnut
    ((250, 245, 235), (150, 120, 190), (80, 60, 100)),  # purple (online style)
]


def available_fonts() -> list[str]:
    fonts = []
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                # The font must really contain chess glyphs: otherwise they all
                # render as the same "missing glyph" box and king and pawn look alike.
                if not np.array_equal(_raw_glyph(path, "k"), _raw_glyph(path, "p")):
                    fonts.append(path)
            except OSError:
                pass
    if not fonts:
        raise RuntimeError("No font with chess glyphs found (e.g. DejaVu Sans).")
    return fonts


def _raw_glyph(font_path: str, piece: str) -> np.ndarray:
    font = ImageFont.truetype(font_path, 200)
    img = Image.new("L", (300, 300), 0)
    ImageDraw.Draw(img).text((50, 20), GLYPHS[piece], font=font, fill=255)
    return np.asarray(img, dtype=np.float32) / 255.0


@lru_cache(maxsize=64)
def glyph_mask(font_path: str, piece: str) -> np.ndarray:
    """Glyph mask (float 0..1), cropped tightly around the piece."""
    arr = _raw_glyph(font_path, piece)
    ys, xs = np.nonzero(arr > 0.05)
    return arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


# --------------------------------------------------------------------------
# Positions
# --------------------------------------------------------------------------


def random_position(rng: random.Random) -> list[list[str]]:
    """A game-like position: start from the initial one and "play" it randomly."""
    for _ in range(200):
        start = fen_to_board(STARTING_BOARD_FEN)
        phase = rng.random()  # 0 = opening, 1 = endgame
        board = [[EMPTY] * 8 for _ in range(8)]
        pieces = [(r, c, start[r][c]) for r in range(8) for c in range(8) if start[r][c] != EMPTY]
        rng.shuffle(pieces)
        for r, c, p in pieces:
            low = p.lower()
            if low != "k" and rng.random() < phase * 0.75:
                continue  # the piece was captured
            white = p.isupper()
            fwd = -1 if white else 1
            if rng.random() < 0.35 * (1 - phase):
                nr, nc = r, c  # it has not moved
            elif low == "p":
                nr = r + fwd * rng.randint(1, 1 + int(5 * phase))
                nc = c + rng.choice([0, 0, 0, -1, 1])
            elif low == "k":
                nr = r + fwd * rng.randint(0, int(1 + 5 * phase * phase))
                nc = c + rng.randint(-2, 2)
            else:
                radius = 2 + int(5 * phase)
                nr = r + fwd * rng.randint(0, radius) + rng.randint(-1, 1)
                nc = c + rng.randint(-radius, radius)
            nr, nc = min(7, max(0, nr)), min(7, max(0, nc))
            if low == "p" and nr in (0, 7):
                p = rng.choice("QNRB") if white else rng.choice("qnrb")
            if board[nr][nc] == EMPTY:
                board[nr][nc] = p
            elif low == "k":
                break  # a king must exist; try again
        if validate(board).ok:
            return board
    return fen_to_board(STARTING_BOARD_FEN)


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _jitter(color: tuple[int, int, int], rng: random.Random, amount: int = 18) -> np.ndarray:
    return np.clip(np.array(color, np.float32) + [rng.uniform(-amount, amount) for _ in range(3)], 0, 255)


def draw_flat_board(rng: random.Random, sq: int, border: int) -> np.ndarray:
    light, dark, frame = PALETTES[rng.randrange(len(PALETTES))]
    light, dark, frame = _jitter(light, rng), _jitter(dark, rng), _jitter(frame, rng, 25)
    size = 8 * sq + 2 * border
    img = np.empty((size, size, 3), np.float32)
    img[:] = frame
    for r in range(8):
        for c in range(8):
            y, x = border + r * sq, border + c * sq
            img[y : y + sq, x : x + sq] = light if (r + c) % 2 == 0 else dark
    # Wood grain / texture: waves plus noise.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = rng.uniform(0, math.pi)
    stripes = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) / rng.uniform(3, 12))
    img += stripes[..., None] * rng.uniform(0, 8)
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(1, 5), img.shape)
    if rng.random() < 0.5:  # a thin line between the frame and the squares
        cv2.rectangle(img, (border - 2, border - 2), (border + 8 * sq + 1, border + 8 * sq + 1), tuple(map(float, frame * 0.6)), 2)
    return np.clip(img, 0, 255)


def draw_background(rng: random.Random, h: int, w: int) -> np.ndarray:
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    c1, c2 = nprng.uniform(20, 230, 3), nprng.uniform(20, 230, 3)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = (c1 * (1 - t) + c2 * t) * np.ones((h, w, 3), np.float32)
    # Some clutter: rectangles and lines (table, papers, cables...).
    for _ in range(rng.randint(0, 6)):
        color = tuple(float(v) for v in nprng.uniform(0, 255, 3))
        p1 = (rng.randrange(w), rng.randrange(h))
        p2 = (p1[0] + rng.randint(-w // 3, w // 3), p1[1] + rng.randint(-h // 3, h // 3))
        if rng.random() < 0.5:
            cv2.rectangle(img, p1, p2, color, -1)
        else:
            cv2.line(img, p1, p2, color, rng.randint(1, 8))
    img += nprng.normal(0, 6, img.shape)
    return np.clip(img, 0, 255)


def camera_quad(rng: random.Random, h: int, w: int) -> np.ndarray:
    """The four playing-area corners in the image (TL, TR, BR, BL), as if the camera faced the board."""
    base = min(h, w) * rng.uniform(0.55, 0.85)
    top_ratio = rng.uniform(0.6, 1.0)  # perspective: the far edge is shorter
    depth = base * rng.uniform(0.5, 1.0)  # how "flattened" the board looks
    quad = np.array(
        [
            [-base * top_ratio / 2, -depth / 2],
            [base * top_ratio / 2, -depth / 2],
            [base / 2, depth / 2],
            [-base / 2, depth / 2],
        ],
        np.float32,
    )
    a = math.radians(rng.uniform(-20, 20))
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
    quad = quad @ rot.T
    # Shift so the board (with frame and pieces) stays in the frame.
    mn, mx = quad.min(0), quad.max(0)
    margin = 0.1 * min(h, w)  # room for the board frame
    top_margin = margin + 0.12 * base  # pieces on the back rank stick up
    lo_x, hi_x = margin - mn[0], w - margin - mx[0]
    lo_y, hi_y = top_margin - mn[1], h - margin - mx[1]
    cx = rng.uniform(lo_x, hi_x) if hi_x > lo_x else w / 2
    cy = rng.uniform(lo_y, hi_y) if hi_y > lo_y else h / 2
    return quad + np.array([cx, cy], np.float32)


def paste_piece(img: np.ndarray, mask: np.ndarray, base_xy: tuple[float, float], height: float, white: bool, rng: random.Random) -> None:
    h = max(4, int(round(height)))
    w = max(3, int(round(mask.shape[1] * h / mask.shape[0])))
    m = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    pad = max(2, h // 12)
    m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    outline = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=max(1, h // 40))
    H, W = m.shape
    x0 = int(round(base_xy[0] - W / 2))
    y0 = int(round(base_xy[1] - H + pad + h * 0.08))

    if white:
        fill = _jitter((232, 222, 200), rng, 15)
        edge = _jitter((70, 60, 50), rng, 15)
    else:
        fill = _jitter((45, 38, 35), rng, 12)
        edge = _jitter((150, 140, 130), rng, 20)
    shade = np.linspace(1.08, 0.82, H, dtype=np.float32)[:, None, None]  # light from above

    # Shadow on the board.
    shadow = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(shadow, (int(base_xy[0] + h * 0.1), int(base_xy[1] + h * 0.02)), (max(2, w // 2), max(1, h // 8)), 0, 0, 360, 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), max(1, h / 20))
    img *= (1 - 0.35 * shadow)[..., None]

    ys, xs = slice(max(0, y0), min(img.shape[0], y0 + H)), slice(max(0, x0), min(img.shape[1], x0 + W))
    my, mx = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
    if ys.start >= ys.stop or xs.start >= xs.stop:
        return
    region = img[ys, xs]
    o = outline[my, mx][..., None]
    f = m[my, mx][..., None]
    s = shade[my]
    region[:] = region * (1 - o) + edge * o
    region[:] = region * (1 - f) + (fill * s) * f


def render(rng: random.Random, board: list[list[str]], fonts: list[str], camera_rot: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    portrait = rng.random() < 0.6
    h, w = (960, 720) if portrait else (720, 960)
    img = draw_background(rng, h, w)

    sq = 64
    border = int(sq * rng.choice([0.0, rng.uniform(0.15, 0.7)]))
    flat = draw_flat_board(rng, sq, border)
    quad = camera_quad(rng, h, w)  # TL, TR, BR, BL in the image

    if camera_rot is None:
        camera_rot = rng.choices([0, 2, 1, 3], weights=[45, 45, 5, 5])[0]
    # Playing-area corners on the flat board: a8, h8, h1, a1.
    play = np.array([[border, border], [border + 8 * sq, border], [border + 8 * sq, border + 8 * sq], [border, border + 8 * sq]], np.float32)
    corners_img = np.array([quad[(i + camera_rot) % 4] for i in range(4)], np.float32)  # a8, h8, h1, a1
    Hm = cv2.getPerspectiveTransform(play, corners_img)

    warped = cv2.warpPerspective(flat, Hm, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones(flat.shape[:2], np.float32), Hm, (w, h))
    img = img * (1 - mask[..., None]) + warped * mask[..., None]

    # Pieces from the farthest to the nearest (larger y = closer to the camera).
    font = rng.choice(fonts)
    items = []
    for r in range(8):
        for c in range(8):
            p = board[r][c]
            if p == EMPTY:
                continue
            cx, cy = border + (c + 0.5) * sq, border + (r + 0.5) * sq
            pts = cv2.perspectiveTransform(np.array([[[cx, cy], [cx - sq / 2, cy], [cx + sq / 2, cy], [cx, cy - sq / 2], [cx, cy + sq / 2]]], np.float32), Hm)[0]
            local_w = max(np.linalg.norm(pts[2] - pts[1]), np.linalg.norm(pts[4] - pts[3]))
            items.append((pts[0][1], pts[0], local_w, p))
    for _, base, local_w, p in sorted(items, key=lambda t: t[0]):
        height = local_w * HEIGHT[p.lower()] * rng.uniform(0.9, 1.1)
        paste_piece(img, glyph_mask(font, p.lower()), (float(base[0] + rng.uniform(-2, 2)), float(base[1])), height, p.isupper(), rng)

    img = photo_effects(img, rng)
    return img, corners_img, camera_rot


def photo_effects(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    nprng = np.random.default_rng(rng.randrange(1 << 30))
    # Uneven lighting (a lamp from the side).
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lx, ly = rng.uniform(-0.5, 1.5) * w, rng.uniform(-0.5, 1.5) * h
    dist = np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / max(h, w)
    light = 1.15 - rng.uniform(0.1, 0.5) * dist
    img = img * light[..., None]
    # Colour temperature, contrast, gamma.
    img = img * np.array([rng.uniform(0.85, 1.1), 1.0, rng.uniform(0.85, 1.1)], np.float32)
    img = (img - 128) * rng.uniform(0.8, 1.2) + 128 + rng.uniform(-25, 25)
    img = np.clip(img, 0, 255)
    img = 255 * (img / 255) ** rng.uniform(0.8, 1.25)
    # Glare.
    if rng.random() < 0.3:
        glare = np.zeros((h, w), np.float32)
        cv2.circle(glare, (rng.randrange(w), rng.randrange(h)), rng.randint(30, 140), 1.0, -1)
        glare = cv2.GaussianBlur(glare, (0, 0), 40)
        img = img + glare[..., None] * rng.uniform(40, 110)
    # Focus / hand shake and sensor noise.
    if rng.random() < 0.6:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.2:
        k = rng.randint(5, 9)
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1 / k
        M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, M, (k, k))
        img = cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))
    img = img + nprng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    # JPEG compression like on a phone.
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(55, 95)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def generate(out_dir: Path, count: int, seed: int, font_filter: list[str] | None = None) -> None:
    rng = random.Random(seed)
    fonts = available_fonts()
    if font_filter:
        # E.g. --fonts FreeMono: a test set with a "piece set" the model has not seen.
        fonts = [f for f in fonts if any(k.lower() in Path(f).name.lower() for k in font_filter)]
        if not fonts:
            raise SystemExit(f"No font matches the filter {font_filter}")
    print("Fonts (piece sets):", [Path(f).name for f in fonts])
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for i in range(count):
            board = random_position(rng)
            img, corners, camera_rot = render(rng, board, fonts)
            name = f"images/{i:06d}.jpg"
            cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            f.write(json.dumps({"image": name, "fen": board_to_fen(board), "corners": corners.round(2).tolist(), "camera_rot": camera_rot}) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{count}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fonts", nargs="*", help="use only fonts whose name contains these words")
    args = ap.parse_args()
    generate(args.out, args.count, args.seed, args.fonts)


if __name__ == "__main__":
    main()
