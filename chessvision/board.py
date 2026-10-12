"""Chessboard detection with classical computer vision (no ML).

Quadrilateral candidates are found from edges and adaptive thresholds; each one
is searched for the 8x8 checkerboard grid and its corners are refined locally.

The result is the four corners of the playing area in image coordinates,
clockwise from the top-left (TL, TR, BR, BL). Which corner is which square
(a1, h1, ...) is decided later from the board content.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = 1000  # detection runs on a downscaled image (longest side)
LATTICE_N = 400  # size of the rectified candidate used for the grid search
REFINE_CELL = 16  # pixels per square during refinement


@dataclass
class BoardDetection:
    corners: np.ndarray  # (4, 2) float32, TL, TR, BR, BL in the original image
    score: float

    def homography(self, cell: int = 1) -> np.ndarray:
        """Homography from the image to a grid where one square is `cell` pixels."""
        dst = np.array([[0, 0], [8 * cell, 0], [8 * cell, 8 * cell], [0, 8 * cell]], np.float32)
        return cv2.getPerspectiveTransform(self.corners.astype(np.float32), dst)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points clockwise: TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(ang)]  # clockwise in image coordinates (y points down)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _quad_ok(q: np.ndarray, img_area: float) -> bool:
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.int32)):
        return False
    area = cv2.contourArea(q.astype(np.float32))
    if area < 0.04 * img_area:
        return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(sides) > 0.2 * max(sides)


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    if hasattr(cv2, "approxPolyN"):  # OpenCV >= 4.11: exactly N vertices
        try:
            q = cv2.approxPolyN(hull, 4)
            return q.reshape(4, 2).astype(np.float32)
        except cv2.error:
            return None
    return None


def candidate_quads(gray: np.ndarray, max_candidates: int = 12) -> list[np.ndarray]:
    h, w = gray.shape
    img_area = float(h * w)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    masks = []
    med = float(np.median(blur))
    for lo, hi in ((0.5 * med, 1.2 * med), (20, 60), (40, 120)):
        e = cv2.Canny(blur, lo, hi)
        masks.append(cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=2))
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    masks.append(255 - th)
    masks.append(th)

    quads: list[tuple[float, np.ndarray]] = []
    for m in masks:
        contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
        for cnt in contours:
            if cv2.contourArea(cnt) < 0.04 * img_area:
                break
            q = _approx_quad(cnt)
            if q is not None and _quad_ok(order_corners(q), img_area):
                quads.append((cv2.contourArea(q), order_corners(q)))

    # Remove duplicates (nearly identical quadrilaterals from different masks).
    unique: list[np.ndarray] = []
    for _, q in sorted(quads, key=lambda t: -t[0]):
        if all(np.abs(q - u).max() > 0.02 * max(h, w) for u in unique):
            unique.append(q)
    return unique[:max_candidates]


# --------------------------------------------------------------------------
# Grid score
# --------------------------------------------------------------------------


def grid_score(gray: np.ndarray, corners: np.ndarray, cell: int = REFINE_CELL) -> float:
    """How well four corners describe an 8x8 checkerboard in the image.

    - A square's value is the median of its pixels, so a piece covering less
      than half of the square does not change it.
    - Neighbouring squares must differ in EVERY row and column; the outer rows
      and columns are checked separately, because a grid shifted by one square
      includes a row of frame or background with no contrast.
    - Edges must be stronger on the grid lines than in the middle of squares
      (this gives sub-square precision).
    """
    size = 8 * cell
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_AREA).astype(np.float32)
    m = 2
    cells = warped.reshape(8, cell, 8, cell)[:, m:-m, :, m:-m].transpose(0, 2, 1, 3).reshape(8, 8, -1)
    v = np.median(cells, axis=2)
    dh = np.abs(v[:, :-1] - v[:, 1:])  # 8 x 7
    dv = np.abs(v[:-1, :] - v[1:, :])  # 7 x 8
    overall = float(np.median(np.concatenate([dh.ravel(), dv.ravel()])))
    boundary = min(float(np.median(dh[0])), float(np.median(dh[7])), float(np.median(dv[:, 0])), float(np.median(dv[:, 7])))

    gx = np.abs(cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3))
    idx = np.arange(1, 8) * cell
    on_lines = gx[:, idx].mean() + gy[idx, :].mean()
    mid = idx - cell // 2
    off_lines = gx[:, mid].mean() + gy[mid, :].mean() + 1e-3
    edge_ratio = min(float(on_lines / off_lines), 4.0)
    return (0.5 * overall + boundary) * edge_ratio**0.5


def find_lattice(gray: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, float]:
    """Find the 8x8 playing area inside a rectified quadrilateral."""
    N = LATTICE_N
    dst = np.array([[0, 0], [N, 0], [N, N], [0, N]], np.float32)
    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, H, (N, N), flags=cv2.INTER_AREA).astype(np.float32)

    best = (-1.0, 0.0, 0.0, N / 8)
    for s in np.arange(N / 8 * 0.62, N / 8 * 1.001, 1.0):
        k = max(3, int(s * 0.5) | 1)
        box = cv2.blur(warped, (k, k))  # local mean around every point
        span = int(8 * s)
        if span > N:
            continue
        lim = N - span
        offs = np.arange(0, lim + 1, 2)
        # Values at the square centres for all offsets at once.
        acc = np.zeros((len(offs), len(offs)), np.float32)
        for r in range(8):
            for c in range(8):
                ys = np.clip((offs + (r + 0.5) * s).astype(int), 0, N - 1)
                xs = np.clip((offs + (c + 0.5) * s).astype(int), 0, N - 1)
                v = box[np.ix_(ys, xs)]
                acc += v if (r + c) % 2 == 0 else -v
        score = np.abs(acc) / 64.0
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        if score[i, j] > best[0]:
            best = (float(score[i, j]), float(offs[j]), float(offs[i]), float(s))

    _, ox, oy, s = best
    inner = np.array([[ox, oy], [ox + 8 * s, oy], [ox + 8 * s, oy + 8 * s], [ox, oy + 8 * s]], np.float32)
    corners = cv2.perspectiveTransform(inner[None], np.linalg.inv(H))[0]
    return corners, grid_score(gray, corners)


def refine_corners(gray: np.ndarray, corners: np.ndarray, iters: int = 3) -> tuple[np.ndarray, float]:
    """Coordinate ascent: move each corner while the grid score improves."""
    corners = corners.astype(np.float32).copy()
    best = grid_score(gray, corners)
    scale = np.linalg.norm(corners[0] - corners[2]) / 8.0  # ~square size
    for step in (0.25 * scale, 0.1 * scale, 0.04 * scale):
        for _ in range(iters):
            improved = False
            for i in range(4):
                for d in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    cand = corners.copy()
                    cand[i] += d
                    sc = grid_score(gray, cand)
                    if sc > best:
                        best, corners, improved = sc, cand, True
            if not improved:
                break
    return corners, best


def detect_board(image: np.ndarray) -> BoardDetection | None:
    """Main entry point: BGR image -> corners of the playing area, or None."""
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    scale = WORK_SIZE / max(gray_full.shape)
    scale = min(scale, 1.0)
    gray = cv2.resize(gray_full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray_full
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Quadrilateral contours + a grid search inside each.
    scored: list[tuple[float, np.ndarray]] = []
    for quad in candidate_quads(gray):
        corners = order_corners(find_lattice(gray, quad)[0])
        if _quad_ok(corners, 0.0):
            scored.append((grid_score(gray, corners), corners))
    if not scored:
        return None
    score, corners = max(scored, key=lambda t: t[0])
    corners, score = refine_corners(gray, corners)
    return BoardDetection(corners=order_corners(corners) / scale, score=score)


def corners_from_user(points: list[list[float]]) -> BoardDetection:
    """Corners given by hand (e.g. dragged by the user in the UI)."""
    corners = order_corners(np.array(points, np.float32))
    return BoardDetection(corners=corners, score=float("nan"))
