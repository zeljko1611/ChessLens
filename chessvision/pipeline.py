"""Full pipeline: photo -> board detection -> square crops -> model -> legal position -> FEN."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .board import BoardDetection, corners_from_user, detect_board
from .classifier import SquareClassifier
from .position import (
    CLASSES,
    Board,
    board_to_fen,
    full_fen,
    repair,
    square_name,
    suggest_side_to_move,
    uncertain_squares,
    validate,
)
from .squares import (
    best_orientation,
    crop_squares,
    grid_points,
    orientation_scores,
    rotate_grid,
    square_brightness,
)

# Board squares whose outer corner is a board corner: a8, h8, h1, a1.
_CORNER_CELLS = [(0, 0), (0, 7), (7, 7), (7, 0)]
# The grid corner (in grid coordinates) that belongs to each corner square of the grid.
_GRID_CORNER_OF_CELL = {(0, 0): 0, (0, 7): 1, (7, 7): 2, (7, 0): 3}


class BoardNotFound(Exception):
    pass


@dataclass
class Recognition:
    fen: str
    placement: str
    board: Board
    side_to_move: str
    side_to_move_hint: str | None
    orientation: int
    orientation_scores: list[float]
    corners: list[list[float]]  # image corners in the order a8, h8, h1, a1
    grid: list[list[list[float]]]  # 9x9 line intersections (in grid order)
    square_centers: dict[str, list[float]]
    confidence: dict[str, float]  # model confidence in the chosen class, per square
    probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    uncertain: list[str] = field(default_factory=list)
    repairs: list[dict] = field(default_factory=list)  # {square, from, to, reason}
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    detection_score: float = 0.0
    model: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def board_cell_to_grid(k: int) -> np.ndarray:
    """For rotation k: an (8, 8, 2) array telling which grid square is board square [r][c]."""
    idx = np.arange(64).reshape(8, 8)
    rot = rotate_grid(idx, k)
    return np.stack([rot // 8, rot % 8], axis=-1)


def recognize(
    image: np.ndarray,
    classifier: SquareClassifier,
    corners: list[list[float]] | None = None,
    orientation: int | None = None,
    side_to_move: str | None = None,
) -> Recognition:
    timings: dict[str, float] = {}
    t = time.perf_counter()

    det: BoardDetection | None = corners_from_user(corners) if corners else detect_board(image)
    if det is None:
        raise BoardNotFound("Board not found.")
    timings["detection"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    crops = crop_squares(image, det.corners)
    probs_grid = classifier.predict_proba(crops.reshape(64, *crops.shape[2:])).reshape(8, 8, len(CLASSES))
    timings["classification"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    brightness = square_brightness(image, det.corners)
    scores = orientation_scores(probs_grid, brightness)
    k = int(orientation) % 4 if orientation is not None else best_orientation(probs_grid, brightness)
    board_probs = rotate_grid(probs_grid, k)

    board, changes = repair(board_probs)
    hint = suggest_side_to_move(board)
    stm = side_to_move or hint or "w"
    validation = validate(board, stm)
    timings["position"] = (time.perf_counter() - t) * 1000

    # Geometry in board coordinates (for drawing in the UI).
    cell_map = board_cell_to_grid(k)
    grid_corner_pts = det.corners
    board_corners = []
    for r, c in _CORNER_CELLS:
        gr, gc = cell_map[r, c]
        board_corners.append(grid_corner_pts[_GRID_CORNER_OF_CELL[(int(gr), int(gc))]].tolist())
    pts = grid_points(det.corners)
    centers, confidence, probabilities = {}, {}, {}
    for r in range(8):
        for c in range(8):
            gr, gc = cell_map[r, c]
            name = square_name(r, c)
            centers[name] = pts[gr : gr + 2, gc : gc + 2].reshape(4, 2).mean(axis=0).round(1).tolist()
            p = board_probs[r, c]
            confidence[name] = round(float(p[CLASSES.index(board[r][c])]), 3)
            top = np.argsort(-p)[:3]
            probabilities[name] = {CLASSES[i]: round(float(p[i]), 3) for i in top}

    return Recognition(
        fen=full_fen(board, stm),
        placement=board_to_fen(board),
        board=board,
        side_to_move=stm,
        side_to_move_hint=hint,
        orientation=k,
        orientation_scores=[round(float(s), 3) for s in scores],
        corners=[[round(float(x), 1), round(float(y), 1)] for x, y in board_corners],
        grid=pts.round(1).tolist(),
        square_centers=centers,
        confidence=confidence,
        probabilities=probabilities,
        uncertain=uncertain_squares(board_probs),
        repairs=changes,
        errors=validation.errors,
        warnings=validation.warnings,
        detection_score=round(float(det.score), 2) if np.isfinite(det.score) else 0.0,
        model=classifier.name,
        timings_ms={k_: round(v, 1) for k_, v in timings.items()},
    )
