"""Computer vision tests on synthetic images (known ground truth)."""

import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))

from eval_detection import corner_error  # noqa: E402
from synth import available_fonts, render  # noqa: E402

from chessvision.board import detect_board, order_corners  # noqa: E402
from chessvision.pipeline import recognize  # noqa: E402
from chessvision.position import CLASS_INDEX, CLASSES, STARTING_BOARD_FEN, board_to_fen, fen_to_board  # noqa: E402
from chessvision.squares import CROP_H, CROP_W, crop_squares, grid_points  # noqa: E402

POSITION = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R"


def synthetic(seed: int, camera_rot: int = 0, fen: str = POSITION):
    board = fen_to_board(fen)
    img, corners, _ = render(random.Random(seed), board, available_fonts(), camera_rot=camera_rot)
    return img, corners, board


def test_order_corners():
    pts = np.array([[10, 90], [90, 85], [12, 5], [95, 10]], np.float32)
    assert order_corners(pts).tolist() == [[12, 5], [95, 10], [90, 85], [10, 90]]


def test_detect_board_on_synthetic_images():
    errors = []
    for seed in range(6):
        img, corners, _ = synthetic(seed)
        det = detect_board(img)
        assert det is not None
        errors.append(corner_error(det.corners, corners))
    # At least 5 of 6 with an error below a quarter of a square.
    assert sum(e < 0.25 for e in errors) >= 5, errors


def test_no_board_on_image_without_board():
    # The contour fallback always finds some quadrilateral; without the threshold this would be a "board".
    img = np.full((600, 800, 3), 128, np.uint8)
    cv2.circle(img, (400, 300), 120, (60, 90, 200), -1)
    cv2.rectangle(img, (100, 80), (700, 520), (40, 40, 40), 3)
    assert detect_board(img) is None


def test_crop_shapes_and_grid():
    img, corners, _ = synthetic(1)
    crops = crop_squares(img, corners)
    assert crops.shape == (8, 8, CROP_H, CROP_W, 3)
    pts = grid_points(corners)
    assert np.allclose(pts[0, 0], corners[0], atol=1e-3)
    assert np.allclose(pts[8, 8], corners[2], atol=1e-3)


class OracleClassifier:
    """A "perfect" model: it knows the truth and returns it for the grid squares it receives.

    This tests everything EXCEPT the model: detection, grid ordering,
    orientation and mapping the corners back to board coordinates.
    """

    name = "oracle"

    def __init__(self, img, gt_corners, board):
        det = detect_board(img)
        H_gt = cv2.getPerspectiveTransform(np.asarray(gt_corners, np.float32), np.array([[0, 0], [8, 0], [8, 8], [0, 8]], np.float32))
        pts = grid_points(det.corners)
        self.probs = np.full((64, len(CLASSES)), 0.01, np.float32)
        for r in range(8):
            for c in range(8):
                center = pts[r : r + 2, c : c + 2].reshape(4, 2).mean(0)
                bx, by = cv2.perspectiveTransform(center.reshape(1, 1, 2).astype(np.float32), H_gt)[0, 0]
                piece = board[int(by)][int(bx)]
                self.probs[r * 8 + c, CLASS_INDEX[piece]] = 0.9

    def predict_proba(self, crops):
        return self.probs


def test_pipeline_orientation_from_both_sides():
    for camera_rot in (0, 2, 1):
        img, corners, board = synthetic(10 + camera_rot, camera_rot=camera_rot)
        res = recognize(img, OracleClassifier(img, corners, board))
        assert res.placement == board_to_fen(board), (camera_rot, res.placement)
        # Returned corners must be a8, h8, h1, a1 (within half a square).
        assert corner_error(np.array(res.corners), corners) < 0.25
        # The a8 in the result is close to the true a8.
        square = np.linalg.norm(corners[0] - corners[1]) / 8
        assert np.linalg.norm(np.array(res.corners[0]) - corners[0]) < 0.5 * square


def test_pipeline_start_position():
    img, corners, board = synthetic(42, camera_rot=2, fen=STARTING_BOARD_FEN)
    res = recognize(img, OracleClassifier(img, corners, board))
    assert res.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert not res.errors
