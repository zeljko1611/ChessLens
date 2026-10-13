"""Backend logic independent of the web framework (easy to test).

User-facing errors are short codes (e.g. "board_not_found"); the frontend
turns them into English or Serbian text.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from chessvision.classifier import SquareClassifier, load_classifier
from chessvision.pipeline import BoardNotFound, recognize
from chessvision.position import fen_to_board, validate

MAX_SIDE = 1600  # larger images are downscaled (phones send 4000 px and more)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
# Corrected positions the user saves in the app (data for fine-tuning).
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", Path(__file__).resolve().parent.parent / "data" / "feedback"))


class BadRequest(Exception):
    """args[0] is an error code that the frontend translates."""


class Service:
    def __init__(self, model_dir: Path = MODEL_DIR, feedback_dir: Path = FEEDBACK_DIR):
        self.classifier: SquareClassifier = load_classifier(model_dir)
        self.feedback_dir = feedback_dir
        self._feedback_lock = threading.Lock()

    @staticmethod
    def decode_image(data: bytes) -> tuple[np.ndarray, float]:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # respects EXIF orientation
        if img is None:
            raise BadRequest("not_an_image")
        scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img, scale

    @staticmethod
    def parse_corners(corners_json: str, scale: float) -> np.ndarray:
        try:
            corners = np.array(json.loads(corners_json), np.float32) * scale
        except (ValueError, TypeError) as e:
            raise BadRequest("bad_corners") from e
        if corners.shape != (4, 2):
            raise BadRequest("bad_corners")
        return corners

    def recognize(self, data: bytes, corners_json: str | None = None, orientation: int | None = None,
                  side_to_move: str | None = None) -> dict:
        img, scale = self.decode_image(data)
        corners = self.parse_corners(corners_json, scale).tolist() if corners_json else None
        if side_to_move not in (None, "w", "b"):
            raise BadRequest("bad_side_to_move")
        try:
            res = recognize(img, self.classifier, corners=corners, orientation=orientation, side_to_move=side_to_move).to_dict()
        except BoardNotFound as e:
            raise BadRequest("board_not_found") from e

        # Return coordinates in the space of the ORIGINAL image the client sent.
        inv = 1.0 / scale
        res["corners"] = [[round(x * inv, 1), round(y * inv, 1)] for x, y in res["corners"]]
        res["grid"] = (np.array(res["grid"]) * inv).round(1).tolist()
        res["square_centers"] = {k: [round(v[0] * inv, 1), round(v[1] * inv, 1)] for k, v in res["square_centers"].items()}
        res["image_size"] = [round(img.shape[1] * inv), round(img.shape[0] * inv)]
        return res

    def save_feedback(self, data: bytes, corners_json: str, placement: str) -> dict:
        """Save a photo with its CORRECTED position as a training example.

        Corners are a8, h8, h1, a1 in pixels of the uploaded image (the same as
        /api/recognize returns). The format matches data/synth.py, so
        training/make_dataset.py reads it as is.
        """
        img, scale = self.decode_image(data)
        corners = self.parse_corners(corners_json, scale)
        try:
            board = fen_to_board(placement)
        except ValueError as e:
            raise BadRequest("bad_position") from e
        if not validate(board).ok:
            raise BadRequest("illegal_position")

        with self._feedback_lock:
            (self.feedback_dir / "images").mkdir(parents=True, exist_ok=True)
            name = f"images/{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.jpg"
            cv2.imwrite(str(self.feedback_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            row = {"image": name, "fen": placement.split()[0], "corners": corners.round(2).tolist(), "source": "app"}
            labels = self.feedback_dir / "labels.jsonl"
            with open(labels, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            count = len(labels.read_text(encoding="utf-8").splitlines())
        return {"saved": name, "count": count}
