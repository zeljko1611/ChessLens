"""Backend: the service layer (always) and the HTTP API (if FastAPI is installed)."""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from synth import available_fonts, render  # noqa: E402

from chessvision.position import fen_to_board  # noqa: E402

MODELS = ROOT / "models"
HAS_MODEL = (MODELS / "square_cnn.onnx").exists()


def _photo_bytes(seed: int = 3) -> tuple[bytes, np.ndarray]:
    board = fen_to_board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R")
    img, corners, _ = render(random.Random(seed), board, available_fonts(), camera_rot=0)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes(), corners


def test_service_recognize_and_rescale():
    if not HAS_MODEL:
        return  # the model is not trained (see README); skip the test
    from backend.service import Service

    svc = Service(MODELS)
    data, corners = _photo_bytes()
    res = svc.recognize(data)
    assert len(res["placement"].split("/")) == 8
    assert res["image_size"] == [720, 960] or res["image_size"] == [960, 720]
    # Corners given by hand (in pixels of the original image) come back in the same
    # coordinates, and the position differs in at most a few squares (corners are not pixel-identical).
    res2 = svc.recognize(data, corners_json=json.dumps(corners.tolist()))
    returned = np.array(res2["corners"])
    assert all(np.linalg.norm(returned - p, axis=1).min() < 1.0 for p in corners)
    b1, b2 = fen_to_board(res["placement"]), fen_to_board(res2["placement"])
    assert sum(b1[r][c] != b2[r][c] for r in range(8) for c in range(8)) <= 4


def test_service_rejects_garbage():
    if not HAS_MODEL:
        return
    from backend.service import BadRequest, Service

    svc = Service(MODELS)
    try:
        svc.recognize(b"not an image")
    except BadRequest:
        pass
    else:
        raise AssertionError("expected BadRequest")


def test_http_api():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        return  # FastAPI is not installed
    if not HAS_MODEL:
        return
    from backend.app import app

    data, _ = _photo_bytes()
    with TestClient(app) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        r = client.post("/api/recognize", files={"image": ("board.jpg", data, "image/jpeg")})
        assert r.status_code == 200, r.text
        assert " w " in r.json()["fen"] or " b " in r.json()["fen"]
        assert client.post("/api/recognize", files={"image": ("x.jpg", b"nope", "image/jpeg")}).status_code == 422

        bad = client.post("/api/recognize", files={"image": ("x.jpg", b"nope", "image/jpeg")})
        assert bad.json()["detail"] == "not_an_image"  # an error code; the frontend makes the text


def test_save_feedback(tmp_path):
    if not HAS_MODEL:
        return
    from backend.service import BadRequest, Service

    svc = Service(MODELS, feedback_dir=tmp_path)
    data, corners = _photo_bytes()
    placement = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R"
    res = svc.save_feedback(data, json.dumps(corners.tolist()), placement)
    assert res["count"] == 1
    row = json.loads((tmp_path / "labels.jsonl").read_text().splitlines()[0])
    assert row["fen"] == placement and (tmp_path / row["image"]).exists()
    try:
        svc.save_feedback(data, json.dumps(corners.tolist()), "8/8/8/8/8/8/8/8")  # no kings
    except BadRequest:
        pass
    else:
        raise AssertionError("an illegal position must not be saved")
