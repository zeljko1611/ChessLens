"""FastAPI server: `uvicorn backend.app:app --reload --port 8000`.

Errors are returned as codes in "detail" (e.g. "board_not_found"). scripts/start_app.sh sets
CHESSVISION_IDLE_TIMEOUT (stop after that many idle seconds) and CHESSVISION_PHONE_PORT (report a LAN address).
"""

from __future__ import annotations

import os
import signal
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service import BadRequest, Service  # noqa: E402
from backend.training import Trainer  # noqa: E402
from chessvision.classifier import load_classifier  # noqa: E402

MAX_UPLOAD = 15 * 1024 * 1024
IDLE_TIMEOUT = float(os.environ.get("CHESSVISION_IDLE_TIMEOUT", "0"))
PHONE_PORT = os.environ.get("CHESSVISION_PHONE_PORT")
service: Service | None = None
trainer: Trainer | None = None
last_request = time.monotonic()
# The latest recognition from any device: {"id", "client", "result"} and the uploaded image.
latest: dict = {"id": 0}
latest_image = b""


def _shutdown_when_idle() -> None:
    while True:
        time.sleep(5)
        if time.monotonic() - last_request > IDLE_TIMEOUT and not (trainer and trainer.running):
            os.kill(os.getpid(), signal.SIGINT)  # same as Ctrl+C
            return


def lan_ip() -> str | None:
    """This computer's address on the home network (a UDP "connect" sends no packets)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service, trainer
    service = Service()
    trainer = Trainer(service.model_dir, service.feedback_dir)
    if IDLE_TIMEOUT > 0:
        threading.Thread(target=_shutdown_when_idle, daemon=True).start()
    yield
    trainer.stop()


app = FastAPI(title="Chess Vision", lifespan=lifespan)
app.add_middleware(  # for `npm run dev` (Vite on port 5173)
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"]
)


@app.middleware("http")
async def track_activity(request, call_next):
    global last_request
    last_request = time.monotonic()
    response = await call_next(request)
    last_request = time.monotonic()
    return response


async def read_image(image: UploadFile) -> bytes:
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image_too_large")
    return data


@app.get("/api/health")
def health() -> dict:
    assert service is not None
    ip = lan_ip() if PHONE_PORT else None
    return {"status": "ok", "model": service.classifier.name, "phone_url": f"http://{ip}:{PHONE_PORT}" if ip else None}


@app.post("/api/recognize")
async def recognize_endpoint(
    image: UploadFile = File(...),
    corners: str | None = Form(None),
    orientation: int | None = Form(None),
    side_to_move: str | None = Form(None),
    client: str | None = Form(None),  # sender id, so the sender does not reload its own photo
) -> dict:
    global latest, latest_image
    assert service is not None
    data = await read_image(image)
    try:
        # Computer vision is CPU-bound work: run it outside the event loop.
        res = await run_in_threadpool(service.recognize, data, corners, orientation, side_to_move)
    except BadRequest as e:
        raise HTTPException(422, str(e)) from e
    latest, latest_image = {"id": latest["id"] + 1, "client": client, "result": res}, data
    return res


@app.get("/api/latest")
def latest_endpoint(since: int = 0) -> dict:
    # No result unless there is something new; clients poll this every 2 seconds.
    return latest if latest["id"] > since else {"id": latest["id"]}


@app.get("/api/latest/image")
def latest_image_endpoint() -> Response:
    if not latest_image:
        raise HTTPException(404, "no_image")
    return Response(latest_image, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/feedback")
async def feedback_endpoint(image: UploadFile = File(...), corners: str = Form(...), placement: str = Form(...)) -> dict:
    assert service is not None
    data = await read_image(image)
    try:
        return await run_in_threadpool(service.save_feedback, data, corners, placement)
    except BadRequest as e:
        raise HTTPException(422, str(e)) from e


def is_local(request: Request) -> bool:
    """Training and model changes are allowed only on the computer running the server, not from phones."""
    return request.client is not None and request.client.host in ("127.0.0.1", "::1")


@app.get("/api/training")
def training_status(request: Request) -> dict:
    assert trainer is not None
    return {**trainer.status(), "local": is_local(request)}


@app.post("/api/training/{action}")
def training_action(action: str, request: Request) -> dict:
    assert service is not None and trainer is not None
    if not is_local(request):
        raise HTTPException(403, "training_local_only")
    try:
        if action == "start":
            trainer.start()
        elif action == "accept":
            trainer.accept()
            service.classifier = load_classifier(service.model_dir)
        elif action == "restore":
            trainer.restore_previous()
            service.classifier = load_classifier(service.model_dir)
        elif action == "discard":
            trainer.discard()
        else:
            raise HTTPException(404, "unknown_action")
    except ValueError as e:
        raise HTTPException(409, str(e)) from e
    return {**trainer.status(), "local": True}


DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
