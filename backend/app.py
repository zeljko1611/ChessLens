"""FastAPI server.

    uvicorn backend.app:app --reload --port 8000

    GET  /api/health      which model is loaded
    POST /api/recognize   multipart: image (+ optional corners, orientation, side_to_move)
    POST /api/feedback    multipart: image, corners (a8, h8, h1, a1), placement

Errors are returned as a code in the "detail" field (e.g. "board_not_found").
If frontend/dist exists (after `npm run build`), the server serves it as well.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.service import BadRequest, Service  # noqa: E402

MAX_UPLOAD = 15 * 1024 * 1024
service: Service | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global service
    service = Service()
    yield


app = FastAPI(title="Chess Vision", lifespan=lifespan)
app.add_middleware(  # for `npm run dev` (Vite on port 5173)
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"]
)


async def read_image(image: UploadFile) -> bytes:
    data = await image.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image_too_large")
    return data


@app.get("/api/health")
def health() -> dict:
    assert service is not None
    return {"status": "ok", "model": service.classifier.name}


@app.post("/api/recognize")
async def recognize_endpoint(
    image: UploadFile = File(...),
    corners: str | None = Form(None),
    orientation: int | None = Form(None),
    side_to_move: str | None = Form(None),
) -> dict:
    assert service is not None
    data = await read_image(image)
    try:
        # Computer vision is CPU-bound work: run it outside the event loop.
        return await run_in_threadpool(service.recognize, data, corners, orientation, side_to_move)
    except BadRequest as e:
        raise HTTPException(422, str(e)) from e


@app.post("/api/feedback")
async def feedback_endpoint(image: UploadFile = File(...), corners: str = Form(...), placement: str = Form(...)) -> dict:
    assert service is not None
    data = await read_image(image)
    try:
        return await run_in_threadpool(service.save_feedback, data, corners, placement)
    except BadRequest as e:
        raise HTTPException(422, str(e)) from e


DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="frontend")
