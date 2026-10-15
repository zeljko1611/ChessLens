"""Runs training/finetune.py in a separate process and keeps its progress for the app.

Training needs PyTorch and the original training crops (data/npz), so it is only available
where the model was trained, not in the Docker image.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from finetune import MIN_PHOTOS, load_rows  # noqa: E402


class Trainer:
    def __init__(self, model_dir: Path, feedback_dir: Path, npz_dir: Path = ROOT / "data" / "npz"):
        self.model_dir = model_dir
        self.feedback_dir = feedback_dir
        self.npz_dir = npz_dir
        self.candidate = model_dir / "candidate"
        self.previous = model_dir / "previous"
        self.available = importlib.util.find_spec("torch") is not None and any(npz_dir.glob("*.npz"))
        self.state = "done" if (self.candidate / "result.json").exists() else "idle"
        self.progress: dict = {}
        self.result: dict | None = self._saved_result()
        self.log: deque[str] = deque(maxlen=20)
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def _saved_result(self) -> dict | None:
        path = self.candidate / "result.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @property
    def running(self) -> bool:
        return self.state == "running"

    def status(self) -> dict:
        return {"available": self.available, "photos": len(load_rows(self.feedback_dir)), "min_photos": MIN_PHOTOS,
                "state": self.state, "progress": self.progress, "result": self.result,
                "has_previous": (self.previous / "square_cnn.onnx").exists(),
                "error": self.log[-1] if self.state == "failed" and self.log else None}

    def start(self) -> None:
        with self._lock:
            if not self.available:
                raise ValueError("training_unavailable")
            if self.running:
                raise ValueError("training_running")
            if len(load_rows(self.feedback_dir)) < MIN_PHOTOS:
                raise ValueError("too_few_photos")
            shutil.rmtree(self.candidate, ignore_errors=True)
            self.state, self.progress, self.result = "running", {"step": "start"}, None
            self.log.clear()
            self.process = subprocess.Popen(
                [sys.executable, str(ROOT / "training" / "finetune.py"), "--feedback", str(self.feedback_dir),
                 "--npz", str(self.npz_dir), "--models", str(self.model_dir), "--out", str(self.candidate)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT,
            )
        threading.Thread(target=self._follow, args=(self.process,), daemon=True).start()

    def _follow(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            self.log.append(line)
            event = json.loads(line) if line.startswith("{") else {}
            if event.get("event") == "step":
                self.progress = {"step": event["step"]}
            elif event.get("event") == "epoch":
                self.progress = {"step": "train", "epoch": event["epoch"], "epochs": event["epochs"]}
            elif event.get("event") == "done":
                self.result = event["result"]
        ok = process.wait() == 0 and self.result is not None
        self.state = "done" if ok else "failed"

    def accept(self) -> None:
        """Use the new model; the current one is kept in models/previous/."""
        if self.state != "done":
            raise ValueError("no_new_model")
        shutil.rmtree(self.previous, ignore_errors=True)
        self.previous.mkdir()
        for name in ("square_cnn.onnx", "square_cnn.pt"):
            shutil.copy2(self.model_dir / name, self.previous / name)
            shutil.copy2(self.candidate / name, self.model_dir / name)
        self.discard()

    def restore_previous(self) -> None:
        """Swap the current and the previous model (a second call undoes the first)."""
        if self.running:
            raise ValueError("training_running")
        if not (self.previous / "square_cnn.onnx").exists():
            raise ValueError("no_previous_model")
        for name in ("square_cnn.onnx", "square_cnn.pt"):
            current, previous = self.model_dir / name, self.previous / name
            tmp = current.with_suffix(".swap")
            current.replace(tmp)
            previous.replace(current)
            tmp.replace(previous)

    def discard(self) -> None:
        if self.running:
            raise ValueError("training_running")
        shutil.rmtree(self.candidate, ignore_errors=True)
        self.state, self.progress, self.result = "idle", {}, None

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
