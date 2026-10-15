"""Fine-tune the CNN on photos saved with "Save for training", mixed with the original training data.

Every fifth photo is held out to compare the old and the new model. The new model goes to --out;
it replaces models/ only when it is accepted in the app. Progress is printed as JSON lines.

    python training/finetune.py --out models/candidate
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chessvision.classifier import OnnxClassifier  # noqa: E402
from chessvision.position import fen_to_board, repair  # noqa: E402
from chessvision.squares import crop_squares  # noqa: E402

MIN_PHOTOS = 10
FEEDBACK_SHARE = 0.2  # share of the saved photos in every batch


def load_rows(feedback_dir: Path) -> list[dict]:
    labels = feedback_dir / "labels.jsonl"
    if not labels.exists():
        return []
    rows = [json.loads(line) for line in labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if r.get("corners")]


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Every fifth photo is held out for the comparison."""
    holdout = rows[4::5]
    return [r for i, r in enumerate(rows) if i % 5 != 4], holdout


def square_accuracy(clf: OnnxClassifier, folder: Path, rows: list[dict]) -> float:
    """Squares correct after repair(), with the saved (correct) corners."""
    correct = total = 0
    for row in rows:
        img = cv2.imread(str(folder / row["image"]))
        if img is None:
            continue
        crops = crop_squares(img, np.array(row["corners"], np.float32))  # corners are a8, h8, h1, a1
        board, _ = repair(clf.predict_proba(crops.reshape(64, *crops.shape[2:])).reshape(8, 8, -1))
        truth = fen_to_board(row["fen"])
        correct += sum(board[r][c] == truth[r][c] for r in range(8) for c in range(8))
        total += 64
    return correct / max(total, 1)


def report(event: str, **data) -> None:
    print(json.dumps({"event": event, **data}), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback", type=Path, default=ROOT / "data" / "feedback")
    ap.add_argument("--npz", type=Path, default=ROOT / "data" / "npz", help="crops of the original training data")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "candidate")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps-per-epoch", type=int, default=400)
    args = ap.parse_args()

    rows = load_rows(args.feedback)
    if len(rows) < MIN_PHOTOS:
        raise SystemExit(f"At least {MIN_PHOTOS} saved photos are needed (now {len(rows)}).")
    base = sorted(p for p in args.npz.glob("*.npz") if p.stem != "feedback")
    if not base:
        raise SystemExit(f"No original training data in {args.npz} (see README, Training).")
    train_rows, holdout = split_rows(rows)

    from make_dataset import build
    from train_cnn import mean_accuracy, train

    report("step", step="crops")
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "feedback"
    tmp.mkdir(exist_ok=True)
    (tmp / "labels.jsonl").write_text(
        "".join(json.dumps({**r, "image": str((args.feedback / r["image"]).resolve())}) + "\n" for r in train_rows), encoding="utf-8")
    feedback_npz = args.out / "feedback.npz"
    np.savez(feedback_npz, **build([tmp], jitter=0.06, seed=0, limit=0, repeat=8))

    report("step", step="train")
    weights = [(1 - FEEDBACK_SHARE) / len(base)] * len(base) + [FEEDBACK_SHARE]
    accs = train(base + [feedback_npz], args.out, weights, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, lr=5e-4,
                 init=args.models / "square_cnn.pt", on_epoch=lambda e, n: report("epoch", epoch=e, epochs=n))
    shutil.rmtree(tmp)
    feedback_npz.unlink()

    report("step", step="evaluate")
    old, new = OnnxClassifier(args.models / "square_cnn.onnx"), OnnxClassifier(args.out / "square_cnn.onnx")
    others = [p.stem for p in base]
    result = {
        "photos": len(rows),
        "held_out": len(holdout),
        "own": {"before": square_accuracy(old, args.feedback, holdout), "after": square_accuracy(new, args.feedback, holdout)},
        "others": {"before": mean_accuracy({k: accs["before"][k] for k in others}),
                   "after": mean_accuracy({k: accs["after"][k] for k in others})},
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report("done", result=result)


if __name__ == "__main__":
    main()
