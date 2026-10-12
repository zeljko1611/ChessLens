"""Split a folder (labels.jsonl + images/) into train and test parts.

The same position (e.g. two photos of the same board) always goes to the same
part, so the test set never contains images the model has practically seen.

    python data/split.py --src data/real/samryan18 --test-frac 0.2
    # -> data/real/samryan18_train, data/real/samryan18_test
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", default="0")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.src / "labels.jsonl", encoding="utf-8") if line.strip()]
    out = {}
    for part in ("train", "test"):
        d = args.src.parent / f"{args.src.name}_{part}"
        (d / "images").mkdir(parents=True, exist_ok=True)
        out[part] = (d, open(d / "labels.jsonl", "w", encoding="utf-8"))
    counts = {"train": 0, "test": 0}
    for r in rows:
        # Deterministic split by position (hash of the FEN), independent of order.
        h = int(hashlib.sha1((args.seed + r["fen"]).encode()).hexdigest(), 16) % 10_000
        part = "test" if h < args.test_frac * 10_000 else "train"
        d, f = out[part]
        link = d / r["image"]
        if not link.exists():
            link.symlink_to((args.src / r["image"]).resolve())
        f.write(json.dumps(r) + "\n")
        counts[part] += 1
    for _, f in out.values():
        f.close()
    print(f"train: {counts['train']} images, test: {counts['test']} images")


if __name__ == "__main__":
    main()
