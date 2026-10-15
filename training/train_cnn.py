"""Train the square classifier CNN in PyTorch and export it to ONNX.

    python training/train_cnn.py --train data/npz/a.npz data/npz/b.npz --weights 0.5 0.5 --out models/
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chessvision.classifier import MEAN, STD  # noqa: E402
from chessvision.position import CLASSES  # noqa: E402

NUM_CLASSES = len(CLASSES)


def split_by_group(group: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split by image, not by crop (otherwise validation would be too easy)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(group)
    rng.shuffle(ids)
    is_val = np.isin(group, ids[: int(len(ids) * val_frac)])
    return np.nonzero(~is_val)[0], np.nonzero(is_val)[0]


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SquareNet(nn.Module):
    """A small VGG-like network (~0.84M parameters), fast on a CPU and in a browser."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32), nn.MaxPool2d(2),  # 32x16
            conv_block(32, 64), nn.MaxPool2d(2),  # 16x8
            conv_block(64, 128), nn.MaxPool2d(2),  # 8x4
            conv_block(128, 192),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(192, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def to_tensor(batch_bgr: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    """uint8 (B, H, W, 3) BGR -> float (B, 3, H, W) RGB in [0, 1]."""
    if isinstance(batch_bgr, np.ndarray):
        batch_bgr = torch.from_numpy(np.ascontiguousarray(batch_bgr))
    x = batch_bgr.to(device).flip(-1)  # BGR -> RGB
    return x.permute(0, 3, 1, 2).float() / 255.0


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x: torch.Tensor) -> torch.Tensor:
    """Augmentation of a whole batch (on the GPU if available)."""
    b = x.shape[0]
    dev = x.device
    # Horizontal flip: pieces are (nearly) symmetric, a knight stays a knight.
    flip = torch.rand(b, device=dev) < 0.5
    x = torch.where(flip.view(b, 1, 1, 1), x.flip(-1), x)
    # Small shift and scale (simulates imprecise corner detection).
    scale = torch.empty(b, device=dev).uniform_(0.92, 1.08)
    theta = torch.zeros(b, 2, 3, device=dev)
    theta[:, 0, 0] = scale
    theta[:, 1, 1] = scale
    theta[:, 0, 2] = torch.empty(b, device=dev).uniform_(-0.1, 0.1)
    theta[:, 1, 2] = torch.empty(b, device=dev).uniform_(-0.06, 0.06)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    # Brightness, contrast, colour, noise.
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0.6, 1.4) + mean
    x = x * torch.empty(b, 1, 1, 1, device=dev).uniform_(0.65, 1.35)
    x = x * torch.empty(b, 3, 1, 1, device=dev).uniform_(0.8, 1.2)  # warm/cold light, board colour
    x = x + torch.empty(b, 3, 1, 1, device=dev).uniform_(-0.06, 0.06)
    # Greyscale (the board colour must not be decisive).
    gray = x.mean(dim=1, keepdim=True).expand_as(x)
    x = torch.where((torch.rand(b, device=dev) < 0.2).view(b, 1, 1, 1), gray, x)
    # Blur (focus, motion): a 3x3 box filter on part of the batch.
    blurred = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    x = torch.where((torch.rand(b, device=dev) < 0.3).view(b, 1, 1, 1), blurred, x)
    # Occlude the top of the crop (a piece on the square behind, a hand) with a grey box.
    occ = torch.rand(b, device=dev) < 0.15
    h, w = x.shape[2], x.shape[3]
    ys = torch.arange(h, device=dev).view(1, 1, h, 1)
    xs = torch.arange(w, device=dev).view(1, 1, 1, w)
    y1 = (torch.rand(b, device=dev) * 0.35 * h).view(b, 1, 1, 1)
    x0 = (torch.rand(b, device=dev) * w).view(b, 1, 1, 1)
    x1 = x0 + (0.3 + 0.5 * torch.rand(b, device=dev)).view(b, 1, 1, 1) * w
    box = (ys < y1) & (xs >= x0) & (xs < x1) & occ.view(b, 1, 1, 1)
    x = torch.where(box, torch.rand(b, 3, 1, 1, device=dev), x)
    x = x + torch.randn_like(x) * torch.empty(b, 1, 1, 1, device=dev).uniform_(0, 0.05)
    return x.clamp(0, 1)


@torch.no_grad()
def evaluate(model: nn.Module, X, y, device: torch.device, bs: int = 512) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(X), bs):
        logits = model(normalize(to_tensor(X[i : i + bs], device)))
        correct += int((logits.argmax(1).cpu() == torch.as_tensor(y[i : i + bs])).sum())
    return correct / max(len(X), 1)


def export_onnx(model: nn.Module, path: Path, X_check: np.ndarray) -> None:
    model = model.cpu().eval()
    dummy = torch.zeros(1, 3, X_check.shape[1], X_check.shape[2])
    kwargs = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)  # newer PyTorch versions
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kwargs)

    # Check: ONNX and PyTorch must give the same probabilities.
    from chessvision.classifier import OnnxClassifier, softmax

    onnx_p = OnnxClassifier(path).predict_proba(X_check)
    with torch.no_grad():
        torch_p = softmax(model(normalize(to_tensor(X_check, torch.device("cpu")))).numpy())
    diff = float(np.abs(onnx_p - torch_p).max())
    print(f"ONNX exported to {path} (max difference from PyTorch: {diff:.2e})")
    if diff > 1e-3:
        raise RuntimeError("The ONNX model does not match the PyTorch model!")


def load_sources(paths: list[Path], val_frac: float, device: torch.device) -> list[dict]:
    """Load each .npz, split it by image and move it to the device as uint8 right away.

    This way RAM never holds more than one file, and crops (64x32x3 bytes) for
    several hundred thousand squares fit in GPU memory.
    """
    sources = []
    for path in paths:
        d = np.load(path)
        X, y, group = d["X"], d["y"], d["group"]
        tr, va = split_by_group(group, val_frac, seed=0)
        src = {"name": path.stem, "X_tr": torch.from_numpy(X[tr]).to(device), "y_tr": torch.from_numpy(y[tr]).to(device),
               "X_va": torch.from_numpy(X[va]), "y_va": torch.from_numpy(y[va])}
        del X, y, group, d
        print(f"  {path.stem}: {len(tr)} train, {len(va)} validation", flush=True)
        sources.append(src)
    return sources


def mean_accuracy(accs: dict[str, float | None]) -> float:
    """Every source with a validation split counts equally."""
    vals = [v for v in accs.values() if v is not None]
    return float(np.mean(vals)) if vals else 0.0


def validate_sources(model: nn.Module, sources: list[dict], device: torch.device) -> dict[str, float | None]:
    return {s["name"]: evaluate(model, s["X_va"], s["y_va"], device) if len(s["y_va"]) else None for s in sources}


def train(paths: list[Path], out: Path, weights: list[float] | None = None, epochs: int = 15, steps_per_epoch: int = 0,
          batch_size: int = 256, lr: float = 2e-3, val_frac: float = 0.15, init: Path | None = None,
          on_epoch=None) -> dict:
    """Train, keep the best epoch in out/square_cnn.pt and export it to ONNX.

    Returns the validation accuracy per source before (only with `init`) and after training.
    """
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    sources = load_sources(paths, val_frac, device)
    sizes = np.array([len(s["y_tr"]) for s in sources], np.float64)
    w = np.array(weights, np.float64) if weights else sizes
    if len(w) != len(sources):
        raise SystemExit("--weights needs exactly one value per --train file")
    w = w / w.sum()
    print("Batch share: " + ", ".join(f"{s['name']} {x:.2f}" for s, x in zip(sources, w)))

    model = SquareNet().to(device)
    before = None
    if init:
        model.load_state_dict(torch.load(init, map_location=device))
        before = validate_sources(model, sources, device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    steps_per_epoch = steps_per_epoch or math.ceil(sizes.sum() / batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    out.mkdir(parents=True, exist_ok=True)
    best_acc, best_accs, best_path = -1.0, {}, out / "square_cnn.pt"
    gen = torch.Generator(device=device).manual_seed(0)
    for epoch in range(epochs):
        model.train()
        t0, total = time.time(), 0.0
        for _ in range(steps_per_epoch):
            # Every batch mixes the sources in the given proportions.
            counts = np.random.multinomial(batch_size, w)
            xs, ts = [], []
            for src, k in zip(sources, counts):
                if k:
                    idx = torch.randint(len(src["y_tr"]), (int(k),), device=device, generator=gen)
                    xs.append(src["X_tr"][idx])
                    ts.append(src["y_tr"][idx])
            x = normalize(augment(to_tensor(torch.cat(xs), device)))
            loss = loss_fn(model(x), torch.cat(ts))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        accs = validate_sources(model, sources, device)
        acc = mean_accuracy(accs)
        detail = "  ".join(f"{k} {v * 100:.2f}%" for k, v in accs.items() if v is not None)
        print(f"epoch {epoch + 1}/{epochs}  loss {total / steps_per_epoch:.4f}  val {acc * 100:.2f}% [{detail}]  ({time.time() - t0:.0f} s)",
              flush=True)
        if acc > best_acc:
            best_acc, best_accs = acc, accs
            torch.save(model.state_dict(), best_path)
        if on_epoch:
            on_epoch(epoch + 1, epochs)

    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"Best mean square accuracy (validation): {best_acc * 100:.2f}%")
    first = sources[0]
    export_onnx(model, out / "square_cnn.onnx", (first["X_va"] if len(first["y_va"]) else first["X_tr"])[:64].cpu().numpy())
    return {"before": before, "after": best_accs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="one or more .npz files (make_dataset.py)")
    ap.add_argument("--weights", type=float, nargs="*", help="share of each file in a batch (default: proportional to size)")
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--steps-per-epoch", type=int, default=0, help="0 = dataset size / batch size")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--init", type=Path, help="start from existing weights (fine-tuning on real photos)")
    args = ap.parse_args()
    train(args.train, args.out, args.weights, args.epochs, args.steps_per_epoch, args.batch_size, args.lr, args.val_frac, args.init)


if __name__ == "__main__":
    main()
