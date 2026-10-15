# ChessLens

[![CI](https://github.com/zeljko1611/ChessLens/actions/workflows/ci.yml/badge.svg)](https://github.com/zeljko1611/ChessLens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ChessLens recognizes a chess position from a photo of a physical board, converts it to FEN and shows the best move (Stockfish). A misrecognized square can be fixed with one click, and if the detection misses the board, its four corners can be dragged into place. You can play on from the recognized position on the same board, and one button restores the position from the photo. The interface is available in English (default) and Serbian.

![ChessLens](docs/screenshot-desktop.png)

**Stack:** Python (OpenCV, NumPy, PyTorch → ONNX), FastAPI, React + TypeScript (Vite), Stockfish in the browser.

## Highlights

- Board detection without machine learning: saddle points (X-corners) fitted with RANSAC into the 8×8 grid, robust to pieces covering the board edge.
- A small CNN (0.84M parameters) trained in PyTorch and served with onnxruntime: 99.3 % of squares and 91.8 % of complete boards correct on real photos.
- The model's probabilities are turned into the most likely legal position (one king per side, no pawns on the back ranks, ...).
- Works from a phone on the same Wi-Fi network (QR code), and a photo taken on the phone appears on the computer.
- The model can be fine-tuned on your own board from the app, with a before/after comparison and a way back to the previous model.

## How it works

![Pipeline](docs/pipeline.jpg)

1. **Board detection** (`chessvision/board.py`, no ML): find the points where four squares meet (saddle points), then RANSAC grows a single-square hypothesis into the full 8×8 grid. Board contours alone are not enough because pieces on the back rank cover the edge.
2. **Square crops** (`squares.py`): an upright 32×64 crop is cut from the photo for every square, tall enough to contain the whole piece.
3. **Classification** (`classifier.py`): a small CNN (0.84M parameters) predicts one of 13 classes (empty or one of 12 pieces). It is trained in PyTorch and runs on the server with onnxruntime.
4. **Orientation** (`squares.py`): h1 is a light square, and White is usually on the near side. When the rule is wrong, one button flips the board.
5. **Legal position** (`position.py`): the model's probabilities are turned into the most likely legal board (exactly one king per side, no pawns on the back ranks, ...).
6. **Analysis and play:** Stockfish runs in the browser (Web Worker), and [chess.js](https://github.com/jhlywa/chess.js) checks moves in game mode.

## Results

The whole system (detection + model + orientation), measured with `training/evaluate.py` on images not used for training:

| Test set | Square accuracy | Fully correct boards |
|---|---|---|
| Real photos, [samryan18](https://github.com/samryan18/chess-dataset) (98) | 99.3 % | 91.8 % |
| [chesscog](https://osf.io/xf3ka/) renders (342) | 99.5 % | 81.0 % |
| Synthetic images (300) | 98.5 % | 81.3 % |

Training data: chesscog (4,400 renders), 379 real photos from samryan18 and 1,500 synthetic images; 25 epochs on a GTX 1650. The previous model (HOG + MLP) recognized 65 % of the squares and no complete board on the chesscog test set.

**Limitation:** the real photos show a single board shot from above, and chesscog renders a single piece set. Accuracy is noticeably lower on other piece sets, at steep angles or with a diagonally rotated board. That is why the app has a *Save for training* button: corrected photos are stored in `data/feedback/`, and once there are at least 10, *Train on my photos* fine-tunes the model in a few minutes and shows the accuracy of the current and the new model before you switch.

## Challenges

- **Board contours were not enough.** Pieces on the back rank break the outline of the board, so detection moved to the X-corners inside it.
- **The first model did not generalize.** A HOG + MLP baseline recognized 65 % of the squares on chesscog renders; the CNN with heavy augmentation reaches 99.5 % on the same set.
- **The grid was shifted by one row at steep angles,** because border lines often have no X-corners. Scoring each side of the board separately (checkerboard pattern inside, none outside) raised detection on chesscog renders from 83 % to 91 %.
- **Black pieces looked white in Samsung Internet's dark mode,** which recolours text. Pieces are now drawn as SVG paths instead of Unicode glyphs.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000

cd frontend && npm install && npm run dev    # http://localhost:5173
```

Without a terminal (Linux): `scripts/install_desktop.sh` adds *ChessLens* to the application menu. Clicking it starts the server and opens the browser, and the server stops by itself a few minutes after the app is closed. The server is also reachable from phones on the same Wi-Fi network: the *On phone* button shows a QR code with the address. A photo taken on the phone appears on the computer within a couple of seconds.

Docker: `docker build -t chesslens . && docker run -p 8000:8000 chesslens`

## Training

```bash
pip install -r requirements-train.txt

# data: synthetic images and chesscog (https://osf.io/xf3ka/)
python data/synth.py --out data/generated/synth_train --count 1500 --seed 2024
python data/chesscog.py --src ~/datasets/chesscog/train --out data/chesscog_train

# square crops, training, ONNX export
python training/make_dataset.py --data data/chesscog_train --out data/npz/chesscog.npz
python training/make_dataset.py --data data/generated/synth_train --out data/npz/synth.npz
python training/train_cnn.py --train data/npz/chesscog.npz data/npz/synth.npz --epochs 25 --out models/

python training/evaluate.py --data data/chesscog_test --models models/
```

Fine-tuning on your own board: `python training/finetune.py` mixes the photos saved in the app (at least 10) with the original crops in `data/npz/`, holds out every fifth photo and writes the new model with a before/after comparison to `models/candidate/`. The *Train on my photos* button runs the same script (only where PyTorch and `data/npz/` are installed).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Project structure

```
chessvision/   board detection, square crops, model, chess rules, full pipeline (pipeline.py)
backend/       FastAPI server
frontend/      React app
data/          synthetic image generator and dataset converters
training/      crop dataset builder, CNN training, evaluation
models/        trained model (square_cnn.onnx, square_cnn.pt)
scripts/       launcher without a terminal (Linux)
tests/
```

## Credits

- Piece shapes come from [Noto Sans Symbols 2](https://fonts.google.com/noto/specimen/Noto+Sans+Symbols+2) (SIL Open Font License 1.1).
- Rendered training and test images: [chesscog](https://osf.io/xf3ka/) by Wölflein and Arandjelović (CC BY 4.0).
- Real photos: [samryan18/chess-dataset](https://github.com/samryan18/chess-dataset).
- Analysis: [Stockfish](https://stockfishchess.org/) (stockfish.js), game rules in the browser: [chess.js](https://github.com/jhlywa/chess.js).

## License

MIT, see [LICENSE](LICENSE).
