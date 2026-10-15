#!/usr/bin/env bash
# Adds "ChessLens" to the Linux application menu (--remove removes it).
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/applications/chess-vision.desktop"

if [ "${1:-}" = "--remove" ]; then
    rm -f "$DEST" && echo "Removed: $DEST"
    exit 0
fi

chmod +x "$ROOT/scripts/start_app.sh" "$ROOT/scripts/stop_app.sh"
mkdir -p "$(dirname "$DEST")"
cat >"$DEST" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=ChessLens
Comment=Recognize a chess position from a photo and show the best move
Comment[sr]=Prepoznaj šahovsku poziciju sa fotografije i vidi najbolji potez
Exec="$ROOT/scripts/start_app.sh"
Icon=$ROOT/scripts/icon.svg
Terminal=false
Categories=Game;BoardGame;
Keywords=chess;šah;FEN;stockfish;
StartupNotify=false
Actions=stop;

[Desktop Action stop]
Name=Stop server
Name[sr]=Zaustavi server
Exec="$ROOT/scripts/stop_app.sh"
DESKTOP
chmod +x "$DEST"
command -v update-desktop-database >/dev/null && update-desktop-database "$(dirname "$DEST")" 2>/dev/null || true
echo "Added to the application menu: $DEST"
