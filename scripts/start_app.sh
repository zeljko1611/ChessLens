#!/usr/bin/env bash
# Starts ChessLens from the application menu: builds the frontend if needed, starts the
# server in the background (it stops when idle) and opens the browser.
set -uo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
PORT="${CHESSVISION_PORT:-8765}"
URL="http://127.0.0.1:$PORT"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/chess-vision"
mkdir -p "$STATE"

say() {
    echo "$1"
    command -v notify-send >/dev/null && notify-send -a "ChessLens" -i "$ROOT/scripts/icon.svg" "ChessLens" "$1" || true
}
running() { curl -fs --max-time 2 "$URL/api/health" >/dev/null 2>&1; }

cd "$ROOT" || exit 1

if ! running; then
    # Menu launchers do not see the terminal PATH, so load nvm (Node) manually.
    if ! command -v npm >/dev/null && [ -s "$HOME/.nvm/nvm.sh" ]; then
        # shellcheck disable=SC1091
        . "$HOME/.nvm/nvm.sh" >/dev/null
    fi
    if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html -newer frontend/dist/index.html -print -quit)" ]; then
        if command -v npm >/dev/null; then
            say "Preparing the interface…"
            if ! (cd frontend && npm run build) >"$STATE/build.log" 2>&1; then
                say "Building the interface failed. Details: $STATE/build.log"
                exit 1
            fi
        elif [ ! -f frontend/dist/index.html ]; then
            say "npm (Node.js) was not found and the interface has not been built yet."
            exit 1
        fi
    fi

    say "Starting the server…"
    CHESSVISION_IDLE_TIMEOUT="${CHESSVISION_IDLE_TIMEOUT:-300}" CHESSVISION_PHONE_PORT="$PORT" \
        nohup "$ROOT/.venv/bin/python" -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT" >"$STATE/server.log" 2>&1 &
    echo $! >"$STATE/server.pid"
    for _ in $(seq 1 120); do
        running && break
        if ! kill -0 "$(cat "$STATE/server.pid")" 2>/dev/null; then
            say "The server did not start. Details: $STATE/server.log"
            exit 1
        fi
        sleep 0.5
    done
    running || { say "The server is not responding. Details: $STATE/server.log"; exit 1; }
    PHONE_URL="$(curl -fs --max-time 2 "$URL/api/health" | sed -n 's/.*"phone_url":"\([^"]*\)".*/\1/p')"
    [ -n "$PHONE_URL" ] && say "Running. On your phone (same Wi-Fi): $PHONE_URL"
fi

xdg-open "$URL" >/dev/null 2>&1 &
