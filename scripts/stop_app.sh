#!/usr/bin/env bash
# Stops the server started by the launcher (scripts/start_app.sh).
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/chess-vision"
PID_FILE="$STATE/server.pid"
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill -INT "$(cat "$PID_FILE")" && echo "The server has been stopped."
    command -v notify-send >/dev/null && notify-send -a "ChessLens" "ChessLens" "The server has been stopped." || true
else
    echo "The server is not running."
fi
rm -f "$PID_FILE"
