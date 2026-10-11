"""Position representation, FEN, legality checks and probability-based repair.

A board is an 8x8 list of strings in board coordinates:
board[0][0] = a8, board[0][7] = h8, board[7][0] = a1, board[7][7] = h1.
An empty square is "."; pieces use FEN letters (uppercase = White).

Deliberately without python-chess, so the backend stays light and the logic
is fully under our control (and easy to test).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Class order shared by the model and the rest of the pipeline.
CLASSES = [".", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
EMPTY = "."

FILES = "abcdefgh"
STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

Board = list[list[str]]


def square_name(r: int, c: int) -> str:
    return f"{FILES[c]}{8 - r}"


def parse_square(name: str) -> tuple[int, int]:
    return 8 - int(name[1]), FILES.index(name[0])


def empty_board() -> Board:
    return [[EMPTY] * 8 for _ in range(8)]


def board_to_fen(board: Board) -> str:
    rows = []
    for r in range(8):
        out, run = "", 0
        for c in range(8):
            p = board[r][c]
            if p == EMPTY:
                run += 1
            else:
                if run:
                    out += str(run)
                    run = 0
                out += p
        if run:
            out += str(run)
        rows.append(out)
    return "/".join(rows)


def fen_to_board(fen: str) -> Board:
    placement = fen.split()[0]
    rows = placement.split("/")
    if len(rows) != 8:
        raise ValueError(f"FEN must have 8 ranks: {fen!r}")
    board = []
    for row in rows:
        out: list[str] = []
        for ch in row:
            if ch.isdigit():
                out.extend([EMPTY] * int(ch))
            elif ch in CLASS_INDEX and ch != EMPTY:
                out.append(ch)
            else:
                raise ValueError(f"Unknown character {ch!r} in FEN")
        if len(out) != 8:
            raise ValueError(f"Rank {row!r} does not have 8 squares")
        board.append(out)
    return board


# --------------------------------------------------------------------------
# Attacks (needed to know who is in check)
# --------------------------------------------------------------------------

_KNIGHT = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
_KING = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _inside(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def is_attacked(board: Board, r: int, c: int, by_white: bool) -> bool:
    """Whether the side `by_white` attacks square (r, c)."""
    pawn, knight, king = ("P", "N", "K") if by_white else ("p", "n", "k")
    rook_like = {"R", "Q"} if by_white else {"r", "q"}
    bishop_like = {"B", "Q"} if by_white else {"b", "q"}

    # A white pawn attacks "upwards" (towards smaller r), so look for it below the square.
    pr = r + 1 if by_white else r - 1
    for dc in (-1, 1):
        if _inside(pr, c + dc) and board[pr][c + dc] == pawn:
            return True
    for dr, dc in _KNIGHT:
        if _inside(r + dr, c + dc) and board[r + dr][c + dc] == knight:
            return True
    for dr, dc in _KING:
        if _inside(r + dr, c + dc) and board[r + dr][c + dc] == king:
            return True
    for dirs, attackers in ((_ROOK_DIRS, rook_like), (_BISHOP_DIRS, bishop_like)):
        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            while _inside(rr, cc):
                p = board[rr][cc]
                if p != EMPTY:
                    if p in attackers:
                        return True
                    break
                rr += dr
                cc += dc
    return False


def find_piece(board: Board, piece: str) -> list[tuple[int, int]]:
    return [(r, c) for r in range(8) for c in range(8) if board[r][c] == piece]


def in_check(board: Board, white: bool) -> bool:
    kings = find_piece(board, "K" if white else "k")
    if len(kings) != 1:
        return False
    r, c = kings[0]
    return is_attacked(board, r, c, by_white=not white)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@dataclass
class Validation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(board: Board, side_to_move: str | None = None) -> Validation:
    """Check whether a position is possible. Errors mean it is certainly illegal."""
    v = Validation()
    counts = {p: 0 for p in CLASSES}
    for row in board:
        for p in row:
            counts[p] += 1

    for king, name in (("K", "white"), ("k", "black")):
        if counts[king] != 1:
            v.errors.append(f"There must be exactly one {name} king (found {counts[king]}).")

    for c in range(8):
        for r in (0, 7):
            if board[r][c] in ("P", "p"):
                v.errors.append(f"A pawn cannot stand on {square_name(r, c)}.")

    for color, pieces, pawn in (("White", "PNBRQK", "P"), ("Black", "pnbrqk", "p")):
        if counts[pawn] > 8:
            v.errors.append(f"{color} has more than 8 pawns.")
        total = sum(counts[p] for p in pieces)
        if total > 16:
            v.errors.append(f"{color} has more than 16 pieces.")
        # Extra pieces must come from promotion, and every promotion uses up a pawn.
        extra = (
            max(0, counts[pieces[1]] - 2)
            + max(0, counts[pieces[2]] - 2)
            + max(0, counts[pieces[3]] - 2)
            + max(0, counts[pieces[4]] - 1)
        )
        if counts[pawn] + extra > 8:
            v.errors.append(f"{color} has too many promoted pieces for its number of pawns.")

    kings_w, kings_b = find_piece(board, "K"), find_piece(board, "k")
    if len(kings_w) == 1 and len(kings_b) == 1:
        (r1, c1), (r2, c2) = kings_w[0], kings_b[0]
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            v.errors.append("The kings cannot stand on adjacent squares.")
        w_check, b_check = in_check(board, True), in_check(board, False)
        if w_check and b_check:
            v.errors.append("Both kings are in check.")
        if side_to_move == "w" and b_check:
            v.errors.append("Black is in check but it is White's move.")
        if side_to_move == "b" and w_check:
            v.errors.append("White is in check but it is Black's move.")
    return v


def suggest_side_to_move(board: Board) -> str | None:
    """If one side is in check, it must be that side's move."""
    if in_check(board, True):
        return "w"
    if in_check(board, False):
        return "b"
    return None


def infer_castling(board: Board) -> str:
    """Castling is possible only if the king and rook are on their starting squares.

    A photo cannot tell whether they have moved, so the most permissive set is
    returned (the user can change it in the UI).
    """
    out = ""
    if board[7][4] == "K":
        if board[7][7] == "R":
            out += "K"
        if board[7][0] == "R":
            out += "Q"
    if board[0][4] == "k":
        if board[0][7] == "r":
            out += "k"
        if board[0][0] == "r":
            out += "q"
    return out or "-"


def full_fen(board: Board, side_to_move: str = "w", castling: str | None = None) -> str:
    if castling is None:
        castling = infer_castling(board)
    return f"{board_to_fen(board)} {side_to_move} {castling} - 0 1"


# --------------------------------------------------------------------------
# Repair based on model probabilities
# --------------------------------------------------------------------------


def _ranked(probs_row: np.ndarray) -> list[int]:
    return list(np.argsort(-probs_row))


def repair(board_probs: np.ndarray, max_changes: int = 6) -> tuple[Board, list[dict]]:
    """Build the most likely *legal* board from (8, 8, 13) probabilities.

    Greedy: start from the argmax and, while a rule is broken, change the square
    whose change "costs" the least (the smallest drop in log-probability).
    Returns the board and a list of changes {square, from, to, reason}, where
    reason is a code (pawn_back_rank, extra_king, missing_king, too_many_pawns)
    translated by the frontend.
    """
    probs = np.clip(board_probs.astype(np.float64), 1e-9, 1.0)
    logp = np.log(probs)
    board: Board = [[CLASSES[int(np.argmax(probs[r, c]))] for c in range(8)] for r in range(8)]
    changes: list[dict] = []

    def set_sq(r: int, c: int, new: str, why: str) -> None:
        old = board[r][c]
        board[r][c] = new
        changes.append({"square": square_name(r, c), "from": old, "to": new, "reason": why})

    def best_alternative(r: int, c: int, forbidden: set[str]) -> tuple[str, float]:
        cur = logp[r, c, CLASS_INDEX[board[r][c]]]
        for idx in _ranked(probs[r, c]):
            cand = CLASSES[idx]
            if cand not in forbidden and cand != board[r][c]:
                return cand, cur - logp[r, c, idx]
        return EMPTY, float("inf")

    for _ in range(max_changes * 4):
        if len(changes) >= max_changes:
            break
        fixed = False

        # 1) Pawns on the first or eighth rank.
        for r in (0, 7):
            for c in range(8):
                if board[r][c] in ("P", "p"):
                    alt, _ = best_alternative(r, c, {"P", "p"})
                    set_sq(r, c, alt, "pawn_back_rank")
                    fixed = True
        if fixed:
            continue

        # 2) Exactly one king per colour.
        for king in ("K", "k"):
            squares = find_piece(board, king)
            if len(squares) > 1:
                # Keep the most confident one and replace the rest with their best alternative.
                squares.sort(key=lambda rc: -logp[rc[0], rc[1], CLASS_INDEX[king]])
                for r, c in squares[1:]:
                    alt, _ = best_alternative(r, c, {king})
                    set_sq(r, c, alt, "extra_king")
                fixed = True
            elif not squares:
                ki = CLASS_INDEX[king]
                cost = np.full((8, 8), np.inf)
                for r in range(8):
                    for c in range(8):
                        if board[r][c] in ("K", "k"):
                            continue
                        cost[r, c] = logp[r, c, CLASS_INDEX[board[r][c]]] - logp[r, c, ki]
                r, c = np.unravel_index(int(np.argmin(cost)), cost.shape)
                set_sq(int(r), int(c), king, "missing_king")
                fixed = True
        if fixed:
            continue

        # 3) Too many pawns of one colour: remove the least confident.
        for pawn in ("P", "p"):
            squares = find_piece(board, pawn)
            if len(squares) > 8:
                squares.sort(key=lambda rc: logp[rc[0], rc[1], CLASS_INDEX[pawn]])
                r, c = squares[0]
                alt, _ = best_alternative(r, c, {pawn})
                set_sq(r, c, alt, "too_many_pawns")
                fixed = True
        if not fixed:
            break

    return board, changes


def uncertain_squares(board_probs: np.ndarray, threshold: float = 0.6) -> list[str]:
    conf = board_probs.max(axis=-1)
    return [square_name(r, c) for r in range(8) for c in range(8) if conf[r, c] < threshold]
