// Chess helpers for the UI (FEN, legality, moves); the rules match chessvision/position.py.
import { tr } from "./i18n";

export type Piece = "P" | "N" | "B" | "R" | "Q" | "K" | "p" | "n" | "b" | "r" | "q" | "k";
export type Cell = Piece | ".";
export type Board = Cell[][]; // [0][0] = a8, [7][7] = h1
export type Side = "w" | "b";

export const FILES = "abcdefgh";
export const PIECES: Piece[] = ["K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"];

/** Piece name in the current language ("white king" / "beli kralj"). */
export const pieceName = (p: Piece) => tr(`piece.${p}`);

// Filled glyphs for both colours; the colour comes from CSS (fill/stroke).
// U+FE0E requests text presentation instead of emoji (♟ is an emoji on phones).
export const GLYPH: Record<string, string> = { k: "♚\uFE0E", q: "♛\uFE0E", r: "♜\uFE0E", b: "♝\uFE0E", n: "♞\uFE0E", p: "♟\uFE0E" };

export const squareName = (r: number, c: number) => `${FILES[c]}${8 - r}`;
export const parseSquare = (s: string): [number, number] => [8 - Number(s[1]), FILES.indexOf(s[0])];
export const isWhite = (p: Cell) => p !== "." && p === p.toUpperCase();

export function boardToPlacement(board: Board): string {
  return board
    .map((row) => {
      let out = "";
      let run = 0;
      for (const p of row) {
        if (p === ".") run++;
        else {
          if (run) out += run;
          run = 0;
          out += p;
        }
      }
      return run ? out + run : out;
    })
    .join("/");
}

export function placementToBoard(placement: string): Board | null {
  const rows = placement.trim().split(" ")[0].split("/");
  if (rows.length !== 8) return null;
  const board: Board = [];
  for (const row of rows) {
    const out: Cell[] = [];
    for (const ch of row) {
      if (/[1-8]/.test(ch)) out.push(...Array<Cell>(Number(ch)).fill("."));
      else if ("PNBRQKpnbrqk".includes(ch)) out.push(ch as Piece);
      else return null;
    }
    if (out.length !== 8) return null;
    board.push(out);
  }
  return board;
}

export function rotate180(board: Board): Board {
  return board.map((row) => [...row].reverse()).reverse();
}

export interface Castling { K: boolean; Q: boolean; k: boolean; q: boolean }

export function possibleCastling(board: Board): Castling {
  return {
    K: board[7][4] === "K" && board[7][7] === "R",
    Q: board[7][4] === "K" && board[7][0] === "R",
    k: board[0][4] === "k" && board[0][7] === "r",
    q: board[0][4] === "k" && board[0][0] === "r",
  };
}

export function castlingString(c: Castling, board: Board): string {
  const allowed = possibleCastling(board);
  const s = (["K", "Q", "k", "q"] as const).filter((k) => c[k] && allowed[k]).join("");
  return s || "-";
}

export function toFen(board: Board, side: Side, castling: Castling): string {
  return `${boardToPlacement(board)} ${side} ${castlingString(castling, board)} - 0 1`;
}

const inside = (r: number, c: number) => r >= 0 && r < 8 && c >= 0 && c < 8;

function attacked(board: Board, r: number, c: number, byWhite: boolean): boolean {
  const [P, N, B, R, Q, K] = byWhite ? ["P", "N", "B", "R", "Q", "K"] : ["p", "n", "b", "r", "q", "k"];
  const pr = byWhite ? r + 1 : r - 1;
  for (const dc of [-1, 1]) if (inside(pr, c + dc) && board[pr][c + dc] === P) return true;
  for (const [dr, dc] of [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]])
    if (inside(r + dr, c + dc) && board[r + dr][c + dc] === N) return true;
  for (let dr = -1; dr <= 1; dr++)
    for (let dc = -1; dc <= 1; dc++)
      if ((dr || dc) && inside(r + dr, c + dc) && board[r + dr][c + dc] === K) return true;
  const rays: [number, number, string[]][] = [
    [-1, 0, [R, Q]], [1, 0, [R, Q]], [0, -1, [R, Q]], [0, 1, [R, Q]],
    [-1, -1, [B, Q]], [-1, 1, [B, Q]], [1, -1, [B, Q]], [1, 1, [B, Q]],
  ];
  for (const [dr, dc, att] of rays) {
    let rr = r + dr, cc = c + dc;
    while (inside(rr, cc)) {
      const p = board[rr][cc];
      if (p !== ".") {
        if (att.includes(p)) return true;
        break;
      }
      rr += dr;
      cc += dc;
    }
  }
  return false;
}

function find(board: Board, piece: Piece): [number, number][] {
  const out: [number, number][] = [];
  board.forEach((row, r) => row.forEach((p, c) => p === piece && out.push([r, c])));
  return out;
}

export function inCheck(board: Board, white: boolean): boolean {
  const kings = find(board, white ? "K" : "k");
  return kings.length === 1 && attacked(board, kings[0][0], kings[0][1], !white);
}

export function validate(board: Board, side: Side): string[] {
  const errors: string[] = [];
  const wk = find(board, "K").length, bk = find(board, "k").length;
  if (wk !== 1) errors.push(tr("err.whiteKings", { n: wk }));
  if (bk !== 1) errors.push(tr("err.blackKings", { n: bk }));
  for (const r of [0, 7])
    for (let c = 0; c < 8; c++)
      if (board[r][c] === "P" || board[r][c] === "p") errors.push(tr("err.pawnBackRank", { square: squareName(r, c) }));
  if (wk === 1 && bk === 1) {
    const [[r1, c1]] = find(board, "K"), [[r2, c2]] = find(board, "k");
    if (Math.max(Math.abs(r1 - r2), Math.abs(c1 - c2)) <= 1) errors.push(tr("err.kingsAdjacent"));
    if (side === "w" && inCheck(board, false)) errors.push(tr("err.blackInCheck"));
    if (side === "b" && inCheck(board, true)) errors.push(tr("err.whiteInCheck"));
  }
  return errors;
}

export interface UciMove { from: [number, number]; to: [number, number]; promotion?: string }

export function parseUci(move: string): UciMove | null {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(move)) return null;
  return { from: parseSquare(move.slice(0, 2)), to: parseSquare(move.slice(2, 4)), promotion: move[4] };
}

/** Readable move notation without full SAN logic: "♞ g8–f6", "e7–e8=Q". */
export function describeMove(board: Board, move: string): string {
  const m = parseUci(move);
  if (!m) return move;
  const piece = board[m.from[0]][m.from[1]];
  const capture = board[m.to[0]][m.to[1]] !== "." ? "×" : "–";
  const glyph = piece !== "." && piece.toLowerCase() !== "p" ? GLYPH[piece.toLowerCase()] + " " : "";
  const promo = m.promotion ? `=${m.promotion.toUpperCase()}` : "";
  return `${glyph}${move.slice(0, 2)}${capture}${move.slice(2, 4)}${promo}`;
}

/** Play a move on a copy of the board (for showing the principal variation). */
export function applyUci(board: Board, move: string): Board {
  const m = parseUci(move);
  if (!m) return board;
  const b = board.map((row) => [...row]);
  let piece = b[m.from[0]][m.from[1]];
  const lower = piece.toLowerCase();
  // Castling: the king moves two squares, so move the rook too.
  if (lower === "k" && Math.abs(m.to[1] - m.from[1]) === 2) {
    const row = m.from[0];
    const [rf, rt] = m.to[1] === 6 ? [7, 5] : [0, 3];
    b[row][rt] = b[row][rf];
    b[row][rf] = ".";
  }
  // En passant: a pawn moves diagonally onto an empty square.
  if (lower === "p" && m.from[1] !== m.to[1] && b[m.to[0]][m.to[1]] === ".") b[m.from[0]][m.to[1]] = ".";
  if (m.promotion) piece = (isWhite(piece) ? m.promotion.toUpperCase() : m.promotion) as Piece;
  b[m.to[0]][m.to[1]] = piece;
  b[m.from[0]][m.from[1]] = ".";
  return b;
}

export function lichessUrl(fen: string): string {
  return `https://lichess.org/analysis/standard/${fen.replace(/ /g, "_")}`;
}
