import { isWhite, type Piece as PieceType } from "../lib/chess";
import { PIECE_PATHS } from "../lib/pieces";

/** A piece in a 100×100 box whose top-left corner is (x, y); `size` scales the box. */
export function Piece({ piece, x = 0, y = 0, size = 100, opacity }: { piece: PieceType; x?: number; y?: number; size?: number; opacity?: number }) {
  return (
    <path
      d={PIECE_PATHS[piece.toLowerCase()]}
      transform={`translate(${x} ${y}) scale(${size / 100})`}
      className={isWhite(piece) ? "pc pc-white" : "pc pc-black"}
      opacity={opacity}
    />
  );
}
