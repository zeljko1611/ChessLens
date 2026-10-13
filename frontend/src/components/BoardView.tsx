import { GLYPH, isWhite, parseUci, pieceName, squareName, type Board } from "../lib/chess";
import { tr, useLang } from "../lib/i18n";

interface Props {
  board: Board;
  flipped: boolean;
  bestMove: string | null;
  uncertain: Set<string>;
  selected: string | null;
  onSelect: (square: string) => void;
}

const S = 100; // unit = one square in SVG coordinates

export function BoardView({ board, flipped, bestMove, uncertain, selected, onSelect }: Props) {
  useLang(); // re-render the labels when the language changes
  // Screen position of square (r, c), taking a flipped view into account.
  const at = (r: number, c: number) => (flipped ? [7 - c, 7 - r] : [c, r]);
  const move = bestMove ? parseUci(bestMove) : null;

  const cells = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const [x, y] = at(r, c);
      const name = squareName(r, c);
      const piece = board[r][c];
      const light = (r + c) % 2 === 0;
      cells.push(
        <g
          key={name}
          className="sq"
          transform={`translate(${x * S} ${y * S})`}
          onClick={() => onSelect(name)}
          role="button"
          tabIndex={0}
          aria-label={`${name}, ${piece !== "." ? pieceName(piece) : tr("board.empty")}`}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(name))}
        >
          <rect width={S} height={S} className={light ? "sq-light" : "sq-dark"} />
          {uncertain.has(name) && <rect x={5} y={5} width={S - 10} height={S - 10} className="sq-uncertain" />}
          {selected === name && <rect x={3} y={3} width={S - 6} height={S - 6} className="sq-selected" />}
          {piece !== "." && (
            <text x={S / 2} y={S * 0.8} className={isWhite(piece) ? "pc pc-white" : "pc pc-black"}>
              {GLYPH[piece.toLowerCase()]}
            </text>
          )}
          {y === 7 && <text x={S - 8} y={S - 8} className={`coord ${light ? "on-light" : "on-dark"}`} textAnchor="end">{"abcdefgh"[c]}</text>}
          {x === 0 && <text x={7} y={22} className={`coord ${light ? "on-light" : "on-dark"}`}>{8 - r}</text>}
        </g>,
      );
    }
  }

  let arrow = null;
  if (move) {
    const [x1, y1] = at(...move.from).map((v) => v * S + S / 2);
    const [x2, y2] = at(...move.to).map((v) => v * S + S / 2);
    const len = Math.hypot(x2 - x1, y2 - y1);
    const ux = (x2 - x1) / len, uy = (y2 - y1) / len;
    arrow = (
      <g className="arrow" pointerEvents="none">
        <line x1={x1} y1={y1} x2={x2 - ux * 34} y2={y2 - uy * 34} />
        <polygon
          points={`${x2},${y2} ${x2 - ux * 42 - uy * 26},${y2 - uy * 42 + ux * 26} ${x2 - ux * 42 + uy * 26},${y2 - uy * 42 - ux * 26}`}
        />
      </g>
    );
  }

  return (
    <svg className="board" viewBox={`0 0 ${8 * S} ${8 * S}`} role="grid" aria-label={tr("board.label")}>
      {cells}
      {arrow}
    </svg>
  );
}
