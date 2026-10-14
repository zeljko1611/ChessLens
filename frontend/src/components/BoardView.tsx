import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { parseUci, pieceName, squareName, type Board } from "../lib/chess";
import { tr, useLang } from "../lib/i18n";
import { Piece } from "./Piece";

interface Props {
  board: Board;
  flipped: boolean;
  bestMove: string | null;
  uncertain: Set<string>;
  selected: string | null;
  onSelect: (square: string) => void;
  targets?: Set<string>; // squares the selected piece can move to (game mode)
  lastMove?: [string, string] | null;
  canDrag: (square: string) => boolean;
  onDragStart?: (square: string) => void;
  onDrop: (from: string, to: string | null) => void; // to = null: the piece was dropped off the board
}

type Drag = { from: string; x: number; y: number; moved: boolean };

const S = 100; // unit = one square in SVG coordinates

export function BoardView({ board, flipped, bestMove, uncertain, selected, onSelect, targets, lastMove, canDrag, onDragStart, onDrop }: Props) {
  useLang(); // re-render the labels when the language changes
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  // Screen position of square (r, c), taking a flipped view into account.
  const at = (r: number, c: number) => (flipped ? [7 - c, 7 - r] : [c, r]);
  const move = bestMove ? parseUci(bestMove) : null;

  const toSvg = (e: ReactPointerEvent): [number, number] => {
    const pt = svgRef.current!.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svgRef.current!.getScreenCTM()!.inverse());
    return [p.x, p.y];
  };

  // The square under a point (in SVG coordinates), or null outside the board.
  const squareAt = (x: number, y: number): string | null => {
    const sx = Math.floor(x / S), sy = Math.floor(y / S);
    if (sx < 0 || sx > 7 || sy < 0 || sy > 7) return null;
    return flipped ? squareName(7 - sy, 7 - sx) : squareName(sy, sx);
  };

  // Pressing a square: if the pointer moves, it is a drag;
  // if it is released on the same square, it is a plain click.
  const onPointerDown = (e: ReactPointerEvent) => {
    if (e.button !== 0) return;
    const [x, y] = toSvg(e);
    const from = squareAt(x, y);
    if (!from) return;
    svgRef.current!.setPointerCapture(e.pointerId);
    setDrag({ from, x, y, moved: false });
  };

  const onPointerMove = (e: ReactPointerEvent) => {
    if (!drag || !canDrag(drag.from)) return;
    const [x, y] = toSvg(e);
    const moved = drag.moved || squareAt(x, y) !== drag.from;
    if (moved && !drag.moved) onDragStart?.(drag.from);
    setDrag({ ...drag, x, y, moved });
  };

  const onPointerUp = (e: ReactPointerEvent) => {
    if (!drag) return;
    const [x, y] = toSvg(e);
    const to = squareAt(x, y);
    setDrag(null);
    if (drag.moved && canDrag(drag.from)) onDrop(drag.from, to);
    else if (to === drag.from) onSelect(drag.from);
  };

  const dragging = drag?.moved ? drag : null;

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
          role="button"
          tabIndex={0}
          aria-label={`${name}, ${piece !== "." ? pieceName(piece) : tr("board.empty")}`}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), onSelect(name))}
        >
          <rect width={S} height={S} className={light ? "sq-light" : "sq-dark"} />
          {lastMove?.includes(name) && <rect width={S} height={S} className="sq-last" />}
          {uncertain.has(name) && <rect x={5} y={5} width={S - 10} height={S - 10} className="sq-uncertain" />}
          {selected === name && <rect x={3} y={3} width={S - 6} height={S - 6} className="sq-selected" />}
          {piece !== "." && <Piece piece={piece} opacity={dragging?.from === name ? 0.3 : 1} />}
          {targets?.has(name) && <circle cx={S / 2} cy={S / 2} r={piece !== "." ? S * 0.45 : S * 0.15} className={piece !== "." ? "sq-capture" : "sq-target"} />}
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
    <svg
      ref={svgRef}
      className={dragging ? "board dragging" : "board"}
      viewBox={`0 0 ${8 * S} ${8 * S}`}
      role="grid"
      aria-label={tr("board.label")}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => setDrag(null)}
    >
      {cells}
      {arrow}
      {dragging && (() => {
        const [r, c] = [8 - Number(dragging.from[1]), "abcdefgh".indexOf(dragging.from[0])];
        const piece = board[r][c];
        const size = S * 1.15; // slightly larger while dragged, centred on the pointer
        return piece === "." ? null : <Piece piece={piece} x={dragging.x - size / 2} y={dragging.y - size / 2} size={size} />;
      })()}
    </svg>
  );
}
