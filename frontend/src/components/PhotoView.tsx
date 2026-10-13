import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import type { Recognition } from "../lib/api";

interface Props {
  url: string;
  width: number;
  height: number;
  result: Recognition | null;
  corners: [number, number][] | null; // a8, h8, h1, a1 (may be moved by hand)
  onCornersChange: (c: [number, number][]) => void;
  selected: string | null;
}

const CORNER_LABELS = ["a8", "h8", "h1", "a1"];

/** A 9x9 grid from 4 corners (bilinear, for display while the user drags corners). */
function previewGrid(c: [number, number][]): [number, number][][] {
  const [a8, h8, h1, a1] = c;
  const lerp = (p: [number, number], q: [number, number], t: number): [number, number] => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
  const rows: [number, number][][] = [];
  for (let i = 0; i <= 8; i++) {
    const left = lerp(a8, a1, i / 8), right = lerp(h8, h1, i / 8);
    rows.push(Array.from({ length: 9 }, (_, j) => lerp(left, right, j / 8)));
  }
  return rows;
}

// The parent changes `key` after every recognition, which resets the "moved" state.
export function PhotoView({ url, width, height, result, corners, onCornersChange, selected }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const [moved, setMoved] = useState(false);

  // While the corners are untouched, draw the exact grid from the server (perspective);
  // once the user drags a corner, show an approximate grid from the new corners.
  const grid = corners && (moved || !result) ? previewGrid(corners) : result?.grid;
  const unit = Math.max(width, height) / 400; // line width independent of resolution

  const toSvg = (e: ReactPointerEvent): [number, number] => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX;
    pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return [Math.max(0, Math.min(width, p.x)), Math.max(0, Math.min(height, p.y))];
  };

  const lines = [];
  if (grid) {
    for (let i = 0; i <= 8; i++) {
      const row = grid[i].map((p) => p.join(",")).join(" ");
      const col = grid.map((r) => r[i].join(",")).join(" ");
      lines.push(<polyline key={`r${i}`} points={row} />, <polyline key={`c${i}`} points={col} />);
    }
  }

  return (
    <div className="photo">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className={drag !== null ? "dragging" : undefined}
        onPointerMove={(e) => {
          if (drag === null || !corners) return;
          const next = corners.map((p, i) => (i === drag ? toSvg(e) : p)) as [number, number][];
          setMoved(true);
          onCornersChange(next);
        }}
        onPointerUp={() => setDrag(null)}
        onPointerLeave={() => setDrag(null)}
      >
        <image href={url} width={width} height={height} />
        <g className="grid" strokeWidth={unit * 1.2}>
          {lines}
        </g>
        {result && !moved &&
          result.uncertain.map((sq) => {
            const [x, y] = result.square_centers[sq];
            return <circle key={sq} cx={x} cy={y} r={unit * 7} className="photo-uncertain" strokeWidth={unit * 1.5} />;
          })}
        {result && !moved && selected && result.square_centers[selected] && (
          <circle cx={result.square_centers[selected][0]} cy={result.square_centers[selected][1]} r={unit * 9} className="photo-selected" strokeWidth={unit * 2} />
        )}
        {corners?.map(([x, y], i) => (
          <g key={i} className="handle" onPointerDown={(e) => { (e.target as Element).setPointerCapture?.(e.pointerId); setDrag(i); }}>
            <circle cx={x} cy={y} r={unit * 14} className="handle-hit" />
            <circle cx={x} cy={y} r={unit * 6} className="handle-dot" strokeWidth={unit * 2} />
            <text x={x + unit * 10} y={y - unit * 10} fontSize={unit * 14} className="handle-label">{CORNER_LABELS[i]}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
