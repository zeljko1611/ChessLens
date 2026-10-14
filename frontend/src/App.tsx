import { Chess, type Square } from "chess.js";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BoardView } from "./components/BoardView";
import { PhotoView } from "./components/PhotoView";
import { health, prepareImage, recognize, saveFeedback, type Recognition } from "./lib/api";
import {
  GLYPH,
  PIECES,
  applyUci,
  boardToPlacement,
  describeMove,
  isWhite,
  lichessUrl,
  parseSquare,
  pieceName,
  placementToBoard,
  possibleCastling,
  toFen,
  validate,
  type Board,
  type Castling,
  type Cell,
  type Side,
} from "./lib/chess";
import { analyze, whitePerspective, type Analysis } from "./lib/engine";
import { LANGS, setLang, tr, useLang, type Lang } from "./lib/i18n";

type Photo = { blob: Blob; url: string; width: number; height: number };

const DEPTH = 16;

export default function App() {
  const lang = useLang();
  const [photo, setPhoto] = useState<Photo | null>(null);
  const [result, setResult] = useState<Recognition | null>(null);
  const [runId, setRunId] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [board, setBoard] = useState<Board | null>(null);
  const [side, setSide] = useState<Side>("w");
  const [castling, setCastling] = useState<Castling>({ K: true, Q: true, k: true, q: true });
  const [corners, setCorners] = useState<[number, number][] | null>(null);
  const [cornersMoved, setCornersMoved] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [flipped, setFlipped] = useState(false);
  const [fenDraft, setFenDraft] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState<{ state: "idle" | "saving" | "saved" | "error"; text?: string }>({ state: "idle" });
  const analysisId = useRef(0);

  // Playing from the recognized position. playFen === null means edit mode;
  // the edited board (board/side/castling) does not change during a game.
  const chess = useRef<Chess | null>(null);
  const [playFen, setPlayFen] = useState<string | null>(null);
  const [lastMove, setLastMove] = useState<[string, string] | null>(null);
  const playing = playFen !== null;

  useEffect(() => {
    // While the app is open, ping the server every minute. A server started by
    // the launcher (scripts/start_app.sh) stops once the pings stop.
    const beat = setInterval(() => health().catch(() => undefined), 60_000);
    return () => clearInterval(beat);
  }, []);

  const applyResult = useCallback((res: Recognition) => {
    setResult(res);
    setRunId((n) => n + 1);
    setBoard(res.board);
    setSide(res.side_to_move);
    setCastling({ K: true, Q: true, k: true, q: true });
    setCorners(res.corners);
    setCornersMoved(false);
    setSelected(null);
    setFlipped(false);
    setFenDraft(null);
    setFeedback({ state: "idle" });
    setPlayFen(null);
  }, []);

  const run = useCallback(
    async (p: Photo, opts: { corners?: [number, number][]; orientation?: number } = {}) => {
      setBusy(true);
      setError(null);
      try {
        applyResult(await recognize(p.blob, opts));
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [applyResult],
  );

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setError(null);
    setResult(null);
    setBoard(null);
    setAnalysis(null);
    try {
      const p = await prepareImage(file);
      if (photo) URL.revokeObjectURL(photo.url);
      setPhoto(p);
      setCorners(null);
      await run(p);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const editFen = board ? toFen(board, side, castling) : null;
  const fen = playFen ?? editFen;
  const shownBoard = playFen ? placementToBoard(playFen) : board;
  const shownSide: Side = playFen ? (playFen.split(" ")[1] as Side) : side;
  // Error messages are in the current language, so recompute them when it changes.
  const errors = useMemo(() => (board && !playing ? validate(board, side) : []), [board, side, lang, playing]);
  const allowed = board ? possibleCastling(board) : null;
  const uncertain = useMemo(() => new Set(result?.uncertain ?? []), [result]);

  // Analyse automatically whenever the position changes (and is legal).
  useEffect(() => {
    if (!fen || errors.length) {
      setAnalysis(null);
      return;
    }
    const id = ++analysisId.current;
    setAnalyzing(true);
    setAnalysisError(null);
    const t = setTimeout(() => {
      analyze(fen, DEPTH, (partial) => id === analysisId.current && setAnalysis(partial))
        .then((a) => id === analysisId.current && setAnalysis(a))
        .catch((e: Error) => {
          if (id === analysisId.current) setAnalysisError(e.message);
        })
        .finally(() => id === analysisId.current && setAnalyzing(false));
    }, 350);
    return () => clearTimeout(t);
  }, [fen, errors.length]);

  const setPiece = (piece: Cell) => {
    if (!board || !selected) return;
    const [r, c] = parseSquare(selected);
    setBoard(board.map((row, i) => row.map((p, j) => (i === r && j === c ? piece : p))));
  };

  // After a new board edit, the same photo can be saved again.
  useEffect(() => setFeedback((f) => (f.state === "saved" ? { state: "idle" } : f)), [board]);

  const onSaveFeedback = async () => {
    if (!photo || !result || !board) return;
    setFeedback({ state: "saving" });
    try {
      const res = await saveFeedback(photo.blob, result.corners, boardToPlacement(board));
      setFeedback({ state: "saved", text: tr("teach.savedCount", { n: res.count }) });
    } catch (e) {
      setFeedback({ state: "error", text: (e as Error).message });
    }
  };

  // A game starts (or restarts) from the photo position, including corrections.
  const startGame = () => {
    if (!editFen) return;
    try {
      chess.current = new Chess(editFen);
    } catch (e) {
      setError((e as Error).message); // chess.js is stricter than our check (e.g. impossible castling)
      return;
    }
    setPlayFen(chess.current.fen());
    setLastMove(null);
    setSelected(null);
  };

  const stopGame = () => {
    chess.current = null;
    setPlayFen(null);
    setLastMove(null);
    setSelected(null);
  };

  // The first click selects a piece of the side to move, the second plays it.
  const tryMove = (from: string, to: string): boolean => {
    const game = chess.current;
    if (!game || !game.moves({ square: from as Square, verbose: true }).some((m) => m.to === to)) return false;
    game.move({ from, to, promotion: "q" }); // always promote to a queen
    setPlayFen(game.fen());
    setLastMove([from, to]);
    setSelected(null);
    return true;
  };

  const onPlayClick = (sq: string) => {
    const game = chess.current;
    if (!game || (selected && tryMove(selected, sq))) return;
    const piece = game.get(sq as Square);
    setSelected(piece && piece.color === game.turn() && sq !== selected ? sq : null);
  };

  // Dragging: in a game only legal moves by the side to move; in edit mode a piece
  // moves freely, and dropping it off the board removes it.
  const canDrag = (sq: string): boolean => {
    if (playing) return chess.current?.get(sq as Square)?.color === chess.current?.turn();
    const [r, c] = parseSquare(sq);
    return !!board && board[r][c] !== ".";
  };

  const onDrop = (from: string, to: string | null) => {
    setSelected(null);
    if (playing) {
      if (to) tryMove(from, to);
      return;
    }
    if (!board || from === to) return;
    const [fr, fc] = parseSquare(from);
    const next = board.map((row) => [...row]);
    if (to) {
      const [tr2, tc2] = parseSquare(to);
      next[tr2][tc2] = next[fr][fc];
    }
    next[fr][fc] = ".";
    setBoard(next);
  };

  const targets = useMemo(
    () => (playFen && selected && chess.current ? new Set(chess.current.moves({ square: selected as Square, verbose: true }).map((m) => m.to as string)) : undefined),
    [playFen, selected],
  );

  const pv = useMemo(() => {
    if (!shownBoard || !analysis?.pv?.length) return [];
    let b = shownBoard;
    return analysis.pv.slice(0, 8).map((m) => {
      const text = describeMove(b, m);
      b = applyUci(b, m);
      return text;
    });
  }, [shownBoard, analysis]);

  const evalView = analysis && (analysis.score_cp !== null || analysis.mate !== null) ? whitePerspective(analysis, shownSide) : null;

  if (!photo) {
    return (
      <main className="landing">
        <LangSwitch lang={lang} />
        <Intro />
        <Upload onFile={onFile} large />
        {error && <p className="error" role="alert">{error}</p>}
      </main>
    );
  }

  return (
    <div className="app">
      <header className="bar">
        <h1 className="wordmark">{tr("title")}</h1>
        <div className="bar-actions">
          <Upload onFile={onFile} />
          <LangSwitch lang={lang} />
        </div>
      </header>

      <section className="pane pane-photo" aria-label={tr("pane.photo")}>
        <PhotoView
          key={runId}
          url={photo.url}
          width={photo.width}
          height={photo.height}
          result={result}
          corners={corners}
          selected={playing ? null : selected}
          onCornersChange={(c) => {
            setCorners(c);
            setCornersMoved(true);
          }}
        />
        <div className="photo-actions">
          {cornersMoved && corners ? (
            <button className="btn primary" disabled={busy} onClick={() => run(photo, { corners })}>
              {tr("photo.recognizeWithCorners")}
            </button>
          ) : (
            <p className="hint">
              {result ? tr("photo.hintCorners") : busy ? tr("photo.searching") : ""}
            </p>
          )}
          {result && (
            <button className="btn" disabled={busy} onClick={() => run(photo, { corners: result.corners, orientation: (result.orientation + 2) % 4 })}>
              {tr("photo.otherSide")}
            </button>
          )}
        </div>
        {error && (
          <div className="error" role="alert">
            <p>{error}</p>
            {!result && photo && (
              <button className="btn" onClick={() => {
                const w = photo.width, h = photo.height;
                setCorners([[w * 0.25, h * 0.25], [w * 0.75, h * 0.25], [w * 0.75, h * 0.75], [w * 0.25, h * 0.75]]);
                setCornersMoved(true);
                setError(null);
              }}>
                {tr("photo.markCorners")}
              </button>
            )}
          </div>
        )}
      </section>

      <section className="pane pane-board" aria-label={tr("pane.board")} aria-busy={busy}>
        {board && shownBoard ? (
          <>
            <div className="board-wrap">
              <EvalBar share={evalView?.whiteShare ?? 0.5} flipped={flipped} />
              <BoardView
                board={shownBoard}
                flipped={flipped}
                bestMove={errors.length ? null : analysis?.best_move ?? null}
                uncertain={playing ? new Set() : uncertain}
                selected={selected}
                onSelect={playing ? onPlayClick : (sq) => setSelected(sq === selected ? null : sq)}
                targets={targets}
                lastMove={lastMove}
                canDrag={canDrag}
                onDragStart={playing ? setSelected : undefined}
                onDrop={onDrop}
              />
            </div>

            {playing && chess.current ? (
              <PlayPanel game={chess.current} onReset={startGame} onEdit={stopGame} />
            ) : selected ? (
              <div className="picker" role="toolbar" aria-label={tr("picker.label", { square: selected })}>
                <span className="picker-title">{selected}</span>
                {PIECES.map((p) => (
                  <button key={p} className={`pick ${isWhite(p) ? "pc-white" : "pc-black"}`} title={pieceName(p)} aria-label={pieceName(p)} onClick={() => setPiece(p)}>
                    {GLYPH[p.toLowerCase()]}
                  </button>
                ))}
                <button className="pick pick-empty" title={tr("picker.empty")} aria-label={tr("picker.empty")} onClick={() => setPiece(".")}>
                  ×
                </button>
                {result?.probabilities[selected] && (
                  <span className="picker-note">
                    {tr("picker.model")}: {Object.entries(result.probabilities[selected]).map(([k, v]) => `${k === "." ? tr("picker.emptyShort") : k} ${Math.round(v * 100)}%`).join(", ")}
                  </span>
                )}
              </div>
            ) : (
              <p className="hint">
                {uncertain.size === 1 ? tr("hint.uncertainOne") : uncertain.size > 1 ? tr("hint.uncertainMany", { n: uncertain.size }) : ""}
                {tr("hint.clickToFix")}
              </p>
            )}

            {!playing && (
            <>
            <div className="controls">
              <div className="segmented" role="radiogroup" aria-label={tr("side.label")}>
                <button role="radio" aria-checked={side === "w"} className={side === "w" ? "on" : ""} onClick={() => setSide("w")}>{tr("side.white")}</button>
                <button role="radio" aria-checked={side === "b"} className={side === "b" ? "on" : ""} onClick={() => setSide("b")}>{tr("side.black")}</button>
              </div>
              <button className="btn" onClick={() => setFlipped(!flipped)}>{tr("flip")}</button>
              <button className="btn primary" disabled={errors.length > 0} onClick={startGame}>{tr("play.start")}</button>
            </div>

            {allowed && (allowed.K || allowed.Q || allowed.k || allowed.q) && (
              <fieldset className="castling">
                <legend>{tr("castling.legend")}</legend>
                {(["K", "Q", "k", "q"] as const).map((k) =>
                  allowed[k] ? (
                    <label key={k}>
                      <input type="checkbox" checked={castling[k]} onChange={(e) => setCastling({ ...castling, [k]: e.target.checked })} />
                      {tr(`castling.${k}`)}
                    </label>
                  ) : null,
                )}
              </fieldset>
            )}

            {errors.length > 0 && (
              <ul className="problems" role="alert">
                {errors.map((e) => <li key={e}>{e}</li>)}
              </ul>
            )}
            </>
            )}

            {!errors.length && (
              <div className="analysis" aria-live="polite">
                <div className="analysis-head">
                  <span className="eval">{analysisError ? "–" : evalView?.text ?? "…"}</span>
                  <span className="analysis-meta">
                    {analysisError
                      ? analysisError
                      : analyzing
                        ? `${tr("analysis.thinking")}${analysis?.depth ? ` (${tr("analysis.depth", { n: analysis.depth })})` : ""}…`
                        : analysis
                          ? tr("analysis.done", { n: analysis.depth })
                          : ""}
                  </span>
                </div>
                {pv.length > 0 && (
                  <ol className="pv" start={1}>
                    {pv.map((m, i) => <li key={i} className={i === 0 ? "best" : ""}>{m}</li>)}
                  </ol>
                )}
              </div>
            )}

            <div className="fen">
              <label htmlFor="fen">FEN</label>
              <input
                id="fen"
                spellCheck={false}
                readOnly={playing}
                value={fenDraft ?? fen ?? ""}
                onChange={(e) => setFenDraft(e.target.value)}
                onBlur={() => {
                  if (fenDraft === null) return;
                  const b = placementToBoard(fenDraft);
                  if (b) {
                    setBoard(b);
                    const s = fenDraft.trim().split(/\s+/)[1];
                    if (s === "w" || s === "b") setSide(s);
                  }
                  setFenDraft(null);
                }}
              />
              <button className="btn" onClick={() => fen && navigator.clipboard?.writeText(fen).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); })}>
                {copied ? tr("copied") : tr("copy")}
              </button>
              {fen && <a className="btn" href={lichessUrl(fen)} target="_blank" rel="noreferrer">{tr("lichess")}</a>}
            </div>

            {result && !playing && (
              <div className="teach">
                <button
                  className="btn"
                  disabled={errors.length > 0 || cornersMoved || feedback.state === "saving" || feedback.state === "saved"}
                  onClick={onSaveFeedback}
                >
                  {feedback.state === "saved" ? tr("teach.saved") : tr("teach.save")}
                </button>
                <p className={feedback.state === "error" ? "error" : "hint"}>
                  {feedback.text ?? tr("teach.hint")}
                </p>
              </div>
            )}

          </>
        ) : (
          <div className="board-placeholder">{busy ? tr("placeholder.busy") : tr("placeholder.idle")}</div>
        )}
      </section>
    </div>
  );
}

/** Game status, the moves played and the game buttons. */
function PlayPanel({ game, onReset, onEdit }: { game: Chess; onReset: () => void; onEdit: () => void }) {
  const turn = game.turn() === "w" ? tr("side.white") : tr("side.black");
  const status = game.isCheckmate()
    ? tr("play.mate", { winner: game.turn() === "w" ? tr("play.black") : tr("play.white") })
    : game.isStalemate()
      ? tr("play.stalemate")
      : game.isDraw()
        ? tr("play.draw")
        : game.inCheck()
          ? `${tr("play.check")} ${turn}`
          : turn;
  const moves = game.history();
  const start = game.history({ verbose: true })[0]?.before.split(" ")[1] === "b" ? 1 : 0; // when Black moves first
  const numbered = moves.map((m, i) => ((i + start) % 2 === 0 ? `${Math.floor((i + start) / 2) + 1}. ${m}` : i === 0 ? `1… ${m}` : m));
  return (
    <div className="play">
      <p className="play-status" aria-live="polite">{status}</p>
      <p className="hint">{tr("play.hint")}</p>
      {moves.length > 0 && <p className="play-moves">{numbered.join(" ")}</p>}
      <div className="controls">
        <button className="btn" onClick={onReset}>{tr("play.reset")}</button>
        <button className="btn" onClick={onEdit}>{tr("play.edit")}</button>
      </div>
    </div>
  );
}

/** Language switch: SR | EN. */
function LangSwitch({ lang }: { lang: Lang }) {
  return (
    <div className="segmented lang-switch" role="radiogroup" aria-label={tr("lang.label")}>
      {LANGS.map((l) => (
        <button key={l} role="radio" aria-checked={lang === l} className={lang === l ? "on" : ""} onClick={() => setLang(l)} lang={l === "sr" ? "sr-Latn" : "en"}>
          {l.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function EvalBar({ share, flipped }: { share: number; flipped: boolean }) {
  // The white part of the bar grows from the bottom (or from the top when flipped).
  return (
    <div className={`evalbar${flipped ? " flipped" : ""}`} aria-hidden="true">
      <div className="evalbar-white" style={{ height: `${Math.round(share * 100)}%` }} />
    </div>
  );
}

function Upload({ onFile, large = false }: { onFile: (f: File | undefined) => void; large?: boolean }) {
  return (
    <div className={large ? "upload upload-large" : "upload"}>
      <label className="btn primary">
        <input type="file" accept="image/*" capture="environment" onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
        {tr("upload.camera")}
      </label>
      <label className="btn">
        <input type="file" accept="image/*" onChange={(e) => { onFile(e.target.files?.[0]); e.target.value = ""; }} />
        {tr("upload.file")}
      </label>
    </div>
  );
}

function Intro() {
  return (
    <section className="intro">
      <PerspectiveBoard />
      <div className="intro-text">
        <h1 className="wordmark big">{tr("title")}</h1>
        <p className="lede">{tr("intro.lede")}</p>
        <p className="small">{tr("intro.small")}</p>
      </div>
    </section>
  );
}

/** A board in perspective with the detection grid, illustrating what the app does. */
function PerspectiveBoard() {
  const tl: [number, number] = [70, 40], tr: [number, number] = [330, 40], br: [number, number] = [390, 250], bl: [number, number] = [10, 250];
  const lerp = (p: [number, number], q: [number, number], t: number): [number, number] => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
  // Approximate perspective: rows closer to the camera are taller.
  const rowT = (i: number) => (Math.pow(1.18, i) - 1) / (Math.pow(1.18, 8) - 1);
  const pt = (i: number, j: number) => lerp(lerp(tl, bl, rowT(i)), lerp(tr, br, rowT(i)), j / 8);
  const cells = [];
  for (let i = 0; i < 8; i++)
    for (let j = 0; j < 8; j++) {
      const q = [pt(i, j), pt(i, j + 1), pt(i + 1, j + 1), pt(i + 1, j)].map((p) => p.join(",")).join(" ");
      cells.push(<polygon key={`${i}-${j}`} points={q} className={(i + j) % 2 ? "sq-dark" : "sq-light"} />);
    }
  const lines = [];
  for (let k = 0; k <= 8; k++) {
    lines.push(<line key={`h${k}`} x1={pt(k, 0)[0]} y1={pt(k, 0)[1]} x2={pt(k, 8)[0]} y2={pt(k, 8)[1]} />);
    lines.push(<line key={`v${k}`} x1={pt(0, k)[0]} y1={pt(0, k)[1]} x2={pt(8, k)[0]} y2={pt(8, k)[1]} />);
  }
  const pieces: [number, number, string, boolean][] = [
    [7, 4, "k", true], [6, 3, "p", true], [6, 5, "p", true], [5, 2, "n", true], [0, 4, "k", false], [1, 3, "p", false], [2, 5, "n", false], [3, 4, "p", false],
  ];
  return (
    <svg className="hero-board" viewBox="0 0 400 290" aria-hidden="true">
      {cells}
      {pieces.map(([i, j, p, white]) => {
        const [x, y] = lerp(lerp(pt(i, j), pt(i, j + 1), 0.5), lerp(pt(i + 1, j), pt(i + 1, j + 1), 0.5), 0.55);
        const size = 22 + i * 3.2;
        return <text key={`${i}${j}`} x={x} y={y} fontSize={size} textAnchor="middle" className={white ? "pc pc-white" : "pc pc-black"}>{GLYPH[p]}</text>;
      })}
      <g className="hero-grid">{lines}</g>
      {[tl, tr, br, bl].map(([x, y], i) => <circle key={i} cx={x} cy={y} r={5} className="hero-corner" />)}
    </svg>
  );
}
