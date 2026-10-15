// Stockfish in the browser (Web Worker).
import { tr } from "./i18n";

export interface Analysis {
  best_move: string | null;
  score_cp: number | null;
  mate: number | null;
  depth: number;
  pv: string[];
}

// Stockfish compiled to JavaScript (asm.js), loaded from cdnjs. It runs in a Web
// Worker, so it never blocks the UI.
const STOCKFISH_CDN = "https://cdnjs.cloudflare.com/ajax/libs/stockfish.js/10.0.2/stockfish.js";

let worker: Worker | null = null;
let busy = false;

function getWorker(): Worker {
  if (!worker) {
    // A worker from another origin cannot be started directly, so create a small
    // local worker that loads the script with importScripts.
    const src = `importScripts(${JSON.stringify(STOCKFISH_CDN)});`;
    worker = new Worker(URL.createObjectURL(new Blob([src], { type: "application/javascript" })));
  }
  return worker;
}

export function parseInfoLine(line: string): Partial<Analysis> | null {
  const parts = line.trim().split(/\s+/);
  if (parts[0] !== "info" || !parts.includes("pv") || !parts.includes("score")) return null;
  const out: Partial<Analysis> = {};
  for (let i = 1; i < parts.length; i++) {
    if (parts[i] === "depth") out.depth = Number(parts[++i]);
    else if (parts[i] === "multipv" && parts[i + 1] !== "1") return null; // main line only
    else if (parts[i] === "score") {
      const kind = parts[++i];
      const val = Number(parts[++i]);
      if (kind === "cp") out.score_cp = val;
      else out.mate = val;
    } else if (parts[i] === "pv") {
      out.pv = parts.slice(i + 1);
      break;
    }
  }
  return out;
}

function analyzeInBrowser(fen: string, depth: number, onProgress?: (a: Analysis) => void): Promise<Analysis> {
  if (busy && worker) {
    // The previous analysis is still running: the simplest fix is to terminate the
    // worker so its "bestmove" cannot arrive as the answer for the new position.
    worker.terminate();
    worker = null;
  }
  busy = true;
  return new Promise<Analysis>((resolve, reject) => {
    let w: Worker;
    try {
      w = getWorker();
    } catch (e) {
      reject(new Error(tr("engine.start")));
      return;
    }
    const current: Analysis = { best_move: null, score_cp: null, mate: null, depth: 0, pv: [] };
    const fail = (msg: string) => {
      clearTimeout(timer);
      w.removeEventListener("message", onMessage);
      // Never reuse a broken worker; the next analysis creates a new one.
      w.terminate();
      if (worker === w) worker = null;
      reject(new Error(msg));
    };
    const timer = setTimeout(() => fail(tr("engine.timeout")), 60000);
    const onMessage = (e: MessageEvent) => {
      const line = String(e.data);
      const info = parseInfoLine(line);
      if (info) {
        Object.assign(current, { score_cp: null, mate: null }, info);
        current.best_move = current.pv[0] ?? null;
        onProgress?.({ ...current });
      } else if (line.startsWith("bestmove")) {
        clearTimeout(timer);
        w.removeEventListener("message", onMessage);
        const bm = line.split(/\s+/)[1];
        resolve({ ...current, best_move: bm && bm !== "(none)" ? bm : null });
      }
    };
    w.addEventListener("message", onMessage);
    w.onerror = () => fail(tr("engine.load"));
    w.postMessage("ucinewgame");
    w.postMessage(`position fen ${fen}`);
    w.postMessage(`go depth ${depth}`);
  });
}

export async function analyze(fen: string, depth: number, onProgress?: (a: Analysis) => void): Promise<Analysis> {
  try {
    return await analyzeInBrowser(fen, depth, onProgress);
  } finally {
    busy = false;
  }
}

/** Evaluation from White's point of view, for the eval bar and text (+1.2, −M3...). */
export function whitePerspective(a: Analysis, side: "w" | "b"): { text: string; whiteShare: number } {
  const sign = side === "w" ? 1 : -1;
  if (a.mate !== null && a.mate !== undefined) {
    const m = a.mate * sign;
    return { text: `${m > 0 ? "+" : "−"}M${Math.abs(m)}`, whiteShare: m > 0 ? 1 : 0 };
  }
  const cp = (a.score_cp ?? 0) * sign;
  const share = 1 / (1 + Math.exp(-cp / 250)); // smooth mapping to 0..1
  const pawns = (cp / 100).toFixed(1);
  return { text: `${cp > 0 ? "+" : cp < 0 ? "−" : ""}${pawns.replace("-", "")}`, whiteShare: share };
}
