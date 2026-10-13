// UI translations (Serbian / English). The language is global because api.ts,
// engine.ts and chess.ts use it too; components read it with useLang().
import { useSyncExternalStore } from "react";

export type Lang = "sr" | "en";
export const LANGS: Lang[] = ["sr", "en"];

const sr = {
  title: "ChessLens",
  "lang.label": "Jezik",
  "intro.lede": "Slikaj šahovsku tablu telefonom. Aplikacija pronađe tablu, prepozna svaku figuru, pretvori poziciju u FEN i pokaže najbolji potez.",
  "intro.small": "Najbolje radi kad se vidi cela tabla, iz ugla igrača, bez jakog odsjaja. Pogrešno prepoznata polja možeš da ispraviš jednim klikom.",
  "upload.camera": "Slikaj tablu",
  "upload.file": "Izaberi fotografiju",
  "pane.photo": "Fotografija",
  "pane.board": "Pozicija",
  "photo.recognizeWithCorners": "Prepoznaj sa ovim uglovima",
  "photo.hintCorners": "Ako mreža ne leži na tabli, prevuci uglove na ivice polja za igru.",
  "photo.searching": "Tražim tablu na fotografiji…",
  "photo.otherSide": "Beli su na drugoj strani",
  "photo.markCorners": "Označi uglove ručno",
  "picker.label": "Figura na polju {square}",
  "picker.empty": "prazno polje",
  "picker.model": "model",
  "picker.emptyShort": "prazno",
  "hint.uncertainOne": "Model nije siguran za 1 polje (označeno žuto). ",
  "hint.uncertainMany": "Model nije siguran za {n} polja (označena žuto). ",
  "hint.clickToFix": "Klikni na polje da ispraviš figuru.",
  "side.label": "Na potezu",
  "side.white": "Beli na potezu",
  "side.black": "Crni na potezu",
  flip: "Okreni prikaz",
  "castling.legend": "Rokade su još moguće",
  "castling.K": "beli kratka",
  "castling.Q": "beli duga",
  "castling.k": "crni kratka",
  "castling.q": "crni duga",
  "analysis.thinking": "Stockfish razmišlja",
  "analysis.depth": "dubina {n}",
  "analysis.done": "Stockfish, dubina {n}",
  copy: "Kopiraj",
  copied: "Kopirano",
  lichess: "Otvori na Lichess-u",
  "teach.save": "Sačuvaj za učenje",
  "teach.saved": "Sačuvano za učenje",
  "teach.savedCount": "Sačuvano. Primera za učenje: {n}.",
  "teach.hint": "Kad je pozicija na tabli tačna, sačuvaj je. Sa 10–20 takvih fotografija model se dotrenira na tvoje figure.",
  "placeholder.busy": "Prepoznajem figure…",
  "placeholder.idle": "Pozicija će se pojaviti ovde.",
  "board.label": "Prepoznata pozicija",
  "board.empty": "prazno",
  // chess.ts
  "piece.K": "beli kralj", "piece.Q": "bela dama", "piece.R": "beli top", "piece.B": "beli lovac", "piece.N": "beli skakač", "piece.P": "beli pijun",
  "piece.k": "crni kralj", "piece.q": "crna dama", "piece.r": "crni top", "piece.b": "crni lovac", "piece.n": "crni skakač", "piece.p": "crni pijun",
  "err.whiteKings": "Potreban je tačno jedan beli kralj (sada: {n}).",
  "err.blackKings": "Potreban je tačno jedan crni kralj (sada: {n}).",
  "err.pawnBackRank": "Pijun ne može stajati na {square}.",
  "err.kingsAdjacent": "Kraljevi stoje na susednim poljima.",
  "err.blackInCheck": "Crni je u šahu, a beli je na potezu.",
  "err.whiteInCheck": "Beli je u šahu, a crni je na potezu.",
  // error codes from the server (backend/service.py)
  "server.board_not_found": "Tabla nije pronađena. Probaj sliku iz drugog ugla ili ručno označi uglove.",
  "server.not_an_image": "Fajl nije slika (podržani su JPEG, PNG, WebP).",
  "server.image_too_large": "Slika je prevelika (najviše 15 MB).",
  "server.bad_corners": "Neispravni uglovi table.",
  "server.bad_side_to_move": "Neispravna strana na potezu.",
  "server.bad_position": "Neispravna pozicija.",
  "server.illegal_position": "Pozicija nije legalna; ispravi je pre čuvanja.",
  "api.serverError": "Server je vratio grešku {status}.",
  "api.imagePrepare": "Slika nije mogla da se pripremi.",
  "engine.start": "Engine u browseru nije mogao da se pokrene.",
  "engine.timeout": "Engine u browseru ne odgovara.",
  "engine.load": "Engine u browseru nije mogao da se učita (proveri internet vezu). Poziciju možeš analizirati na Lichess-u.",
};

type Key = keyof typeof sr;

const en: Record<Key, string> = {
  title: "ChessLens",
  "lang.label": "Language",
  "intro.lede": "Take a photo of a chessboard with your phone. The app finds the board, recognizes every piece, turns the position into FEN and shows the best move.",
  "intro.small": "Works best when the whole board is visible, from a player's side, without strong glare. Misrecognized squares can be fixed with one click.",
  "upload.camera": "Take a photo",
  "upload.file": "Choose a photo",
  "pane.photo": "Photo",
  "pane.board": "Position",
  "photo.recognizeWithCorners": "Recognize with these corners",
  "photo.hintCorners": "If the grid does not sit on the board, drag the corners to the edges of the playing area.",
  "photo.searching": "Looking for the board in the photo…",
  "photo.otherSide": "White is on the other side",
  "photo.markCorners": "Mark corners manually",
  "picker.label": "Piece on {square}",
  "picker.empty": "empty square",
  "picker.model": "model",
  "picker.emptyShort": "empty",
  "hint.uncertainOne": "The model is unsure about 1 square (highlighted in yellow). ",
  "hint.uncertainMany": "The model is unsure about {n} squares (highlighted in yellow). ",
  "hint.clickToFix": "Click a square to correct the piece.",
  "side.label": "Side to move",
  "side.white": "White to move",
  "side.black": "Black to move",
  flip: "Flip board",
  "castling.legend": "Castling still possible",
  "castling.K": "White short",
  "castling.Q": "White long",
  "castling.k": "Black short",
  "castling.q": "Black long",
  "analysis.thinking": "Stockfish is thinking",
  "analysis.depth": "depth {n}",
  "analysis.done": "Stockfish, depth {n}",
  copy: "Copy",
  copied: "Copied",
  lichess: "Open on Lichess",
  "teach.save": "Save for training",
  "teach.saved": "Saved for training",
  "teach.savedCount": "Saved. Training examples: {n}.",
  "teach.hint": "When the position on the board is correct, save it. With 10–20 such photos the model can be fine-tuned on your pieces.",
  "placeholder.busy": "Recognizing pieces…",
  "placeholder.idle": "The position will appear here.",
  "board.label": "Recognized position",
  "board.empty": "empty",
  "piece.K": "white king", "piece.Q": "white queen", "piece.R": "white rook", "piece.B": "white bishop", "piece.N": "white knight", "piece.P": "white pawn",
  "piece.k": "black king", "piece.q": "black queen", "piece.r": "black rook", "piece.b": "black bishop", "piece.n": "black knight", "piece.p": "black pawn",
  "err.whiteKings": "There must be exactly one white king (now: {n}).",
  "err.blackKings": "There must be exactly one black king (now: {n}).",
  "err.pawnBackRank": "A pawn cannot stand on {square}.",
  "err.kingsAdjacent": "The kings stand on adjacent squares.",
  "err.blackInCheck": "Black is in check, but it is White's move.",
  "err.whiteInCheck": "White is in check, but it is Black's move.",
  "server.board_not_found": "Board not found. Try a photo from a different angle or mark the corners manually.",
  "server.not_an_image": "The file is not an image (JPEG, PNG and WebP are supported).",
  "server.image_too_large": "The image is too large (max 15 MB).",
  "server.bad_corners": "Invalid board corners.",
  "server.bad_side_to_move": "Invalid side to move.",
  "server.bad_position": "Invalid position.",
  "server.illegal_position": "The position is not legal; fix it before saving.",
  "api.serverError": "The server returned error {status}.",
  "api.imagePrepare": "The image could not be prepared.",
  "engine.start": "The in-browser engine could not be started.",
  "engine.timeout": "The in-browser engine is not responding.",
  "engine.load": "The in-browser engine could not be loaded (check your internet connection). You can analyze the position on Lichess.",
};

const DICTS: Record<Lang, Record<Key, string>> = { sr, en };

// The app always opens in English; choosing Serbian lasts until it is closed.
let current: Lang = "en";
const listeners = new Set<() => void>();

function applyToDocument() {
  document.documentElement.lang = current === "sr" ? "sr-Latn" : "en";
  document.title = DICTS[current].title;
}
applyToDocument();

export function setLang(lang: Lang) {
  if (lang === current) return;
  current = lang;
  applyToDocument();
  listeners.forEach((l) => l());
}

/** Translation in the current language; {name} in the text is replaced from `vars`. */
export function tr(key: Key, vars: Record<string, string | number> = {}): string {
  const text = DICTS[current][key] ?? sr[key] ?? key;
  return text.replace(/\{(\w+)\}/g, (m, name: string) => (name in vars ? String(vars[name]) : m));
}

/** React: re-render the component when the language changes. */
export function useLang(): Lang {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
  );
}

export type { Key as MessageKey };
