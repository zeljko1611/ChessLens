import type { Board, Side } from "./chess";
import { tr, type MessageKey } from "./i18n";

export interface Recognition {
  fen: string;
  placement: string;
  board: Board;
  side_to_move: Side;
  side_to_move_hint: Side | null;
  orientation: number;
  corners: [number, number][]; // a8, h8, h1, a1 in pixels of the uploaded image
  grid: [number, number][][]; // 9x9 line intersections
  square_centers: Record<string, [number, number]>;
  confidence: Record<string, number>;
  probabilities: Record<string, Record<string, number>>;
  uncertain: string[];
  repairs: { square: string; from: string; to: string; reason: string }[];
  errors: string[];
  detection_score: number;
  model: string;
  image_size: [number, number];
  timings_ms: Record<string, number>;
}

export interface Health {
  status: string;
  model: string;
  phone_url: string | null; // only when the server runs on the home network
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = tr("api.serverError", { status: res.status });
    try {
      const body = await res.json();
      // The server returns an error code (e.g. "board_not_found"); translate it if known.
      if (typeof body.detail === "string") {
        const key = `server.${body.detail}` as MessageKey;
        detail = tr(key) === key ? body.detail : tr(key);
      }
    } catch {
      /* the body is not JSON */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function health(): Promise<Health> {
  return asJson(await fetch("/api/health"));
}

// Id of this tab: the server returns it with the latest recognition so a tab does not
// reload its own photo. (crypto.randomUUID is unavailable over plain http on phones.)
export const CLIENT_ID = Math.random().toString(36).slice(2);

export async function recognize(
  image: Blob,
  opts: { corners?: [number, number][]; orientation?: number } = {},
): Promise<Recognition> {
  const form = new FormData();
  form.append("image", image, "board.jpg");
  form.append("client", CLIENT_ID);
  if (opts.corners) form.append("corners", JSON.stringify(opts.corners));
  if (opts.orientation !== undefined) form.append("orientation", String(opts.orientation));
  return asJson(await fetch("/api/recognize", { method: "POST", body: form }));
}

/** Save a photo with its corrected position as an example for fine-tuning the model. */
export async function saveFeedback(image: Blob, corners: [number, number][], placement: string): Promise<{ saved: string; count: number }> {
  const form = new FormData();
  form.append("image", image, "board.jpg");
  form.append("corners", JSON.stringify(corners));
  form.append("placement", placement);
  return asJson(await fetch("/api/feedback", { method: "POST", body: form }));
}

export interface TrainingStatus {
  available: boolean; // PyTorch and the original training data are installed
  local: boolean; // training can be started only on the computer running the server
  photos: number;
  min_photos: number;
  state: "idle" | "running" | "done" | "failed";
  progress: { step?: "start" | "crops" | "train" | "evaluate"; epoch?: number; epochs?: number };
  // Square accuracy before and after: on held-out saved photos and on the original validation data.
  result: { photos: number; held_out: number; own: { before: number; after: number }; others: { before: number; after: number } } | null;
  error: string | null;
  has_previous: boolean; // a model replaced by fine-tuning is kept in models/previous/
}

export async function trainingStatus(): Promise<TrainingStatus> {
  return asJson(await fetch("/api/training"));
}

export async function trainingAction(action: "start" | "accept" | "discard" | "restore"): Promise<TrainingStatus> {
  return asJson(await fetch(`/api/training/${action}`, { method: "POST" }));
}

export interface Latest {
  id: number;
  client?: string | null;
  result?: Recognition; // only if there is something newer than `since`
}

/** The latest recognition from any device (e.g. a photo taken on a phone). */
export async function latest(since: number): Promise<Latest> {
  return asJson(await fetch(`/api/latest?since=${since}`));
}

export async function latestImage(): Promise<Blob> {
  const res = await fetch("/api/latest/image", { cache: "no-store" });
  if (!res.ok) throw new Error(tr("api.serverError", { status: res.status }));
  return res.blob();
}

/**
 * Downscale the photo before upload (phones take 12+ MP images) and bake in the
 * EXIF rotation, so coordinates in the browser and on the server are the same.
 */
export async function prepareImage(file: Blob, maxSide = 1600): Promise<{ blob: Blob; url: string; width: number; height: number }> {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.round(bitmap.width * scale);
  const height = Math.round(bitmap.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error(tr("api.imagePrepare")))), "image/jpeg", 0.9),
  );
  return { blob, url: URL.createObjectURL(blob), width, height };
}
