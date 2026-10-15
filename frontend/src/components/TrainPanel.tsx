import { useEffect, useState } from "react";
import { trainingAction, trainingStatus, type TrainingStatus } from "../lib/api";
import { tr } from "../lib/i18n";

const pct = (x: number) => `${(x * 100).toFixed(1)} %`;

/** Fine-tunes the model on the saved photos (only where the server can train). */
export function TrainPanel({ savedCount }: { savedCount: number }) {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<"accepted" | "restored" | null>(null);

  useEffect(() => {
    trainingStatus().then(setStatus).catch(() => undefined);
  }, [savedCount]);

  const running = status?.state === "running";
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => trainingStatus().then(setStatus).catch(() => undefined), 3000);
    return () => clearInterval(id);
  }, [running]);

  if (!status?.available) return null;

  const act = async (action: "start" | "accept" | "discard" | "restore") => {
    setError(null);
    try {
      setStatus(await trainingAction(action));
      setNotice(action === "accept" ? "accepted" : action === "restore" ? "restored" : null);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const { progress, result } = status;
  if (!status.local) {
    return (
      <div className="train">
        <p className="hint">{status.state === "running" ? tr("train.runningElsewhere") : tr("train.onComputer", { n: status.photos })}</p>
      </div>
    );
  }
  return (
    <div className="train">
      {status.state === "running" ? (
        <p className="hint" role="status">
          {progress.step === "train" && progress.epoch
            ? tr("train.epoch", { n: progress.epoch, total: progress.epochs ?? 0 })
            : tr(progress.step === "evaluate" ? "train.evaluating" : "train.preparing")}
        </p>
      ) : status.state === "done" && result ? (
        <>
          <table className="train-result">
            <thead>
              <tr><th /><th>{tr("train.current")}</th><th>{tr("train.new")}</th></tr>
            </thead>
            <tbody>
              <tr><th>{tr("train.own", { n: result.held_out })}</th><td>{pct(result.own.before)}</td><td>{pct(result.own.after)}</td></tr>
              <tr><th>{tr("train.others")}</th><td>{pct(result.others.before)}</td><td>{pct(result.others.after)}</td></tr>
            </tbody>
          </table>
          <button className="btn primary" onClick={() => act("accept")}>{tr("train.accept")}</button>
          <button className="btn" onClick={() => act("discard")}>{tr("train.discard")}</button>
        </>
      ) : (
        <>
          <button className="btn" disabled={status.photos < status.min_photos} onClick={() => act("start")}>
            {tr("train.start", { n: status.photos })}
          </button>
          {status.has_previous && <button className="btn" onClick={() => act("restore")}>{tr("train.restore")}</button>}
          <p className={status.state === "failed" ? "error" : "hint"}>
            {status.state === "failed"
              ? tr("train.failed", { detail: status.error ?? "" })
              : notice
                ? tr(notice === "accepted" ? "train.accepted" : "train.restored")
                : status.photos < status.min_photos
                  ? tr("train.needMore", { n: status.min_photos })
                  : tr("train.hint")}
          </p>
        </>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
