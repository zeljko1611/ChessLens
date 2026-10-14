import qrcode from "qrcode-generator";
import { useState } from "react";
import { tr } from "../lib/i18n";

/** A button that shows a QR code and the address for opening the app on a phone. */
export function PhoneLink({ url }: { url: string }) {
  const [open, setOpen] = useState(false);
  const qr = qrcode(0, "M");
  qr.addData(url);
  qr.make();
  const n = qr.getModuleCount();
  const cells = [];
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) if (qr.isDark(r, c)) cells.push(`M${c} ${r}h1v1h-1z`);

  return (
    <div className="phone">
      <button className="btn" aria-expanded={open} onClick={() => setOpen(!open)}>{tr("phone.button")}</button>
      {open && (
        <div className="phone-panel" role="dialog" aria-label={tr("phone.button")}>
          <svg viewBox={`-2 -2 ${n + 4} ${n + 4}`} className="phone-qr" role="img" aria-label={url}>
            <rect x={-2} y={-2} width={n + 4} height={n + 4} fill="#fff" />
            <path d={cells.join("")} fill="#000" />
          </svg>
          <p className="phone-url">{url}</p>
          <p className="hint">{tr("phone.hint")}</p>
        </div>
      )}
    </div>
  );
}
