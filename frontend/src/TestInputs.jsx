import { useCallback, useEffect, useRef, useState } from "react";
import {
  deleteTestInput,
  getTestSummary,
  listTestInputs,
  testInputThumbUrl,
  uploadTestInput,
} from "./api";

export default function TestInputs({ onOpenLightbox, onUseInput, onUseBackground }) {
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState("");
  const [error, setError] = useState(false);
  const [busy, setBusy] = useState(false);
  const inputRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const [data, summary] = await Promise.all([
        listTestInputs(),
        getTestSummary().catch(() => null),
      ]);
      setItems(Array.isArray(data.items) ? data.items : []);
      setTotal(
        Number(summary?.inputs ?? data.total ?? data.items?.length ?? 0)
      );
    } catch (err) {
      setStatus(String(err.message || err));
      setError(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function onUpload(files) {
    if (!files?.length) return;
    setBusy(true);
    setError(false);
    try {
      let n = 0;
      for (const file of files) {
        if (!file.type.startsWith("image/")) continue;
        await uploadTestInput(file);
        n += 1;
      }
      setStatus(
        n
          ? `Saved ${n} start photo${n === 1 ? "" : "s"}.`
          : "No images selected."
      );
      await refresh();
    } catch (err) {
      setStatus(String(err.message || err));
      setError(true);
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function onDelete(id) {
    if (!id) return;
    if (!window.confirm("Remove this start photo?")) return;
    setBusy(true);
    try {
      await deleteTestInput(id);
      setStatus("Start photo removed.");
      setError(false);
      await refresh();
    } catch (err) {
      setStatus(String(err.message || err));
      setError(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="section refs-section">
      <div className="section-head">
        <h2 className="section-title">Input images</h2>
        <span className="setting-hint">
          {total} stored · reuse as the start photo
        </span>
      </div>
      <p className="refs-lead">
        Keep the photos you generate from here. Tap Use to load one into
        Generation — nothing extra is left on this phone.
      </p>

      <label className="drop ref-drop">
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          multiple
          hidden
          disabled={busy}
          onChange={(e) => onUpload(e.target.files)}
        />
        <div className="drop-inner">
          <span className="drop-title">
            {busy ? "Uploading…" : "Tap to add start photos"}
          </span>
          <span className="drop-sub">
            Stored in this PIN’s cloud library — not on this PC
          </span>
        </div>
      </label>

      {items.length === 0 ? (
        <p className="muted">No start photos stored yet.</p>
      ) : (
        <div className="gallery ref-gallery">
          {items.map((item) => (
            <div key={item.id} className="ref-card">
              <button
                type="button"
                className="gallery-item"
                onClick={() =>
                  onOpenLightbox?.({ ...item, _testInput: true, kind: "img" })
                }
              >
                <img src={testInputThumbUrl(item)} alt="" loading="lazy" />
              </button>
              <div className="ref-card-actions">
                <button
                  type="button"
                  className="ghost"
                  disabled={busy}
                  onClick={() =>
                    onUseInput?.({ ...item, _testInput: true, kind: "img" })
                  }
                >
                  Use
                </button>
                <button
                  type="button"
                  className="ghost"
                  disabled={busy}
                  title="Put a girl into this photo (Text to Image)"
                  onClick={() =>
                    onUseBackground?.({ ...item, _testInput: true, kind: "img" })
                  }
                >
                  Background
                </button>
                <button
                  type="button"
                  className="ghost ref-del"
                  disabled={busy}
                  onClick={() => onDelete(item.id)}
                >
                  Remove
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {status ? (
        <p className={`status${error ? " error" : ""}`} role="status">
          {status}
        </p>
      ) : null}
    </section>
  );
}
