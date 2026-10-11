import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelJob,
  deleteGeneration,
  downloadGeneration,
  fetchMediaAsFile,
  getJob,
  listActiveJobs,
  listGenerations,
  mediaUrl,
  startJob,
  thumbUrl,
  wakeApi,
} from "./api";
import { clearSession } from "./auth";

const PROMPT_MAX = 800;
const ASPECTS = [
  ["landscape", "Landscape"],
  ["portrait", "Portrait"],
  ["square", "Square"],
];
const IDEAS = [
  "a 40 year old blonde woman in an apron, standing in a sunny suburban kitchen, hand on hip, warm smile",
  "a 24 year old woman with long black hair in a red bikini on a sunny beach, waving at the viewer",
  "a 30 year old librarian with glasses and a bun, leaning on the library desk, playful smile",
  "a 27 year old redhead in black lingerie sitting on a bed at night, bedside lamp, looking at the viewer",
];
const isVideo = (item) => item?.kind === "vid" || (item?.content_type || "").startsWith("video/");

export default function SagaStudio() {
  const [prompt, setPrompt] = useState("");
  const [aspect, setAspect] = useState("landscape");
  const [jobs, setJobs] = useState([]);
  const [library, setLibrary] = useState([]);
  const [libraryTotal, setLibraryTotal] = useState(0);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [viewer, setViewer] = useState(null);
  const [motion, setMotion] = useState("");
  const [seconds, setSeconds] = useState(5);
  const pollRef = useRef(null);

  const refreshLibrary = useCallback(async () => {
    try {
      const data = await listGenerations({ limit: 60 });
      setLibrary(data.items);
      setLibraryTotal(data.total);
    } catch (err) {
      setError(String(err.message || err));
    }
  }, []);

  async function loadMore() {
    try {
      const data = await listGenerations({ limit: 60, skip: library.length });
      setLibrary((prev) => [...prev, ...data.items.filter((x) => !prev.some((p) => p.id === x.id))]);
      setLibraryTotal(data.total);
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  useEffect(() => {
    refreshLibrary();
    listActiveJobs()
      .then((data) => setJobs(data.items.map((j) => ({ id: j.id, mode: j.mode, status: j.status, prompt: j.prompt }))))
      .catch(() => {});
  }, [refreshLibrary]);

  useEffect(() => {
    const active = jobs.filter((j) => j.status === "queued" || j.status === "running");
    if (!active.length) return undefined;
    pollRef.current = window.setTimeout(async () => {
      let finished = false;
      const next = await Promise.all(
        jobs.map(async (j) => {
          if (j.status !== "queued" && j.status !== "running") return j;
          try {
            const fresh = await getJob(j.id);
            if (fresh.status === "done") finished = true;
            return { ...j, status: fresh.status, error: fresh.error, position: fresh.queue_position };
          } catch {
            return j;
          }
        })
      );
      setJobs(next.filter((j) => j.status !== "done" && j.status !== "cancelled"));
      if (finished) refreshLibrary();
    }, 3000);
    return () => window.clearTimeout(pollRef.current);
  }, [jobs, refreshLibrary]);

  async function enqueue(params, label) {
    setBusy(true);
    setError("");
    try {
      await wakeApi();
      const started = await startJob(params);
      setJobs((prev) => [...prev, { id: started.id, mode: params.mode, status: "queued", prompt: label }]);
      return true;
    } catch (err) {
      setError(String(err.message || err));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function onGenerate(e) {
    e.preventDefault();
    const text = prompt.trim();
    if (!text) return;
    await enqueue({ mode: "saga", prompt: text, aspect }, text);
  }

  async function onAnimate() {
    if (!viewer) return;
    const text = motion.trim() || "she smiles, blinks and shifts her weight, hair sways gently, the camera stays still";
    setBusy(true);
    let file;
    try {
      file = await fetchMediaAsFile(viewer);
    } catch (err) {
      setError(String(err.message || err));
      setBusy(false);
      return;
    }
    const ok = await enqueue({ mode: "saga_vid", prompt: text, file, videoSeconds: seconds }, `Animate: ${text}`);
    if (ok) {
      setViewer(null);
      setMotion("");
    }
  }

  async function onDelete(item) {
    if (!window.confirm("Delete this from the library?")) return;
    try {
      await deleteGeneration(item.id);
      setViewer(null);
      setLibrary((prev) => prev.filter((x) => x.id !== item.id));
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  async function onCancel(job) {
    try {
      await cancelJob(job.id);
      setJobs((prev) => prev.filter((j) => j.id !== job.id));
    } catch (err) {
      setError(String(err.message || err));
    }
  }

  function lock() {
    clearSession();
    window.location.reload();
  }

  return (
    <>
      <div className="noise" aria-hidden="true" />
      <main className="shell">
        <header className="app-header">
          <div className="app-brand">
            <span className="app-mark" aria-hidden="true">S</span>
            <span className="app-brand-text">
              <span className="app-name">Saga</span>
              <span className="app-tagline">Summertime Saga style studio</span>
            </span>
          </div>
          <div className="app-header-actions">
            <button type="button" className="ghost" onClick={lock}>
              Lock
            </button>
          </div>
        </header>

        <section className="panel view-panel">
          <form className="form" onSubmit={onGenerate}>
            <div className="section">
              <div className="section-head">
                <h2 className="section-title">Describe the scene</h2>
                <span className="char-count">
                  {prompt.length}/{PROMPT_MAX}
                </span>
              </div>
              <textarea
                className="prompt-area"
                rows={4}
                maxLength={PROMPT_MAX}
                placeholder="e.g. a 35 year old woman with short red hair in a green sweater, reading on a park bench in autumn, smiling at the viewer…"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value.slice(0, PROMPT_MAX))}
              />
              <div className="saga-ideas">
                {IDEAS.map((idea) => (
                  <button key={idea} type="button" className="ghost saga-idea" onClick={() => setPrompt(idea)}>
                    {idea.split(",")[0]}
                  </button>
                ))}
              </div>
            </div>

            <div className="section">
              <div className="setting-row">
                <div className="setting-label">
                  <span>Shape</span>
                  <span className="setting-hint">Landscape matches the game screen</span>
                </div>
                <div className="seg" role="tablist" aria-label="Shape">
                  {ASPECTS.map(([key, label]) => (
                    <button
                      key={key}
                      type="button"
                      className={`seg-btn${aspect === key ? " active" : ""}`}
                      onClick={() => setAspect(key)}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <button type="submit" className="go" disabled={busy || !prompt.trim()}>
              <span className="go-main">
                <span>{busy ? "Queuing…" : "Generate image"}</span>
                {busy && <span className="spinner" />}
              </span>
              <span className="go-sub">about 30 seconds · tap any image to animate it</span>
            </button>
            {error && <p className="status error">{error}</p>}
          </form>

          {jobs.length > 0 && (
            <ul className="saga-jobs">
              {jobs.map((j) => (
                <li key={j.id} className={`saga-job${j.status === "failed" ? " failed" : ""}`}>
                  <span className="saga-job-kind">{j.mode === "saga_vid" ? "Video" : "Image"}</span>
                  <span className="saga-job-text">
                    {j.status === "failed" ? j.error || "Failed" : j.prompt}
                  </span>
                  <span className="saga-job-status">
                    {j.status === "running" ? (
                      <span className="spinner" />
                    ) : j.status === "queued" ? (
                      "queued"
                    ) : null}
                  </span>
                  <button
                    type="button"
                    className="ghost"
                    onClick={() =>
                      j.status === "failed" ? setJobs((p) => p.filter((x) => x.id !== j.id)) : onCancel(j)
                    }
                  >
                    {j.status === "failed" ? "dismiss" : "cancel"}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="panel view-panel">
          <h2 className="section-title">Library{libraryTotal ? ` (${libraryTotal})` : ""}</h2>
          <ul className="gallery">
            {!library.length && <li className="empty gallery-empty">Nothing here yet.</li>}
            {library.map((item) => (
              <li key={item.id} className="gallery-item">
                <button type="button" className="gallery-cell" onClick={() => setViewer(item)}>
                  {isVideo(item) ? (
                    <span className="gallery-thumb gallery-thumb-video">
                      <img src={thumbUrl(item)} alt="" loading="lazy" decoding="async" />
                      <span className="gallery-play" aria-hidden="true">▶</span>
                    </span>
                  ) : (
                    <img className="gallery-thumb" src={thumbUrl(item)} alt="" loading="lazy" decoding="async" />
                  )}
                  <span className="gallery-kind">{item.kind}</span>
                </button>
              </li>
            ))}
          </ul>
          {library.length < libraryTotal && (
            <button type="button" className="ghost saga-more" onClick={loadMore}>
              Load more
            </button>
          )}
        </section>
      </main>

      {viewer && (
        <div className="lightbox" role="dialog" aria-modal="true" onClick={() => setViewer(null)}>
          <div className="lightbox-panel" onClick={(e) => e.stopPropagation()}>
            <div className="lightbox-toolbar">
              <button type="button" className="ghost" onClick={() => setViewer(null)}>
                close
              </button>
              <div className="lightbox-actions">
                <button type="button" className="action" onClick={() => downloadGeneration(viewer)}>
                  download
                </button>
                <button type="button" className="ghost danger" onClick={() => onDelete(viewer)}>
                  delete
                </button>
              </div>
            </div>
            <div className="lightbox-stage">
              {isVideo(viewer) ? (
                <video className="lightbox-media" src={mediaUrl(viewer)} controls playsInline autoPlay loop />
              ) : (
                <img className="lightbox-media" src={mediaUrl(viewer)} alt={viewer.prompt || ""} draggable={false} />
              )}
            </div>
            <p className="lightbox-caption">{viewer.prompt || "—"}</p>
            {!isVideo(viewer) && (
              <div className="saga-animate">
                <textarea
                  className="prompt-area"
                  rows={2}
                  maxLength={PROMPT_MAX}
                  placeholder="How should she move? e.g. she waves at the viewer and laughs, hair blowing in the breeze"
                  value={motion}
                  onChange={(e) => setMotion(e.target.value.slice(0, PROMPT_MAX))}
                />
                <div className="saga-animate-row">
                  <div className="seg" role="tablist" aria-label="Length">
                    {[3, 5].map((s) => (
                      <button
                        key={s}
                        type="button"
                        className={`seg-btn${seconds === s ? " active" : ""}`}
                        onClick={() => setSeconds(s)}
                      >
                        {s}s
                      </button>
                    ))}
                  </div>
                  <button type="button" className="action" disabled={busy} onClick={onAnimate}>
                    {busy ? "Queuing…" : "Animate"}
                  </button>
                </div>
                <p className="setting-hint">About 3–5 minutes on the GPU</p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
