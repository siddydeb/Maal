import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelJob,
  deleteGeneration,
  downloadGeneration,
  fetchMediaAsFile,
  getJob,
  getSagaConfig,
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
const QUALITIES = [
  ["hd", "HD"],
  ["fast", "Fast"],
];
const LAYOUTS = [
  ["scene", "Scene"],
  ["sprite", "Game sprite"],
  ["part", "Part"],
];
const STRENGTHS = [
  ["light", "Light"],
  ["medium", "Medium"],
  ["strong", "Strong"],
];
const VIDEO_QUALITIES = [
  ["smooth", "Smooth"],
  ["hd", "720p (slow)"],
];
const IDEAS = [
  "a 40 year old blonde woman in an apron, standing in a sunny suburban kitchen, hand on hip, warm smile",
  "a 24 year old woman with long black hair in a red bikini on a sunny beach, waving at the viewer",
  "a 30 year old librarian with glasses and a bun, leaning on the library desk, playful smile",
  "a 27 year old redhead in black lingerie sitting on a bed at night, bedside lamp, looking at the viewer",
];
const CHAR_IDEAS = [
  "standing in a sunny suburban kitchen, hand on hip, warm smile, looking at viewer",
  "sitting on a living room couch at night, relaxed smile, lamp light",
  "on a sunny beach in a swimsuit, waving at the viewer",
];
const BACKGROUND_IDEAS = ["suburban kitchen, morning light", "living room at night", "school hallway", "sunny beach"];

function Segment({ label, options, value, onChange }) {
  return (
    <div className="seg" role="tablist" aria-label={label}>
      {options.map(([key, text]) => (
        <button
          key={key}
          type="button"
          className={`seg-btn${value === key ? " active" : ""}`}
          onClick={() => onChange(key)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
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
  const [videoQuality, setVideoQuality] = useState("smooth");
  const [characters, setCharacters] = useState([]);
  const [character, setCharacter] = useState("");
  const [outfit, setOutfit] = useState(true);
  const [quality, setQuality] = useState("hd");
  const [layout, setLayout] = useState("scene");
  const [background, setBackground] = useState("");
  const [photo, setPhoto] = useState(null);
  const [photoUrl, setPhotoUrl] = useState("");
  const [strength, setStrength] = useState("medium");
  const pollRef = useRef(null);
  const photoInput = useRef(null);
  const sprite = layout === "sprite";
  const part = layout === "part";
  const picked = characters.find((c) => c.id === character);
  const ideas = character ? CHAR_IDEAS : IDEAS;

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
    getSagaConfig()
      .then((cfg) => setCharacters(cfg.characters || []))
      .catch(() => {});
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

  const viewerOpen = Boolean(viewer);
  useEffect(() => {
    if (!viewerOpen) return undefined;
    window.history.pushState({ sagaViewer: true }, "");
    const close = () => setViewer(null);
    const onKey = (e) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("popstate", close);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("popstate", close);
      window.removeEventListener("keydown", onKey);
      if (window.history.state?.sagaViewer) window.history.back();
    };
  }, [viewerOpen]);

  useEffect(() => {
    if (!photo) {
      setPhotoUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(photo);
    setPhotoUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [photo]);

  function onPickPhoto(e) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) setPhoto(file);
  }

  const canGenerate = part ? Boolean(photo) : Boolean(prompt.trim()) || (photo && !sprite);

  function onPickLayout(next) {
    setLayout(next);
    if (next === "part") setStrength("light");
  }

  async function onGenerate(e) {
    e.preventDefault();
    const text = prompt.trim();
    if (!canGenerate) return;
    if (sprite && !background.trim() && !photo) {
      setError("Game sprite mode needs a background — say where she stands, or add a background photo.");
      return;
    }
    const saga = {
      aspect,
      character: character || undefined,
      outfit: character ? outfit : undefined,
      quality,
      layout,
      background: sprite && background.trim() ? background.trim() : undefined,
      strength: photo ? strength : undefined,
    };
    const base = text || (part ? "Part in Saga style" : "Photo in Saga style");
    const label = picked ? `${picked.label}: ${base}` : base;
    await enqueue({ mode: "saga", prompt: text, saga, file: photo || undefined }, label);
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
    const ok = await enqueue(
      { mode: "saga_vid", prompt: text, file, videoSeconds: seconds, saga: { quality: videoQuality } },
      `Animate: ${text}`
    );
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
          <div className="app-header-actions saga-header-actions">
            <button type="button" className="ghost" onClick={lock} title="Log out / switch PIN">
              Log out
            </button>
          </div>
        </header>

        <section className="panel view-panel">
          <form className="form" onSubmit={onGenerate}>
            {characters.length > 0 && (
              <div className="section">
                <label className="t2i-field">
                  <span>Character</span>
                  <select value={character} onChange={(e) => setCharacter(e.target.value)}>
                    <option value="">Original (describe her)</option>
                    {characters.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.label}
                      </option>
                    ))}
                  </select>
                </label>
                {picked && (
                  <label className="train-auto">
                    <input type="checkbox" checked={outfit} onChange={(e) => setOutfit(e.target.checked)} />
                    <span>Her usual outfit ({picked.outfit})</span>
                  </label>
                )}
              </div>
            )}

            <div className="section">
              <div className="section-head">
                <h2 className="section-title">
                  {part ? "Part to restyle (face, hand, body…)" : "Start from a photo (optional)"}
                </h2>
              </div>
              {part && !photo && (
                <p className="setting-hint">
                  Transparent PNGs come back transparent with the exact same outline, and the result has the same
                  pixel size as your file, so the layers line up.
                </p>
              )}
              <input ref={photoInput} type="file" accept="image/*" hidden onChange={onPickPhoto} />
              {photo ? (
                <div className="saga-photo">
                  <img className="saga-photo-preview saga-checker" src={photoUrl} alt="" />
                  <div className="saga-photo-side">
                    <p className="setting-hint">
                      {part
                        ? "This part gets redrawn in the game style, same size and outline as your file."
                        : sprite
                          ? "This photo becomes the game background she stands in."
                          : "This photo gets redrawn in the Summertime Saga style."}
                    </p>
                    <Segment label="Strength" options={STRENGTHS} value={strength} onChange={setStrength} />
                    <p className="setting-hint">
                      {part
                        ? "Light keeps the shape closest (fingers, pose) · Strong changes more. Saying what the part is helps a lot."
                        : "Light keeps more of the photo · Strong looks more like the game. A short description below gives a much stronger game look."}
                    </p>
                    <button type="button" className="ghost" onClick={() => setPhoto(null)}>
                      {part ? "remove image" : "remove photo"}
                    </button>
                  </div>
                </div>
              ) : (
                <button type="button" className="ghost" onClick={() => photoInput.current?.click()}>
                  {part ? "Add part image" : "Add photo"}
                </button>
              )}
            </div>

            <div className="section">
              <div className="section-head">
                <h2 className="section-title">
                  {part
                    ? "What is this part? (optional)"
                    : photo && !sprite
                    ? "Describe the photo (optional)"
                    : picked
                      ? `What is ${picked.label} doing?`
                      : "Describe the scene"}
                </h2>
                <span className="char-count">
                  {prompt.length}/{PROMPT_MAX}
                </span>
              </div>
              <textarea
                className="prompt-area"
                rows={4}
                maxLength={PROMPT_MAX}
                placeholder={
                  part
                    ? "e.g. left hand, open palm, red nail polish · or: woman's face, smiling, brown eyes, long dark hair"
                    : photo && !sprite
                    ? "Optional but recommended: say what's in it, e.g. woman with long dark hair laughing, red turtleneck, city street…"
                    : picked
                      ? "e.g. standing in the kitchen holding a coffee mug, warm smile, morning light…"
                      : "e.g. a 35 year old woman with short red hair in a green sweater, reading on a park bench in autumn, smiling at the viewer…"
                }
                value={prompt}
                onChange={(e) => setPrompt(e.target.value.slice(0, PROMPT_MAX))}
              />
              <div className="saga-ideas" hidden={part}>
                {ideas.map((idea) => (
                  <button key={idea} type="button" className="ghost saga-idea" onClick={() => setPrompt(idea)}>
                    {idea.split(",")[0]}
                  </button>
                ))}
              </div>
            </div>

            {sprite && (
              <div className="section">
                <h2 className="section-title">
                  {photo ? "Background notes (optional)" : "Background (where she stands)"}
                </h2>
                <input
                  className="saga-bg-input"
                  type="text"
                  maxLength={200}
                  placeholder="e.g. suburban kitchen, morning light"
                  value={background}
                  onChange={(e) => setBackground(e.target.value)}
                />
                <div className="saga-ideas">
                  {BACKGROUND_IDEAS.map((idea) => (
                    <button key={idea} type="button" className="ghost saga-idea" onClick={() => setBackground(idea)}>
                      {idea}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="section">
              {!photo && (
                <div className="setting-row">
                  <div className="setting-label">
                    <span>Shape</span>
                    <span className="setting-hint">Landscape matches the game screen</span>
                  </div>
                  <Segment label="Shape" options={ASPECTS} value={aspect} onChange={setAspect} />
                </div>
              )}
              <div className="setting-row">
                <div className="setting-label">
                  <span>Quality</span>
                  <span className="setting-hint">HD: sharper, bigger, faces and hands cleaned up</span>
                </div>
                <Segment label="Quality" options={QUALITIES} value={quality} onChange={setQuality} />
              </div>
              <div className="setting-row">
                <div className="setting-label">
                  <span>Layout</span>
                  <span className="setting-hint">
                    Game sprite: cut-out character on a game background · Part: restyle one piece (face, hand…)
                  </span>
                </div>
                <Segment label="Layout" options={LAYOUTS} value={layout} onChange={onPickLayout} />
              </div>
            </div>

            <button type="submit" className="go" disabled={busy || !canGenerate}>
              <span className="go-main">
                <span>
                  {busy ? "Queuing…" : part ? "Restyle part" : photo && !sprite ? "Restyle photo" : "Generate image"}
                </span>
                {busy && <span className="spinner" />}
              </span>
              <span className="go-sub">
                {quality === "hd" ? "about 1–2 minutes" : "about 30 seconds"} · tap any image to animate it
              </span>
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
                <img
                  className="lightbox-media saga-checker"
                  src={mediaUrl(viewer)}
                  alt={viewer.prompt || ""}
                  draggable={false}
                />
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
                  <Segment
                    label="Length"
                    options={(videoQuality === "hd" ? [3, 5] : [3, 5, 8, 10]).map((s) => [s, `${s}s`])}
                    value={seconds}
                    onChange={setSeconds}
                  />
                  <Segment
                    label="Video quality"
                    options={VIDEO_QUALITIES}
                    value={videoQuality}
                    onChange={(q) => {
                      setVideoQuality(q);
                      if (q === "hd" && seconds > 5) setSeconds(5);
                    }}
                  />
                  <button type="button" className="action" disabled={busy} onClick={onAnimate}>
                    {busy ? "Queuing…" : "Animate"}
                  </button>
                </div>
                <p className="setting-hint">
                  {videoQuality === "hd"
                    ? "720p, 32 fps · about 15–20 minutes"
                    : `32 fps${seconds > 5 ? " · two chained clips" : ""} · about ${seconds > 5 ? "8–11" : "4–6"} minutes`}
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
