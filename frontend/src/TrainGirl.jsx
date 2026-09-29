import { useCallback, useEffect, useState } from "react";
import {
  authedUrl,
  cancelTraining,
  createTraining,
  deleteTrainImage,
  deleteTraining,
  getTrainStatus,
  getTraining,
  listTrainHeroes,
  startTraining,
} from "./api";

const LABELS = {
  dataset_queued: "Waiting to build shots",
  dataset_building: "Building shots",
  review: "Review shots",
  train_queued: "Waiting to train",
  training: "Training",
  done: "Ready",
  failed: "Failed",
  cancelled: "Cancelled",
};
const OPEN = ["dataset_queued", "dataset_building", "review", "train_queued", "training"];
const BUSY = ["dataset_building", "training"];

function hours(min) {
  if (!min) return "";
  return min >= 60 ? `~${Math.floor(min / 60)}h ${min % 60}m` : `~${min}m`;
}

export default function TrainGirl() {
  const [worker, setWorker] = useState(null);
  const [trainings, setTrainings] = useState([]);
  const [minImages, setMinImages] = useState(10);
  const [detail, setDetail] = useState(null);
  const [heroes, setHeroes] = useState([]);
  const [heroId, setHeroId] = useState("");
  const [name, setName] = useState("");
  const [autoStart, setAutoStart] = useState(true);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState(false);

  const openTraining = trainings.find((t) => OPEN.includes(t.status));

  const refresh = useCallback(async () => {
    try {
      const data = await getTrainStatus();
      setWorker(data.worker);
      setMinImages(data.min_images || 10);
      const list = Array.isArray(data.trainings) ? data.trainings : [];
      setTrainings(list);
      const open = list.find((t) => OPEN.includes(t.status));
      setDetail(open && open.status === "review" ? await getTraining(open.id) : null);
    } catch (err) {
      setStatus(String(err.message || err));
      setError(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    listTrainHeroes()
      .then((d) => setHeroes(Array.isArray(d.items) ? d.items : []))
      .catch(() => {});
    const t = setInterval(refresh, 10000);
    return () => clearInterval(t);
  }, [refresh]);

  async function run(fn, okMsg) {
    setBusy(true);
    setError(false);
    try {
      await fn();
      if (okMsg) setStatus(okMsg);
      await refresh();
    } catch (err) {
      setStatus(String(err.message || err));
      setError(true);
    } finally {
      setBusy(false);
    }
  }

  function onCreate() {
    if (!name.trim() || !heroId) {
      setStatus("Give her a name and tap a hero photo.");
      setError(true);
      return;
    }
    run(async () => {
      await createTraining(name.trim(), heroId, autoStart);
      setName("");
      setHeroId("");
    }, autoStart ? "Queued — shots, then training, run on their own." : "Queued — the trainer PC will build her shots.");
  }

  return (
    <section className="section refs-section train-section">
      <div className="section-head">
        <h2 className="section-title">Train a girl</h2>
        <span className={`train-worker${worker?.online ? " online" : ""}`}>
          {worker?.online ? "Trainer PC online" : "Trainer PC offline"}
        </span>
      </div>
      <p className="refs-lead">
        Make her in Generation → Text to Image, then pick that photo here. The trainer
        PC builds ~24 shots of her into the cloud, you remove bad ones, then it trains
        her face so Text to Image can make her again and again. Nothing is kept on the PC.
      </p>

      {!openTraining && (
        <div className="train-new">
          <label className="t2i-field">
            <span>Her name</span>
            <div className="t2i-seed">
              <input
                type="text"
                maxLength={16}
                placeholder="e.g. Zara"
                value={name}
                onChange={(e) => setName(e.target.value.replace(/[^A-Za-z0-9]/g, ""))}
              />
            </div>
          </label>
          <div className="t2i-field">
            <span>Hero photo — clear face, good light</span>
            {heroes.length === 0 ? (
              <p className="muted">No Text to Image results yet — make her first.</p>
            ) : (
              <div className="gallery train-heroes">
                {heroes.map((h) => (
                  <button
                    key={h.id}
                    type="button"
                    className={`gallery-item train-hero${heroId === h.id ? " picked" : ""}`}
                    title={h.prompt || ""}
                    onClick={() => setHeroId(h.id)}
                  >
                    <img src={authedUrl(h.thumb_url)} alt="" loading="lazy" />
                  </button>
                ))}
              </div>
            )}
          </div>
          <label className="train-auto">
            <input type="checkbox" checked={autoStart} onChange={(e) => setAutoStart(e.target.checked)} />
            <span>Start training automatically (skip review; broken shots are dropped)</span>
          </label>
          <button type="button" className="go" disabled={busy || !worker?.online} onClick={onCreate}>
            <span className="go-main">Build her shots</span>
            <span className="go-sub">
              {worker?.online ? "~24 photos · 15–30 min on the GPU" : "Turn on the GPU PC first"}
            </span>
          </button>
        </div>
      )}

      {trainings.map((t) => {
        const pct =
          t.progress?.total > 0 ? Math.min(100, Math.round((100 * t.progress.done) / t.progress.total)) : 0;
        const shots = detail && detail.id === t.id ? detail.images : null;
        return (
          <div key={t.id} className={`train-card status-${t.status}`}>
            <div className="train-card-head">
              <strong>{t.name}</strong>
              <span className="train-badge">{LABELS[t.status] || t.status}</span>
            </div>
            {(OPEN.includes(t.status) || t.status === "done") && t.progress?.message && (
              <p className="setting-hint">{t.progress.message}</p>
            )}
            {BUSY.includes(t.status) && t.progress?.total > 0 && (
              <div className="train-bar" aria-label={`${pct}%`}>
                <span style={{ width: `${pct}%` }} />
              </div>
            )}
            {t.error && <p className="status error">{t.error}</p>}
            {t.warnings?.length > 0 && t.status === "review" && (
              <p className="setting-hint">Skipped: {t.warnings.join("; ")}</p>
            )}

            {shots && (
              <>
                <p className="setting-hint">
                  {shots.length} shots · remove any where her face looks different. Need at least {minImages}.
                </p>
                <div className="gallery ref-gallery">
                  {shots.map((img) => (
                    <div key={img.id} className="ref-card">
                      <a className="gallery-item" href={authedUrl(img.media_url)} target="_blank" rel="noreferrer">
                        <img src={authedUrl(img.thumb_url)} alt={img.caption || ""} loading="lazy" />
                      </a>
                      <div className="ref-card-actions">
                        <button
                          type="button"
                          className="ghost ref-del"
                          disabled={busy}
                          onClick={() => run(() => deleteTrainImage(img.id), "Shot removed.")}
                        >
                          Remove
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  type="button"
                  className="go"
                  disabled={busy || shots.length < minImages}
                  onClick={() => {
                    if (
                      window.confirm(
                        `Train ${t.name} now? Takes ${hours(t.train_minutes_estimate)}. New app gens wait in the queue until it finishes.`
                      )
                    )
                      run(() => startTraining(t.id), "Training queued.");
                  }}
                >
                  <span className="go-main">Start training</span>
                  <span className="go-sub">
                    {hours(t.train_minutes_estimate)} · app queue pauses meanwhile
                  </span>
                </button>
              </>
            )}

            <div className="ref-card-actions">
              {OPEN.includes(t.status) && (
                <button
                  type="button"
                  className="ghost ref-del"
                  disabled={busy || t.cancel_requested}
                  onClick={() => {
                    if (window.confirm(`Cancel ${t.name}?`)) run(() => cancelTraining(t.id), "Cancelling…");
                  }}
                >
                  {t.cancel_requested ? "Cancelling…" : "Cancel"}
                </button>
              )}
              {!OPEN.includes(t.status) && (
                <button
                  type="button"
                  className="ghost ref-del"
                  disabled={busy}
                  onClick={() => {
                    const msg =
                      t.status === "done"
                        ? `Delete ${t.name}'s training shots from the cloud? She stays usable in Text to Image.`
                        : `Delete this ${LABELS[t.status]?.toLowerCase()} training?`;
                    if (window.confirm(msg)) run(() => deleteTraining(t.id), "Deleted.");
                  }}
                >
                  {t.status === "done" ? "Delete shots" : "Delete"}
                </button>
              )}
            </div>
          </div>
        );
      })}

      {status ? (
        <p className={`status${error ? " error" : ""}`} role="status">
          {status}
        </p>
      ) : null}
    </section>
  );
}
