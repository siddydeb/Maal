import { clearSession, getAccessToken, getOwnerId } from "./auth";

const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

function url(path) {
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

function authHeaders(extra = {}) {
  const token = getAccessToken();
  const headers = { ...extra };
  if (token) headers["X-Wan-Token"] = token;
  return headers;
}

function withAuthQuery(path) {
  const token = getAccessToken();
  if (!token) return path;
  const join = path.includes("?") ? "&" : "?";
  return `${path}${join}t=${encodeURIComponent(token)}`;
}

function detailMessage(data, status) {
  const d = data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  return data?.message || `HTTP ${status}`;
}

async function readJson(res) {
  const data = await res.json().catch(() => ({}));
  if (res.status === 401) {
    clearSession();
  }
  if (!res.ok) throw new Error(detailMessage(data, res.status));
  return data;
}

export async function getAuthStatus() {
  const res = await fetch(url("/api/auth/status"), { cache: "no-store" });
  return readJson(res);
}

export async function unlockAccount(pin) {
  const res = await fetch(url("/api/auth/unlock"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pin: String(pin || "").trim() }),
    cache: "no-store",
  });
  return readJson(res);
}

export async function getHealth() {
  const res = await fetch(url("/api/health"), { cache: "no-store" });
  return res.json();
}

export async function listPresets() {
  const res = await fetch(url("/api/presets"), {
    headers: authHeaders(),
    cache: "no-store",
  });
  return readJson(res);
}

export async function listReviewBins() {
  const res = await fetch(url("/api/test/review-bins"), {
    headers: authHeaders(),
    cache: "no-store",
  });
  return readJson(res);
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Cancelled"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new Error("Cancelled"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function isTransientNetworkError(err) {
  const msg = String(err?.message || err || "");
  return /fetch|network|Failed to fetch|timed?\s*out|TimeoutError|AbortError|networkerror/i.test(
    msg
  );
}

function isTransientComfyError(err) {
  const msg = String(err?.message || err || "");
  return /ComfyUI is not reachable|GPU (is )?offline|comfyui.*not reachable/i.test(
    msg
  );
}

/**
 * Ping health until Render wakes (free tier can sleep ~30–60s+).
 * Returns health JSON or throws after retries.
 */
export async function wakeApi({ onStatus, signal, attempts = 5 } = {}) {
  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    if (signal?.aborted) throw new Error("Cancelled");
    onStatus?.({
      status: "waking",
      message:
        i === 0
          ? "Waking API server (after idle this can take up to a minute)…"
          : `API still waking… retry ${i + 1}/${attempts}`,
    });
    const ac = new AbortController();
    const onOuterAbort = () => ac.abort();
    signal?.addEventListener("abort", onOuterAbort);
    const timer = setTimeout(() => ac.abort(), 55_000);
    try {
      const res = await fetch(url("/api/health"), {
        cache: "no-store",
        signal: ac.signal,
      });
      const data = await res.json().catch(() => ({}));
      if (res.ok && (data.ok === true || data.mongodb != null)) {
        return data;
      }
      lastErr = new Error(`Health HTTP ${res.status}`);
    } catch (err) {
      lastErr = err;
      if (signal?.aborted) throw new Error("Cancelled");
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onOuterAbort);
    }
    if (i < attempts - 1) {
      await sleep(2000, signal);
    }
  }
  throw new Error(
    lastErr && isTransientNetworkError(lastErr)
      ? "API server timed out waking up. Wait 30–60s and tap Generate again (Render free tier sleeps when idle)."
      : String(lastErr?.message || lastErr || "API unreachable")
  );
}

export async function listGenerations(limitOrOpts = 30, maybeSkip = 0) {
  const opts =
    typeof limitOrOpts === "object" && limitOrOpts != null
      ? limitOrOpts
      : { limit: limitOrOpts, skip: maybeSkip };
  const limit = Math.min(Math.max(1, Number(opts.limit) || 30), 100);
  const skip = Math.max(0, Number(opts.skip) || 0);
  const params = new URLSearchParams({
    limit: String(limit),
    skip: String(skip),
  });
  if (opts.test_run === 1 || opts.test_run === true) params.set("test_run", "1");
  if (opts.preset_id) params.set("preset_id", String(opts.preset_id));
  const res = await fetch(url(`/api/generations?${params}`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (res.status === 401) {
    clearSession();
    window.location.reload();
    throw new Error("Session expired");
  }
  if (!res.ok) throw new Error("Failed to load library");
  const data = await res.json();
  // Older API returned a bare array
  if (Array.isArray(data)) {
    return { items: data, total: data.length, limit, skip };
  }
  return {
    items: Array.isArray(data.items) ? data.items : [],
    total: Number(data.total) || 0,
    limit: Number(data.limit) || limit,
    skip: Number(data.skip) || skip,
  };
}

export async function getGeneration(id) {
  const res = await fetch(url(`/api/generations/${id}`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function updateGeneration(id, { prompt, meta } = {}) {
  const body = {};
  if (prompt != null) body.prompt = prompt;
  if (meta != null) body.meta = meta;
  const res = await fetch(url(`/api/generations/${id}`), {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(body),
  });
  return readJson(res);
}

export async function markGenerationOpened(id) {
  const res = await fetch(url(`/api/generations/${encodeURIComponent(id)}/opened`), {
    method: "POST",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function deleteGeneration(id) {
  const res = await fetch(url(`/api/generations/${id}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function generate({
  mode,
  prompt,
  file,
  negative,
  seed,
  videoSeconds,
  presetId,
  testRun,
  onStatus,
  signal,
}) {
  await wakeApi({ onStatus, signal });
  onStatus?.({ status: "queued", message: "API awake — submitting job…" });
  const started = await startJob({
    mode,
    prompt,
    file,
    negative,
    seed,
    videoSeconds,
    presetId,
    testRun,
    signal,
    onStatus,
  });
  // Persist immediately so a refresh mid-poll can resume
  if (started?.id) saveActiveJobId(started.id, { mode });
  return waitForJob(started.id, { onStatus, signal });
}

export async function startJob({
  mode,
  prompt,
  file,
  negative,
  seed,
  videoSeconds,
  presetId,
  testRun,
  signal,
  onStatus,
  attempts = 3,
}) {
  const buildBody = () => {
    const body = new FormData();
    body.append("mode", mode);
    body.append("prompt", prompt);
    body.append("image", file, file.name || "input.png");
    if (negative) body.append("negative", negative);
    if (seed != null && seed !== "") body.append("seed", String(seed));
    if (mode === "vid" && videoSeconds != null && videoSeconds !== "") {
      body.append("video_seconds", String(videoSeconds));
    }
    if (presetId) body.append("preset_id", String(presetId));
    if (testRun) body.append("test_run", "1");
    return body;
  };

  let lastErr = null;
  for (let i = 0; i < attempts; i++) {
    if (signal?.aborted) throw new Error("Cancelled");
    try {
      const res = await fetch(url("/api/jobs"), {
        method: "POST",
        body: buildBody(),
        headers: authHeaders(),
        signal,
      });
      const data = await readJson(res);
      if (data?.id) saveActiveJobId(data.id, { mode });
      return data;
    } catch (err) {
      lastErr = err;
      if (signal?.aborted) throw new Error("Cancelled");
      const retryable =
        isTransientNetworkError(err) || isTransientComfyError(err);
      if (!retryable) throw err;
      if (i < attempts - 1) {
        onStatus?.({
          status: "queued",
          message: "GPU blip — retrying…",
        });
        await sleep(1500 + i * 1000, signal);
      }
    }
  }
  throw new Error(
    isTransientNetworkError(lastErr)
      ? "Could not reach API to start the job (timeout). Wait a moment and try again — nothing was queued."
      : String(lastErr?.message || lastErr || "Failed to start job")
  );
}

export async function getJob(jobId) {
  const res = await fetch(url(`/api/jobs/${jobId}`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function listActiveJobs(limit = 30) {
  const res = await fetch(
    url(`/api/jobs?active=1&limit=${Math.max(1, Math.min(100, limit))}`),
    { cache: "no-store", headers: authHeaders() }
  );
  const data = await readJson(res);
  if (Array.isArray(data)) return { items: data, total: data.length };
  return {
    items: Array.isArray(data.items) ? data.items : [],
    total: Number(data.total) || 0,
  };
}

export async function cancelJob(jobId) {
  const res = await fetch(url(`/api/jobs/${jobId}/cancel`), {
    method: "POST",
    headers: authHeaders(),
  });
  return readJson(res);
}

const JOB_STORAGE_KEY = "wan_active_job_v2";

function jobStorageKey() {
  const owner = getOwnerId();
  return owner ? `${JOB_STORAGE_KEY}:${owner}` : JOB_STORAGE_KEY;
}

/** Persist across phone kills / PWA restarts (sessionStorage is wiped too often). */
export function saveActiveJobId(id, meta = {}) {
  try {
    const key = jobStorageKey();
    if (id) {
      const payload = JSON.stringify({
        id,
        startedAt: Date.now(),
        ...meta,
      });
      localStorage.setItem(key, payload);
      // Also write unscoped key so resume works if owner lookup glitches
      localStorage.setItem(JOB_STORAGE_KEY, payload);
      sessionStorage.removeItem("wan_active_job");
    } else {
      localStorage.removeItem(key);
      localStorage.removeItem(JOB_STORAGE_KEY);
      sessionStorage.removeItem("wan_active_job");
    }
  } catch {
    /* private mode */
  }
}

export function loadActiveJobId() {
  try {
    for (const key of [jobStorageKey(), JOB_STORAGE_KEY]) {
      const raw = localStorage.getItem(key);
      if (!raw) continue;
      const parsed = JSON.parse(raw);
      if (parsed?.id) return parsed.id;
    }
    return sessionStorage.getItem("wan_active_job") || null;
  } catch {
    return null;
  }
}

export async function ensureNotificationPermission() {
  if (typeof Notification === "undefined") return false;
  if (Notification.permission === "granted") return true;
  if (Notification.permission === "denied") return false;
  try {
    const p = await Notification.requestPermission();
    return p === "granted";
  } catch {
    return false;
  }
}

export function notifyJobDone(result) {
  try {
    if (typeof Notification === "undefined") return;
    if (Notification.permission !== "granted") return;
    const kind = result?.kind === "vid" ? "Video" : "Image";
    const n = new Notification("Wan Studio", {
      body: `${kind} ready — open the app to view it.`,
      icon: "/icons/icon-192.png",
      tag: `wan-job-${result?.id || "done"}`,
    });
    n.onclick = () => {
      window.focus();
      n.close();
    };
  } catch {
    /* ignore */
  }
}

/**
 * Poll until done/failed. Job id lives in localStorage for PWA relaunch.
 * "Failed to fetch" while sleeping is treated as reconnect, not failure.
 */
export async function waitForJob(jobId, { onStatus, signal } = {}) {
  saveActiveJobId(jobId);
  const started = Date.now();
  const maxMs = 45 * 60 * 1000;
  let transientFails = 0;

  while (Date.now() - started < maxMs) {
    if (signal?.aborted) {
      throw new Error("Cancelled");
    }
    let job;
    try {
      job = await getJob(jobId);
      transientFails = 0;
    } catch (err) {
      const msg = String(err?.message || err);
      if (/Unlock with your PIN|Wrong PIN|401/i.test(msg)) {
        saveActiveJobId(null);
        throw err;
      }
      transientFails += 1;
      onStatus?.({
        status: "waiting",
        message:
          "Still generating on the server. Your phone briefly lost the status link — reconnecting…",
        transientFails,
      });
      await sleepVisible(Math.min(8000, 2000 + transientFails * 500), signal);
      continue;
    }

    const st = job.status;
    onStatus?.(job);

    if (st === "done") {
      saveActiveJobId(null);
      if (!job.result) throw new Error("Job finished with no result");
      notifyJobDone(job.result);
      return job.result;
    }
    if (st === "failed") {
      saveActiveJobId(null);
      throw new Error(job.error || "Generation failed");
    }

    await sleepVisible(st === "queued" ? 1500 : 2500, signal);
  }

  throw new Error(
    "Timed out waiting for the server. If Comfy was busy, wait and check Library — otherwise try again."
  );
}

function sleepVisible(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new Error("Cancelled"));
      return;
    }
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      cleanup();
      resolve();
    };
    const onAbort = () => {
      if (done) return;
      done = true;
      cleanup();
      reject(new Error("Cancelled"));
    };
    const onVis = () => {
      if (document.visibilityState === "visible") finish();
    };
    const timer = setTimeout(finish, ms);
    const cleanup = () => {
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVis);
      signal?.removeEventListener("abort", onAbort);
    };
    document.addEventListener("visibilitychange", onVis);
    signal?.addEventListener("abort", onAbort);
  });
}

export function mediaUrl(itemOrId) {
  if (!itemOrId) return "";
  const id = typeof itemOrId === "string" ? itemOrId : itemOrId.id;
  if (typeof itemOrId !== "string" && itemOrId._testRef && id) {
    return url(withAuthQuery(`/api/test/refs/${id}/media`));
  }
  if (typeof itemOrId !== "string" && itemOrId._testInput && id) {
    return url(withAuthQuery(`/api/test/inputs/${id}/media`));
  }
  if (id) return url(withAuthQuery(`/api/media/${id}`));
  if (typeof itemOrId !== "string" && itemOrId.media_url) {
    const m = String(itemOrId.media_url);
    if (/cloudinary/i.test(m)) return "";
    const abs = m.startsWith("http") ? m : url(m);
    return withAuthQuery(abs);
  }
  return "";
}

/** Library-sized JPEG from GridFS (images and videos). */
export function thumbUrl(itemOrId) {
  if (!itemOrId) return "";
  const id = typeof itemOrId === "string" ? itemOrId : itemOrId.id;
  if (typeof itemOrId !== "string" && itemOrId._testRef && id) {
    return url(withAuthQuery(`/api/test/refs/${id}/thumb?w=240`));
  }
  if (typeof itemOrId !== "string" && itemOrId._testInput && id) {
    return url(withAuthQuery(`/api/test/inputs/${id}/thumb?w=240`));
  }
  if (id) return url(withAuthQuery(`/api/media/${id}/thumb?w=240`));
  return "";
}

function mediaExt(type, kind) {
  const t = (type || "").toLowerCase();
  if (t.includes("jpeg") || t.includes("jpg")) return "jpg";
  if (t.includes("webp")) return "webp";
  if (t.includes("gif")) return "gif";
  if (t.includes("webm")) return "webm";
  if (t.includes("mp4") || kind === "vid") return "mp4";
  if (t.includes("png") || kind === "img") return "png";
  return kind === "vid" ? "mp4" : "png";
}

/** Download a generation's media and return it as a File for the drop zone / generate. */
export async function fetchMediaAsFile(itemOrId) {
  const src = mediaUrl(itemOrId);
  const res = await fetch(src, { cache: "no-store", headers: authHeaders() });
  if (!res.ok) throw new Error("Failed to load media for editing");
  const blob = await res.blob();
  const type = blob.type || "image/png";
  if (!type.startsWith("image/")) {
    throw new Error("Only images can be used as input. Pick an img generation.");
  }
  const id =
    typeof itemOrId === "string" ? itemOrId : itemOrId?.id || "edit";
  const ext = mediaExt(type, "img");
  return new File([blob], `wan_edit_${id}.${ext}`, { type });
}

function triggerFileDownload(href, name) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.rel = "noopener";
  a.style.display = "none";
  document.body.appendChild(a);
  a.click();
  a.remove();
}

/** Save media to the device Downloads folder — never open the share sheet. */
export async function downloadGeneration(item) {
  if (!item?.id) throw new Error("Nothing to download");
  const type = item.content_type || "application/octet-stream";
  const ext = mediaExt(type, item.kind);
  const name = `wan_${item.kind || "out"}_${item.id}.${ext}`;
  const inlineSrc = mediaUrl(item);
  const attachmentSrc = url(withAuthQuery(`/api/media/${item.id}?download=1`));

  try {
    const res = await fetch(inlineSrc, { cache: "no-store", headers: authHeaders() });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const blob = await res.blob();
    const href = URL.createObjectURL(blob);
    triggerFileDownload(href, name);
    setTimeout(() => URL.revokeObjectURL(href), 2_000);
    return;
  } catch {
    /* fall through to Content-Disposition: attachment */
  }

  triggerFileDownload(attachmentSrc, name);
}

export async function getOpsCapabilities() {
  const res = await fetch(url("/api/ops/capabilities"), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function opsRestartApi() {
  const res = await fetch(url("/api/ops/restart-api"), {
    method: "POST",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function opsSetTunnel(tunnelUrl) {
  const res = await fetch(url("/api/ops/set-tunnel"), {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ url: String(tunnelUrl || "").trim() }),
  });
  return readJson(res);
}

export async function opsRestartComfy() {
  const res = await fetch(url("/api/ops/restart-comfy"), {
    method: "POST",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function opsScrub() {
  const res = await fetch(url("/api/scrub"), {
    method: "POST",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function getTestSummary() {
  const res = await fetch(url("/api/test/summary"), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function setTestReviewBin(id, bin) {
  const res = await fetch(url("/api/test/review"), {
    method: "POST",
    headers: { ...authHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ id, bin: bin || null }),
  });
  return readJson(res);
}

export async function listTestRefs(presetId) {
  const q = presetId ? `?preset_id=${encodeURIComponent(presetId)}` : "";
  const res = await fetch(url(`/api/test/refs${q}`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function uploadTestRef(presetId, file) {
  const body = new FormData();
  body.append("preset_id", presetId);
  body.append("image", file, file.name || "ref.png");
  const res = await fetch(url("/api/test/refs"), {
    method: "POST",
    body,
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function deleteTestRef(id) {
  const res = await fetch(url(`/api/test/refs/${id}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return readJson(res);
}

export function testRefMediaUrl(itemOrId) {
  const id = typeof itemOrId === "string" ? itemOrId : itemOrId?.id;
  if (!id) return "";
  return url(withAuthQuery(`/api/test/refs/${id}/media`));
}

export function testRefThumbUrl(itemOrId) {
  const id = typeof itemOrId === "string" ? itemOrId : itemOrId?.id;
  if (!id) return "";
  return url(withAuthQuery(`/api/test/refs/${id}/thumb?w=240`));
}

export async function listTestInputs() {
  const res = await fetch(url("/api/test/inputs"), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

export async function uploadTestInput(file) {
  const body = new FormData();
  body.append("image", file, file.name || "input.png");
  const res = await fetch(url("/api/test/inputs"), {
    method: "POST",
    headers: authHeaders(),
    body,
  });
  return readJson(res);
}

export async function deleteTestInput(id) {
  const res = await fetch(url(`/api/test/inputs/${id}`), {
    method: "DELETE",
    headers: authHeaders(),
  });
  return readJson(res);
}

export function testInputThumbUrl(itemOrId) {
  const id = typeof itemOrId === "string" ? itemOrId : itemOrId?.id;
  if (!id) return "";
  return url(withAuthQuery(`/api/test/inputs/${id}/thumb?w=240`));
}

export function testInputMediaUrl(itemOrId) {
  const id = typeof itemOrId === "string" ? itemOrId : itemOrId?.id;
  if (!id) return "";
  return url(withAuthQuery(`/api/test/inputs/${id}/media`));
}

export async function fetchTestInputAsFile(item) {
  const res = await fetch(testInputMediaUrl(item), {
    cache: "no-store",
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error("Could not load image");
  const blob = await res.blob();
  const name = item.filename || "input.jpg";
  return new File([blob], name, { type: item.content_type || blob.type || "image/jpeg" });
}

export async function listTestRuns({ presetId, limit = 20 } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (presetId) params.set("preset_id", presetId);
  const res = await fetch(url(`/api/test/runs?${params}`), {
    cache: "no-store",
    headers: authHeaders(),
  });
  return readJson(res);
}

/** Poll health until API answers (after restart). */
export async function waitForApiHealth({
  attempts = 20,
  pauseMs = 4000,
  onTick,
  signal,
} = {}) {
  let last = null;
  for (let i = 1; i <= attempts; i++) {
    if (signal?.aborted) throw new Error("Cancelled");
    try {
      const h = await getHealth();
      last = h;
      onTick?.({ attempt: i, attempts, health: h });
      if (h && (h.ok === true || h.mongodb != null)) return h;
    } catch (err) {
      last = { error: String(err?.message || err) };
      onTick?.({ attempt: i, attempts, health: last });
    }
    if (i < attempts) await sleep(pauseMs, signal);
  }
  throw new Error(
    `API did not recover after ${attempts} checks. Last: ${JSON.stringify(last)}`
  );
}
