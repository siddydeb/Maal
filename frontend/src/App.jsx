import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelJob,
  deleteGeneration,
  downloadGeneration,
  ensureNotificationPermission,
  fetchMediaAsFile,
  generate,
  getHealth,
  getJob,
  getOpsCapabilities,
  listActiveJobs,
  listGenerations,
  loadActiveJobId,
  mediaUrl,
  opsRestartApi,
  opsRestartComfy,
  opsScrub,
  opsSetTunnel,
  thumbUrl,
  saveActiveJobId,
  waitForApiHealth,
  waitForJob,
  updateGeneration,
  markGenerationOpened,
  listTestRefs,
  testRefThumbUrl,
  listPresets,
  listReviewBins,
} from "./api";
import { clearSession, getOwnerId, isAdminOwner, isTesterOwner } from "./auth";
import { presetById, presetsForMode, setActionPresets } from "./presets";
import { setReviewBins } from "./reviewBins";
import TestRefs from "./TestRefs.jsx";
import TestInputs from "./TestInputs.jsx";
import TestReview from "./TestReview.jsx";

const PROMPT_MAX = 3000;

function IconSpark({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 3.5 13.8 9.2 19.5 11 13.8 12.8 12 18.5 10.2 12.8 4.5 11 10.2 9.2 12 3.5Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconImage({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="3.5" y="4.5" width="17" height="15" rx="2.5" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="9" cy="10" r="1.6" fill="currentColor" />
      <path d="M4.5 16.5 9 13l3.2 2.4L16 12l3.5 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

function IconVideo({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="3.5" y="6" width="12.5" height="12" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M16 10.2 20.5 7.5v9L16 13.8V10.2Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

function IconJobs({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M4 7h16M4 12h16M4 17h10" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function IconFolder({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M4 7.5A1.5 1.5 0 0 1 5.5 6h4.2l1.6 1.8H18.5A1.5 1.5 0 0 1 20 9.3v8.2a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17.5v-10Z"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconReview({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M5 7h14M5 12h10M5 17h12" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="m15.5 11.2 1.6 1.6 3.2-3.4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function IconLibrary({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="4" y="4.5" width="6.5" height="15" rx="1.5" stroke="currentColor" strokeWidth="1.6" />
      <rect x="13.5" y="4.5" width="6.5" height="15" rx="1.5" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function IconLock({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <rect x="5.5" y="10.5" width="13" height="9" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M8.5 10.5V8.2a3.5 3.5 0 0 1 7 0v2.3" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function IconControls({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.6" />
      <path
        d="M12 3.5v2.2M12 18.3v2.2M3.5 12h2.2M18.3 12h2.2M6.1 6.1l1.6 1.6M16.3 16.3l1.6 1.6M17.9 6.1l-1.6 1.6M7.7 16.3l-1.6 1.6"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}

function IconUpload({ className }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path d="M12 15.5V6.8M12 6.8 8.6 10.2M12 6.8l3.4 3.4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M5 15.2v2.3A1.5 1.5 0 0 0 6.5 19h11a1.5 1.5 0 0 0 1.5-1.5v-2.3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function formatElapsed(fromIso) {
  const t = fromIso ? Date.parse(fromIso) : NaN;
  const start = Number.isFinite(t) ? t : Date.now();
  const sec = Math.max(0, Math.floor((Date.now() - start) / 1000));
  const m = Math.floor(sec / 60);
  const r = sec % 60;
  return m > 0 ? `${m}m ${String(r).padStart(2, "0")}s` : `${r}s`;
}

function statusForJob(j) {
  if (!j) return null;
  if (j.status === "waking") {
    return (
      j.message ||
      "Waking API server (after idle this can take up to a minute)…"
    );
  }
  if (j.status === "running") {
    const elapsed = formatElapsed(j.started_at || j.created_at);
    return `Generating… ${elapsed} — keeps going if you lock or turn off the phone`;
  }
  if (j.status === "queued") {
    if ((j.resume_count || 0) > 0) {
      return "Server restarted — resuming your generation…";
    }
    return j.message || "Queued on server…";
  }
  if (j.status === "waiting") {
    return (
      j.message ||
      "Still generating on the server. Updating status when the phone reconnects…"
    );
  }
  return null;
}

export default function App() {
  const [mode, setMode] = useState("img");
  const [videoSeconds, setVideoSeconds] = useState(5);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState(null);
  const [status, setStatus] = useState("");
  const [statusError, setStatusError] = useState(false);
  const [result, setResult] = useState(null);
  const [editPrompt, setEditPrompt] = useState("");
  const [editing, setEditing] = useState(false);
  const [library, setLibrary] = useState([]);
  const [libraryTotal, setLibraryTotal] = useState(0);
  const [libraryPage, setLibraryPage] = useState(0);
  const [ongoing, setOngoing] = useState([]);
  const LIBRARY_PAGE_SIZE = 30;
  const [healthText, setHealthText] = useState("—");
  const [view, setView] = useState("generation"); // generation | ongoing | library | refs | inputs | review
  const [dragging, setDragging] = useState(false);
  const [lightbox, setLightbox] = useState(null); // generation item for full preview
  const inputRef = useRef(null);
  const promptRef = useRef(null);
  const wakeLockRef = useRef(null);
  const pollAbortRef = useRef(null);
  const touchStartRef = useRef(null);
  const [installHint, setInstallHint] = useState(false);
  const [admin, setAdmin] = useState(() => isAdminOwner());
  const [tester, setTester] = useState(() => isTesterOwner());
  const [resultRefs, setResultRefs] = useState([]);
  const [lastPresetId, setLastPresetId] = useState(null);
  const [opsCaps, setOpsCaps] = useState(null);
  const [opsBusy, setOpsBusy] = useState(null); // action id
  const [tunnelInput, setTunnelInput] = useState("");
  const [opsOpen, setOpsOpen] = useState(true);
  const [, setPresetRev] = useState(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await listPresets();
        if (!cancelled && data?.presets?.length) setActionPresets(data.presets);
      } catch {
        /* keep generic fallbacks */
      }
      try {
        const bins = await listReviewBins();
        if (!cancelled && bins?.bins?.length) setReviewBins(bins.bins);
      } catch {
        /* tester-only */
      }
      if (!cancelled) setPresetRev((n) => n + 1);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshLibrary = useCallback(async (page = libraryPage) => {
    const listOpts = {
      limit: LIBRARY_PAGE_SIZE,
      // Tester PIN library = tagged test runs (batch outputs land here).
      ...(isTesterOwner() ? { test_run: 1 } : {}),
    };
    try {
      const skip = Math.max(0, page) * LIBRARY_PAGE_SIZE;
      const data = await listGenerations({
        ...listOpts,
        skip,
      });
      const totalPages = Math.max(
        1,
        Math.ceil((data.total || 0) / LIBRARY_PAGE_SIZE)
      );
      // If current page is past the end (e.g. after deletes), snap back
      if (page > 0 && page >= totalPages && data.total > 0) {
        const last = totalPages - 1;
        setLibraryPage(last);
        const again = await listGenerations({
          ...listOpts,
          skip: last * LIBRARY_PAGE_SIZE,
        });
        setLibrary(again.items);
        setLibraryTotal(again.total);
        return;
      }
      setLibrary(data.items);
      setLibraryTotal(data.total);
    } catch {
      setLibrary([]);
      setLibraryTotal(0);
    }
  }, [libraryPage]);

  const refreshOngoing = useCallback(async () => {
    try {
      const data = await listActiveJobs(40);
      setOngoing(data.items || []);
    } catch {
      /* keep previous list on transient errors */
    }
  }, []);

  useEffect(() => {
    if (view !== "ongoing") return undefined;
    refreshOngoing();
    const timer = setInterval(refreshOngoing, 4000);
    return () => clearInterval(timer);
  }, [view, refreshOngoing]);

  // Keep the Ongoing badge fresh even when on other tabs
  useEffect(() => {
    refreshOngoing();
    const timer = setInterval(refreshOngoing, 12000);
    return () => clearInterval(timer);
  }, [refreshOngoing]);

  const refreshHealth = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealthText(
        [
          h.mongodb ? "Library storage OK" : "Library storage down",
          h.comfyui ? "GPU ready" : "GPU offline",
          h.zero_residue ? "Auto-clean on" : null,
        ]
          .filter(Boolean)
          .join("  ·  ")
      );
      if (h.comfyui_url) {
        setTunnelInput((prev) => prev || String(h.comfyui_url));
      }
      return h;
    } catch {
      setHealthText("Server unreachable");
      return null;
    }
  }, []);

  function friendlyOpsError(err) {
    const msg = String(err?.message || err || "");
    if (/RENDER_API_KEY|render api key|isn’t configured yet/i.test(msg)) {
      return "Server restart isn’t set up yet. Try again later, or ask for it to be configured.";
    }
    if (/GPU_AGENT|GPU helper|GPU restart isn’t set up/i.test(msg)) {
      return "GPU restart isn’t set up yet. The GPU helper app needs to be running first.";
    }
    if (/timed?\s*out|Failed to fetch|network|unreachable|waking/i.test(msg)) {
      return "Can’t reach the server right now. Wait a minute and try again.";
    }
    if (/doesn’t look right|Tunnel URL must be https|Paste a full https/i.test(msg)) {
      return "That link doesn’t look right. Paste a full https://…trycloudflare.com address.";
    }
    if (/HTTP \d{3}|env-vars|Deploy trigger|Render restart/i.test(msg)) {
      return "Something went wrong on the server side. Wait a minute and try again.";
    }
    return msg || "Something went wrong. Try again.";
  }

  async function runOps(action) {
    if (!admin) return;
    setOpsBusy(action);
    setStatusError(false);
    try {
      if (action === "health") {
        setStatus("Checking if everything is online…");
        const h = await refreshHealth();
        if (!h) {
          setStatus("Server is offline or too slow to answer.");
          setStatusError(true);
          return;
        }
        const bits = [];
        bits.push(h.mongodb ? "library storage is OK" : "library storage is down");
        bits.push(h.comfyui ? "GPU is ready" : "GPU is offline");
        setStatus(`Status: ${bits.join("; ")}.`);
        setStatusError(!h.mongodb || !h.comfyui);
        return;
      }
      if (action === "restart-api") {
        if (
          !window.confirm(
            "Restart the cloud server?\n\nAny generation in progress will stop. This usually takes about a minute."
          )
        ) {
          return;
        }
        setStatus("Restarting the cloud server…");
        await opsRestartApi();
        setStatus("Server is restarting. Waiting for it to come back…");
        await waitForApiHealth({
          attempts: 24,
          pauseMs: 5000,
          onTick: ({ attempt, attempts }) => {
            setStatus(
              `Waiting for the server to come back… (${attempt}/${attempts})`
            );
          },
        });
        await refreshHealth();
        await refreshOpsCaps();
        setStatus("Server is back online. You can generate again.");
        return;
      }
      if (action === "restart-comfy") {
        if (
          !window.confirm(
            "Restart the GPU app on your PC?\n\nUse this if generation is stuck or the GPU shows offline."
          )
        ) {
          return;
        }
        setStatus("Restarting the GPU app…");
        const res = await opsRestartComfy();
        setStatus(
          res.agent?.comfy_up
            ? "GPU app is running again."
            : "GPU restart finished, but it may still be starting. Tap Check status in a moment."
        );
        setStatusError(!res.agent?.comfy_up);
        await refreshHealth();
        return;
      }
      if (action === "set-tunnel") {
        const u = tunnelInput.trim();
        if (!u) {
          setStatus("Paste the new tunnel link first (https://…trycloudflare.com).");
          setStatusError(true);
          return;
        }
        if (
          !window.confirm(
            `Connect the cloud server to this GPU tunnel?\n\n${u}\n\nThe server will briefly update — wait about a minute.`
          )
        ) {
          return;
        }
        setStatus("Connecting the new tunnel and updating the server…");
        await opsSetTunnel(u);
        setStatus("Update started. Waiting for the server…");
        await waitForApiHealth({
          attempts: 30,
          pauseMs: 6000,
          onTick: ({ attempt, attempts, health }) => {
            const gpu = health?.comfyui ? "GPU ready" : "waiting for GPU";
            setStatus(
              `Waiting for the update to finish… (${attempt}/${attempts}, ${gpu})`
            );
          },
        });
        await refreshHealth();
        setStatus("Tunnel connected. If GPU still shows offline, restart the GPU app.");
        return;
      }
      if (action === "scrub") {
        if (
          !window.confirm(
            "Clear temporary files on the GPU PC?\n\nThis only removes leftovers from generations — your library stays safe."
          )
        ) {
          return;
        }
        setStatus("Clearing temporary GPU files…");
        const res = await opsScrub();
        const n = res.wiped_files ?? 0;
        setStatus(
          n > 0
            ? `Cleared ${n} temporary file${n === 1 ? "" : "s"}.`
            : "Nothing to clear — temp folders were already clean."
        );
        return;
      }
    } catch (err) {
      setStatus(friendlyOpsError(err));
      setStatusError(true);
    } finally {
      setOpsBusy(null);
    }
  }

  const refreshOpsCaps = useCallback(async () => {
    if (!isAdminOwner()) {
      setAdmin(false);
      setOpsCaps(null);
      return;
    }
    setAdmin(true);
    try {
      const caps = await getOpsCapabilities();
      setOpsCaps(caps);
    } catch {
      setOpsCaps(null);
    }
  }, []);

  useEffect(() => {
    setTester(isTesterOwner());
    refreshHealth();
    refreshLibrary();
    refreshOpsCaps();
    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      window.navigator.standalone === true;
    setInstallHint(!standalone);
  }, [refreshHealth, refreshLibrary, refreshOpsCaps]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  useEffect(() => {
    setEditPrompt(result?.prompt || "");
    setEditing(false);
  }, [result?.id]);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e) => {
      if (e.key === "Escape") setLightbox(null);
      if (e.key === "ArrowLeft") stepLightbox(-1);
      if (e.key === "ArrowRight") stepLightbox(1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, library]);

  function stepLightbox(delta) {
    if (!library.length) return;
    const curId = lightbox?.id;
    const idx = library.findIndex((x) => x.id === curId);
    if (idx < 0) {
      setLightbox(library[0]);
      return;
    }
    const next = (idx + delta + library.length) % library.length;
    setLightbox(library[next]);
  }

  function onLightboxTouchStart(e) {
    const t = e.changedTouches?.[0];
    if (!t) return;
    touchStartRef.current = { x: t.clientX, y: t.clientY };
  }

  function onLightboxTouchEnd(e) {
    const start = touchStartRef.current;
    touchStartRef.current = null;
    const t = e.changedTouches?.[0];
    if (!start || !t) return;
    const dx = t.clientX - start.x;
    const dy = t.clientY - start.y;
    if (Math.abs(dx) < 56 || Math.abs(dx) < Math.abs(dy) * 1.2) return;
    stepLightbox(dx < 0 ? 1 : -1);
  }

  async function requestWakeLock() {
    try {
      if (wakeLockRef.current) return;
      if (!("wakeLock" in navigator)) return;
      wakeLockRef.current = await navigator.wakeLock.request("screen");
      wakeLockRef.current.addEventListener("release", () => {
        wakeLockRef.current = null;
      });
    } catch {
      /* unsupported / denied — job polling still works */
    }
  }

  async function releaseWakeLock() {
    try {
      await wakeLockRef.current?.release();
    } catch {
      /* ignore */
    }
    wakeLockRef.current = null;
  }

  const finishWithResult = useCallback(
    async (data) => {
      setResult(data);
      setStatusError(false);
      setStatus("Saved to MongoDB only. Local residue scrubbed.");
      refreshLibrary();
      if (data.kind === "img" || (data.content_type || "").startsWith("image/")) {
        try {
          const next = await fetchMediaAsFile(data);
          pickFile(next);
          setPrompt("");
          setStatus(
            "Saved. Result loaded as next input — change the prompt to edit again."
          );
        } catch {
          clearInput();
        }
      } else {
        clearInput();
      }
    },
    [refreshLibrary]
  );

  const finishWithResultRef = useRef(finishWithResult);
  finishWithResultRef.current = finishWithResult;

  // Resume in-flight job after phone kill / PWA relaunch / reload.
  // Intentionally mount-once: do not depend on finishWithResult or resume will
  // abort mid-video when library state changes and never restart.
  useEffect(() => {
    const jobId = loadActiveJobId();
    if (!jobId) return undefined;

    let cancelled = false;
    const ac = new AbortController();
    pollAbortRef.current = ac;

    (async () => {
      setLoading(true);
      setStatusError(false);
      setStatus(
        "Resuming generation… Refresh only closes this screen — the GPU job keeps running."
      );
      await requestWakeLock();
      try {
        try {
          const existing = await getJob(jobId);
          if (existing.status === "done" && existing.result) {
            if (!cancelled) await finishWithResultRef.current(existing.result);
            saveActiveJobId(null);
            return;
          }
          if (existing.status === "failed") {
            saveActiveJobId(null);
            if (!cancelled) {
              setStatus(existing.error || "Generation failed");
              setStatusError(true);
            }
            return;
          }
        } catch {
          /* fall through — keep job id, reconnect */
        }
        const data = await waitForJob(jobId, {
          signal: ac.signal,
          onStatus: (j) => {
            if (cancelled) return;
            const msg = statusForJob(j);
            if (msg) setStatus(msg);
          },
        });
        if (!cancelled) await finishWithResultRef.current(data);
      } catch (err) {
        const msg = String(err.message || err);
        // Page refresh / StrictMode remount aborts polling — keep job id for next mount
        if (msg === "Cancelled") return;
        if (!cancelled) {
          if (/fetch|network|Failed to fetch/i.test(msg)) {
            setStatus(
              "Connection dropped — reopen Wan shortly; check Library if it finished."
            );
            setStatusError(false);
          } else {
            setStatus(msg);
            setStatusError(true);
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
        await releaseWakeLock();
      }
    })();

    return () => {
      cancelled = true;
      // Delay abort so React StrictMode remount does not cancel a real resume
      setTimeout(() => ac.abort(), 400);
    };
  }, []);

  useEffect(() => {
    if (!loading) return undefined;
    const onLeave = (e) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", onLeave);
    return () => window.removeEventListener("beforeunload", onLeave);
  }, [loading]);

  function pickFile(f) {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(f);
    });
  }

  function clearInput() {
    setFile(null);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return null;
    });
    if (inputRef.current) inputRef.current.value = "";
  }

  async function useAsInput(item) {
    if (!item?.id) return;
    const isVid =
      item.kind === "vid" || (item.content_type || "").startsWith("video/");
    if (isVid) {
      setResult(item);
      setStatus("Videos can’t be used as input — pick an image generation.");
      setStatusError(true);
      return;
    }

    setBusyId(item.id);
    setStatusError(false);
    setStatus("Loading image into editor…");
    try {
      const f = await fetchMediaAsFile(item);
      pickFile(f);
      setResult(item);
      setMode("img");
      setPrompt("");
      setView("generation");
      setStatus(
        item._testInput
          ? "Loaded start photo — pick a generate button."
          : "Loaded into drop zone — write a new prompt to edit it."
      );
      window.scrollTo({ top: 0, behavior: "smooth" });
      requestAnimationFrame(() => promptRef.current?.focus());
    } catch (err) {
      setStatus(String(err.message || err));
      setStatusError(true);
    } finally {
      setBusyId(null);
    }
  }

  function selectItem(item) {
    // Images and videos both open the lightbox preview
    setLightbox(item);
    if (item?.id && !item?.meta?.opened_at) {
      markGenerationOpened(item.id)
        .then((doc) => {
          if (!doc?.id) return;
          setLibrary((prev) =>
            prev.map((row) =>
              row.id === doc.id
                ? { ...row, meta: { ...(row.meta || {}), ...(doc.meta || {}) } }
                : row
            )
          );
          setLightbox((cur) =>
            cur?.id === doc.id
              ? { ...cur, meta: { ...(cur.meta || {}), ...(doc.meta || {}) } }
              : cur
          );
        })
        .catch(() => {
          /* badge clear is best-effort */
        });
    }
  }

  useEffect(() => {
    if (view === "library") {
      refreshLibrary(libraryPage);
    }
  }, [libraryPage, view, refreshLibrary]);

  function goLibrary() {
    setView("library");
    refreshLibrary(libraryPage);
  }

  function goOngoing() {
    setView("ongoing");
    refreshOngoing();
  }

  async function onCancelJob(job) {
    if (!job?.id) return;
    if (
      !window.confirm(
        "Cancel this generation?\n\nIt will stop on the server/GPU if still running."
      )
    ) {
      return;
    }
    setBusyId(job.id);
    try {
      await cancelJob(job.id);
      if (loadActiveJobId() === job.id) {
        saveActiveJobId(null);
        setLoading(false);
      }
      await refreshOngoing();
      setStatus("Job cancelled.");
      setStatusError(false);
    } catch (err) {
      setStatus(String(err.message || err));
      setStatusError(true);
    } finally {
      setBusyId(null);
    }
  }

  const libraryPageCount = Math.max(
    1,
    Math.ceil((libraryTotal || 0) / LIBRARY_PAGE_SIZE)
  );
  const libraryFrom =
    libraryTotal === 0 ? 0 : libraryPage * LIBRARY_PAGE_SIZE + 1;
  const libraryTo = Math.min(
    libraryTotal,
    (libraryPage + 1) * LIBRARY_PAGE_SIZE
  );

  async function runGeneration({
    mode: genMode,
    prompt: text,
    videoSeconds: secs,
    presetId,
  }) {
    if (!file) {
      setStatus("Add an image first.");
      setStatusError(true);
      return;
    }
    const trimmed = (text || "").trim();
    if (!trimmed) {
      setStatus("Prompt is required.");
      setStatusError(true);
      return;
    }

    pollAbortRef.current?.abort();
    const ac = new AbortController();
    pollAbortRef.current = ac;

    setLoading(true);
    setStatusError(false);
    setStatus(
      "Waking API if needed, then starting job… (first request after idle can take up to a minute)"
    );
    await ensureNotificationPermission();
    await requestWakeLock();

    try {
      const data = await generate({
        mode: genMode,
        prompt: trimmed,
        file,
        videoSeconds: genMode === "vid" ? secs : undefined,
        presetId: presetId || undefined,
        testRun: tester,
        signal: ac.signal,
        onStatus: (j) => {
          const msg = statusForJob(j);
          if (msg) setStatus(msg);
        },
      });
      setLastPresetId(presetId || null);
      if (tester && presetId) {
        try {
          const refData = await listTestRefs(presetId);
          setResultRefs(Array.isArray(refData.items) ? refData.items : []);
        } catch {
          setResultRefs([]);
        }
      } else {
        setResultRefs([]);
      }
      await finishWithResult(data);
      refreshOngoing();
    } catch (err) {
      const msg = String(err.message || err);
      if (msg === "Cancelled") {
        // Refresh / navigation aborted polling — server job keeps running
        setStatus(
          "Status updates paused on this screen — generation keeps running on the GPU. Reopen the app or check Library when it’s done."
        );
        setStatusError(false);
        refreshOngoing();
      } else {
        setStatus(msg);
        setStatusError(true);
      }
    } finally {
      setLoading(false);
      await releaseWakeLock();
      refreshOngoing();
    }
  }

  async function onSubmit(e) {
    e.preventDefault();
    await runGeneration({
      mode,
      prompt: prompt.trim(),
      videoSeconds,
    });
  }

  async function onPreset(presetId) {
    const preset = presetsForMode(mode).find((p) => p.id === presetId);
    if (!preset) return;
    if (!file) {
      setStatus("Add an image first, then tap an action.");
      setStatusError(true);
      return;
    }
    setPrompt(preset.prompt);
    setStatus(
      mode === "vid"
        ? `Starting ${preset.label} (${videoSeconds}s)…`
        : `Starting ${preset.label}…`
    );
    await runGeneration({
      mode,
      prompt: preset.prompt,
      videoSeconds,
      presetId: preset.id,
    });
  }

  async function onSavePrompt() {
    if (!result?.id) return;
    const text = editPrompt.trim();
    if (!text) {
      setStatus("Prompt cannot be empty.");
      setStatusError(true);
      return;
    }
    setBusyId(result.id);
    setStatusError(false);
    try {
      const updated = await updateGeneration(result.id, { prompt: text });
      setResult(updated);
      setEditing(false);
      setStatus("Prompt updated in MongoDB.");
      refreshLibrary();
    } catch (err) {
      setStatus(String(err.message || err));
      setStatusError(true);
    } finally {
      setBusyId(null);
    }
  }

  async function onDelete(id) {
    if (!id) return;
    if (!window.confirm("Delete this generation from MongoDB?")) return;
    setBusyId(id);
    setStatusError(false);
    try {
      await deleteGeneration(id);
      if (result?.id === id) {
        setResult(null);
        setEditing(false);
      }
      setStatus("Deleted from MongoDB.");
      refreshLibrary();
    } catch (err) {
      setStatus(String(err.message || err));
      setStatusError(true);
    } finally {
      setBusyId(null);
    }
  }

  async function onDownload(item) {
    if (!item?.id) return;
    setBusyId(item.id);
    setStatusError(false);
    try {
      await downloadGeneration(item);
      setStatus("Download started.");
    } catch (err) {
      setStatus(String(err.message || err));
      setStatusError(true);
    } finally {
      setBusyId(null);
    }
  }

  const goLabel = loading
    ? "Working…"
    : mode === "vid"
      ? "Generate video"
      : "Generate image";

  const estimateLabel =
    mode === "vid" ? "Estimated time: 2–5 min" : "Estimated time: 1–3 min";

  const resultIsImage =
    result &&
    result.kind !== "vid" &&
    !(result.content_type || "").startsWith("video/");

  function resetComposer() {
    setPrompt("");
    setVideoSeconds(5);
    setStatus("");
    setStatusError(false);
  }

  function lockApp() {
    clearSession();
    window.location.reload();
  }

  function toggleControls() {
    setOpsOpen((v) => !v);
    if (!opsOpen) {
      requestAnimationFrame(() => {
        document
          .getElementById("ops-controls")
          ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
      });
    }
  }

  function goNewGeneration() {
    setView("generation");
    window.scrollTo({ top: 0, behavior: "smooth" });
    requestAnimationFrame(() => inputRef.current?.click());
  }

  const topNav = (
    <div className="top-tabs" role="tablist" aria-label="App views">
      <button
        type="button"
        className={`top-tab${view === "generation" ? " active" : ""}`}
        role="tab"
        aria-selected={view === "generation"}
        onClick={() => setView("generation")}
      >
        <IconSpark className="tab-ico" />
        <span>Generation</span>
      </button>
      {tester && (
        <button
          type="button"
          className={`top-tab${view === "inputs" ? " active" : ""}`}
          role="tab"
          aria-selected={view === "inputs"}
          onClick={() => setView("inputs")}
        >
          <IconImage className="tab-ico" />
          <span>Input images</span>
        </button>
      )}
      {tester && (
        <button
          type="button"
          className={`top-tab${view === "refs" ? " active" : ""}`}
          role="tab"
          aria-selected={view === "refs"}
          onClick={() => setView("refs")}
        >
          <IconFolder className="tab-ico" />
          <span>Reference generations</span>
        </button>
      )}
      {tester && (
        <button
          type="button"
          className={`top-tab${view === "review" ? " active" : ""}`}
          role="tab"
          aria-selected={view === "review"}
          onClick={() => setView("review")}
        >
          <IconReview className="tab-ico" />
          <span>Review</span>
        </button>
      )}
      <button
        type="button"
        className={`top-tab${view === "ongoing" ? " active" : ""}`}
        role="tab"
        aria-selected={view === "ongoing"}
        onClick={goOngoing}
      >
        <IconJobs className="tab-ico" />
        <span>Ongoing Jobs</span>
        {ongoing.length > 0 && (
          <span className="nav-count">{ongoing.length}</span>
        )}
      </button>
      <button
        type="button"
        className={`top-tab${view === "library" ? " active" : ""}`}
        role="tab"
        aria-selected={view === "library"}
        onClick={goLibrary}
      >
        <IconLibrary className="tab-ico" />
        <span>Library</span>
        {libraryTotal > 0 && (
          <span className="nav-count">{libraryTotal}</span>
        )}
      </button>
      <button
        type="button"
        className="top-tab"
        onClick={lockApp}
        title="Lock / switch account"
      >
        <IconLock className="tab-ico" />
        <span>Lock</span>
      </button>
      {admin && (
        <button
          type="button"
          className={`top-tab${opsOpen ? " active" : ""}`}
          onClick={toggleControls}
          title="Stack tools"
        >
          <IconControls className="tab-ico" />
          <span>Controls</span>
        </button>
      )}
    </div>
  );

  return (
    <>
      <div className="noise" aria-hidden="true" />
      <main className="shell">
        <header className="app-header">
          <button
            type="button"
            className="app-brand"
            onClick={() => setView("generation")}
          >
            <span className="app-mark" aria-hidden="true">
              W
            </span>
            <span className="app-brand-text">
              <span className="app-name">Wan</span>
              <span className="app-tagline">Create. Transform. Wow.</span>
            </span>
          </button>
          <div className="app-header-actions">
            <button
              type="button"
              className="icon-btn"
              onClick={lockApp}
              title="Lock"
              aria-label="Lock"
            >
              <IconLock className="icon-btn-svg" />
            </button>
            {admin && (
              <button
                type="button"
                className={`icon-btn${opsOpen ? " active" : ""}`}
                onClick={toggleControls}
                title="Controls"
                aria-label="Controls"
              >
                <IconControls className="icon-btn-svg" />
              </button>
            )}
          </div>
        </header>

        <nav className="nav-desktop" aria-label="Primary">
          {topNav}
        </nav>

        {view === "generation" && (
          <section className="panel view-panel gen-panel">
            {tester && (
              <p className="test-banner">
                Test mode — drop a photo or pick one from Input images, then tap a
                button. Results stay in this PIN’s library next to your reference folders.
              </p>
            )}
            <form className="form gen-form" onSubmit={onSubmit}>
              <div className="gen-primary">
                <div className="section">
                  <h2 className="section-title">Choose Mode</h2>
                  <div className="mode-cards" role="tablist" aria-label="Generation mode">
                    <button
                      type="button"
                      className={`mode-card${mode === "img" ? " active" : ""}`}
                      role="tab"
                      aria-selected={mode === "img"}
                      onClick={() => setMode("img")}
                    >
                      <span className="mode-card-icon">
                        <IconImage />
                      </span>
                      <strong>Image to Image</strong>
                      <span className="mode-card-desc">Transform images with AI</span>
                    </button>
                    <button
                      type="button"
                      className={`mode-card${mode === "vid" ? " active" : ""}`}
                      role="tab"
                      aria-selected={mode === "vid"}
                      onClick={() => setMode("vid")}
                    >
                      <span className="mode-card-icon">
                        <IconVideo />
                      </span>
                      <strong>Image to Video</strong>
                      <span className="mode-card-desc">Animate images with AI</span>
                    </button>
                  </div>
                </div>

                <div className="section">
                  <div className="section-head">
                    <h2 className="section-title">Upload Input</h2>
                    <div className="seg" role="tablist" aria-label="Output type">
                      <button
                        type="button"
                        className={`seg-btn${mode === "img" ? " active" : ""}`}
                        role="tab"
                        aria-selected={mode === "img"}
                        onClick={() => setMode("img")}
                      >
                        Image
                      </button>
                      <button
                        type="button"
                        className={`seg-btn${mode === "vid" ? " active" : ""}`}
                        role="tab"
                        aria-selected={mode === "vid"}
                        onClick={() => setMode("vid")}
                      >
                        Video
                      </button>
                    </div>
                  </div>

                  <label
                    className={`drop${dragging ? " drag" : ""}${previewUrl ? " has-preview" : ""}`}
                    onDragEnter={(e) => {
                      e.preventDefault();
                      setDragging(true);
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragging(true);
                    }}
                    onDragLeave={(e) => {
                      e.preventDefault();
                      setDragging(false);
                    }}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragging(false);
                      pickFile(e.dataTransfer.files?.[0]);
                    }}
                  >
                    <input
                      ref={inputRef}
                      type="file"
                      accept="image/*"
                      hidden
                      onChange={(e) => pickFile(e.target.files?.[0])}
                    />
                    {!previewUrl && (
                      <div className="drop-inner">
                        <span className="drop-ico">
                          <IconUpload />
                        </span>
                        <span className="drop-title">Tap to upload or drag & drop</span>
                        <span className="drop-sub">
                          JPG, PNG, WEBP · or open from Library
                        </span>
                      </div>
                    )}
                    {previewUrl && (
                      <img
                        className="preview"
                        src={previewUrl}
                        alt="Selected input"
                      />
                    )}
                  </label>
                  {previewUrl && (
                    <button
                      type="button"
                      className="ghost clear-input"
                      onClick={(e) => {
                        e.preventDefault();
                        clearInput();
                      }}
                    >
                      Clear image
                    </button>
                  )}
                </div>
              </div>

              <div className="gen-side">
                <div className="section">
                  <div className="section-head">
                    <h2 className="section-title">Prompt (Optional)</h2>
                    <span className="char-count">
                      {prompt.length}/{PROMPT_MAX}
                    </span>
                  </div>
                  <textarea
                    ref={promptRef}
                    className="prompt-area"
                    rows={5}
                    maxLength={PROMPT_MAX}
                    placeholder="Describe the changes you want to make…"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value.slice(0, PROMPT_MAX))}
                  />
                </div>

                <div className="section settings-section">
                  <div className="section-head">
                    <h2 className="section-title">Settings</h2>
                    <button
                      type="button"
                      className="ghost reset-btn"
                      onClick={resetComposer}
                    >
                      Reset
                    </button>
                  </div>

                  {mode === "vid" && (
                    <div
                      className="setting-row"
                      role="group"
                      aria-label="Video length"
                    >
                      <div className="setting-label">
                        <span>Video length</span>
                        <span className="setting-hint">clip duration</span>
                      </div>
                      <div className="duration">
                        {[2, 3, 4, 5].map((sec) => (
                          <button
                            key={sec}
                            type="button"
                            className={`duration-btn${
                              videoSeconds === sec ? " active" : ""
                            }`}
                            aria-pressed={videoSeconds === sec}
                            onClick={() => setVideoSeconds(sec)}
                          >
                            {sec}s
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div
                    className="setting-block"
                    role="group"
                    aria-label={mode === "vid" ? "Video actions" : "Image actions"}
                  >
                    <div className="setting-label">
                      <span>{mode === "vid" ? "Quick video actions" : "Quick image actions"}</span>
                      <span className="setting-hint">
                        {mode === "vid"
                          ? `pic + tap → ${videoSeconds}s clip`
                          : "pic + tap → result"}
                      </span>
                    </div>
                    <div className="preset-grid">
                      {presetsForMode(mode).map((p) => (
                        <button
                          key={p.id}
                          type="button"
                          className="preset-btn"
                          disabled={loading}
                          title={p.hint}
                          onClick={() => onPreset(p.id)}
                        >
                          <span className="preset-btn-label">{p.label}</span>
                          <span className="preset-btn-hint">{p.hint}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                <button
                  type="submit"
                  className="go"
                  disabled={loading || !prompt.trim()}
                >
                  <span className="go-main">
                    <IconSpark className="go-ico" />
                    <span>{goLabel}</span>
                    {loading && <span className="spinner" />}
                  </span>
                  <span className="go-sub">{estimateLabel}</span>
                </button>
              </div>
            </form>

            <p className={`status${statusError ? " error" : ""}`} role="status">
              {status}
            </p>

            {result && (
              <div className="result">
                <div className="result-head">
                  <span>{result.kind === "vid" ? "video" : "image"}</span>
                  <span className="mono">{result.id}</span>
                </div>
                <div className="result-stage">
                  {result.kind === "vid" ||
                  (result.content_type || "").startsWith("video/") ? (
                    <button
                      type="button"
                      className="result-hit"
                      title="View full size"
                      onClick={() => setLightbox(result)}
                    >
                      <video
                        src={mediaUrl(result)}
                        muted
                        playsInline
                        preload="metadata"
                        poster={thumbUrl(result) || undefined}
                      />
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="result-hit"
                      title="View full size"
                      onClick={() => setLightbox(result)}
                    >
                      <img
                        src={mediaUrl(result)}
                        alt={result.prompt || "result"}
                        loading="lazy"
                        decoding="async"
                      />
                    </button>
                  )}
                </div>

                <div className="crud">
                  {editing ? (
                    <label className="field">
                      <span>edit prompt</span>
                      <textarea
                        rows={3}
                        value={editPrompt}
                        onChange={(e) => setEditPrompt(e.target.value)}
                      />
                    </label>
                  ) : (
                    <p className="result-prompt">{result.prompt || "—"}</p>
                  )}
                  <div className="crud-actions">
                    <button
                      type="button"
                      className="action"
                      disabled={busyId === result.id || loading}
                      onClick={() => onDownload(result)}
                    >
                      download
                    </button>
                    {resultIsImage && (
                      <button
                        type="button"
                        className="ghost"
                        disabled={busyId === result.id || loading}
                        onClick={() => useAsInput(result)}
                      >
                        use as input
                      </button>
                    )}
                    {editing ? (
                      <>
                        <button
                          type="button"
                          className="ghost"
                          disabled={busyId === result.id}
                          onClick={() => {
                            setEditPrompt(result.prompt || "");
                            setEditing(false);
                          }}
                        >
                          cancel
                        </button>
                        <button
                          type="button"
                          className="action"
                          disabled={busyId === result.id}
                          onClick={onSavePrompt}
                        >
                          save
                        </button>
                      </>
                    ) : (
                      <button
                        type="button"
                        className="ghost"
                        disabled={busyId === result.id}
                        onClick={() => setEditing(true)}
                      >
                        edit
                      </button>
                    )}
                    <button
                      type="button"
                      className="ghost danger"
                      disabled={busyId === result.id}
                      onClick={() => onDelete(result.id)}
                    >
                      delete
                    </button>
                  </div>
                </div>
                {tester && resultRefs.length > 0 && (
                  <div className="result-refs">
                    <div className="section-head">
                      <h3 className="refs-sub">
                        References
                        {lastPresetId
                          ? ` · ${presetById(lastPresetId)?.label || lastPresetId}`
                          : ""}
                      </h3>
                    </div>
                    <div className="gallery ref-gallery">
                      {resultRefs.map((item) => (
                        <button
                          key={item.id}
                          type="button"
                          className="gallery-item"
                          onClick={() =>
                            setLightbox({ ...item, _testRef: true, kind: "img" })
                          }
                        >
                          <img src={testRefThumbUrl(item)} alt="" loading="lazy" />
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        )}

        {view === "inputs" && tester && (
          <section className="panel view-panel">
            <TestInputs
              onOpenLightbox={(item) => setLightbox(item)}
              onUseInput={(item) => useAsInput(item)}
            />
          </section>
        )}

        {view === "refs" && tester && (
          <section className="panel view-panel">
            <TestRefs onOpenLightbox={(item) => setLightbox(item)} />
          </section>
        )}

        {view === "review" && tester && (
          <section className="panel view-panel">
            <TestReview onOpenLightbox={(item) => setLightbox(item)} />
          </section>
        )}

        {view === "ongoing" && (
          <section className="panel view-panel history">
            <div className="history-head">
              <h2>Ongoing</h2>
              <button type="button" className="ghost" onClick={refreshOngoing}>
                refresh
              </button>
            </div>
            <p className="ongoing-note">
              Live jobs on the server. Cancel frees the GPU for the next one.
            </p>
            <ul className="ongoing-list">
              {!ongoing.length && (
                <li className="empty">No active generations right now.</li>
              )}
              {ongoing.map((job) => {
                const elapsed = formatElapsed(job.started_at || job.created_at);
                return (
                  <li key={job.id} className="ongoing-card">
                    <div className="ongoing-main">
                      <div className="ongoing-top">
                        <strong className="ongoing-mode">{job.mode || "—"}</strong>
                        <span className={`ongoing-status st-${job.status}`}>
                          {job.status}
                        </span>
                        <span className="ongoing-time">{elapsed}</span>
                      </div>
                      <p className="ongoing-prompt">
                        {(job.prompt || "—").slice(0, 160)}
                        {(job.prompt || "").length > 160 ? "…" : ""}
                      </p>
                      {job.mode === "vid" && job.video_seconds != null && (
                        <p className="ongoing-meta">{job.video_seconds}s video</p>
                      )}
                    </div>
                    <button
                      type="button"
                      className="ongoing-cancel"
                      disabled={busyId === job.id}
                      onClick={() => onCancelJob(job)}
                    >
                      {busyId === job.id ? "…" : "Cancel"}
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        )}

        {view === "library" && (
          <section className="panel view-panel history">
            <div className="history-head">
              <h2>Library</h2>
              <button
                type="button"
                className="ghost"
                onClick={() => refreshLibrary(libraryPage)}
              >
                refresh
              </button>
            </div>
            {libraryTotal > 0 && (
              <div className="library-pager" role="navigation" aria-label="Library pages">
                <button
                  type="button"
                  className="ghost pager-btn"
                  disabled={libraryPage <= 0}
                  onClick={() => setLibraryPage((p) => Math.max(0, p - 1))}
                >
                  ← Prev
                </button>
                <span className="pager-meta">
                  {libraryFrom}–{libraryTo} of {libraryTotal}
                  {libraryPageCount > 1
                    ? ` · page ${libraryPage + 1}/${libraryPageCount}`
                    : ""}
                </span>
                <button
                  type="button"
                  className="ghost pager-btn"
                  disabled={libraryPage + 1 >= libraryPageCount}
                  onClick={() =>
                    setLibraryPage((p) =>
                      p + 1 >= libraryPageCount ? p : p + 1
                    )
                  }
                >
                  Next →
                </button>
              </div>
            )}
            <ul className="gallery">
              {!library.length && (
                <li className="empty gallery-empty">No generations in MongoDB yet.</li>
              )}
              {library.map((item) => {
                const isVid =
                  item.kind === "vid" ||
                  (item.content_type || "").startsWith("video/");
                const isRecent = !item?.meta?.opened_at;
                return (
                  <li
                    key={item.id}
                    className={`gallery-item${isRecent ? " is-recent" : ""}`}
                  >
                    <button
                      type="button"
                      className="gallery-cell"
                      title={isVid ? "View video" : "View photo"}
                      onClick={() => selectItem(item)}
                    >
                      {isVid ? (
                        <span className="gallery-thumb gallery-thumb-video">
                          {thumbUrl(item) ? (
                            <img
                              src={thumbUrl(item)}
                              alt=""
                              loading="lazy"
                              decoding="async"
                            />
                          ) : (
                            <span className="gallery-ph" aria-hidden="true" />
                          )}
                          <span className="gallery-play" aria-hidden="true">
                            ▶
                          </span>
                        </span>
                      ) : (
                        <img
                          className="gallery-thumb"
                          src={thumbUrl(item)}
                          alt=""
                          loading="lazy"
                          decoding="async"
                        />
                      )}
                      <span className="gallery-kind">{item.kind}</span>
                      {isRecent && (
                        <span className="gallery-recent-tag">Recently generated</span>
                      )}
                    </button>
                    <div className="gallery-actions">
                      <button
                        type="button"
                        className="gallery-act"
                        title="Download"
                        disabled={busyId === item.id}
                        onClick={() => onDownload(item)}
                      >
                        ↓
                      </button>
                      <button
                        type="button"
                        className="gallery-act danger"
                        title="Delete"
                        disabled={busyId === item.id}
                        onClick={() => onDelete(item.id)}
                      >
                        ×
                      </button>
                    </div>
                  </li>
                );
              })}
            </ul>
            {libraryTotal > LIBRARY_PAGE_SIZE && (
              <div className="library-pager library-pager-bottom" role="navigation" aria-label="Library pages bottom">
                <button
                  type="button"
                  className="ghost pager-btn"
                  disabled={libraryPage <= 0}
                  onClick={() => setLibraryPage((p) => Math.max(0, p - 1))}
                >
                  ← Prev
                </button>
                <span className="pager-meta">
                  page {libraryPage + 1}/{libraryPageCount}
                </span>
                <button
                  type="button"
                  className="ghost pager-btn"
                  disabled={libraryPage + 1 >= libraryPageCount}
                  onClick={() =>
                    setLibraryPage((p) =>
                      p + 1 >= libraryPageCount ? p : p + 1
                    )
                  }
                >
                  Next →
                </button>
              </div>
            )}
          </section>
        )}

        {admin && (
          <section
            id="ops-controls"
            className="ops-panel"
            aria-label="Admin controls"
          >
            <button
              type="button"
              className="ops-toggle"
              aria-expanded={opsOpen}
              onClick={() => setOpsOpen((v) => !v)}
            >
              Controls {opsOpen ? "▾" : "▸"}
            </button>
            {opsOpen && (
              <div className="ops-body">
                <p className="ops-note">
                  Fixes for when generation won’t start or the GPU looks offline.
                  If the whole app is unreachable, restart from your other PC instead.
                </p>
                <div className="ops-actions">
                  <button
                    type="button"
                    className="ghost"
                    disabled={Boolean(opsBusy)}
                    onClick={() => runOps("health")}
                  >
                    {opsBusy === "health" ? "Checking…" : "Check status"}
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    disabled={Boolean(opsBusy) || opsCaps?.restart_api === false}
                    title={
                      opsCaps?.restart_api === false
                        ? "Not set up yet on the cloud server"
                        : "Restart the cloud server"
                    }
                    onClick={() => runOps("restart-api")}
                  >
                    {opsBusy === "restart-api" ? "Restarting…" : "Restart server"}
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    disabled={
                      Boolean(opsBusy) || opsCaps?.restart_comfy === false
                    }
                    title={
                      opsCaps?.restart_comfy === false
                        ? "GPU helper isn’t set up yet"
                        : "Restart the GPU app on your PC"
                    }
                    onClick={() => runOps("restart-comfy")}
                  >
                    {opsBusy === "restart-comfy" ? "Restarting…" : "Restart GPU"}
                  </button>
                  <button
                    type="button"
                    className="ghost"
                    disabled={Boolean(opsBusy)}
                    onClick={() => runOps("scrub")}
                  >
                    {opsBusy === "scrub" ? "Clearing…" : "Clear temp files"}
                  </button>
                </div>
                <label className="ops-tunnel">
                  <span>New GPU tunnel link</span>
                  <div className="ops-tunnel-row">
                    <input
                      type="url"
                      value={tunnelInput}
                      placeholder="https://….trycloudflare.com"
                      onChange={(e) => setTunnelInput(e.target.value)}
                      disabled={Boolean(opsBusy)}
                    />
                    <button
                      type="button"
                      className="action"
                      disabled={
                        Boolean(opsBusy) || opsCaps?.set_tunnel === false
                      }
                      onClick={() => runOps("set-tunnel")}
                    >
                      {opsBusy === "set-tunnel" ? "Saving…" : "Connect"}
                    </button>
                  </div>
                </label>
                {opsCaps && (
                  <p className="ops-caps">
                    Available:{" "}
                    {[
                      opsCaps.restart_api ? "restart server" : null,
                      opsCaps.restart_comfy ? "restart GPU" : null,
                      opsCaps.set_tunnel ? "connect tunnel" : null,
                      "clear temp files",
                    ]
                      .filter(Boolean)
                      .join(" · ") || "check status only"}
                  </p>
                )}
              </div>
            )}
          </section>
        )}

        <footer className="foot">
          <span>{healthText}</span>
          {installHint && (
            <span className="install-hint">
              Install: browser menu → Add to Home Screen (runs as app, keeps job id)
            </span>
          )}
        </footer>
      </main>

      <nav
        className={`bottom-nav${tester ? " bottom-nav-tester" : ""}`}
        aria-label="Mobile"
      >
        <button
          type="button"
          className={`bottom-item${view === "generation" ? " active" : ""}`}
          onClick={() => setView("generation")}
        >
          <IconSpark className="bottom-ico" />
          <span>Generate</span>
        </button>
        {tester && (
          <button
            type="button"
            className={`bottom-item${view === "inputs" ? " active" : ""}`}
            onClick={() => setView("inputs")}
          >
            <IconImage className="bottom-ico" />
            <span>Inputs</span>
          </button>
        )}
        {tester && (
          <button
            type="button"
            className={`bottom-item${view === "refs" ? " active" : ""}`}
            onClick={() => setView("refs")}
          >
            <IconFolder className="bottom-ico" />
            <span>Refs</span>
          </button>
        )}
        {tester && (
          <button
            type="button"
            className={`bottom-item${view === "review" ? " active" : ""}`}
            onClick={() => setView("review")}
          >
            <IconReview className="bottom-ico" />
            <span>Review</span>
          </button>
        )}
        <button
          type="button"
          className={`bottom-item${view === "ongoing" ? " active" : ""}`}
          onClick={goOngoing}
        >
          <IconJobs className="bottom-ico" />
          <span>Jobs</span>
          {ongoing.length > 0 && (
            <span className="bottom-badge">{ongoing.length}</span>
          )}
        </button>
        <button
          type="button"
          className="bottom-fab"
          onClick={goNewGeneration}
          aria-label="New generation"
        >
          <span aria-hidden="true">+</span>
        </button>
        <button
          type="button"
          className={`bottom-item${view === "library" ? " active" : ""}`}
          onClick={goLibrary}
        >
          <IconLibrary className="bottom-ico" />
          <span>Library</span>
        </button>
        {admin ? (
          <button
            type="button"
            className={`bottom-item${opsOpen ? " active" : ""}`}
            onClick={toggleControls}
          >
            <IconControls className="bottom-ico" />
            <span>Controls</span>
          </button>
        ) : (
          <button type="button" className="bottom-item" onClick={lockApp}>
            <IconLock className="bottom-ico" />
            <span>Lock</span>
          </button>
        )}
      </nav>

      {lightbox && (
        <div
          className="lightbox"
          role="dialog"
          aria-modal="true"
          aria-label="Gallery preview"
          onClick={() => setLightbox(null)}
        >
          <div
            className="lightbox-panel"
            onClick={(e) => e.stopPropagation()}
            onTouchStart={onLightboxTouchStart}
            onTouchEnd={onLightboxTouchEnd}
          >
            <div className="lightbox-toolbar">
              <button
                type="button"
                className="ghost"
                onClick={() => setLightbox(null)}
              >
                close
              </button>
              <span className="lightbox-pos">
                {Math.max(
                  1,
                  library.findIndex((x) => x.id === lightbox.id) + 1
                )}
                /{Math.max(library.length, 1)}
              </span>
              <div className="lightbox-actions">
                <button
                  type="button"
                  className="action"
                  disabled={busyId === lightbox.id}
                  onClick={() => onDownload(lightbox)}
                >
                  download
                </button>
                {(lightbox.kind !== "vid" &&
                  !(lightbox.content_type || "").startsWith("video/")) && (
                  <button
                    type="button"
                    className="ghost"
                    disabled={busyId === lightbox.id || loading}
                    onClick={() => {
                      setLightbox(null);
                      useAsInput(lightbox);
                    }}
                  >
                    use as input
                  </button>
                )}
              </div>
            </div>
            <div className="lightbox-stage">
              {library.length > 1 && (
                <button
                  type="button"
                  className="lightbox-nav prev"
                  aria-label="Previous"
                  onClick={() => stepLightbox(-1)}
                >
                  ‹
                </button>
              )}
              {lightbox.kind === "vid" ||
              (lightbox.content_type || "").startsWith("video/") ? (
                <video
                  className="lightbox-media"
                  src={mediaUrl(lightbox)}
                  controls
                  playsInline
                  autoPlay
                />
              ) : (
                <img
                  className="lightbox-media"
                  src={mediaUrl(lightbox)}
                  alt={lightbox.prompt || "preview"}
                  draggable={false}
                />
              )}
              {library.length > 1 && (
                <button
                  type="button"
                  className="lightbox-nav next"
                  aria-label="Next"
                  onClick={() => stepLightbox(1)}
                >
                  ›
                </button>
              )}
            </div>
            <p className="lightbox-caption">{lightbox.prompt || "—"}</p>
            <p className="lightbox-hint">Swipe or use ← → to browse</p>
          </div>
        </div>
      )}
    </>
  );
}
