/**
 * Wipe the media this app generated on this PC — nothing else. Does not touch MongoDB / Atlas / GridFS.
 *
 * temp_assets/ is the one folder the app, tests and Mongo pulls write media into; it is emptied
 * (except .gitkeep). Also cleared, because the app or ComfyUI writes there directly:
 *   - legacy scratch: tmp_test/ sub-folders + root media, repo outputs/ temp/ tmp/ temptest_assets/
 *   - ComfyUI input/output/temp (install + shared) and ComfyUI history
 *   - E:\LoraTraining\work (old trainer work dir), OS temp wan_thumb_* dirs
 *   - files the site downloaded into ~/Downloads (wan_*, flux_i2i*, flux_kontext*, <id>_input/_thumb.jpg)
 * ComfyUI dirs and temp_assets/train are skipped while a gen or training is running
 * (it would break that run) unless --force is given.
 *
 * Usage (repo root):  npm run wipe
 * Dry run:            npm run wipe -- --dry-run
 * Even mid-run:       npm run wipe -- --force
 */
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, "..");
const DRY = process.argv.includes("--dry-run");
const FORCE = process.argv.includes("--force");

const MEDIA_EXT = new Set([
  ".png",
  ".jpg",
  ".jpeg",
  ".webp",
  ".gif",
  ".mp4",
  ".webm",
  ".mov",
  ".avi",
  ".mkv",
  ".bmp",
]);

const TEMP_ASSETS = path.join(REPO, "temp_assets");
const DEFAULT_COMFY =
  "E:\\Comfy-Desktop\\ComfyUI-Installs\\Khelukhiladi\\ComfyUI";
const COMFY_SHARED = "E:\\Comfy-Desktop\\ComfyUI-Shared";
const COMFY_URL = "http://127.0.0.1:8188";
const LORA_WORK = "E:\\LoraTraining\\work";

function readEnv(key) {
  const envPath = path.join(REPO, ".env");
  if (!fs.existsSync(envPath)) return "";
  const text = fs.readFileSync(envPath, "utf8");
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const i = line.indexOf("=");
    if (i < 1) continue;
    if (line.slice(0, i).trim() === key) {
      return line.slice(i + 1).trim().replace(/^["']|["']$/g, "");
    }
  }
  return "";
}

const stats = { files: 0, dirs: 0, skipped: 0, errors: 0 };

function log(msg) {
  console.log(msg);
}

const handled = new Set();

function unlinkFile(file) {
  const key = path.resolve(file).toLowerCase();
  if (handled.has(key)) return;
  handled.add(key);
  try {
    if (DRY) {
      log(`  dry  ${file}`);
    } else {
      try {
        fs.chmodSync(file, 0o666);
      } catch {
        /* ignore */
      }
      fs.unlinkSync(file);
      log(`  del  ${file}`);
    }
    stats.files += 1;
  } catch (err) {
    stats.errors += 1;
    log(`  err  ${file}  (${err.message})`);
  }
}

function rmdirIfEmpty(dir) {
  try {
    if (!fs.existsSync(dir)) return;
    const left = fs.readdirSync(dir);
    if (left.length) return;
    if (!DRY) fs.rmdirSync(dir);
    stats.dirs += 1;
  } catch {
    /* keep parent folders Comfy expects */
  }
}

function walkFiles(root) {
  const out = [];
  if (!fs.existsSync(root)) return out;
  const stack = [root];
  while (stack.length) {
    const cur = stack.pop();
    let entries;
    try {
      entries = fs.readdirSync(cur, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of entries) {
      const full = path.join(cur, e.name);
      if (e.isDirectory()) stack.push(full);
      else if (e.isFile()) out.push(full);
    }
  }
  return out;
}

function wipeTree(root, { keepRoot = true, filter = null } = {}) {
  if (!fs.existsSync(root)) {
    stats.skipped += 1;
    log(`skip  ${root}  (missing)`);
    return;
  }
  log(`${DRY ? "scan" : "wipe"} ${root}`);
  const files = walkFiles(root);
  const dirs = [];
  for (const file of files) {
    if (filter && !filter(file)) continue;
    unlinkFile(file);
    dirs.push(path.dirname(file));
  }
  const uniqueDirs = [...new Set(dirs)].sort(
    (a, b) => b.length - a.length || b.localeCompare(a),
  );
  for (const dir of uniqueDirs) {
    if (path.resolve(dir) === path.resolve(root) && keepRoot) continue;
    rmdirIfEmpty(dir);
  }
  if (!keepRoot && fs.existsSync(root)) {
    try {
      if (!DRY) fs.rmSync(root, { recursive: true, force: true });
      stats.dirs += 1;
      log(`  rmdir ${root}`);
    } catch (err) {
      stats.errors += 1;
      log(`  err  ${root}  (${err.message})`);
    }
  }
}

function isMedia(file) {
  return MEDIA_EXT.has(path.extname(file).toLowerCase());
}

function isWanDownload(file) {
  const name = path.basename(file);
  if (!isMedia(file)) return false;
  return (
    /^(wan_|flux_i2i|flux_kontext|garment_mask)/i.test(name) ||
    /^[0-9a-f]{24}_(input|thumb)\.jpe?g$/i.test(name)
  );
}

async function comfyBusy() {
  try {
    const res = await fetch(`${COMFY_URL}/queue`, { signal: AbortSignal.timeout(5000) });
    const q = await res.json();
    return (q.queue_running || []).length + (q.queue_pending || []).length > 0;
  } catch {
    return false;
  }
}

function trainingRunning() {
  try {
    const out = execFileSync(
      "powershell",
      [
        "-NoProfile",
        "-Command",
        "(Get-CimInstance Win32_Process -Filter \"CommandLine like '%flux_train_network%' and Name <> 'powershell.exe'\" | Measure-Object).Count",
      ],
      { encoding: "utf8", timeout: 60000 },
    );
    return Number(out.trim()) > 0;
  } catch {
    return false;
  }
}

async function clearComfyHistory() {
  if (DRY) {
    log(`  dry  ${COMFY_URL}/history (clear)`);
    return;
  }
  try {
    await fetch(`${COMFY_URL}/history`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ clear: true }),
      signal: AbortSignal.timeout(5000),
    });
    log(`  clr  ${COMFY_URL}/history`);
  } catch {
    stats.skipped += 1;
    log(`skip  ${COMFY_URL}/history  (ComfyUI not running)`);
  }
}

/** Root-level generation leftovers under tmp_test (not tunnel/watchdog logs or helper scripts). */
function isTmpTestRootTrace(file) {
  const name = path.basename(file).toLowerCase();
  if (isMedia(file)) return true;
  if (name.endsWith(".pid")) return true;
  if (name.endsWith("_run.log") || name.endsWith("_run.txt")) return true;
  if (name === "job_status.txt") return true;
  if (name === "manifest.json" || name === "manifest_all.json") return true;
  return false;
}

const comfyDir = readEnv("COMFYUI_DIR") || DEFAULT_COMFY;
const training = !FORCE && trainingRunning();
const comfyRunning = !FORCE && (await comfyBusy());

log(DRY ? "Local media wipe (dry run) — Mongo/Atlas not touched" : "Local media wipe — Mongo/Atlas not touched");
log("");

// temp_assets: everything except .gitkeep. train/ holds the live dataset while training.
fs.mkdirSync(TEMP_ASSETS, { recursive: true });
log(`${DRY ? "scan" : "wipe"} ${TEMP_ASSETS}`);
for (const e of fs.readdirSync(TEMP_ASSETS, { withFileTypes: true })) {
  const full = path.join(TEMP_ASSETS, e.name);
  if (e.name === ".gitkeep") continue;
  if (e.isDirectory()) {
    if (e.name === "train" && training) {
      stats.skipped += 1;
      log(`skip  ${full}  (training is running — rerun later or use --force)`);
      continue;
    }
    wipeTree(full, { keepRoot: false });
  } else if (e.isFile()) {
    unlinkFile(full);
  }
}

// Legacy tmp_test scratch: sub-folders are generation dumps; root keeps logs, tunnel URLs, helpers.
const tmpTest = path.join(REPO, "tmp_test");
if (fs.existsSync(tmpTest)) {
  let wipedRootTrace = false;
  for (const e of fs.readdirSync(tmpTest, { withFileTypes: true })) {
    const full = path.join(tmpTest, e.name);
    if (e.isDirectory()) {
      wipeTree(full, { keepRoot: false });
    } else if (e.isFile() && isTmpTestRootTrace(full)) {
      if (!wipedRootTrace) {
        log(`${DRY ? "scan" : "wipe"} ${tmpTest} (root traces)`);
        wipedRootTrace = true;
      }
      unlinkFile(full);
    }
  }
}

for (const folder of ["outputs", "temp", "tmp", "temptest_assets"]) {
  const full = path.join(REPO, folder);
  if (fs.existsSync(full)) wipeTree(full, { keepRoot: false });
}

if (comfyRunning || training) {
  stats.skipped += 1;
  log("skip  ComfyUI + trainer dirs  (a gen or training is running — rerun later or use --force)");
} else {
  const comfyRoots = [...new Set([comfyDir, COMFY_SHARED].map((p) => path.resolve(p)))];
  for (const root of comfyRoots) {
    for (const side of ["input", "output", "temp"]) {
      wipeTree(path.join(root, side), { keepRoot: true });
    }
  }
  await clearComfyHistory();
  if (fs.existsSync(LORA_WORK)) wipeTree(LORA_WORK, { keepRoot: true });
}

const tempRoot = os.tmpdir();
for (const e of fs.existsSync(tempRoot) ? fs.readdirSync(tempRoot, { withFileTypes: true }) : []) {
  if (e.isDirectory() && e.name.toLowerCase().startsWith("wan_thumb_")) {
    wipeTree(path.join(tempRoot, e.name), { keepRoot: false });
  }
}

const downloads = path.join(os.homedir(), "Downloads");
wipeTree(downloads, { keepRoot: true, filter: isWanDownload });

log("");
log(
  `${DRY ? "Would remove" : "Removed"} ${stats.files} file(s), ${stats.dirs} dir(s). skipped=${stats.skipped} errors=${stats.errors}`,
);
log("MongoDB Atlas / GridFS was not modified.");
if (stats.errors) process.exit(1);
