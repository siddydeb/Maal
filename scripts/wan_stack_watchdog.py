#!/usr/bin/env python3
"""Keep Wan GPU stack alive so you can recover from another PC.

Run ON the GPU PC (leave it on / awake). This watchdog:

1. Ensures ComfyUI is up on :8188
2. Ensures a Cloudflare quick tunnel to Comfy; if the URL changes, updates
   Render ``COMFYUI_URL`` automatically
3. Ensures ``gpu_agent`` on :8799 (Restart GPU from Controls / phone)
4. Ensures a second tunnel for the agent; updates Render ``GPU_AGENT_URL``
5. Ensures Prowler Control (UI+API) on :8010 + a third Cloudflare tunnel
   (URL saved as ``prowler_url`` in tokens&cmd)
6. Ensures Scapper dashboard on :8000 + a fourth Cloudflare tunnel
   (URL saved as ``scapper_url`` in tokens&cmd))

Requires gitignored ``tokens&cmd`` with at least ``render=<api key>``.
Optional keys (auto-filled on first run if missing):
  gpu_agent_secret=…
  GPU_COMFY_CMD=…   (Windows launch command for Comfy)

Usage:
  python scripts/wan_stack_watchdog.py
  python scripts/wan_stack_watchdog.py --once   # single heal pass

Install at login (PowerShell as user):
  powershell -ExecutionPolicy Bypass -File scripts/install_wan_watchdog_task.ps1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _dpapi(data: bytes, protect: bool) -> bytes:
    """Windows DPAPI (current user) — runtime config is unreadable to other accounts."""
    import ctypes
    from ctypes import wintypes

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    buf = ctypes.create_string_buffer(data, len(data))
    inp = _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = _Blob()
    crypt32 = ctypes.windll.crypt32
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not fn(ctypes.byref(inp), None, None, None, None, 0x1, ctypes.byref(out)):
        raise ctypes.WinError()
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out.pbData)


def _cfg_read(path: Path) -> dict[str, str]:
    try:
        return json.loads(_dpapi(path.read_bytes(), protect=False).decode("utf-8"))
    except Exception:
        return {}


# The repo may be ACL-locked (VaultLock) while the stack runs. A copy of this
# script + gpu_agent + a DPAPI-encrypted subset of tokens&cmd lives in RUNTIME
# (hidden, neutral name) so restarts and tunnel healing never need the repo.
HERE = Path(__file__).resolve().parent
RUNTIME_CFG_NAME = "cfg.bin"
RUNTIME_MAIN_NAME = "svc_main.py"
RUNTIME_AGENT_NAME = "svc_agent.py"
IN_RUNTIME = (HERE / RUNTIME_CFG_NAME).is_file()
if IN_RUNTIME:
    RUNTIME = HERE
    REPO = Path(_cfg_read(HERE / RUNTIME_CFG_NAME).get("_repo") or HERE)
else:
    REPO = HERE.parent
    RUNTIME = Path(os.environ.get("WAN_RUNTIME_DIR") or (REPO.parent / "svc"))
TOKENS = REPO / "tokens&cmd"
RUNTIME_CFG = RUNTIME / RUNTIME_CFG_NAME
# Only what the heal loop needs — never github/vercel tokens.
RUNTIME_KEYS = {
    "render",
    "gpu_agent_secret",
    "GPU_COMFY_CMD",
    "gpu_comfy_cmd",
    "named_tunnel",
    "gpu_comfy_url",
    "COMFYUI_URL",
    "gpu_agent",
    "prowler_url",
    "prowler_pin",
    "PROWLER_PIN",
    "prowler_auth_secret",
    "PROWLER_AUTH_SECRET",
    "scapper_url",
    "scapper_url_pushed",
    "scapper_render_service",
}
COMFY_ROOT = Path(r"E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI")
PROWLER_ROOT = Path(r"D:\prowler")
PROWLER_API = PROWLER_ROOT / "apps" / "api"
PROWLER_PYTHON = PROWLER_API / ".venv" / "Scripts" / "python.exe"
SCAPPER_ROOT = Path(r"D:\Scapper")
SCAPPER_PYTHON = SCAPPER_ROOT / ".venv" / "Scripts" / "python.exe"
SCAPPER_TOKENS = SCAPPER_ROOT / "tokens&cmd"
NAMED_TUNNEL_CONFIG = SCAPPER_ROOT / "deploy" / "tunnel" / "config.yml"
CLOUDFLARED = Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe")
RENDER_API = "https://api.render.com/v1"
SERVICE_ID = "srv-d9ot8spt0dsc73bqjv0g"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)
# Back off creating new quick tunnels after Cloudflare 429s
_QUICK_TUNNEL_COOLDOWN_UNTIL = 0.0

DEFAULT_COMFY_CMD = (
    f'set TQDM_DISABLE=1&& "{COMFY_ROOT / ".venv" / "Scripts" / "python.exe"}" '
    f'main.py --listen 0.0.0.0 --port 8188'
)


def log(msg: str) -> None:
    print(f"[watchdog] {msg}", flush=True)


def _cfg_write(values: dict[str, str]) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RUNTIME_CFG.write_bytes(_dpapi(json.dumps(values).encode("utf-8"), protect=True))


def _cfg_update(key: str, value: str) -> None:
    if key not in RUNTIME_KEYS or not RUNTIME_CFG.is_file():
        return
    cfg = _cfg_read(RUNTIME_CFG)
    cfg[key] = value
    _cfg_write(cfg)


def tokens_from_repo() -> bool:
    try:
        TOKENS.stat()
        return True
    except OSError:
        return False


def load_tokens() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = TOKENS.read_text(encoding="utf-8", errors="replace")
    except OSError:
        cfg = _cfg_read(RUNTIME_CFG)
        if not cfg:
            raise SystemExit(f"Cannot read {TOKENS} and no runtime config at {RUNTIME_CFG}")
        log("tokens&cmd unreadable (repo locked?) — using runtime config")
        return {k: v for k, v in cfg.items() if not k.startswith("_")}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _upsert_token_file(path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    found = False
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith(f"{key}="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def upsert_token(key: str, value: str) -> None:
    try:
        _upsert_token_file(TOKENS, key, value)
    except OSError as exc:
        # Repo locked: keep the in-memory value; Render still gets pushed.
        log(f"could not write {key} to tokens&cmd: {exc}")
    try:
        _cfg_update(key, value)
    except Exception as exc:
        log(f"could not write {key} to runtime config: {exc}")
    # Keep Scapper's gitignored tokens file in sync for scapper_url (and shared keys).
    if key == "scapper_url" and SCAPPER_TOKENS != TOKENS:
        try:
            _upsert_token_file(SCAPPER_TOKENS, key, value)
        except OSError as exc:
            log(f"could not sync {key} to Scapper tokens&cmd: {exc}")


def ensure_secrets(tokens: dict[str, str]) -> dict[str, str]:
    if not (tokens.get("gpu_agent_secret") or "").strip():
        secret = secrets.token_urlsafe(24)
        upsert_token("gpu_agent_secret", secret)
        tokens["gpu_agent_secret"] = secret
        log("wrote gpu_agent_secret=… into tokens&cmd")
    if not (tokens.get("GPU_COMFY_CMD") or tokens.get("gpu_comfy_cmd") or "").strip():
        upsert_token("GPU_COMFY_CMD", DEFAULT_COMFY_CMD)
        tokens["GPU_COMFY_CMD"] = DEFAULT_COMFY_CMD
        log("wrote GPU_COMFY_CMD default into tokens&cmd")
    return tokens


def port_open(port: int) -> bool:
    import socket

    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def http_ok(url: str, timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return int(r.status) == 200
    except Exception:
        return False


def _no_window_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _new_console_flags() -> int:
    return int(getattr(subprocess, "CREATE_NEW_CONSOLE", 0))


def ensure_comfy(tokens: dict[str, str]) -> None:
    if http_ok("http://127.0.0.1:8188/system_stats"):
        return
    log("Comfy down — starting…")
    cmd = (
        tokens.get("GPU_COMFY_CMD")
        or tokens.get("gpu_comfy_cmd")
        or DEFAULT_COMFY_CMD
    )
    env = os.environ.copy()
    env["TQDM_DISABLE"] = "1"
    subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(COMFY_ROOT if COMFY_ROOT.is_dir() else REPO),
        env=env,
        # Hidden — phone stack must not depend on a visible Comfy console.
        creationflags=_no_window_flags() or _new_console_flags(),
    )
    for _ in range(60):
        time.sleep(2)
        if http_ok("http://127.0.0.1:8188/system_stats"):
            log("Comfy is up")
            return
    log("WARNING: Comfy did not become healthy in time")


def _cloudflared_pids_for(target_port: int) -> list[int]:
    """PIDs of cloudflared quick tunnels pointed at 127.0.0.1:port.

    Prefer PowerShell CIM — ``wmic`` is missing/broken on many Win11 installs,
    which made the watchdog think no tunnel existed and spawn duplicates.
    """
    if sys.platform != "win32":
        return []
    needle = f"127.0.0.1:{int(target_port)}"
    # ProcessId|CommandLine per line (CommandLine may contain commas)
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\" "
        "| ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    )
    out = ""
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            errors="replace",
            creationflags=_no_window_flags(),
        )
    except Exception:
        out = ""
    if not out.strip():
        # Last-resort fallback for older boxes that still have wmic
        try:
            out = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    "name='cloudflared.exe'",
                    "get",
                    "ProcessId,CommandLine",
                    "/FORMAT:LIST",
                ],
                text=True,
                errors="replace",
                creationflags=_no_window_flags(),
            )
        except Exception:
            return []
        pids: list[int] = []
        cur_cmd = ""
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("CommandLine="):
                cur_cmd = line.split("=", 1)[1]
            elif line.startswith("ProcessId="):
                raw = line.split("=", 1)[1].strip()
                if raw.isdigit() and needle in cur_cmd:
                    pids.append(int(raw))
                cur_cmd = ""
        return pids

    pids = []
    for line in out.splitlines():
        line = line.strip()
        if "|" not in line:
            continue
        raw_pid, cmd = line.split("|", 1)
        if raw_pid.isdigit() and needle in cmd:
            pids.append(int(raw_pid))
    return pids


def _kill_cloudflared_for(target_port: int, *, keep_pid: int | None = None) -> None:
    """Kill cloudflared processes for a local port (optionally keep one)."""
    if sys.platform != "win32":
        return
    for cur_pid in _cloudflared_pids_for(target_port):
        if keep_pid is not None and cur_pid == keep_pid:
            continue
        subprocess.run(
            ["taskkill", "/PID", str(cur_pid), "/F"],
            check=False,
            capture_output=True,
            creationflags=_no_window_flags(),
        )
        log(f"killed stale cloudflared pid {cur_pid} (→:{target_port})")


def _dedupe_cloudflared(target_port: int) -> int | None:
    """Ensure at most one tunnel per port. Returns surviving PID (or None)."""
    pids = _cloudflared_pids_for(target_port)
    if not pids:
        return None
    if len(pids) == 1:
        return pids[0]
    # Keep oldest PID — usually the URL already stored on Render.
    keep = min(pids)
    _kill_cloudflared_for(target_port, keep_pid=keep)
    log(f"deduped cloudflared :{target_port} — kept pid {keep}, removed {len(pids) - 1}")
    return keep


def tunnel_probe_url(local_port: int, base: str) -> str:
    base = base.rstrip("/")
    if local_port == 8188:
        return f"{base}/system_stats"
    if local_port == 8799:
        return f"{base}/status"
    if local_port == 8010:
        return f"{base}/api/health"
    if local_port == 8000:
        return f"{base}/health"
    return base


def start_quick_tunnel(local_port: int, log_path: Path) -> subprocess.Popen:
    global _QUICK_TUNNEL_COOLDOWN_UNTIL
    if time.time() < _QUICK_TUNNEL_COOLDOWN_UNTIL:
        raise RuntimeError(
            f"quick-tunnel cooldown active until {_QUICK_TUNNEL_COOLDOWN_UNTIL:.0f} "
            f"(Cloudflare rate limit) — not creating :{local_port}"
        )
    if not CLOUDFLARED.is_file():
        raise SystemExit(f"cloudflared not found at {CLOUDFLARED}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        [
            str(CLOUDFLARED),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{local_port}",
        ],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=_no_window_flags(),
    )
    log(f"started cloudflared -> :{local_port} (pid {proc.pid})")
    return proc


def wait_tunnel_url(log_path: Path, timeout: float = 45.0) -> str | None:
    global _QUICK_TUNNEL_COOLDOWN_UNTIL
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log_path.is_file():
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                text = ""
            if "429" in text or "1015" in text or "Too Many Requests" in text:
                _QUICK_TUNNEL_COOLDOWN_UNTIL = time.time() + 30 * 60
                log("Cloudflare quick-tunnel rate limit — cooldown 30m")
                return None
            m = URL_RE.findall(text)
            if m:
                return m[-1].rstrip("/")
        time.sleep(1)
    return None


def named_tunnel_mode(tokens: dict[str, str]) -> bool:
    flag = (tokens.get("named_tunnel") or "").strip().lower()
    return flag in {"1", "true", "yes", "on"} and NAMED_TUNNEL_CONFIG.is_file()


def _cloudflared_pids_named() -> list[int]:
    """PIDs of cloudflared running our named-tunnel config.yml."""
    if sys.platform != "win32":
        return []
    needle = str(NAMED_TUNNEL_CONFIG).replace("/", "\\").lower()
    needle2 = "deploy\\tunnel\\config.yml"
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='cloudflared.exe'\" "
        "| ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            errors="replace",
            creationflags=_no_window_flags(),
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        raw_pid, cmd = line.split("|", 1)
        cl = cmd.lower().replace("/", "\\")
        if raw_pid.isdigit() and (needle in cl or needle2 in cl):
            pids.append(int(raw_pid))
    return pids


def ensure_named_tunnel(tokens: dict[str, str], procs: dict[str, subprocess.Popen]) -> None:
    """One cloudflared process; hostnames in config.yml map to :8000/:8010/:8188/:8799."""
    scapper_u = (tokens.get("scapper_url") or "").rstrip("/")
    prowler_u = (tokens.get("prowler_url") or "").rstrip("/")
    comfy_u = (tokens.get("gpu_comfy_url") or tokens.get("COMFYUI_URL") or "").rstrip("/")
    # If stable public URLs already healthy, keep the named process only.
    healthy = True
    if scapper_u and not http_ok(f"{scapper_u}/health", timeout=12):
        healthy = False
    if prowler_u and not http_ok(f"{prowler_u}/api/auth/status", timeout=12):
        healthy = False
    if comfy_u and not http_ok(f"{comfy_u}/system_stats", timeout=12):
        healthy = False

    pids = _cloudflared_pids_named()
    proc = procs.get("named_tunnel")
    alive = proc is not None and proc.poll() is None
    if (alive or pids) and healthy:
        _dedupe_named(keep=min(pids) if pids else None)
        return

    if not CLOUDFLARED.is_file():
        log(f"WARNING: cloudflared missing at {CLOUDFLARED}")
        return
    if not NAMED_TUNNEL_CONFIG.is_file():
        log(f"WARNING: named tunnel config missing ({NAMED_TUNNEL_CONFIG})")
        return

    # Kill duplicate named runners, then start one
    for pid in pids:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            check=False,
            capture_output=True,
            creationflags=_no_window_flags(),
        )
    log_path = RUNTIME / "named_tunnel.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "ab", buffering=0)
    procs["named_tunnel"] = subprocess.Popen(
        [str(CLOUDFLARED), "tunnel", "--config", str(NAMED_TUNNEL_CONFIG), "run"],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        creationflags=_no_window_flags(),
    )
    log(f"started named tunnel (pid {procs['named_tunnel'].pid})")
    # Give DNS/edge a moment
    for _ in range(20):
        time.sleep(2)
        if scapper_u and http_ok(f"{scapper_u}/health", timeout=8):
            log(f"named tunnel healthy — {scapper_u}")
            return
    log("WARNING: named tunnel started but hostnames not healthy yet (DNS/propagation?)")


def _dedupe_named(*, keep: int | None) -> None:
    pids = _cloudflared_pids_named()
    if len(pids) <= 1:
        return
    keep_pid = keep if keep is not None else min(pids)
    for pid in pids:
        if pid == keep_pid:
            continue
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            check=False,
            capture_output=True,
            creationflags=_no_window_flags(),
        )


def render_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def list_env(token: str) -> list[dict]:
    req = urllib.request.Request(
        f"{RENDER_API}/services/{SERVICE_ID}/env-vars?limit=100",
        headers=render_headers(token),
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        rows = json.load(r)
    out: list[dict] = []
    for row in rows if isinstance(rows, list) else []:
        ev = row.get("envVar") if isinstance(row, dict) else None
        if not isinstance(ev, dict) or "key" not in ev:
            continue
        item: dict = {"key": str(ev["key"])}
        if ev.get("generateValue"):
            item["generateValue"] = True
        else:
            item["value"] = "" if ev.get("value") is None else str(ev["value"])
        out.append(item)
    return out


def set_env_and_deploy(token: str, updates: dict[str, str]) -> None:
    """Update only the given keys (safe single-var PUTs) then deploy.

    Never bulk-replace all env vars — Render often returns blank secret values
    on list, and a full PUT would wipe Mongo/Cloudinary/etc.
    """
    changed = False
    for key, value in updates.items():
        value = (value or "").strip()
        if not value:
            continue
        # Skip no-op when we can read the current public value
        try:
            current = {str(i["key"]): i for i in list_env(token)}
            prev = (current.get(key) or {}).get("value")
            if prev == value:
                continue
        except Exception:
            pass
        req = urllib.request.Request(
            f"{RENDER_API}/services/{SERVICE_ID}/env-vars/{key}",
            data=json.dumps({"value": value}).encode(),
            headers=render_headers(token),
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=90) as r:
            if int(r.status) not in (200, 201):
                raise RuntimeError(f"env update {key} HTTP {r.status}")
        changed = True
        if key in ("GPU_AGENT_SECRET", "RENDER_API_KEY") or "SECRET" in key.upper() or "TOKEN" in key.upper():
            log(f"Render env {key} -> (set)")
        else:
            log(f"Render env {key} -> {value}")
    if not changed:
        return
    dep = urllib.request.Request(
        f"{RENDER_API}/services/{SERVICE_ID}/deploys",
        data=json.dumps({"clearCache": "do_not_clear"}).encode(),
        headers=render_headers(token),
        method="POST",
    )
    try:
        with urllib.request.urlopen(dep, timeout=90) as r:
            log(f"Render deploy triggered HTTP {r.status}")
    except Exception as exc:
        # Private GitHub repo → Render deploy 404; restart still reloads env vars.
        log(f"Render deploy failed ({exc}); trying service restart…")
        restart = urllib.request.Request(
            f"{RENDER_API}/services/{SERVICE_ID}/restart",
            data=b"",
            headers=render_headers(token),
            method="POST",
        )
        with urllib.request.urlopen(restart, timeout=90) as r:
            log(f"Render restart triggered HTTP {r.status}")


def ensure_tunnel(
    *,
    local_port: int,
    state_key: str,
    render_env_key: str,
    tokens: dict[str, str],
    procs: dict[str, subprocess.Popen],
) -> None:
    log_path = RUNTIME / f"tunnel_{local_port}.log"
    state_path = RUNTIME / f"tunnel_{local_port}.url"

    known = ""
    if state_path.is_file():
        known = state_path.read_text(encoding="utf-8").strip().rstrip("/")

    api_url = ""
    api_comfy_ok = False
    if local_port == 8188 and http_ok("https://wan-studio-api.onrender.com/api/health", timeout=20):
        try:
            with urllib.request.urlopen(
                "https://wan-studio-api.onrender.com/api/health", timeout=20
            ) as r:
                health = json.load(r)
            api_comfy_ok = bool(health.get("comfyui"))
            api_url = str(health.get("comfyui_url") or "").rstrip("/")
        except Exception:
            pass

    # Only trust the API URL when Comfy is actually reachable through it
    if local_port == 8188 and api_comfy_ok and api_url and http_ok(f"{api_url}/system_stats", timeout=15):
        known = api_url
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(known + "\n", encoding="utf-8")
        log(f"comfy tunnel healthy via API ({known})")
        _dedupe_cloudflared(local_port)
        return

    # Stale quick-tunnel hostname (common after PC sleep / cloudflared restart)
    if known and not http_ok(tunnel_probe_url(local_port, known), timeout=12):
        log(f"stale tunnel URL for :{local_port} ({known}) — recycling")
        known = ""
        _kill_cloudflared_for(local_port)
        time.sleep(1)

    # Always collapse duplicate quick tunnels for this port
    surviving = _dedupe_cloudflared(local_port)

    proc = procs.get(state_key)
    alive = proc is not None and proc.poll() is None
    unmanaged = surviving is not None

    if (alive or unmanaged) and known:
        # Keep existing healthy tunnel
        pass
    else:
        _kill_cloudflared_for(local_port)
        time.sleep(1)
        if log_path.is_file():
            try:
                log_path.unlink()
            except OSError:
                pass
        procs[state_key] = start_quick_tunnel(local_port, log_path)
        url = wait_tunnel_url(log_path)
        if not url:
            log(f"WARNING: no trycloudflare URL yet for :{local_port}")
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(url + "\n", encoding="utf-8")
        known = url
        log(f"tunnel :{local_port} -> {url}")

    # Refresh known from log if process exists but state file empty
    if not known:
        known = (wait_tunnel_url(log_path, timeout=5.0) or "").rstrip("/")
        if known:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(known + "\n", encoding="utf-8")

    render_tok = (tokens.get("render") or "").strip()
    if not render_tok or not known or not (render_env_key or "").strip():
        if local_port == 8010 and known:
            upsert_token("prowler_url", known)
            tokens["prowler_url"] = known
        if local_port == 8000 and known:
            upsert_token("scapper_url", known)
            tokens["scapper_url"] = known
        return
    # Push to Render when API still points at a dead/different hostname
    if local_port == 8188 and api_url == known and api_comfy_ok:
        return
    try:
        updates = {render_env_key: known}
        if render_env_key == "GPU_AGENT_URL":
            updates["GPU_AGENT_SECRET"] = tokens["gpu_agent_secret"]
        set_env_and_deploy(render_tok, updates)
        if render_env_key == "GPU_AGENT_URL":
            upsert_token("gpu_agent", known)
            tokens["gpu_agent"] = known
        if local_port == 8010:
            upsert_token("prowler_url", known)
            tokens["prowler_url"] = known
        if local_port == 8000:
            upsert_token("scapper_url", known)
            tokens["scapper_url"] = known
    except Exception as exc:
        log(f"Render update failed ({render_env_key}): {exc}")


def ensure_gpu_agent(tokens: dict[str, str]) -> None:
    if port_open(8799) and http_ok("http://127.0.0.1:8799/status"):
        return
    log("gpu_agent down — starting…")
    env = os.environ.copy()
    env["GPU_AGENT_SECRET"] = tokens["gpu_agent_secret"]
    env["GPU_COMFY_CMD"] = (
        tokens.get("GPU_COMFY_CMD")
        or tokens.get("gpu_comfy_cmd")
        or DEFAULT_COMFY_CMD
    )
    env["GPU_COMFY_URL"] = "http://127.0.0.1:8188"
    agent = RUNTIME / RUNTIME_AGENT_NAME
    if not agent.is_file():
        agent = REPO / "scripts" / "gpu_agent.py"
    RUNTIME.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(
        [sys.executable, str(agent)],
        cwd=str(RUNTIME),
        env=env,
        creationflags=_no_window_flags(),
    )
    for _ in range(20):
        time.sleep(1)
        if http_ok("http://127.0.0.1:8799/status"):
            log("gpu_agent is up")
            return
    log("WARNING: gpu_agent did not become healthy")


TRAINER_HEARTBEAT = Path(r"E:\LoraTraining\trainer_worker.heartbeat")


def ensure_trainer() -> None:
    """Keep the character-LoRA trainer worker alive (it exits if one is already running)."""
    if not (Path(r"E:\LoraTraining\sd-scripts") / "venv").is_dir():
        return
    try:
        if time.time() - TRAINER_HEARTBEAT.stat().st_mtime < 90:
            return
    except OSError:
        pass
    worker = REPO / "scripts" / "character" / "trainer_worker.py"
    try:
        worker.stat()
    except OSError:
        # Trainer needs the repo (backend package, .env, temp_assets) — wait for unlock.
        return
    log("trainer worker down — starting…")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    logf = open(RUNTIME / "trainer_worker.log", "a", encoding="utf-8")
    subprocess.Popen(
        [sys.executable, str(worker)],
        cwd=str(REPO),
        stdout=logf,
        stderr=subprocess.STDOUT,
        creationflags=_no_window_flags(),
    )


def ensure_prowler(tokens: dict[str, str]) -> None:
    """Keep Prowler Control (FastAPI + built UI) listening on :8010."""
    if http_ok("http://127.0.0.1:8010/api/auth/status"):
        return
    if not PROWLER_API.is_dir():
        log(f"WARNING: Prowler API dir missing ({PROWLER_API}) — skip")
        return
    py = PROWLER_PYTHON if PROWLER_PYTHON.is_file() else Path(sys.executable)
    log("Prowler down — starting…")
    env = os.environ.copy()
    pin = (tokens.get("prowler_pin") or tokens.get("PROWLER_PIN") or "").strip()
    if pin:
        env["PROWLER_PIN"] = pin
    secret = (tokens.get("prowler_auth_secret") or tokens.get("PROWLER_AUTH_SECRET") or "").strip()
    if secret:
        env["PROWLER_AUTH_SECRET"] = secret
    subprocess.Popen(
        [
            str(py),
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8010",
        ],
        cwd=str(PROWLER_API),
        env=env,
        creationflags=_no_window_flags(),
    )
    for _ in range(30):
        time.sleep(1)
        if http_ok("http://127.0.0.1:8010/api/auth/status"):
            log("Prowler is up")
            return
    log("WARNING: Prowler did not become healthy in time")


def ensure_scapper(tokens: dict[str, str]) -> None:
    """Keep Scapper dashboard (FastAPI) listening on :8000 — GPU stays on this PC."""
    if http_ok("http://127.0.0.1:8000/health"):
        return
    if not SCAPPER_ROOT.is_dir():
        log(f"WARNING: Scapper dir missing ({SCAPPER_ROOT}) — skip")
        return
    py = SCAPPER_PYTHON if SCAPPER_PYTHON.is_file() else Path(sys.executable)
    log("Scapper down — starting…")
    env = os.environ.copy()
    # Prefer offline Hub once models are cached (faster restarts on tunnel heal).
    env.setdefault("HF_HUB_OFFLINE", "1")
    subprocess.Popen(
        [
            str(py),
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(SCAPPER_ROOT),
        env=env,
        creationflags=_no_window_flags(),
    )
    for _ in range(45):
        time.sleep(2)
        if http_ok("http://127.0.0.1:8000/health"):
            log("Scapper is up")
            return
    log("WARNING: Scapper did not become healthy in time")


def _push_scapper_to_render(tokens: dict[str, str]) -> None:
    """Update scapper-gateway Render env SCAPPER_URL (stable public URL for users)."""
    url = (tokens.get("scapper_url") or "").strip().rstrip("/")
    service = (tokens.get("scapper_render_service") or "").strip()
    render_tok = (tokens.get("render") or "").strip()
    if not url or not service or not render_tok:
        return
    last = (tokens.get("scapper_url_pushed") or "").strip().rstrip("/")
    if last == url:
        return
    # Reuse Wan env PUT helper against the Scapper service id
    global SERVICE_ID
    prev = SERVICE_ID
    try:
        SERVICE_ID = service
        set_env_and_deploy(render_tok, {"SCAPPER_URL": url})
        upsert_token("scapper_url_pushed", url)
        tokens["scapper_url_pushed"] = url
        log(f"Render scapper-gateway SCAPPER_URL -> {url}")
    except Exception as exc:
        log(f"Render scapper push failed: {exc}")
    finally:
        SERVICE_ID = prev


def _step(name: str, fn, *args, **kwargs) -> None:
    """Run one heal step; a failure (e.g. repo locked) must not skip tunnel healing."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        log(f"{name} failed: {exc}")


def heal_once(tokens: dict[str, str], procs: dict[str, subprocess.Popen]) -> None:
    _step("comfy", ensure_comfy, tokens)
    _step("gpu_agent", ensure_gpu_agent, tokens)
    _step("prowler", ensure_prowler, tokens)
    _step("scapper", ensure_scapper, tokens)
    _step("trainer", ensure_trainer)

    if named_tunnel_mode(tokens):
        # One cloudflared → hostnames for :8000 / :8010 / :8188 / :8799
        ensure_named_tunnel(tokens, procs)
        # Stable URLs already in tokens&cmd from setup_named_tunnel.ps1
        _push_scapper_to_render(tokens)
        # Keep Wan Render envs pointed at stable comfy/agent hostnames
        render_tok = (tokens.get("render") or "").strip()
        comfy_u = (tokens.get("gpu_comfy_url") or "").strip().rstrip("/")
        agent_u = (tokens.get("gpu_agent") or "").strip().rstrip("/")
        if render_tok and comfy_u:
            try:
                set_env_and_deploy(
                    render_tok,
                    {
                        "COMFYUI_URL": comfy_u,
                        **(
                            {"GPU_AGENT_URL": agent_u, "GPU_AGENT_SECRET": tokens["gpu_agent_secret"]}
                            if agent_u and tokens.get("gpu_agent_secret")
                            else {}
                        ),
                    },
                )
            except Exception as exc:
                log(f"Render COMFY/agent update failed: {exc}")
        return

    # Legacy: one quick tunnel per port (rate-limited if recreated too often)
    for port, key, env_key in (
        (8188, "comfy_tunnel", "COMFYUI_URL"),
        (8799, "agent_tunnel", "GPU_AGENT_URL"),
        (8010, "prowler_tunnel", ""),
        (8000, "scapper_tunnel", ""),
    ):
        _step(
            f"tunnel :{port}",
            ensure_tunnel,
            local_port=port,
            state_key=key,
            render_env_key=env_key,
            tokens=tokens,
            procs=procs,
        )
    _push_scapper_to_render(tokens)


def _watchdog_already_running() -> bool:
    """True if another wan_stack_watchdog.py heal loop is alive (this PID excluded)."""
    if sys.platform != "win32":
        return False
    me = os.getpid()
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" "
        "| ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps],
            text=True,
            errors="replace",
            creationflags=_no_window_flags(),
        )
    except Exception:
        return False
    needles = ("wan_stack_watchdog.py", RUNTIME_MAIN_NAME)
    for line in out.splitlines():
        if "|" not in line or not any(n in line for n in needles):
            continue
        raw_pid = line.split("|", 1)[0].strip()
        if raw_pid.isdigit() and int(raw_pid) != me:
            return True
    return False


_START_VBS = '''' Starts the heal loop hidden unless it is already running.
Set fso = CreateObject("Scripting.FileSystemObject")
here = fso.GetParentFolderName(WScript.ScriptFullName)
Set procs = GetObject("winmgmts:").ExecQuery( _
  "SELECT ProcessId FROM Win32_Process WHERE CommandLine LIKE '%{main}%' OR CommandLine LIKE '%wan_stack_watchdog.py%'")
If procs.Count > 0 Then WScript.Quit
CreateObject("WScript.Shell").Run "cmd /c cd /d """ & here & """ && """ & "{python}" & """ """ & here & "\\{main}"" >> """ & here & "\\svc.log"" 2>&1", 0, False
'''


def _same_bytes(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except OSError:
        return False


def sync_runtime(tokens: dict[str, str]) -> None:
    """Refresh the runtime copy (scripts + encrypted tokens) while the repo is readable."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetFileAttributesW(str(RUNTIME), 0x2)  # hidden
    pairs = (
        (REPO / "scripts" / "wan_stack_watchdog.py", RUNTIME / RUNTIME_MAIN_NAME),
        (REPO / "scripts" / "gpu_agent.py", RUNTIME / RUNTIME_AGENT_NAME),
    )
    for src, dst in pairs:
        if not _same_bytes(src, dst):
            shutil.copyfile(src, dst)
            log(f"runtime: updated {dst.name}")
    cfg = {k: v for k, v in tokens.items() if k in RUNTIME_KEYS}
    cfg["_repo"] = str(REPO)
    if _cfg_read(RUNTIME_CFG) != cfg:
        _cfg_write(cfg)
        log("runtime: updated encrypted config")
    vbs = _START_VBS.format(main=RUNTIME_MAIN_NAME, python=sys.executable)
    vbs_path = RUNTIME / "start.vbs"
    try:
        current = vbs_path.read_text(encoding="utf-8")
    except OSError:
        current = ""
    if current != vbs:
        vbs_path.write_text(vbs, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--sync-runtime",
        action="store_true",
        help="Only refresh the runtime copy (scripts + encrypted config) and exit",
    )
    ap.add_argument("--interval", type=int, default=45)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Start even if another watchdog instance is already running",
    )
    args = ap.parse_args()

    if args.sync_runtime:
        if not tokens_from_repo():
            raise SystemExit(f"Cannot read {TOKENS} — unlock the repo first")
        sync_runtime(ensure_secrets(load_tokens()))
        log(f"runtime ready at {RUNTIME}")
        return 0

    if not args.once and not args.force and _watchdog_already_running():
        log("another watchdog already running — exit (use --force to override)")
        return 0

    tokens = ensure_secrets(load_tokens())
    if not (tokens.get("render") or "").strip():
        raise SystemExit("tokens&cmd needs render=<Render API key>")
    if tokens_from_repo():
        try:
            sync_runtime(tokens)
        except Exception as exc:
            log(f"runtime sync failed: {exc}")

    procs: dict[str, subprocess.Popen] = {}
    log("starting heal loop (leave this PC on / awake)")
    while True:
        try:
            heal_once(tokens, procs)
        except Exception as exc:
            log(f"heal error: {exc}")
        if args.once:
            return 0
        time.sleep(max(15, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
