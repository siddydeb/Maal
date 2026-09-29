#!/usr/bin/env python3
"""Upload gitignored private/ overlay files to Render as secret files.

Tracked repo stays generic. Overlay payloads never go through git.

Usage (repo root):
  python scripts/sync_runtime_overlay.py
  python scripts/sync_runtime_overlay.py --status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS_FILE = REPO_ROOT / "tokens&cmd"
PRIVATE_DIR = REPO_ROOT / "private"
RENDER_API = "https://api.render.com/v1"
SERVICE_ID = "srv-d9ot8spt0dsc73bqjv0g"
API_HEALTH = "https://wan-studio-api.onrender.com/api/health"

OVERLAY_FILES = (
    "presets.json",
    "review_bins.json",
    "planner_rules.py",
    "edit_runner.py",
    "motion.py",
    "lora_stack.py",
    "video_v1.py",
    "lora_files.py",
    "catalog_loras.py",
    "t2i.json",
)


def _load_render_token() -> str:
    if not TOKENS_FILE.is_file():
        raise SystemExit(f"Missing {TOKENS_FILE.name}")
    for line in TOKENS_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.lower().startswith("render=") and "=" in line:
            return line.split("=", 1)[1].strip()
    raise SystemExit("tokens&cmd missing render=")


def _http_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: object | None = None,
    timeout: float = 120.0,
) -> tuple[int, object]:
    data = None
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = int(resp.status)
            if not raw:
                return code, None
            try:
                return code, json.loads(raw)
            except json.JSONDecodeError:
                return code, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return int(exc.code), payload
    except Exception as exc:
        return 0, {"error": str(exc)}


def list_secret_names(token: str) -> list[str]:
    code, payload = _http_json(
        "GET",
        f"{RENDER_API}/services/{SERVICE_ID}/secret-files?limit=100",
        token=token,
    )
    if code != 200:
        print(f"List secret-files HTTP {code}: {payload}")
        return []
    names: list[str] = []
    if isinstance(payload, list):
        for row in payload:
            sf = row.get("secretFile") if isinstance(row, dict) else None
            if isinstance(sf, dict) and sf.get("name"):
                names.append(str(sf["name"]))
            elif isinstance(row, dict) and row.get("name"):
                names.append(str(row["name"]))
    return names


def put_secret_file(token: str, name: str, content: str) -> int:
    quoted = urllib.parse.quote(name, safe="")
    code, payload = _http_json(
        "PUT",
        f"{RENDER_API}/services/{SERVICE_ID}/secret-files/{quoted}",
        token=token,
        body={"content": content},
    )
    if code not in (200, 201):
        print(f"  FAIL {name} HTTP {code} {payload}")
    else:
        print(f"  OK {name} ({len(content.encode('utf-8'))} bytes) HTTP {code}")
    return code


def trigger_deploy_or_restart(token: str) -> str:
    code, payload = _http_json(
        "POST",
        f"{RENDER_API}/services/{SERVICE_ID}/deploys",
        token=token,
        body={"clearCache": "do_not_clear"},
    )
    if code in (200, 201, 202):
        print("Deploy triggered.")
        return "deploy"
    print(f"Deploy HTTP {code}; restarting instead.")
    code2, payload2 = _http_json(
        "POST",
        f"{RENDER_API}/services/{SERVICE_ID}/restart",
        token=token,
    )
    if code2 not in (200, 202):
        raise SystemExit(f"Restart failed HTTP {code2}: {payload2}")
    print("Restart accepted.")
    return "restart"


def fetch_health() -> tuple[bool, dict]:
    code, payload = _http_json("GET", API_HEALTH, timeout=55.0)
    if code == 200 and isinstance(payload, dict):
        return True, payload
    return False, payload if isinstance(payload, dict) else {"error": payload, "http": code}


def poll_health(*, attempts: int = 24, pause_sec: float = 8.0) -> dict:
    last: dict = {}
    for i in range(1, attempts + 1):
        ok, data = fetch_health()
        last = data if isinstance(data, dict) else {"raw": data}
        overlay = last.get("overlay") if isinstance(last.get("overlay"), dict) else {}
        print(
            f"[{i}/{attempts}] http_ok={ok} mongo={last.get('mongodb')} "
            f"comfy={last.get('comfyui')} overlay={overlay}"
        )
        if ok and overlay:
            return last
        if ok and i >= 3:
            return last
        time.sleep(pause_sec)
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument(
        "--existing",
        action="store_true",
        help="Upload only overlay files that exist locally; leave others on Render",
    )
    ap.add_argument(
        "--files",
        nargs="+",
        default=(),
        help="Upload only these overlay filenames",
    )
    args = ap.parse_args()

    if args.status:
        ok, data = fetch_health()
        overlay = data.get("overlay") if isinstance(data, dict) else None
        print(json.dumps({"ok": ok, "overlay": overlay, "comfyui": data.get("comfyui") if isinstance(data, dict) else None, "mongodb": data.get("mongodb") if isinstance(data, dict) else None}, indent=2))
        return 0 if ok else 2

    token = _load_render_token()
    names = list(OVERLAY_FILES)
    if args.existing or args.files:
        wanted = (
            tuple(Path(x).name for x in args.files)
            if args.files
            else OVERLAY_FILES
        )
        names = [n for n in wanted if (PRIVATE_DIR / n).is_file()]
        missing = [n for n in wanted if n not in names]
        if missing:
            print(f"Skipping missing overlay files: {missing}")
        if not names:
            raise SystemExit("No overlay files to upload")
    else:
        missing = [n for n in OVERLAY_FILES if not (PRIVATE_DIR / n).is_file()]
        if missing:
            raise SystemExit(f"Missing overlay files in private/: {missing}")

    print("Existing secret files:", list_secret_names(token) or "(none)")
    print("Uploading overlay files…")
    failed = 0
    for name in names:
        text = (PRIVATE_DIR / name).read_text(encoding="utf-8")
        code = put_secret_file(token, name, text)
        if code not in (200, 201):
            failed += 1
    if failed:
        raise SystemExit(f"{failed} overlay file(s) failed to upload")

    print("Secret files now:", list_secret_names(token))
    trigger_deploy_or_restart(token)
    time.sleep(5)
    health = poll_health()
    overlay = health.get("overlay") if isinstance(health.get("overlay"), dict) else {}
    needed = ("edit_runner", "planner_rules", "presets")
    if not all(overlay.get(k) for k in needed):
        print("Overlay flags not all true yet. Wait a minute and re-run --status.")
        return 3
    print("Overlay mounted on API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
