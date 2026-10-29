#!/usr/bin/env python3
"""Create the failover Wan API on Render account 2 (tokens&cmd ``render2=`` / ``RENDER2=``).

Copies every env var from the primary service plus the private/ overlay as secret
files, boots it once against an isolated Mongo DB and a dummy Comfy URL (its startup
clears the Comfy queue and re-queues running jobs — must not touch the live stack),
then suspends it and switches it to the real env. The GPU-PC watchdog resumes it
when the primary goes down. Writes render2_service= / render2_url= into tokens&cmd.

Usage (repo root, repo unlocked):
  python scripts/setup_backup_api.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sync_runtime_overlay as overlay  # noqa: E402
from sync_runtime_overlay import RENDER_API, _http_json  # noqa: E402

PRIMARY = overlay.SERVICE_ID
NAME = "wan-studio-api"
# Throwaway values for the first boot only
PROBE_ENV = {"MONGODB_DB": "wan_studio_backup_probe", "COMFYUI_URL": "http://127.0.0.1:9"}


def _ok(code: int, payload: object, what: str) -> object:
    if code not in (200, 201, 202, 204):
        raise SystemExit(f"{what} failed HTTP {code}: {payload}")
    return payload


def _upsert_token(key: str, value: str) -> None:
    path = overlay.TOKENS_FILE
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            break
    else:
        lines += ["", f"{key}={value}"]
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _set_env(token: str, sid: str, key: str, value: str) -> None:
    code, payload = _http_json(
        "PUT", f"{RENDER_API}/services/{sid}/env-vars/{key}", token=token, body={"value": value}
    )
    _ok(code, payload, f"env {key}")


def main() -> int:
    tokens = overlay._load_tokens()
    tok1, tok2 = tokens.get("render"), tokens.get("render2")
    if not tok1 or not tok2:
        raise SystemExit("tokens&cmd needs render= and render2= (or RENDER2=)")
    if tokens.get("render2_service"):
        raise SystemExit(f"Backup already set up: {tokens['render2_service']} ({tokens.get('render2_url')})")

    primary = _ok(*_http_json("GET", f"{RENDER_API}/services/{PRIMARY}", token=tok1), "read primary")
    details = primary["serviceDetails"]
    rows = _ok(*_http_json("GET", f"{RENDER_API}/services/{PRIMARY}/env-vars?limit=100", token=tok1), "read env")
    real_env = {r["envVar"]["key"]: r["envVar"].get("value") or "" for r in rows}
    real_env["RENDER_API_KEY"] = tok2
    owners = _ok(*_http_json("GET", f"{RENDER_API}/owners?limit=5", token=tok2), "read owners")
    owner_id = owners[0]["owner"]["id"]

    secret_files = []
    for name in overlay.OVERLAY_FILES:
        path = overlay.PRIVATE_DIR / name
        if path.is_file():
            secret_files.append({"name": name, "content": path.read_text(encoding="utf-8")})

    probe_env = {**real_env, **PROBE_ENV}
    body = {
        "type": "web_service",
        "name": NAME,
        "ownerId": owner_id,
        "repo": primary["repo"],
        "branch": primary.get("branch") or "main",
        "autoDeploy": "no",
        "envVars": [{"key": k, "value": v} for k, v in probe_env.items() if v != ""],
        "secretFiles": secret_files,
        "serviceDetails": {
            "runtime": details.get("runtime") or "python",
            "plan": "free",
            "region": details.get("region") or "oregon",
            "healthCheckPath": details.get("healthCheckPath") or "/api/healthz",
            "envSpecificDetails": details["envSpecificDetails"],
        },
    }
    created = _ok(*_http_json("POST", f"{RENDER_API}/services", token=tok2, body=body), "create service")
    svc = created.get("service", created)
    sid = svc["id"]
    url = svc["serviceDetails"]["url"].rstrip("/")
    print(f"Created {sid} -> {url}")
    _upsert_token("render2_service", sid)
    _upsert_token("render2_url", url)
    _set_env(tok2, sid, "RENDER_SERVICE_ID", sid)

    print("Waiting for first (isolated) boot…")
    healthy = False
    for i in range(60):
        time.sleep(15)
        code, _ = _http_json("GET", f"{url}/api/healthz", timeout=30)
        print(f"  [{i + 1}/60] healthz HTTP {code}")
        if code == 200:
            healthy = True
            break

    _ok(*_http_json("POST", f"{RENDER_API}/services/{sid}/suspend", token=tok2), "suspend")
    print("Suspended backup (standby).")
    for key in PROBE_ENV:
        _set_env(tok2, sid, key, real_env.get(key, ""))
    print("Switched backup to the real Mongo DB / Comfy URL for failover.")
    if not healthy:
        print("WARNING: backup never answered healthz — check its deploy logs on Render account 2.")
        return 2
    print("Backup API ready. Re-run scripts/install_mobile_autostart.ps1 so the watchdog gets the new keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
