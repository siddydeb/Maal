"""Download latest test-mode run + references into temp_assets, then wipe.

Uses the tester owner token from WAN_AUTH_SECRET (not the PIN).
Media stays in Mongo; this folder is only for local visual review.

Usage (repo root):
  python scripts/review_test.py
  python scripts/review_test.py --preset enhance
  python scripts/review_test.py --wipe
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "temp_assets" / "review"
API = "https://wan-studio-api.onrender.com"

sys.path.insert(0, str(REPO))
from backend.owners import make_token  # noqa: E402
from backend.config import get_settings  # noqa: E402


def _http(path: str, token: str) -> bytes:
    req = urllib.request.Request(
        f"{API}{path}",
        headers={"Accept": "application/json", "X-Wan-Token": token},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def wipe() -> None:
    if OUT.is_dir():
        shutil.rmtree(OUT, ignore_errors=True)
    print(f"wiped {OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="")
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()
    if args.wipe:
        wipe()
        return 0

    settings = get_settings()
    owner = (settings.wan_tester_owner or "").strip()
    if not owner:
        raise SystemExit("WAN_TESTER_OWNER is not set")
    token = make_token(owner)
    q = f"?preset_id={args.preset}&limit=5" if args.preset else "?limit=5"
    raw = _http(f"/api/test/runs{q}", token)
    payload = json.loads(raw.decode("utf-8"))
    items = payload.get("items") or []
    refs = payload.get("refs") or []
    if not items and not refs:
        print("No test runs/refs yet.")
        return 0

    wipe()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    for i, item in enumerate(items, 1):
        data = _http(f"/api/media/{item['id']}", token)
        ext = "mp4" if item.get("kind") == "vid" else "png"
        (OUT / f"run_{i}_{item['id'][:8]}.{ext}").write_bytes(data)
    for i, item in enumerate(refs, 1):
        data = _http(f"/api/test/refs/{item['id']}/media", token)
        (OUT / f"ref_{i}_{item['id'][:8]}.png").write_bytes(data)
    print(f"wrote {OUT}  runs={len(items)} refs={len(refs)}")
    print("Review, then: python scripts/review_test.py --wipe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
