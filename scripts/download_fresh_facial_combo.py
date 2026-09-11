"""Download fresh facial LoRA combo via Civitai token."""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LORAS = Path(r"E:\Comfy-Desktop\ComfyUI-Shared\models\loras")
TOK = re.search(
    r"(?im)^\s*civitai=(.+)$",
    (REPO / "tokens&cmd").read_text(encoding="utf-8", errors="ignore"),
).group(1).strip()

# version_id -> dest filename (from fresh search details)
DOWNLOADS = [
    (1398441, "flux_cum_facial_meower.safetensors"),  # wait - use correct versions
]

# Correct mapping from candidate_details / API version ids:
ITEMS = [
    # (model_id, preferred version_id or None, dest)
    (1315268, None, "flux_char_friendly_facial.safetensors"),
    (866273, None, "flux_facial_cum_massive.safetensors"),
    (655732, 733630, "flux_cum_on_face_cuminator.safetensors"),
    (730199, None, "flux_cumhere_v1.safetensors"),
    (691681, None, "flux_not_another_facial.safetensors"),
    (858262, 1032060, "flux_facial_fluid_v1.safetensors"),  # NFA — confirm present
]


def api_model(mid: int) -> dict:
    req = urllib.request.Request(
        f"https://civitai.com/api/v1/models/{mid}",
        headers={"Authorization": f"Bearer {TOK}", "User-Agent": "Wan/1"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        import json

        return json.loads(r.read().decode())


def pick_ver(m: dict, prefer: int | None) -> dict:
    vers = m.get("modelVersions") or []
    if prefer:
        for v in vers:
            if v.get("id") == prefer:
                return v
    for v in vers:
        if "flux.1 d" in (v.get("baseModel") or "").lower():
            return v
    for v in vers:
        if "flux" in (v.get("baseModel") or "").lower():
            return v
    return vers[0]


def download(ver_id: int, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 5_000_000:
        print(f"HAVE {dest.name} ({dest.stat().st_size // 1_000_000} MB)")
        return True
    print(f"GET ver={ver_id} -> {dest.name}")
    req = urllib.request.Request(
        f"https://civitai.com/api/download/models/{ver_id}",
        headers={"Authorization": f"Bearer {TOK}", "User-Agent": "Wan/1"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read()
    if len(data) < 1_000_000 or data[:20].lstrip().lower().startswith(b"<!doctype"):
        print(f"BAD size={len(data)}")
        return False
    dest.write_bytes(data)
    print(f"OK {dest.name} ({dest.stat().st_size // 1_000_000} MB)")
    return True


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    LORAS.mkdir(parents=True, exist_ok=True)
    ok = 0
    for mid, prefer, fname in ITEMS:
        try:
            m = api_model(mid)
            v = pick_ver(m, prefer)
            print(f"MODEL {mid} {m.get('name')} -> ver {v.get('id')} {v.get('name')} {v.get('baseModel')}")
            # Fix wrong prefer if needed — use resolved id
            if download(int(v["id"]), LORAS / fname):
                ok += 1
        except Exception as e:
            print(f"FAIL {mid}: {type(e).__name__}: {e}")
    print(f"DONE ok={ok}/{len(ITEMS)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
