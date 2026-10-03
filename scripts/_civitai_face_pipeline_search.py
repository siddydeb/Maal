"""Civitai search: realism/face LoRAs for the new T2I models (Z-Image, Flux.2 Klein) and
identity/edit LoRAs for the edit models (Qwen-Image-Edit, FireRed, Klein edit).

Reads civitai= from tokens&cmd. Never prints the token.
Writes temp_assets/civitai_face_pipeline/report.json and prints ranked shortlists per base.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "temp_assets" / "civitai_face_pipeline"
UA = "Mozilla/5.0 (compatible; WanStudioFacePipeline/1.0)"
BASE_PATTERN = re.compile(r"(?i)z.?image|qwen|klein|flux\.2|firered")
QUERIES = [
    "realism", "realistic", "photorealistic", "skin", "skin texture", "amateur", "iphone", "candid",
    "face", "portrait", "indian", "detail", "lightning", "consistency", "consistent character",
    "identity", "same face", "face swap", "head swap", "character", "outfit", "clothes", "try on",
    "relight", "pose", "nsfw", "nude", "body",
]


def token() -> str:
    txt = (REPO / "tokens&cmd").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?im)^\s*civitai=(.+)$", txt)
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def get(url: str, tok: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def base_names(tok: str) -> list[str]:
    names: set[str] = set()
    for qy in ("z-image", "zimage", "z image turbo", "qwen edit", "qwen image edit 2511", "firered", "klein", "flux 2"):
        for m in search(tok, query=qy, sort="Most Downloaded"):
            for v in m.get("modelVersions") or []:
                b = v.get("baseModel") or ""
                if BASE_PATTERN.search(b):
                    names.add(b)
    return sorted(names)


def search(tok: str, **params) -> list[dict]:
    q = {"limit": "40", "nsfw": "true", "types": "LORA", **params}
    for i in range(3):
        try:
            return get("https://civitai.com/api/v1/models?" + urllib.parse.urlencode(q), tok).get("items") or []
        except Exception:
            time.sleep(1.5 * (i + 1))
    return []


def safe(s: str) -> str:
    return (s or "").encode("ascii", "replace").decode("ascii")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    bases = base_names(tok)
    print("bases:", bases, flush=True)
    found: dict[int, dict] = {}
    for base in bases:
        for sort in ("Most Downloaded", "Highest Rated"):
            for m in search(tok, baseModels=base, sort=sort, period="AllTime", limit="100"):
                found.setdefault(m["id"], m)
        for qy in QUERIES:
            for m in search(tok, query=qy, baseModels=base, sort="Most Downloaded"):
                found.setdefault(m["id"], m)
            time.sleep(0.1)
        print(f"  {base}: total {len(found)}", flush=True)

    rows = []
    for mid, m in found.items():
        vers = [v for v in (m.get("modelVersions") or []) if (v.get("baseModel") or "") in bases]
        if not vers:
            continue
        v0 = vers[0]
        stats = m.get("stats") or {}
        rows.append(
            {
                "id": mid,
                "name": safe(m.get("name")),
                "url": f"https://civitai.com/models/{mid}",
                "base": v0.get("baseModel"),
                "version": safe(v0.get("name")),
                "versionId": v0.get("id"),
                "published": (v0.get("publishedAt") or v0.get("createdAt") or "")[:10],
                "downloads": int(stats.get("downloadCount") or 0),
                "thumbs": int(stats.get("thumbsUpCount") or 0),
                "nsfw": bool(m.get("nsfw")),
                "trainedWords": [safe(w) for w in (v0.get("trainedWords") or [])][:4],
                "sizeMB": int(sum((f.get("sizeKB") or 0) for f in (v0.get("files") or [])[:1]) / 1024),
                "desc": safe(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.get("description") or "")))[:600],
            }
        )
    rows.sort(key=lambda r: (r["thumbs"], r["downloads"]), reverse=True)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for base in bases:
        sub = [r for r in rows if r["base"] == base]
        print(f"\n=== {base} ({len(sub)}) ===", flush=True)
        for r in sub[:35]:
            print(
                f"{r['thumbs']:>6} up {r['downloads']:>7} dl {r['published']} id={r['id']:<8} "
                f"{r['name'][:60]:<60} {'NSFW ' if r['nsfw'] else ''}{r['sizeMB']}MB trig={','.join(r['trainedWords'])[:30]}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
