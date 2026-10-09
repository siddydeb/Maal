"""Civitai search: realistic synthetic-girl / consistent character / outfit LoRAs for Flux Dev + Kontext.

Reads civitai= from tokens&cmd. Never prints the token.
Writes temp_assets/civitai_realism/report.json and prints ranked shortlists per bucket.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "temp_assets" / "civitai_realism"
UA = "Mozilla/5.0 (compatible; WanStudioRealism/1.0)"

BUCKETS: dict[str, list[str]] = {
    "realism": [
        "realism", "ultra realistic", "photorealistic", "amateur photo", "amateur photography",
        "realistic skin", "skin texture", "instagram", "iphone photo", "boring reality",
        "snapshot", "raw photo", "natural skin", "film photo", "candid",
    ],
    "consistent_face": [
        "consistent character", "same face", "consistent face", "character consistency",
        "ai influencer", "instagirl", "virtual influencer", "character sheet", "face consistency",
        "synthetic person", "ai model girl", "unique face", "sameface fix",
    ],
    "kontext_identity_outfit": [
        "kontext character", "kontext consistent", "kontext face", "kontext outfit",
        "kontext clothes", "kontext try on", "outfit transfer", "clothes swap", "virtual try on",
        "put it here", "kontext pose", "kontext photo", "kontext realism", "kontext relight",
        "kontext multi image", "kontext reference",
    ],
}
BASES = ["Flux.1 D", "Flux.1 Kontext"]


def token() -> str:
    txt = (REPO / "tokens&cmd").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?im)^\s*civitai=(.+)$", txt)
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def api(url: str, tok: str):
    last = None
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last  # type: ignore[misc]


def search(tok: str, **params) -> list[dict]:
    q = {"limit": "40", "nsfw": "true", "types": "LORA", **params}
    try:
        return api("https://civitai.com/api/v1/models?" + urllib.parse.urlencode(q), tok).get("items") or []
    except Exception as exc:
        print(f"  FAIL {params}: {exc}", flush=True)
        return []


def safe(s: str) -> str:
    return (s or "").encode("ascii", "replace").decode("ascii")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    print("Auth OK.", flush=True)
    found: dict[int, dict] = {}
    bucket_of: dict[int, set[str]] = {}
    for bucket, queries in BUCKETS.items():
        for q in queries:
            for base in BASES:
                for sort in ("Most Downloaded", "Highest Rated"):
                    for m in search(tok, query=q, baseModels=base, sort=sort):
                        mid = m.get("id")
                        if isinstance(mid, int):
                            found.setdefault(mid, m)
                            bucket_of.setdefault(mid, set()).add(bucket)
                    time.sleep(0.15)
        print(f"  bucket {bucket}: total {len(found)}", flush=True)
    # Popular browse per base (realism LoRAs often have odd names).
    for base in BASES:
        for sort in ("Most Downloaded", "Highest Rated"):
            for m in search(tok, baseModels=base, sort=sort, period="AllTime", limit="100"):
                mid = m.get("id")
                if isinstance(mid, int) and mid not in found:
                    found[mid] = m
                    bucket_of.setdefault(mid, set()).add("browse")

    rows = []
    for mid, m in found.items():
        vers = [v for v in (m.get("modelVersions") or []) if (v.get("baseModel") or "") in BASES]
        if not vers:
            continue
        v0 = vers[0]
        stats = m.get("stats") or {}
        tags = [t if isinstance(t, str) else t.get("name", "") for t in (m.get("tags") or [])]
        files = [
            {"name": f.get("name"), "sizeMB": int((f.get("sizeKB") or 0) / 1024), "versionId": v0.get("id")}
            for f in (v0.get("files") or [])
        ]
        rows.append(
            {
                "id": mid,
                "name": m.get("name"),
                "url": f"https://civitai.com/models/{mid}",
                "creator": (m.get("creator") or {}).get("username"),
                "base": v0.get("baseModel"),
                "version": v0.get("name"),
                "versionId": v0.get("id"),
                "downloads": int(stats.get("downloadCount") or 0),
                "thumbs": int(stats.get("thumbsUpCount") or 0),
                "trainedWords": v0.get("trainedWords") or [],
                "tags": tags[:12],
                "buckets": sorted(bucket_of.get(mid, [])),
                "files": files,
                "desc": safe(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.get("description") or "")))[:900],
            }
        )
    rows.sort(key=lambda r: (r["thumbs"], r["downloads"]), reverse=True)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    for bucket in ("realism", "consistent_face", "kontext_identity_outfit", "browse"):
        sub = [r for r in rows if bucket in r["buckets"]]
        print(f"\n=== {bucket} ({len(sub)}) ===", flush=True)
        for r in sub[:30]:
            print(
                f"{r['thumbs']:>6} up {r['downloads']:>7} dl  id={r['id']:<8} "
                f"{safe(r['name'])[:62]:<62} [{safe(r['creator'] or '')[:16]}] {r['base']} "
                f"trig={safe(','.join(r['trainedWords'])[:40])}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
