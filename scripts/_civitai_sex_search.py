"""Focused Civitai search: Wan 2.2 I2V penetration/sex LoRAs (Oral-JFJ quality bar).

Reads civitai= from tokens&cmd. Never prints the token.
Writes temp_assets/civitai_sex/report.json and prints a ranked shortlist.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKENS = REPO / "tokens&cmd"
OUT = REPO / "temp_assets" / "civitai_sex"
UA = "Mozilla/5.0 (compatible; WanStudioSexSearch/1.0)"

QUERIES = [
    "missionary",
    "missionary sex",
    "missionary pov",
    "sex",
    "penetration",
    "vaginal",
    "fucking",
    "pov sex",
    "pov missionary",
    "thrusting",
    "intercourse",
    "general nsfw",
    "nsfw",
    "jfj",
    "deepthroat",
    "cowgirl",
    "doggy",
    "mating press",
    "legs up",
    "sex positions",
]
BASES = ["Wan Video 2.2 I2V-A14B", "Wan Video 14B i2v 720p", "Wan Video 14B i2v 480p"]
SEX_KEYS = (
    "missionary", "sex", "penetrat", "vaginal", "fuck", "thrust", "intercourse",
    "mating", "legs up", "pov", "position", "nsfw", "cowgirl", "doggy",
)


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def api(url: str, tok: str):
    last = None
    for i in range(4):
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {tok}", "User-Agent": UA}
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (i + 1))
    raise last  # type: ignore[misc]


def search(tok: str, **params) -> list[dict]:
    q = {"limit": "50", "nsfw": "true", "types": "LORA", **params}
    url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(q)
    try:
        return api(url, tok).get("items") or []
    except Exception as exc:
        print(f"  FAIL {params}: {exc}", flush=True)
        return []


def safe(s: str) -> str:
    return (s or "").encode("ascii", "replace").decode("ascii")


def wan22_i2v_versions(m: dict) -> list[dict]:
    out = []
    for v in m.get("modelVersions") or []:
        base = (v.get("baseModel") or "").lower()
        if "wan" in base and "i2v" in base:
            out.append(v)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    print("Auth OK. Searching...", flush=True)
    seen: dict[int, dict] = {}

    for q in QUERIES:
        for base in BASES:
            for sort in ("Most Downloaded", "Highest Rated"):
                items = search(tok, query=q, baseModels=base, sort=sort)
                for m in items:
                    if isinstance(m.get("id"), int):
                        seen.setdefault(m["id"], m)
                time.sleep(0.2)
        print(f"  q={q!r} -> total {len(seen)}", flush=True)

    # Browse top Wan 2.2 I2V LoRAs without query (popular sex LoRAs often have odd names).
    for sort in ("Most Downloaded", "Highest Rated", "Most Liked"):
        for page_cursor in (None,):
            items = search(tok, baseModels=BASES[0], sort=sort, period="AllTime")
            for m in items:
                if isinstance(m.get("id"), int):
                    seen.setdefault(m["id"], m)
    print(f"Unique: {len(seen)}", flush=True)

    rows = []
    for mid, m in seen.items():
        vers = wan22_i2v_versions(m)
        if not vers:
            continue
        name = m.get("name") or ""
        tags = " ".join(t if isinstance(t, str) else t.get("name", "") for t in (m.get("tags") or []))
        blob = f"{name} {tags} " + " ".join(
            " ".join(v.get("trainedWords") or []) + " " + (v.get("name") or "") for v in vers
        )
        blob_l = blob.lower()
        if not any(k in blob_l for k in SEX_KEYS):
            continue
        stats = m.get("stats") or {}
        dl = int(stats.get("downloadCount") or 0)
        thumbs = int(stats.get("thumbsUpCount") or 0)
        v0 = vers[0]
        files = []
        for v in vers[:4]:
            for f in v.get("files") or []:
                files.append(
                    {
                        "version": v.get("name"),
                        "versionId": v.get("id"),
                        "base": v.get("baseModel"),
                        "name": f.get("name"),
                        "sizeMB": int((f.get("sizeKB") or 0) / 1024),
                        "url": f.get("downloadUrl"),
                    }
                )
        vnames = " ".join((v.get("name") or "").lower() for v in vers) + " " + " ".join(
            (f["name"] or "").lower() for f in files
        )
        dual = ("high" in vnames) and ("low" in vnames)
        rows.append(
            {
                "id": mid,
                "name": name,
                "url": f"https://civitai.com/models/{mid}",
                "creator": (m.get("creator") or {}).get("username"),
                "downloads": dl,
                "thumbs": thumbs,
                "dual_high_low": dual,
                "base": v0.get("baseModel"),
                "trainedWords": v0.get("trainedWords") or [],
                "published": v0.get("publishedAt") or v0.get("createdAt"),
                "versions": [
                    {"id": v.get("id"), "name": v.get("name"), "base": v.get("baseModel")}
                    for v in vers[:6]
                ],
                "files": files,
                "desc": safe(re.sub(r"<[^>]+>", " ", m.get("description") or ""))[:700],
            }
        )

    rows.sort(key=lambda r: (r["thumbs"], r["downloads"]), reverse=True)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("\n=== Wan I2V sex-ish LoRAs (by thumbs) ===", flush=True)
    for r in rows[:60]:
        print(
            f"{r['thumbs']:>6} up {r['downloads']:>7} dl  dual={int(r['dual_high_low'])}  "
            f"id={r['id']:<8} {safe(r['name'])[:70]}  [{safe(r['creator'] or '')}]  "
            f"base={r['base']}  trig={safe(','.join(r['trainedWords'])[:50])}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
