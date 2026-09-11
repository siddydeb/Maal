"""Deep Civitai search for realistic Flux img2img LoRAs that fit Wan Studio.

Reads civitai= from tokens&cmd. Never prints the token.
Writes ranked report to tmp_test/civitai_deep_fit/.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKENS = REPO / "tokens&cmd"
OUT = REPO / "tmp_test" / "civitai_deep_fit"
UA = "Mozilla/5.0 (compatible; WanStudioDeepFit/1.0)"

# Project needs: photoreal Flux img2img, facial opaque gel / tongue concepts,
# identity + clothes + scene retention (NFA-style preferred).
QUERIES = [
    # fluid / facial
    "cum on face",
    "cumonface",
    "facial cumshot",
    "cum facial",
    "add cum",
    "covered in cum",
    "bukkake",
    "non-face-altering",
    "non face altering",
    "NFA",
    "Cumifier",
    "COF",
    "wet messy cum",
    "semen on face",
    "facial fluid",
    # tongue / mouth
    "tongue out",
    "female tongue mouth teeth",
    "mouth open tongue",
    "ahegao tongue realistic",
    # realism helpers that preserve identity in edits
    "photorealistic detail flux",
    "skin texture flux",
    "face fix flux",
]

BASES = [
    "Flux.1 D",
    "Flux.1 Kontext",
    "Flux",
]

TAGS = [
    "cum",
    "facial",
    "cumshot",
    "tongue",
    "realistic",
    "photorealistic",
]


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def api(url: str, tok: str, retries: int = 3):
    headers = {"Authorization": f"Bearer {tok}", "User-Agent": UA}
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            last = exc
            time.sleep(0.8 * (i + 1))
    raise last  # type: ignore[misc]


def safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def text_blob(m: dict, v: dict) -> str:
    parts = [
        m.get("name") or "",
        m.get("description") or "",
        v.get("name") or "",
        v.get("description") or "",
        " ".join(m.get("tags") or []) if isinstance(m.get("tags"), list) else "",
        " ".join(t.get("name", "") if isinstance(t, dict) else str(t) for t in (m.get("tags") or [])),
    ]
    # trigger words
    for tw in v.get("trainedWords") or []:
        parts.append(str(tw))
    return re.sub(r"<[^>]+>", " ", " ".join(parts)).lower()


def score_model(m: dict, v: dict, query: str) -> tuple[int, list[str]]:
    """Higher = better fit for our realistic Flux img2img facial/identity pipeline."""
    reasons: list[str] = []
    s = 0
    name = (m.get("name") or "").lower()
    base = (v.get("baseModel") or "").lower()
    blob = text_blob(m, v)
    dl = int((v.get("stats") or {}).get("downloadCount") or (m.get("stats") or {}).get("downloadCount") or 0)
    rating = float((v.get("stats") or {}).get("rating") or (m.get("stats") or {}).get("rating") or 0)
    rating_count = int((v.get("stats") or {}).get("ratingCount") or (m.get("stats") or {}).get("ratingCount") or 0)

    # Base model fit (critical)
    if "flux.1 d" in base or base in ("flux", "flux.1"):
        s += 80
        reasons.append("Flux.1 D base")
    elif "kontext" in base:
        s += 70
        reasons.append("Flux Kontext (instruction img2img)")
    elif "flux.2" in base or "klein" in base:
        s += 25
        reasons.append("Flux.2/Klein (not our current stack)")
    elif any(x in base for x in ("pony", "sd 1.5", "sd1.5", "illustrious", "sdxl", "noobai", "pdxl")):
        s -= 100
        reasons.append(f"wrong base:{v.get('baseModel')}")
    else:
        s -= 40
        reasons.append(f"unknown base:{v.get('baseModel')}")

    # Concept fit
    fluid_keys = (
        "cum on face",
        "cumonface",
        "facial",
        "cumshot",
        "bukkake",
        "add cum",
        "covered in cum",
        "semen",
        "cof",
        "cumifier",
        "facial cum",
    )
    tongue_keys = ("tongue", "mouth open", "teeth", "ahegao")
    identity_keys = (
        "non-face",
        "non face",
        "nfa",
        "does not change face",
        "doesn't change face",
        "preserve face",
        "identity",
        "img2img",
        "i2i",
        "inpaint",
        "keep face",
        "face lock",
    )
    realist_keys = (
        "realistic",
        "photoreal",
        "photo real",
        "realism",
        "photograph",
        "natural skin",
    )
    anime_keys = ("anime", "manga", "cartoon", "illustrious", "pony", "2.5d", "hentai")

    fluid_hit = sum(1 for k in fluid_keys if k in name or k in blob)
    tongue_hit = sum(1 for k in tongue_keys if k in name or k in blob)
    id_hit = sum(1 for k in identity_keys if k in name or k in blob)
    real_hit = sum(1 for k in realist_keys if k in name or k in blob)
    anime_hit = sum(1 for k in anime_keys if k in name or k in blob)

    if fluid_hit:
        s += 40 + min(fluid_hit, 4) * 5
        reasons.append(f"fluid/facial concept x{fluid_hit}")
    if tongue_hit and not fluid_hit:
        s += 25 + min(tongue_hit, 3) * 4
        reasons.append(f"tongue/mouth concept x{tongue_hit}")
    if id_hit:
        s += 55 + min(id_hit, 3) * 8
        reasons.append(f"identity/img2img friendly x{id_hit}")
    if real_hit:
        s += 20 + min(real_hit, 3) * 4
        reasons.append(f"realistic cues x{real_hit}")
    if anime_hit and real_hit == 0:
        s -= 35
        reasons.append("anime-leaning")

    # Explicit NFA / non-face-altering bonus (our gold standard)
    if "non-face" in name or "non face" in name or "nfa" in name.split() or "[non-face" in name:
        s += 60
        reasons.append("NFA / non-face-altering named")

    # Popularity (soft)
    s += min(dl // 300, 30)
    if rating >= 4.5 and rating_count >= 10:
        s += 15
        reasons.append(f"rating {rating:.1f}/{rating_count}")
    elif rating >= 4.0 and rating_count >= 5:
        s += 8

    # Query match soft boost
    q = query.lower()
    if q and (q in name or q in blob):
        s += 8

    # Penalize off-topic facial (piercing, expression packs) if no fluid/tongue
    if not fluid_hit and not tongue_hit:
        if any(x in name for x in ("piercing", "expression", "contouring", "kissing")):
            s -= 50
            reasons.append("off-topic facial")

    # Type must be LORA
    if (m.get("type") or "").upper() != "LORA":
        s -= 80
        reasons.append(f"type={m.get('type')}")

    return s, reasons


def fetch_pages(tok: str, params: dict, max_pages: int = 3) -> list[dict]:
    items: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(p)
        try:
            data = api(url, tok)
        except Exception as exc:
            print(safe(f"PAGE_FAIL {params.get('query') or params.get('tag')}: {exc}"))
            break
        batch = data.get("items") or []
        if not batch:
            break
        items.extend(batch)
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
        time.sleep(0.25)
    return items


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)

    me = api("https://civitai.com/api/v1/me", tok)
    print(safe(f"AUTH_OK user={me.get('username')}"))

    seen: set[int] = set()
    catalog: dict[int, dict] = {}

    # 1) Query × base sweeps
    for q in QUERIES:
        for base in BASES:
            params = {
                "query": q,
                "types": "LORA",
                "limit": 40,
                "nsfw": "true",
                "sort": "Most Downloaded",
                "baseModels": base,
            }
            print(safe(f"SEARCH q={q!r} base={base}"))
            for m in fetch_pages(tok, params, max_pages=2):
                mid = m.get("id")
                if mid is None or mid in seen:
                    continue
                seen.add(mid)
                vers = m.get("modelVersions") or []
                # pick best Flux-ish version
                pick = None
                for v in vers:
                    b = (v.get("baseModel") or "").lower()
                    if "flux.1 d" in b or b == "flux":
                        pick = v
                        break
                if pick is None:
                    for v in vers:
                        if "kontext" in (v.get("baseModel") or "").lower():
                            pick = v
                            break
                if pick is None and vers:
                    pick = vers[0]
                if not pick:
                    continue
                sc, reasons = score_model(m, pick, q)
                catalog[mid] = {
                    "id": mid,
                    "name": m.get("name"),
                    "url": f"https://civitai.com/models/{mid}",
                    "type": m.get("type"),
                    "base": pick.get("baseModel"),
                    "version_id": pick.get("id"),
                    "version_name": pick.get("name"),
                    "downloads": int((pick.get("stats") or {}).get("downloadCount") or 0),
                    "rating": (pick.get("stats") or {}).get("rating"),
                    "ratingCount": (pick.get("stats") or {}).get("ratingCount"),
                    "file": ((pick.get("files") or [{}])[0]).get("name"),
                    "sizeKB": ((pick.get("files") or [{}])[0]).get("sizeKB"),
                    "triggers": pick.get("trainedWords") or [],
                    "score": sc,
                    "reasons": reasons,
                    "matched_query": q,
                    "nsfw": m.get("nsfw"),
                }

    # 2) Tag sweeps on Flux.1 D
    for tag in TAGS:
        params = {
            "tag": tag,
            "types": "LORA",
            "limit": 40,
            "nsfw": "true",
            "sort": "Most Downloaded",
            "baseModels": "Flux.1 D",
        }
        print(safe(f"TAG tag={tag!r} base=Flux.1 D"))
        for m in fetch_pages(tok, params, max_pages=2):
            mid = m.get("id")
            if mid is None:
                continue
            vers = m.get("modelVersions") or []
            pick = None
            for v in vers:
                b = (v.get("baseModel") or "").lower()
                if "flux" in b:
                    pick = v
                    break
            if not pick and vers:
                pick = vers[0]
            if not pick:
                continue
            sc, reasons = score_model(m, pick, tag)
            if mid in catalog:
                if sc > catalog[mid]["score"]:
                    catalog[mid]["score"] = sc
                    catalog[mid]["reasons"] = reasons
                    catalog[mid]["matched_query"] = f"tag:{tag}"
                continue
            seen.add(mid)
            catalog[mid] = {
                "id": mid,
                "name": m.get("name"),
                "url": f"https://civitai.com/models/{mid}",
                "type": m.get("type"),
                "base": pick.get("baseModel"),
                "version_id": pick.get("id"),
                "version_name": pick.get("name"),
                "downloads": int((pick.get("stats") or {}).get("downloadCount") or 0),
                "rating": (pick.get("stats") or {}).get("rating"),
                "ratingCount": (pick.get("stats") or {}).get("ratingCount"),
                "file": ((pick.get("files") or [{}])[0]).get("name"),
                "sizeKB": ((pick.get("files") or [{}])[0]).get("sizeKB"),
                "triggers": pick.get("trainedWords") or [],
                "score": sc,
                "reasons": reasons,
                "matched_query": f"tag:{tag}",
                "nsfw": m.get("nsfw"),
            }

    # 3) Known high-value IDs (force-fetch full metadata)
    known_ids = [
        858262,  # NFA
        655732,  # Cum On Face FLUX
        924374,
        1240792,
        725999,  # COF
        1750558,  # Cumifier Kontext
        693749,  # Tongue Flux
        721066,  # Tongue Pony (negative control)
        248823,  # csenhance SD15 (negative)
        1753290,  # YACL
        696681,
        1035573,
        694850,  # Wet and Messy FLUX
        1401578,
        1823449,
    ]
    for mid in known_ids:
        try:
            m = api(f"https://civitai.com/api/v1/models/{mid}", tok)
        except Exception as e:
            print(safe(f"KNOWN_FAIL {mid}: {e}"))
            continue
        vers = m.get("modelVersions") or []
        pick = None
        for v in vers:
            b = (v.get("baseModel") or "").lower()
            if "flux.1 d" in b or b == "flux":
                pick = v
                break
        if pick is None:
            for v in vers:
                if "kontext" in (v.get("baseModel") or "").lower():
                    pick = v
                    break
        if pick is None and vers:
            pick = vers[0]
        if not pick:
            continue
        sc, reasons = score_model(m, pick, "known")
        reasons.append("force-known")
        row = {
            "id": mid,
            "name": m.get("name"),
            "url": f"https://civitai.com/models/{mid}",
            "type": m.get("type"),
            "base": pick.get("baseModel"),
            "version_id": pick.get("id"),
            "version_name": pick.get("name"),
            "downloads": int((pick.get("stats") or {}).get("downloadCount") or 0),
            "rating": (pick.get("stats") or {}).get("rating"),
            "ratingCount": (pick.get("stats") or {}).get("ratingCount"),
            "file": ((pick.get("files") or [{}])[0]).get("name"),
            "sizeKB": ((pick.get("files") or [{}])[0]).get("sizeKB"),
            "triggers": pick.get("trainedWords") or [],
            "score": sc,
            "reasons": reasons,
            "matched_query": "known",
            "nsfw": m.get("nsfw"),
            "description_snip": re.sub(
                r"<[^>]+>", " ", (m.get("description") or "")[:500]
            ).strip(),
        }
        prev = catalog.get(mid)
        if not prev or row["score"] >= prev.get("score", -999):
            catalog[mid] = row

    ranked = sorted(catalog.values(), key=lambda r: (r["score"], r["downloads"]), reverse=True)

    # Buckets
    facial = [r for r in ranked if r["score"] >= 100 and any(
        k in " ".join(r.get("reasons") or []).lower() + (r.get("name") or "").lower()
        for k in ("fluid", "facial", "cum", "nfa", "cof", "cumifier")
    )]
    tongue = [r for r in ranked if "tongue" in (r.get("name") or "").lower() or any(
        "tongue" in x.lower() for x in (r.get("reasons") or [])
    )]
    flux_ok = [r for r in ranked if "flux" in (r.get("base") or "").lower()]
    reject = [r for r in ranked if r["score"] < 40]

    report = {
        "auth_user": me.get("username"),
        "total_unique": len(ranked),
        "top_overall": ranked[:40],
        "top_facial_fluid": facial[:20],
        "top_tongue": tongue[:15],
        "top_flux": flux_ok[:40],
        "reject_sample": reject[:15],
        "recommendation": [],
    }

    # Build recommendation narrative rows
    recs = []
    for r in ranked:
        base = (r.get("base") or "").lower()
        name = (r.get("name") or "").lower()
        if "flux" not in base:
            continue
        if not any(k in name or k in " ".join(r.get("reasons") or []).lower() for k in (
            "cum", "facial", "nfa", "cof", "cumifier", "tongue", "fluid", "wet"
        )):
            continue
        if r["score"] < 90:
            continue
        recs.append(r)
        if len(recs) >= 12:
            break
    report["recommendation"] = recs

    (OUT / "full_ranked.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Human markdown
    lines = [
        "# Civitai deep fit report",
        "",
        f"User: `{me.get('username')}`  ",
        f"Unique models scored: **{len(ranked)}**  ",
        "",
        "## Best for our project (realistic Flux img2img + facial/identity)",
        "",
        "| Rank | Score | Model | Base | DL | Why |",
        "|---:|---:|---|---|---:|---|",
    ]
    for i, r in enumerate(recs[:12], 1):
        why = "; ".join(r.get("reasons") or [])[:120]
        lines.append(
            f"| {i} | {r['score']} | [{r['name']}]({r['url']}) | {r.get('base')} | {r.get('downloads')} | {why} |"
        )
    lines += [
        "",
        "## Top facial / fluid",
        "",
    ]
    for r in facial[:10]:
        lines.append(f"- **{r['score']}** [{r['name']}]({r['url']}) — `{r.get('base')}` — {', '.join(r.get('reasons') or [])}")
    lines += ["", "## Top tongue / mouth", ""]
    for r in tongue[:8]:
        lines.append(f"- **{r['score']}** [{r['name']}]({r['url']}) — `{r.get('base')}`")
    lines += [
        "",
        "## Do not use for this pipeline",
        "",
        "- Anything **Pony / SD1.5 / Illustrious / SDXL-only** (identity remake on our Flux path).",
        "- Anime-primary facial enhancers without photoreal cues.",
        "- Flux.2 Klein LoRAs until we migrate the UNET stack.",
        "",
        f"Full JSON: `{OUT / 'full_ranked.json'}`",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== TOP RECOMMENDATIONS ===")
    for i, r in enumerate(recs[:12], 1):
        print(safe(
            f"{i:2d}. score={r['score']:3d} dl={r['downloads']:6d} "
            f"id={r['id']} base={r.get('base')} | {r['name']}"
        ))
        print(safe(f"    {r['url']}"))
        print(safe(f"    reasons: {'; '.join(r.get('reasons') or [])}"))

    print(f"\nWROTE {OUT / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
