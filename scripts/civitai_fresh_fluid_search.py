"""Fresh deep Civitai search — best facial fluid LoRA / combo for photoreal img2img.

Ignores prior project picks. Uses civitai= from tokens&cmd.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKENS = REPO / "tokens&cmd"
OUT = REPO / "tmp_test" / "civitai_fresh_fluid"
UA = "Mozilla/5.0 (compatible; WanFreshFluid/2.0)"

QUERIES = [
    "cum on face",
    "cumonface",
    "facial cumshot",
    "cum facial",
    "bukkake realistic",
    "covered in cum photoreal",
    "add cum flux",
    "cumifier",
    "sticky cum face",
    "excessive cum face",
    "opaque cum face",
    "cum splatter face",
    "facial semen",
    "messy facial",
    "cum drip face",
]

BASES = ["Flux.1 D", "Flux.1 Kontext", "Flux"]


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing civitai=")
    return m.group(1).strip().strip('"').strip("'")


def api(url: str, tok: str, retries: int = 4):
    headers = {"Authorization": f"Bearer {tok}", "User-Agent": UA}
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            last = exc
            time.sleep(0.6 * (i + 1))
    raise last  # type: ignore[misc]


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "")


def blob_of(m: dict, v: dict) -> str:
    bits = [
        m.get("name") or "",
        strip_html(m.get("description") or ""),
        v.get("name") or "",
        strip_html(v.get("description") or ""),
        " ".join(str(t) for t in (v.get("trainedWords") or [])),
    ]
    tags = m.get("tags") or []
    for t in tags:
        if isinstance(t, dict):
            bits.append(str(t.get("name") or ""))
        else:
            bits.append(str(t))
    return " ".join(bits).lower()


def pick_version(m: dict) -> dict | None:
    vers = m.get("modelVersions") or []
    for prefer in ("flux.1 d", "flux.1 kontext", "flux"):
        for v in vers:
            b = (v.get("baseModel") or "").lower()
            if prefer == "flux":
                if "flux" in b and "flux.2" not in b and "klein" not in b:
                    return v
            elif prefer in b:
                return v
    return vers[0] if vers else None


def score(m: dict, v: dict) -> tuple[int, list[str], dict]:
    """Score for: photoreal facial fluid coverage on img2img without total remake."""
    reasons: list[str] = []
    flags = {
        "is_flux_d": False,
        "is_kontext": False,
        "is_wrong_base": False,
        "heavy_coverage": False,
        "identity_friendly": False,
        "photoreal": False,
        "anime": False,
        "act_specific": False,
    }
    s = 0
    name = (m.get("name") or "").lower()
    base = (v.get("baseModel") or "").lower()
    blob = blob_of(m, v)
    dl = int((v.get("stats") or {}).get("downloadCount") or 0)
    thumbs = int((v.get("stats") or {}).get("thumbsUpCount") or (m.get("stats") or {}).get("thumbsUpCount") or 0)
    rating = float((v.get("stats") or {}).get("rating") or 0)
    rc = int((v.get("stats") or {}).get("ratingCount") or 0)

    # Base
    if "flux.1 d" in base or base == "flux":
        s += 70
        flags["is_flux_d"] = True
        reasons.append("Flux.1 D")
    elif "kontext" in base:
        s += 75  # instruction edit is naturally img2img
        flags["is_kontext"] = True
        reasons.append("Flux Kontext (native edit)")
    elif any(x in base for x in ("pony", "sd 1.5", "sd1.5", "illustrious", "sdxl", "noobai", "pdxl", "flux.2", "klein")):
        s -= 120
        flags["is_wrong_base"] = True
        reasons.append(f"reject-base:{v.get('baseModel')}")
        return s, reasons, flags
    else:
        s -= 50
        reasons.append(f"base:{v.get('baseModel')}")

    # Must be about facial fluid — hard gate
    facial_terms = (
        "cum on face",
        "cumonface",
        "cum facial",
        "facial cum",
        "facial",
        "bukkake",
        "cumshot",
        "covered in cum",
        "add cum",
        "cumifier",
        "cof",
        "semen on",
        "cum over her face",
        "cum across",
    )
    facial_hits = sum(1 for t in facial_terms if t in name or t in blob)
    if facial_hits == 0 and "cum" not in name:
        s -= 80
        reasons.append("not-facial-fluid")
        return s, reasons, flags
    s += 35 + min(facial_hits, 5) * 6
    reasons.append(f"facial-fluid x{facial_hits}")

    # Heavy / opaque / excessive (project want)
    heavy = ("excessive", "thick", "opaque", "heavy", "covered", "splatter", "dripping", "messy", "load", "sticky", "viscous")
    heavy_hits = sum(1 for t in heavy if t in blob or t in name)
    if heavy_hits:
        s += 15 + min(heavy_hits, 4) * 4
        flags["heavy_coverage"] = True
        reasons.append(f"heavy-coverage cues x{heavy_hits}")

    # Identity / img2img
    id_terms = (
        "non-face",
        "non face",
        "nfa",
        "does not change face",
        "doesn't change face",
        "without altering",
        "keep face",
        "preserve",
        "likeness",
        "character",
        "img2img",
        "i2i",
        "inpaint",
        "kontext",
        "edit",
        "favorite character",
    )
    id_hits = sum(1 for t in id_terms if t in blob or t in name)
    if id_hits:
        s += 25 + min(id_hits, 4) * 8
        flags["identity_friendly"] = True
        reasons.append(f"identity/edit friendly x{id_hits}")

    # Photoreal
    real = ("realistic", "photoreal", "photo real", "photograph", "realism", "real life")
    anime = ("anime", "manga", "cartoon", "hentai", "2.5d", "illustrious", "pony")
    rh = sum(1 for t in real if t in blob or t in name)
    ah = sum(1 for t in anime if t in blob or t in name)
    if rh:
        s += 12 + rh * 3
        flags["photoreal"] = True
        reasons.append("photoreal cues")
    if ah and rh == 0:
        s -= 25
        flags["anime"] = True
        reasons.append("anime-leaning")

    # Penalize act LoRAs that aren't primarily facial gel on a still portrait
    acts = ("footjob", "missionary", "doggystyle", "cowgirl", "titjob", "handjob", "blowjob", "pov blow")
    if any(a in name for a in acts) and "cum on face" not in name and "facial" not in name:
        s -= 90
        flags["act_specific"] = True
        reasons.append("act-lora-not-facial")

    # Popularity soft
    s += min(dl // 400, 25)
    s += min(thumbs // 50, 15)
    if rating >= 4.5 and rc >= 8:
        s += 12
        reasons.append(f"rating {rating:.1f} n={rc}")
    elif thumbs >= 100:
        s += 8

    # Type
    if (m.get("type") or "").upper() != "LORA":
        s -= 100
        reasons.append(f"type={m.get('type')}")

    return s, reasons, flags


def paginate(tok: str, params: dict, pages: int = 3) -> list[dict]:
    out = []
    cursor = None
    for _ in range(pages):
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(p)
        try:
            data = api(url, tok)
        except Exception as e:
            print(f"FAIL {params.get('query')}: {e}")
            break
        batch = data.get("items") or []
        if not batch:
            break
        out.extend(batch)
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
        time.sleep(0.2)
    return out


def fetch_detail(tok: str, mid: int) -> dict | None:
    try:
        return api(f"https://civitai.com/api/v1/models/{mid}", tok)
    except Exception:
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)
    me = api("https://civitai.com/api/v1/me", tok)
    print(f"AUTH_OK {me.get('username')}")

    seen: set[int] = set()
    catalog: dict[int, dict] = {}

    for q in QUERIES:
        for base in BASES:
            print(f"SEARCH {q!r} / {base}")
            items = paginate(
                tok,
                {
                    "query": q,
                    "types": "LORA",
                    "limit": 50,
                    "nsfw": "true",
                    "sort": "Most Downloaded",
                    "baseModels": base,
                },
                pages=3,
            )
            for m in items:
                mid = m.get("id")
                if mid is None or mid in seen:
                    continue
                seen.add(mid)
                v = pick_version(m)
                if not v:
                    continue
                sc, reasons, flags = score(m, v)
                catalog[mid] = {
                    "id": mid,
                    "name": m.get("name"),
                    "url": f"https://civitai.com/models/{mid}",
                    "creator": (m.get("creator") or {}).get("username"),
                    "base": v.get("baseModel"),
                    "version_id": v.get("id"),
                    "version_name": v.get("name"),
                    "downloads": int((v.get("stats") or {}).get("downloadCount") or 0),
                    "thumbs": int((v.get("stats") or {}).get("thumbsUpCount") or 0),
                    "triggers": v.get("trainedWords") or [],
                    "file": ((v.get("files") or [{}])[0]).get("name"),
                    "sizeKB": ((v.get("files") or [{}])[0]).get("sizeKB"),
                    "score": sc,
                    "reasons": reasons,
                    "flags": flags,
                    "query": q,
                }

    # Tag sweep
    for tag in ("cum", "facial", "cumshot", "bukkake"):
        print(f"TAG {tag}")
        items = paginate(
            tok,
            {
                "tag": tag,
                "types": "LORA",
                "limit": 50,
                "nsfw": "true",
                "sort": "Highest Rated",
                "baseModels": "Flux.1 D",
            },
            pages=2,
        )
        for m in items:
            mid = m.get("id")
            if mid is None:
                continue
            v = pick_version(m)
            if not v:
                continue
            sc, reasons, flags = score(m, v)
            if mid in catalog and catalog[mid]["score"] >= sc:
                continue
            seen.add(mid)
            catalog[mid] = {
                "id": mid,
                "name": m.get("name"),
                "url": f"https://civitai.com/models/{mid}",
                "creator": (m.get("creator") or {}).get("username"),
                "base": v.get("baseModel"),
                "version_id": v.get("id"),
                "version_name": v.get("name"),
                "downloads": int((v.get("stats") or {}).get("downloadCount") or 0),
                "thumbs": int((v.get("stats") or {}).get("thumbsUpCount") or 0),
                "triggers": v.get("trainedWords") or [],
                "file": ((v.get("files") or [{}])[0]).get("name"),
                "sizeKB": ((v.get("files") or [{}])[0]).get("sizeKB"),
                "score": sc,
                "reasons": reasons,
                "flags": flags,
                "query": f"tag:{tag}",
            }

    ranked = sorted(catalog.values(), key=lambda r: (r["score"], r["downloads"]), reverse=True)
    # Keep only positive facial flux candidates for deep detail
    candidates = [
        r
        for r in ranked
        if r["score"] >= 80
        and not r["flags"].get("is_wrong_base")
        and not r["flags"].get("act_specific")
        and "flux" in (r.get("base") or "").lower()
    ][:25]

    print(f"\nDeep-detailing top {len(candidates)}…")
    detailed = []
    for r in candidates:
        m = fetch_detail(tok, int(r["id"]))
        time.sleep(0.15)
        if not m:
            detailed.append(r)
            continue
        v = pick_version(m) or {}
        desc = " ".join(strip_html(m.get("description") or "").split())[:900]
        # re-score with full description
        sc, reasons, flags = score(m, v if v else {"baseModel": r.get("base"), "stats": {}, "trainedWords": r.get("triggers")})
        row = {
            **r,
            "score": sc,
            "reasons": reasons,
            "flags": flags,
            "description": desc,
            "triggers": v.get("trainedWords") or r.get("triggers") or [],
            "all_versions": [
                {
                    "id": vv.get("id"),
                    "name": vv.get("name"),
                    "base": vv.get("baseModel"),
                    "dl": (vv.get("stats") or {}).get("downloadCount"),
                }
                for vv in (m.get("modelVersions") or [])[:5]
            ],
        }
        detailed.append(row)

    detailed.sort(key=lambda r: (r["score"], r["downloads"]), reverse=True)

    # Combo proposals from top set
    identity = [r for r in detailed if r["flags"].get("identity_friendly")]
    heavy = [r for r in detailed if r["flags"].get("heavy_coverage")]
    kontext = [r for r in detailed if r["flags"].get("is_kontext")]
    flux_d = [r for r in detailed if r["flags"].get("is_flux_d")]

    combos = []
    if identity and heavy:
        a, b = identity[0], None
        for h in heavy:
            if h["id"] != a["id"]:
                b = h
                break
        if b:
            combos.append(
                {
                    "name": "Identity primary + coverage booster",
                    "primary": {"id": a["id"], "name": a["name"], "url": a["url"], "weight": "0.85–1.1"},
                    "secondary": {"id": b["id"], "name": b["name"], "url": b["url"], "weight": "0.35–0.55"},
                    "notes": "Primary locks face; secondary only if coverage thin. Never both at full strength.",
                }
            )
    if kontext:
        k = kontext[0]
        combos.append(
            {
                "name": "Kontext instruction edit only",
                "primary": {"id": k["id"], "name": k["name"], "url": k["url"], "weight": "0.7–1.0"},
                "secondary": None,
                "notes": "Use Flux Kontext UNET + edit instruction; best structural fit for img2img retain.",
            }
        )
    if flux_d:
        f = flux_d[0]
        combos.append(
            {
                "name": "Single best Flux.1 D facial",
                "primary": {"id": f["id"], "name": f["name"], "url": f["url"], "weight": "0.9–1.2"},
                "secondary": None,
                "notes": "Simplest path: one LoRA, modest denoise, face mask.",
            }
        )

    report = {
        "user": me.get("username"),
        "unique_scanned": len(catalog),
        "top_detailed": detailed[:15],
        "combos": combos,
        "reject_examples": [r for r in ranked if r["score"] < 0][:10],
    }
    (OUT / "full_ranked.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    (OUT / "top_detailed.json").write_text(json.dumps(detailed[:20], indent=2), encoding="utf-8")
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Fresh Civitai fluid search (no prior bias)",
        "",
        f"Scanned unique LoRAs: **{len(catalog)}**",
        "",
        "## Top candidates",
        "",
        "| # | Score | DL | Base | Model | Why |",
        "|---:|---:|---:|---|---|---|",
    ]
    for i, r in enumerate(detailed[:12], 1):
        why = "; ".join(r.get("reasons") or [])[:100]
        lines.append(
            f"| {i} | {r['score']} | {r['downloads']} | {r.get('base')} | [{r['name']}]({r['url']}) | {why} |"
        )
    lines += ["", "## Suggested combinations", ""]
    for c in combos:
        lines.append(f"### {c['name']}")
        lines.append(f"- Primary: [{c['primary']['name']}]({c['primary']['url']}) @ {c['primary']['weight']}")
        if c.get("secondary"):
            s = c["secondary"]
            lines.append(f"- Secondary: [{s['name']}]({s['url']}) @ {s['weight']}")
        lines.append(f"- {c['notes']}")
        lines.append("")
    if detailed:
        best = detailed[0]
        lines += [
            "## Verdict",
            "",
            f"**Best single:** [{best['name']}]({best['url']}) (score {best['score']})",
            "",
            f"Triggers: `{best.get('triggers')}`",
            "",
            (best.get("description") or "")[:500],
            "",
        ]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== TOP 12 ===")
    for i, r in enumerate(detailed[:12], 1):
        print(f"{i:2d}. score={r['score']:3d} dl={r['downloads']:6d} {r.get('base')} | {r['name']}")
        print(f"    {r['url']}")
        print(f"    {'; '.join(r.get('reasons') or [])}")
    print("\n=== COMBOS ===")
    for c in combos:
        print(c["name"], "->", c["primary"]["name"], "+", (c.get("secondary") or {}).get("name"))
    print("WROTE", OUT / "REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
