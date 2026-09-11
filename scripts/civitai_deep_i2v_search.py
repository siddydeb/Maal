"""Deep Civitai search for Wan I2V NSFW sex models/LoRAs.

Project fit: Wan 2.2 I2V 14B high/low dual-stage, realistic face retention,
prefer pretrained all-in-one that does motion+anatomy without stacking hell.

Reads civitai= from tokens&cmd. Never prints the token.
Writes ranked report to tmp_test/civitai_deep_i2v/.
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
OUT = REPO / "tmp_test" / "civitai_deep_i2v"
UA = "Mozilla/5.0 (compatible; WanStudioDeepI2V/1.0)"

QUERIES = [
    # core i2v / wan
    "wan 2.2 i2v nsfw",
    "wan2.2 i2v sex",
    "wan 2.2 i2v",
    "wan22 i2v",
    "wan i2v nsfw",
    "wan video i2v nsfw",
    "wan2.1 i2v nsfw",
    "image to video nsfw wan",
    "img2vid wan nsfw",
    # all-in-one / do-everything
    "dr34ml4y",
    "dreamlay wan",
    "all in one nsfw wan",
    "aio nsfw wan i2v",
    "uncensored wan i2v",
    "nsfw enhancer wan",
    # sex acts (high demand)
    "missionary wan i2v",
    "cowgirl wan i2v",
    "doggy wan i2v",
    "blowjob wan i2v",
    "deepthroat wan i2v",
    "handjob wan i2v",
    "cumshot wan i2v",
    "facial wan i2v",
    "sex wan 2.2",
    "penetration wan i2v",
    "vaginal wan i2v",
    "anal wan i2v",
    "pov sex wan",
    "riding wan i2v",
    # face retention / identity
    "face lock wan",
    "face retain wan i2v",
    "consistent face wan",
    "identity wan i2v",
    "character consistency wan video",
    "ip adapter wan video",
    "face enhancer wan video",
    # anatomy / realism helpers for video
    "male genitalia wan",
    "female genitalia wan",
    "penis lora wan 2.2",
    "realistic sex wan 2.2",
    "photorealistic wan i2v nsfw",
    # known creators / packs from project
    "f4c3spl4sh",
    "blink blowjob",
    "oral insertion wan",
    "assertive cowgirl",
    "berninai wan",
    "bernini wan",
    "lightx2v nsfw",
    "igo on wan",
    "igoon wan",
]

BASES = [
    "Wan Video 2.2 I2V-A14B",
    "Wan Video 2.2 TI2V-5B",
    "Wan Video 2.1 I2V-14B",
    "Wan Video",
]

TAGS = [
    "wan",
    "i2v",
    "nsfw",
    "sex",
    "video",
]


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def api(url: str, tok: str, retries: int = 4):
    headers = {"Authorization": f"Bearer {tok}", "User-Agent": UA}
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception as exc:
            last = exc
            time.sleep(1.0 * (i + 1))
    raise last  # type: ignore[misc]


def safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def text_blob(m: dict, v: dict) -> str:
    tags = m.get("tags") or []
    tag_str = " ".join(
        t.get("name", "") if isinstance(t, dict) else str(t) for t in tags
    )
    parts = [
        m.get("name") or "",
        m.get("description") or "",
        v.get("name") or "",
        v.get("description") or "",
        tag_str,
    ]
    for tw in v.get("trainedWords") or []:
        parts.append(str(tw))
    return re.sub(r"<[^>]+>", " ", " ".join(parts)).lower()


def best_version(m: dict) -> dict | None:
    versions = m.get("modelVersions") or []
    if not versions:
        return None
    # Prefer Wan 2.2 I2V, then Wan 2.1 I2V, then any Wan
    scored = []
    for v in versions:
        base = (v.get("baseModel") or "").lower()
        pts = 0
        if "2.2" in base and "i2v" in base:
            pts += 100
        elif "2.2" in base:
            pts += 70
        elif "2.1" in base and "i2v" in base:
            pts += 50
        elif "wan" in base:
            pts += 30
        if "14b" in base or "a14b" in base:
            pts += 15
        if "5b" in base or "ti2v" in base:
            pts -= 10  # weaker than 14B for our pipeline
        dl = int((v.get("stats") or {}).get("downloadCount") or 0)
        pts += min(dl // 500, 20)
        scored.append((pts, v))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else versions[0]


def score_model(m: dict, v: dict, query: str) -> tuple[int, list[str]]:
    reasons: list[str] = []
    s = 0
    name = (m.get("name") or "").lower()
    base = (v.get("baseModel") or "").lower()
    mtype = (m.get("type") or "").lower()
    blob = text_blob(m, v)
    stats_m = m.get("stats") or {}
    stats_v = v.get("stats") or {}
    dl = int(stats_v.get("downloadCount") or stats_m.get("downloadCount") or 0)
    thumbs = int(stats_v.get("thumbsUpCount") or stats_m.get("thumbsUpCount") or 0)
    rating = float(stats_v.get("rating") or stats_m.get("rating") or 0)
    rating_count = int(stats_v.get("ratingCount") or stats_m.get("ratingCount") or 0)

    # --- Base model fit (critical for Comfy Wan 2.2 I2V) ---
    if "wan video 2.2 i2v" in base or ("2.2" in base and "i2v" in base):
        s += 120
        reasons.append("base=Wan2.2-I2V")
    elif "wan video 2.2" in base or ("wan" in base and "2.2" in base):
        s += 90
        reasons.append("base=Wan2.2")
    elif "wan video 2.1 i2v" in base or ("2.1" in base and "i2v" in base):
        s += 55
        reasons.append("base=Wan2.1-I2V")
    elif "wan" in base:
        s += 35
        reasons.append("base=Wan")
    else:
        s -= 80
        reasons.append("base!=Wan")

    if "14b" in base or "a14b" in base:
        s += 25
        reasons.append("14B")
    if "5b" in base or "ti2v" in base:
        s -= 15
        reasons.append("5B/TI2V-penalty")

    # Type
    if mtype == "lora":
        s += 25
        reasons.append("type=LORA")
    elif mtype == "checkpoint":
        s += 10
        reasons.append("type=Checkpoint")
    elif mtype in ("motionmodule", "motion"):
        s += 15
        reasons.append(f"type={mtype}")

    # Dual high/low presence (our stack needs both stages)
    files = " ".join(
        (f.get("name") or "").lower() for f in (v.get("files") or [])
    )
    ver_name = (v.get("name") or "").lower()
    if ("high" in files or "high" in ver_name or "high" in blob) and (
        "low" in files or "low" in ver_name or "low" in blob
    ):
        s += 40
        reasons.append("has-high+low")

    # NSFW / sex relevance
    nsfw_hits = [
        "nsfw",
        "sex",
        "porn",
        "penetration",
        "intercourse",
        "blowjob",
        "oral",
        "missionary",
        "cowgirl",
        "doggy",
        "anal",
        "vaginal",
        "cumshot",
        "facial",
        "handjob",
        "deepthroat",
        "genitals",
        "penis",
        "pussy",
        "nude",
        "uncensored",
    ]
    nsfw_n = sum(1 for k in nsfw_hits if k in blob or k in name)
    if nsfw_n:
        s += min(nsfw_n * 8, 48)
        reasons.append(f"nsfw-hits={nsfw_n}")
    else:
        s -= 25
        reasons.append("weak-nsfw")

    # All-in-one / do-the-job
    aio_keys = [
        "all in one",
        "all-in-one",
        "aio",
        "dr34ml4y",
        "dreamlay",
        "uncensored",
        "nsfw enhancer",
        "complete",
        "everything",
        "general nsfw",
        "universal",
    ]
    if any(k in blob or k in name for k in aio_keys):
        s += 45
        reasons.append("all-in-one-signal")

    # Face retention / identity (critical user ask)
    face_keys = [
        "face lock",
        "face retain",
        "retain face",
        "preserves face",
        "preserve face",
        "keeps face",
        "keep face",
        "face consistency",
        "consistent face",
        "identity",
        "likeness",
        "character consistency",
        "does not alter face",
        "face intact",
        "same face",
        "facial features",
        "no face change",
        "face fidelity",
        "reference face",
        "ip-adapter",
        "ip adapter",
        "instantid",
        "pulid",
    ]
    face_n = sum(1 for k in face_keys if k in blob)
    if face_n:
        s += min(face_n * 18, 54)
        reasons.append(f"face-retain={face_n}")

    # Anti-face-wipe warnings (still useful but score carefully)
    wipe_keys = ["changes face", "face morph", "face drift", "alters face", "destroys face"]
    if any(k in blob for k in wipe_keys):
        s -= 30
        reasons.append("warns-face-change")

    # Motion quality / i2v explicit
    if "i2v" in name or "i2v" in blob or "image to video" in blob or "img2vid" in blob:
        s += 20
        reasons.append("explicit-i2v")

    # Popularity
    if dl >= 20000:
        s += 35
        reasons.append(f"dl={dl}")
    elif dl >= 5000:
        s += 25
        reasons.append(f"dl={dl}")
    elif dl >= 1000:
        s += 15
        reasons.append(f"dl={dl}")
    elif dl >= 200:
        s += 8
        reasons.append(f"dl={dl}")

    if thumbs >= 500:
        s += 20
    elif thumbs >= 100:
        s += 12
    elif thumbs >= 30:
        s += 6
    if thumbs:
        reasons.append(f"thumbs={thumbs}")

    if rating >= 4.5 and rating_count >= 10:
        s += 15
        reasons.append(f"rating={rating:.1f}n={rating_count}")
    elif rating >= 4.0 and rating_count >= 5:
        s += 8
        reasons.append(f"rating={rating:.1f}n={rating_count}")

    # Query match bonus
    q = query.lower()
    if q and (q in name or q in blob):
        s += 10

    # Prefer models already known good in project (but still discover new)
    known_boost = [
        "dr34ml4y",
        "f4c3spl4sh",
        "blink blowjob",
        "oral insertion",
        "missionary",
        "assertive cowgirl",
        "penislora",
        "lightx2v",
        "bernin",
    ]
    if any(k in name or k in blob for k in known_boost):
        s += 8
        reasons.append("known-family")

    # NSFW level from API
    nsfw_level = m.get("nsfwLevel") or v.get("nsfwLevel") or 0
    if isinstance(nsfw_level, int) and nsfw_level >= 4:
        s += 10
        reasons.append(f"nsfwLevel={nsfw_level}")

    return s, reasons


def search_models(tok: str, query: str, base: str | None = None, tag: str | None = None, limit: int = 40):
    params = {
        "limit": str(limit),
        "query": query,
        "nsfw": "true",
        "sort": "Most Downloaded",
        "period": "AllTime",
    }
    if base:
        params["baseModels"] = base
    if tag:
        params["tag"] = tag
    # Prefer LORA but also get checkpoints
    url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(params)
    data = api(url, tok)
    return data.get("items") or []


def fetch_model(tok: str, model_id: int) -> dict:
    return api(f"https://civitai.com/api/v1/models/{model_id}", tok)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = token()
    print("Auth OK (token loaded). Deep I2V NSFW search starting...", flush=True)

    seen: dict[int, dict] = {}
    hits_log: list[dict] = []

    # Phase 1: query sweeps with Wan bases
    for q in QUERIES:
        for base in [None] + BASES:
            label = f"q={q!r} base={base or '*'}"
            try:
                items = search_models(tok, q, base=base, limit=30)
            except Exception as exc:
                print(f"  FAIL {safe(label)}: {exc}", flush=True)
                time.sleep(0.5)
                continue
            print(f"  {safe(label)} -> {len(items)}", flush=True)
            for m in items:
                mid = m.get("id")
                if not isinstance(mid, int):
                    continue
                if mid not in seen:
                    seen[mid] = m
                hits_log.append({"query": q, "base": base, "id": mid, "name": m.get("name")})
            time.sleep(0.25)

    # Phase 2: tag sweeps on Wan bases
    for tag in TAGS:
        for base in BASES[:2]:
            label = f"tag={tag} base={base}"
            try:
                items = search_models(tok, query="", base=base, tag=tag, limit=50)
            except Exception as exc:
                print(f"  FAIL {safe(label)}: {exc}", flush=True)
                continue
            print(f"  {safe(label)} -> {len(items)}", flush=True)
            for m in items:
                mid = m.get("id")
                if isinstance(mid, int) and mid not in seen:
                    seen[mid] = m
            time.sleep(0.25)

    # Phase 3: empty query Most Downloaded on Wan 2.2 I2V
    for base in BASES:
        label = f"browse base={base}"
        try:
            items = search_models(tok, query="", base=base, limit=100)
        except Exception as exc:
            print(f"  FAIL {safe(label)}: {exc}", flush=True)
            continue
        print(f"  {safe(label)} -> {len(items)}", flush=True)
        for m in items:
            mid = m.get("id")
            if isinstance(mid, int) and mid not in seen:
                seen[mid] = m
        time.sleep(0.3)

    print(f"Unique models collected: {len(seen)}", flush=True)

    # Enrich top candidates with full model detail
    ranked_raw = []
    for mid, m in seen.items():
        v = best_version(m)
        if not v:
            continue
        # Use first query that hit, or generic
        qhit = next((h["query"] for h in hits_log if h["id"] == mid), "")
        score, reasons = score_model(m, v, qhit)
        ranked_raw.append((score, mid, m, v, reasons, qhit))

    ranked_raw.sort(key=lambda x: x[0], reverse=True)

    # Deep-fetch top 60 for richer descriptions / files / trainedWords
    top_ids = [mid for _, mid, *_ in ranked_raw[:60]]
    detailed: dict[int, dict] = {}
    for i, mid in enumerate(top_ids):
        try:
            detailed[mid] = fetch_model(tok, mid)
            print(f"  detail {i+1}/{len(top_ids)} id={mid}", flush=True)
        except Exception as exc:
            print(f"  detail FAIL id={mid}: {exc}", flush=True)
        time.sleep(0.2)

    # Re-score with detailed payloads
    final = []
    for score0, mid, m0, v0, reasons0, qhit in ranked_raw:
        m = detailed.get(mid) or m0
        v = best_version(m) or v0
        score, reasons = score_model(m, v, qhit)
        files = []
        for f in v.get("files") or []:
            files.append(
                {
                    "name": f.get("name"),
                    "sizeKB": f.get("sizeKB"),
                    "downloadUrl": f.get("downloadUrl"),
                    "primary": f.get("primary"),
                }
            )
        creator = (m.get("creator") or {}).get("username")
        final.append(
            {
                "score": score,
                "id": mid,
                "name": m.get("name"),
                "url": f"https://civitai.com/models/{mid}",
                "type": m.get("type"),
                "creator": creator,
                "baseModel": v.get("baseModel"),
                "versionName": v.get("name"),
                "versionId": v.get("id"),
                "trainedWords": v.get("trainedWords") or [],
                "downloadCount": int(
                    (v.get("stats") or {}).get("downloadCount")
                    or (m.get("stats") or {}).get("downloadCount")
                    or 0
                ),
                "thumbsUp": int(
                    (v.get("stats") or {}).get("thumbsUpCount")
                    or (m.get("stats") or {}).get("thumbsUpCount")
                    or 0
                ),
                "rating": float(
                    (v.get("stats") or {}).get("rating")
                    or (m.get("stats") or {}).get("rating")
                    or 0
                ),
                "nsfwLevel": m.get("nsfwLevel") or v.get("nsfwLevel"),
                "files": files,
                "reasons": reasons,
                "queryHit": qhit,
                "descriptionSnippet": safe(
                    re.sub(r"<[^>]+>", " ", (m.get("description") or ""))[:600]
                ),
            }
        )

    final.sort(key=lambda x: x["score"], reverse=True)

    # Bucket recommendations
    aio = [x for x in final if "all-in-one-signal" in " ".join(x["reasons"]) and x["score"] >= 100]
    face = [x for x in final if any(r.startswith("face-retain") for r in x["reasons"]) and x["score"] >= 80]
    wan22 = [x for x in final if "Wan2.2" in " ".join(x["reasons"]) and x["score"] >= 120]
    sex_act = [
        x
        for x in final
        if x["score"] >= 100
        and any(
            k in (x["name"] or "").lower()
            for k in (
                "missionary",
                "cowgirl",
                "doggy",
                "blowjob",
                "oral",
                "handjob",
                "cumshot",
                "sex",
                "anal",
                "riding",
            )
        )
    ]

    # Pick #1: prefer Wan2.2 + AIO + face + high downloads
    def pick_best(pool: list[dict]) -> dict | None:
        if not pool:
            return None
        return sorted(
            pool,
            key=lambda x: (
                x["score"],
                x["downloadCount"],
                x["thumbsUp"],
            ),
            reverse=True,
        )[0]

    best_overall = pick_best([x for x in final if x["score"] >= 140]) or (final[0] if final else None)
    best_aio = pick_best(aio)
    best_face = pick_best(face)

    report = {
        "scope": "Wan 2.2 I2V NSFW sex + face retention, prefer AIO pretrained",
        "uniqueModels": len(seen),
        "scored": len(final),
        "bestOverall": best_overall,
        "bestAllInOne": best_aio,
        "bestFaceRetention": best_face,
        "top25": final[:25],
        "aioCandidates": aio[:15],
        "faceCandidates": face[:15],
        "wan22Strong": wan22[:20],
        "sexActTop": sex_act[:20],
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "full_ranked.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    (OUT / "search_hits.json").write_text(json.dumps(hits_log, indent=2), encoding="utf-8")

    # Markdown recommendation
    lines = [
        "# Civitai deep I2V NSFW search — Wan Studio fit",
        "",
        "Scope: **Wan 2.2 I2V 14B** dual high/low, realistic NSFW sex, **face retention**, prefer one powerful pretrained AIO.",
        "",
        f"Searched with authenticated API. Unique models scored: **{len(final)}** (from {len(seen)} collected).",
        "",
    ]
    if best_overall:
        lines += [
            "## #1 Best overall for this project",
            "",
            f"**[{best_overall['name']}]({best_overall['url']})** — score {best_overall['score']}",
            "",
            f"- Creator: `{best_overall.get('creator')}`",
            f"- Base: `{best_overall.get('baseModel')}` / version `{best_overall.get('versionName')}`",
            f"- Downloads: {best_overall.get('downloadCount')} | thumbs: {best_overall.get('thumbsUp')}",
            f"- Triggers: `{best_overall.get('trainedWords')}`",
            f"- Why: {', '.join(best_overall.get('reasons') or [])}",
            f"- Files: {', '.join(f.get('name') or '?' for f in (best_overall.get('files') or [])[:6])}",
            "",
            (best_overall.get("descriptionSnippet") or "")[:400],
            "",
        ]
    if best_aio and (not best_overall or best_aio["id"] != best_overall["id"]):
        lines += [
            "## Best all-in-one (do the job)",
            "",
            f"**[{best_aio['name']}]({best_aio['url']})** — score {best_aio['score']}",
            f"- Base: `{best_aio.get('baseModel')}` | dl={best_aio.get('downloadCount')}",
            f"- Why: {', '.join(best_aio.get('reasons') or [])}",
            "",
        ]
    if best_face and (not best_overall or best_face["id"] != best_overall["id"]):
        lines += [
            "## Best face-retention signal",
            "",
            f"**[{best_face['name']}]({best_face['url']})** — score {best_face['score']}",
            f"- Why: {', '.join(best_face.get('reasons') or [])}",
            "",
        ]

    lines += ["## Top 15 ranked", ""]
    for i, x in enumerate(final[:15], 1):
        lines.append(
            f"{i}. [{x['name']}]({x['url']}) — score {x['score']} | "
            f"`{x.get('baseModel')}` | dl={x.get('downloadCount')} | "
            f"{', '.join(x.get('reasons') or [])[:120]}"
        )
    lines += ["", "## Sex-act specialists (stack only if needed)", ""]
    for i, x in enumerate(sex_act[:12], 1):
        lines.append(
            f"{i}. [{x['name']}]({x['url']}) — score {x['score']} | `{x.get('baseModel')}` | dl={x.get('downloadCount')}"
        )
    lines += [
        "",
        "## Fit notes for Wan Studio",
        "",
        "- Pipeline already uses Wan 2.2 I2V high/low + anatomy + pose LoRAs.",
        "- Prefer **one AIO NSFW** (e.g. DR34ML4Y-class) at low strength over stacking 8 act LoRAs (face drift).",
        "- Dual-stage high+low files required; skip single-file-only unless tested.",
        "- Flux facial LoRAs are **img2img only** — not for I2V.",
        "",
    ]
    (OUT / "FINAL_RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== DONE ===", flush=True)
    print(f"Wrote {OUT / 'FINAL_RECOMMENDATION.md'}", flush=True)
    if best_overall:
        print(f"BEST: {safe(best_overall['name'])} score={best_overall['score']} {best_overall['url']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
