"""Second-pass deep dive on AIO + face signals for Wan I2V NSFW."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKENS = REPO / "tokens&cmd"
OUT = REPO / "tmp_test" / "civitai_deep_i2v"
UA = "Mozilla/5.0 (compatible; WanStudioDeepI2V2/1.0)"


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing civitai=")
    return m.group(1).strip()


def api(url: str, tok: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def main() -> int:
    tok = token()
    focus = [1811313, 1307155, 2206794, 2053259, 2003153, 2040641, 2048863, 2918508, 1295758]
    queries = [
        "general nsfw wan 2.2",
        "cubey general nsfw",
        "k3nk allinone",
        "all in one sex wan",
        "nsfw helper wan 2.2",
        "wan nsfw master",
        "sex motion wan aio",
        "universal sex wan",
        "dr34ml4y",
    ]
    extra_ids: set[int] = set()
    for q in queries:
        url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(
            {
                "limit": "20",
                "query": q,
                "nsfw": "true",
                "sort": "Most Downloaded",
                "baseModels": "Wan Video 2.2 I2V-A14B",
            }
        )
        try:
            items = api(url, tok).get("items") or []
        except Exception as exc:
            print("fail", q, exc, flush=True)
            continue
        print(f"Q {q!r} -> {len(items)}", flush=True)
        for m in items:
            mid = m.get("id")
            if isinstance(mid, int):
                extra_ids.add(mid)
                name = (m.get("name") or "")[:70].encode("ascii", "replace").decode()
                dl = (m.get("stats") or {}).get("downloadCount")
                print(f"  {mid} {m.get('type')} {name} dl={dl}", flush=True)
        time.sleep(0.2)

    for mid in list(extra_ids)[:30]:
        if mid not in focus:
            focus.append(mid)

    rows = []
    for mid in focus:
        m = api(f"https://civitai.com/api/v1/models/{mid}", tok)
        desc = re.sub(r"<[^>]+>", " ", m.get("description") or "")
        desc = re.sub(r"\s+", " ", desc).strip()
        wan_vers = []
        for v in m.get("modelVersions") or []:
            base = v.get("baseModel") or ""
            if "Wan" not in base:
                continue
            files = [{"name": f.get("name"), "sizeKB": f.get("sizeKB")} for f in (v.get("files") or [])]
            wan_vers.append(
                {
                    "name": v.get("name"),
                    "base": base,
                    "id": v.get("id"),
                    "dl": (v.get("stats") or {}).get("downloadCount"),
                    "thumbs": (v.get("stats") or {}).get("thumbsUpCount"),
                    "trainedWords": v.get("trainedWords") or [],
                    "files": files,
                }
            )
        dlow = desc.lower()
        name_l = (m.get("name") or "").lower()
        signals = {
            "mentions_face": any(
                k in dlow for k in ["face", "likeness", "identity", "character consistency", "retain", "preserv"]
            ),
            "warns_likeness_change": any(
                k in dlow
                for k in [
                    "influence the likeness",
                    "changes the face",
                    "face morph",
                    "alter the face",
                    "destroys face",
                ]
            ),
            "claims_preserve": any(
                k in dlow
                for k in [
                    "preserv",
                    "retain",
                    "keep the face",
                    "same face",
                    "does not alter",
                    "least amount of influence",
                    "likeness",
                ]
            ),
            "aio": any(
                k in dlow or k in name_l
                for k in ["all-in-one", "all in one", "aio", "general nsfw", "4llinone", "helper"]
            ),
            "multi_pose": sum(
                1
                for k in ["missionary", "cowgirl", "doggy", "blowjob", "handjob", "anal", "oral"]
                if k in dlow
            ),
        }
        rows.append(
            {
                "id": mid,
                "name": m.get("name"),
                "type": m.get("type"),
                "creator": (m.get("creator") or {}).get("username"),
                "url": f"https://civitai.com/models/{mid}",
                "stats": m.get("stats"),
                "signals": signals,
                "desc": desc[:1500],
                "wan_versions": wan_vers[:10],
            }
        )
        time.sleep(0.15)

    # Rank for project: LoRA > Checkpoint (drop-in), AIO, Wan2.2 I2V, multi_pose, downloads
    scored = []
    for r in rows:
        pts = 0
        why = []
        if r["type"] == "LORA":
            pts += 50
            why.append("lora-dropin")
        elif r["type"] == "Checkpoint":
            pts += 15
            why.append("checkpoint-swap")
        if r["signals"]["aio"]:
            pts += 40
            why.append("aio")
        if r["signals"]["multi_pose"] >= 3:
            pts += 35
            why.append(f"poses={r['signals']['multi_pose']}")
        elif r["signals"]["multi_pose"] >= 1:
            pts += 10
        if r["signals"]["claims_preserve"] and not r["signals"]["warns_likeness_change"]:
            pts += 25
            why.append("face-friendly")
        if r["signals"]["warns_likeness_change"]:
            pts -= 15
            why.append("likeness-risk")
        i2v = any("I2V" in (v.get("base") or "") for v in r["wan_versions"])
        if i2v:
            pts += 30
            why.append("has-i2v")
        high_low = False
        for v in r["wan_versions"]:
            blob = (v.get("name") or "").lower() + " " + " ".join(f.get("name") or "" for f in v.get("files") or [])
            if "high" in blob:
                high_low = True
        # need both high and low somewhere
        names = " ".join(
            (v.get("name") or "") + " " + " ".join(f.get("name") or "" for f in v.get("files") or [])
            for v in r["wan_versions"]
        ).lower()
        if "high" in names and "low" in names:
            pts += 25
            why.append("high+low")
        wan_dl = sum(int(v.get("dl") or 0) for v in r["wan_versions"])
        if wan_dl >= 100000:
            pts += 40
        elif wan_dl >= 30000:
            pts += 25
        elif wan_dl >= 10000:
            pts += 15
        why.append(f"wan_dl={wan_dl}")
        # already in project stack
        if r["id"] == 1811313:
            pts += 20
            why.append("already-wired")
        scored.append((pts, why, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    payload = {
        "ranked": [
            {
                "fitScore": pts,
                "why": why,
                **{k: r[k] for k in ("id", "name", "type", "creator", "url", "signals", "stats", "desc", "wan_versions")},
            }
            for pts, why, r in scored
        ]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aio_face_deep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Final pick — Wan 2.2 I2V NSFW (Civitai authenticated search)",
        "",
        "Constraint: drop into existing Wan Studio I2V high/low stack, maximize sex coverage,",
        "minimize face wipe, prefer one pretrained AIO over act stacking.",
        "",
    ]
    best = scored[0][2] if scored else None
    if best:
        pts, why, _ = scored[0]
        md += [
            "## #1 Use this (best project fit)",
            "",
            f"**[{best['name']}]({best['url']})** — fitScore {pts}",
            "",
            f"- Type: `{best['type']}` | Creator: `{best.get('creator')}`",
            f"- Why: {', '.join(why)}",
            f"- Signals: `{best['signals']}`",
            "",
            "### Wan versions / files",
            "",
        ]
        for v in best.get("wan_versions") or []:
            files = ", ".join(f.get("name") or "?" for f in (v.get("files") or [])[:4])
            md.append(
                f"- `{v.get('base')}` / **{v.get('name')}** — dl={v.get('dl')} — triggers `{v.get('trainedWords')}` — files: {files}"
            )
        md += ["", (best.get("desc") or "")[:700], ""]

    md += ["## Ranked AIO / general contenders", ""]
    for i, (pts, why, r) in enumerate(scored[:12], 1):
        md.append(
            f"{i}. [{r['name']}]({r['url']}) — fit {pts} | `{r['type']}` | {', '.join(why)}"
        )

    md += [
        "",
        "## Verdict for Wan Studio",
        "",
        "1. **Primary AIO LoRA (do-the-job):** DR34ML4Y All-In-One NSFW — already in `private/lora_stack.py`.",
        "   Covers BJ / missionary / cowgirl / doggy in one LoRA. Keep strength modest for face lock.",
        "2. **Strongest alternative AIO LoRA:** CubeyAI WAN General NSFW (`1307155`) — highest downloads;",
        "   experimental but community default for general uncensored I2V.",
        "3. **Runner-up AIO:** K3NK 4llinOne NSFW Helper (`2206794`) — same creator family as F4C3SPL4SH.",
        "4. **If willing to swap UNets (more work):** WAN 2.2 Enhanced NSFW / Remix checkpoints — not drop-in LoRAs.",
        "5. **Do not** stack many act LoRAs if face retention matters — that is what drifts identity.",
        "6. Face retention on I2V is mostly the start image + light LoRA strength; no magic face-lock LoRA beat base I2V.",
        "",
    ]
    (OUT / "FINAL_PICK.md").write_text("\n".join(md), encoding="utf-8")
    print("=== TOP 8 ===", flush=True)
    for pts, why, r in scored[:8]:
        nm = (r["name"] or "")[:65].encode("ascii", "replace").decode()
        print(f"{pts:3d} {r['type']:12s} {r['id']} {nm}", flush=True)
        print(f"     {why}", flush=True)
    print("Wrote", OUT / "FINAL_PICK.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
