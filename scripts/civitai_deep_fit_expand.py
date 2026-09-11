"""Expand NFA creator series + version details for deep fit report."""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKENS = REPO / "tokens&cmd"
OUT = REPO / "tmp_test" / "civitai_deep_fit"


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing token")
    return m.group(1).strip().strip('"').strip("'")


def get(url: str, tok: str):
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {tok}", "User-Agent": "WanStudio/1.0"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)

    user = "valentinkognito365"
    items = []
    cursor = None
    for _ in range(6):
        p = {
            "username": user,
            "types": "LORA",
            "limit": 50,
            "nsfw": "true",
            "sort": "Most Downloaded",
        }
        if cursor:
            p["cursor"] = cursor
        data = get("https://civitai.com/api/v1/models?" + urllib.parse.urlencode(p), tok)
        batch = data.get("items") or []
        if not batch:
            break
        items.extend(batch)
        cursor = (data.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break

    series = []
    for m in items:
        name = m.get("name") or ""
        pick = None
        for v in m.get("modelVersions") or []:
            if "flux" in (v.get("baseModel") or "").lower():
                pick = v
                break
        if not pick:
            continue
        st = pick.get("stats") or {}
        series.append(
            {
                "id": m.get("id"),
                "name": name,
                "url": f"https://civitai.com/models/{m.get('id')}",
                "base": pick.get("baseModel"),
                "ver": pick.get("id"),
                "dl": st.get("downloadCount") or 0,
                "triggers": pick.get("trainedWords") or [],
                "nfa": ("non-face" in name.lower()) or ("non face" in name.lower()),
                "facial_fluid": any(
                    k in name.lower() for k in ("cum", "facial", "cof", "bukkake", "messy")
                ),
                "tongue": "tongue" in name.lower() or "ahegao" in name.lower(),
            }
        )
    series.sort(key=lambda r: r["dl"], reverse=True)
    (OUT / "nfa_creator_series.json").write_text(json.dumps(series, indent=2), encoding="utf-8")
    print(f"NFA_CREATOR_COUNT {len(series)}")
    for s in series:
        mark = "*" if s["nfa"] else " "
        print(f"{mark} dl={s['dl']:5d} id={s['id']} | {s['name']}")

    versions = {}
    for mid in [725999, 655732, 858262, 1240792, 924374, 1753290, 693749, 854713]:
        m = get(f"https://civitai.com/api/v1/models/{mid}", tok)
        vs = []
        for v in m.get("modelVersions") or []:
            f = (v.get("files") or [{}])[0]
            vs.append(
                {
                    "id": v.get("id"),
                    "name": v.get("name"),
                    "base": v.get("baseModel"),
                    "dl": (v.get("stats") or {}).get("downloadCount"),
                    "file": f.get("name"),
                    "sizeKB": f.get("sizeKB"),
                    "triggers": v.get("trainedWords") or [],
                }
            )
        versions[str(mid)] = {"name": m.get("name"), "versions": vs}
        print("VERS", mid, m.get("name"))
        for v in vs[:6]:
            print(" ", v["id"], v["name"], v["base"], "dl", v["dl"], v["file"])

    (OUT / "version_matrix.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")

    # Final recommendation memo
    memo = {
        "best_overall_facial_identity": {
            "id": 858262,
            "name": "Cum on face - FLUX - [Non-Face Altering]",
            "url": "https://civitai.com/models/858262",
            "why": [
                "Only major Flux facial LoRA explicitly trained Non-Face Altering",
                "V2 retrained on De-distilled Flux (matches our Dedistilled UNET path)",
                "Rich triggers for placement/opacity (COF + sticky white reflections)",
                "Same creator family as other reliable NFA LoRAs",
            ],
            "use": "Primary fluid LoRA on Flux.1 D / Dedistilled img2img + face mask",
            "trigger": "COF",
            "already_on_disk": "flux_facial_fluid_v1.safetensors",
        },
        "best_heavy_coverage_secondary": {
            "id": 655732,
            "name": "Cum On Face FLUX (Cuminator)",
            "url": "https://civitai.com/models/655732",
            "why": [
                "Highest downloads among Flux facial (~15.6k)",
                "Strong visible coverage triggers",
                "NOT NFA — expect more identity/scene drift than 858262",
            ],
            "use": "A/B only when NFA coverage too weak; keep denoise low + face lock",
        },
        "best_likeness_prompt_style": {
            "id": 725999,
            "name": "COF Cum On Flux (mawedesign)",
            "url": "https://civitai.com/models/725999",
            "why": [
                "Author docs: whitish translucent splatter keeps likeness",
                "Flexible natural-language prompting",
                "V6 previously glitched in our stack — treat carefully / older epoch",
            ],
            "use": "Prompt recipe source; weight cautiously if retested",
        },
        "best_dense_drops_alt": {
            "id": 1240792,
            "name": "Cum Facial / Cum on face - FLUX",
            "url": "https://civitai.com/models/1240792",
            "why": ["Author trained because others were too weak/tiny drops"],
            "use": "Coverage boost A/B behind NFA",
        },
        "best_tongue_realistic": {
            "id": 693749,
            "name": "Female Tongue, Mouth and Teeth - FLUX",
            "url": "https://civitai.com/models/693749",
            "why": [
                "Official Flux sister of the Pony model you linked",
                "600+ photoreal mouth/tongue/teeth images @1024",
                "Author recommends CFG 1.8–2.5 — matches our Flux edit guidance",
            ],
            "use": "Tongue edits on Flux img2img; low denoise for identity",
        },
        "best_tongue_nfa_combo": {
            "id": 854713,
            "name": "Ahegao - FLUX - [Non-Face Altering]",
            "url": "https://civitai.com/models/854713",
            "why": [
                "Same NFA creator as facial COF",
                "Designed to combine with other NFA LoRAs without face rewrite",
            ],
            "use": "Optional tongue-out expression with identity lock",
        },
        "reject": [
            {
                "id": 248823,
                "name": "Cumshot Enhancer SD1.5",
                "why": "Wrong base — remakes face/clothes/scene on our Flux path",
            },
            {
                "id": 721066,
                "name": "Tongue Pony",
                "why": "Pony only — use Flux sister 693749 instead",
            },
            {
                "id": 1035573,
                "name": "POV Footjob Cumshot NFA",
                "why": "Wrong act; NFA label inflated score — not facial gel for portraits",
            },
        ],
        "civitai_red_note": "civitai.red mirror returned 409; authoritative API used was civitai.com with your token (same model IDs).",
    }
    (OUT / "FINAL_RECOMMENDATION.json").write_text(json.dumps(memo, indent=2), encoding="utf-8")

    md = [
        "# Final Civitai recommendation (project fit)",
        "",
        "Scope: realistic **Flux.1 D / Kontext img2img**, facial opaque gel, face+clothes+scene retention.",
        "",
        "## #1 Best overall — use this",
        "",
        "**[Cum on face - FLUX - Non-Face Altering](https://civitai.com/models/858262)** (valentinkognito365)",
        "",
        "- Explicit NFA training; V2 on **De-distilled Flux**",
        "- Trigger `COF` + sticky/white-reflection prompts",
        "- Already on disk as `flux_facial_fluid_v1.safetensors`",
        "",
        "## Runners-up (facial)",
        "",
        "1. [Cum On Face FLUX / Cuminator](https://civitai.com/models/655732) — heaviest coverage, more drift risk",
        "2. [Cum Facial FLUX](https://civitai.com/models/1240792) — denser drops than weak LoRAs",
        "3. [COF Cum On Flux](https://civitai.com/models/725999) — likeness-friendly translucent prompt style",
        "4. [YACL](https://civitai.com/models/1753290) — small niche alt (`cumonface`, weight 1.0–1.5)",
        "",
        "## Tongue",
        "",
        "1. **[Female Tongue Mouth Teeth - FLUX](https://civitai.com/models/693749)** — best realistic Flux mouth LoRA",
        "2. [Ahegao NFA](https://civitai.com/models/854713) — same NFA creator; stackable with COF",
        "",
        "## Reject for this pipeline",
        "",
        "- SD1.5 Cumshot Enhancer, Pony tongue, act-specific NFA (footjob/missionary/etc.)",
        "- Flux.2 Klein LoRAs until UNET migration",
        "",
        "## Stack recipe",
        "",
        "1. UNET: Dedistilled Flux (or Dev fp8)",
        "2. Primary LoRA: NFA COF `858262` @ ~0.9–1.2",
        "3. Optional: Cum Facial `1240792` low weight if coverage weak",
        "4. Face mask + gel-only composite (existing identity lock)",
        "5. Tongue path: `693749` alone at low denoise — do not mix with SD1.5/Pony",
        "",
    ]
    (OUT / "FINAL_RECOMMENDATION.md").write_text("\n".join(md), encoding="utf-8")
    print("WROTE", OUT / "FINAL_RECOMMENDATION.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
