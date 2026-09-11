"""Pull full descriptions for top fresh facial candidates."""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOK = re.search(
    r"(?im)^\s*civitai=(.+)$",
    (REPO / "tokens&cmd").read_text(encoding="utf-8", errors="ignore"),
).group(1).strip()
OUT = REPO / "tmp_test" / "civitai_fresh_fluid"
IDS = [
    655732,  # Cum On Face FLUX
    858262,  # NFA
    691681,  # Not Another Facial
    725999,  # COF mawedesign
    866273,  # Facial cum massive
    1315268,  # Character Friendly Facial
    730199,  # CumHereV1
    924374,
    1240792,
    1753290,  # YACL
]


def get(mid: int) -> dict:
    req = urllib.request.Request(
        f"https://civitai.com/api/v1/models/{mid}",
        headers={"Authorization": f"Bearer {TOK}", "User-Agent": "Wan/1"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for mid in IDS:
        try:
            m = get(mid)
        except Exception as e:
            print("FAIL", mid, e)
            continue
        desc = re.sub(r"<[^>]+>", " ", m.get("description") or "")
        desc = " ".join(desc.split())
        vers = m.get("modelVersions") or []
        flux = [v for v in vers if "flux" in (v.get("baseModel") or "").lower()]
        v = flux[0] if flux else (vers[0] if vers else {})
        st = v.get("stats") or {}
        row = {
            "id": mid,
            "name": m.get("name"),
            "url": f"https://civitai.com/models/{mid}",
            "creator": (m.get("creator") or {}).get("username"),
            "base": v.get("baseModel"),
            "ver": v.get("id"),
            "ver_name": v.get("name"),
            "dl": st.get("downloadCount"),
            "thumbs": st.get("thumbsUpCount"),
            "triggers": v.get("trainedWords") or [],
            "file": ((v.get("files") or [{}])[0]).get("name"),
            "sizeKB": ((v.get("files") or [{}])[0]).get("sizeKB"),
            "desc": desc[:1200],
        }
        rows.append(row)
        print("=" * 60)
        print(mid, m.get("name"), "|", row["creator"], "|", row["base"], "| dl", row["dl"])
        print("triggers:", row["triggers"])
        print("file:", row["file"], row["sizeKB"])
        print(desc[:600])
        print()

    # also search cumifier by name
    req = urllib.request.Request(
        "https://civitai.com/api/v1/models?query=Cumifier&types=LORA&limit=20&nsfw=true",
        headers={"Authorization": f"Bearer {TOK}", "User-Agent": "Wan/1"},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        data = json.loads(r.read().decode())
    print("CUMIFIER HITS:")
    for m in data.get("items") or []:
        v = (m.get("modelVersions") or [{}])[0]
        print(
            " ",
            m.get("id"),
            m.get("name"),
            v.get("baseModel"),
            (v.get("stats") or {}).get("downloadCount"),
        )

    (OUT / "candidate_details.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Human verdict memo
    memo = {
        "fresh_verdict": {
            "best_single_for_visible_coverage": {
                "id": 655732,
                "name": "Cum On Face FLUX",
                "note": "Highest DL; strongest visible facial load among Flux.1 D. Not NFA — needs low denoise + face lock.",
            },
            "best_single_for_identity": {
                "id": 858262,
                "name": "Cum on face FLUX Non-Face Altering v2",
                "note": "Best face retention claims; V2 de-distilled. Coverage often weaker than Cuminator in practice.",
            },
            "best_new_discovery": {
                "id": 691681,
                "name": "Not Another Facial LoRA",
                "note": "Strong fresh hit — photoreal + identity cues; investigate before locking stack.",
            },
            "best_massive_coverage": {
                "id": 866273,
                "name": "Facial cum massive FLUX/KREA2",
                "note": "Name/target = massive facial; good coverage candidate.",
            },
            "best_character_friendly_alt": {
                "id": 1315268,
                "name": "Character Friendly Facial - FLUX",
                "note": "Explicit character-friendly positioning — good combo primary.",
            },
            "body_clothes_not_face": {
                "id": 730199,
                "name": "CumHereV1 (clothes, hair, non-facials)",
                "note": "Use as SECONDARY for clothes/hair mess, not face primary.",
            },
            "recommended_combo_fresh": [
                {
                    "role": "face identity + facial gel",
                    "model": 858262,
                    "weight": "0.9–1.15",
                    "or_alt": 1315268,
                },
                {
                    "role": "coverage booster if weak",
                    "model": 866273,
                    "weight": "0.3–0.5",
                    "or_alt": 655732,
                },
                {
                    "role": "optional clothes/hair only",
                    "model": 730199,
                    "weight": "0.4–0.6",
                },
            ],
            "do_not_use": ["SD1.5 enhancers", "Pony", "act NFA LoRAs", "getphat Reality NSFW as facial primary"],
        }
    }
    (OUT / "VERDICT.json").write_text(json.dumps(memo, indent=2), encoding="utf-8")
    print("WROTE", OUT / "VERDICT.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
