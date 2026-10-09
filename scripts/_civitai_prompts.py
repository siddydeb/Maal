"""Print example gallery prompts / LoRA weights for given Civitai model version ids."""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UA = "Mozilla/5.0 (compatible; WanStudioPrompts/1.0)"


def token() -> str:
    txt = (REPO / "tokens&cmd").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?im)^\s*civitai=(.+)$", txt)
    if not m:
        raise SystemExit("missing civitai=")
    return m.group(1).strip()


def api(url: str, tok: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def main() -> int:
    tok = token()
    for vid in sys.argv[1:]:
        q = urllib.parse.urlencode({"modelVersionId": vid, "limit": "30", "nsfw": "X", "sort": "Most Reactions"})
        items = api(f"https://civitai.com/api/v1/images?{q}", tok).get("items") or []
        print("=" * 100)
        print(f"version {vid}: {len(items)} images")
        shown = 0
        for it in items:
            meta = it.get("meta") or {}
            prompt = meta.get("prompt") or ""
            if not prompt:
                continue
            res = meta.get("resources") or meta.get("civitaiResources") or []
            loras = []
            for r in res if isinstance(res, list) else []:
                if isinstance(r, dict) and (r.get("type") or "").lower() == "lora":
                    loras.append(f"{r.get('name') or r.get('modelVersionId')}@{r.get('weight') or r.get('strength')}")
            print("-" * 60)
            print("PROMPT:", prompt[:700].encode("ascii", "replace").decode())
            neg = meta.get("negativePrompt") or ""
            if neg:
                print("NEG:", neg[:200].encode("ascii", "replace").decode())
            extra = {k: meta.get(k) for k in ("steps", "cfg", "sampler", "seed", "Size") if meta.get(k)}
            if extra or loras:
                print("META:", extra, "LORAS:", loras)
            shown += 1
            if shown >= 8:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
