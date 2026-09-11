"""Discover + download Flux-fit facial/tongue LoRAs via Civitai API token."""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOKENS = REPO / "tokens&cmd"
LORAS = Path(r"E:\Comfy-Desktop\ComfyUI-Shared\models\loras")
OUT = REPO / "tmp_test" / "civitai_fit_search"
UA = "Mozilla/5.0 (compatible; WanStudio/1.0)"

# Explicit candidates we already know + user links
KNOWN = [
    # label, model_id, version_id|None, dest, why
    ("nfa_v2", 858262, 1032060, "flux_facial_fluid_v1.safetensors", "Flux.1 D NFA — best prior fit"),
    ("cumifier_kontext", 1750558, None, "flux_kontext_fluid_v1.safetensors", "Kontext Cumifier"),
    ("tongue_flux", 970280, None, "tongue-flux-v2.1.safetensors", "Flux tongue sister of Pony"),
    ("pony_tongue", 721066, 1575055, "Female_Tongue_Mouth_and_Teeth_PONY-v2.safetensors", "user link — Pony only"),
    ("csenhance", 248823, 280765, "csenhance_v1.safetensors", "user link — SD1.5 only"),
]

QUERIES = [
    ("cumonface", "Flux.1 D"),
    ("cum on face", "Flux.1 D"),
    ("Cumifier", "Flux.1 Kontext"),
    ("Cumifier", "Flux.1 D"),
    ("non-face-altering", "Flux.1 D"),
    ("cof", "Flux.1 D"),
    ("add cum", "Flux.1 D"),
    ("tongue out", "Flux.1 D"),
    ("female tongue", "Flux.1 D"),
]


def token() -> str:
    m = re.search(r"(?im)^\s*civitai=(.+)$", TOKENS.read_text(encoding="utf-8", errors="ignore"))
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def api(url: str, tok: str, binary: bool = False):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
        if binary:
            return raw
        return json.loads(raw.decode("utf-8", errors="replace"))


def safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def relevant(name: str) -> bool:
    n = name.lower()
    keys = (
        "cum",
        "facial",
        "tongue",
        "cof",
        "nfa",
        "semen",
        "bukkake",
        "fluid",
        "cumifier",
        "mouth",
        "teeth",
    )
    return any(k in n for k in keys)


def discover(tok: str) -> list[dict]:
    seen: set[int] = set()
    rows: list[dict] = []
    for q, base in QUERIES:
        params = {
            "query": q,
            "types": "LORA",
            "limit": 25,
            "nsfw": "true",
            "sort": "Most Downloaded",
        }
        if base:
            params["baseModels"] = base
        url = "https://civitai.com/api/v1/models?" + urllib.parse.urlencode(params)
        try:
            data = api(url, tok)
        except Exception as exc:
            safe_print(f"SEARCH_FAIL {q}: {exc}")
            continue
        for m in data.get("items") or []:
            mid = m.get("id")
            if mid in seen:
                continue
            name = m.get("name") or ""
            if not relevant(name):
                continue
            seen.add(mid)
            v = (m.get("modelVersions") or [{}])[0]
            row = {
                "id": mid,
                "name": name,
                "base": v.get("baseModel"),
                "version_id": v.get("id"),
                "downloads": (v.get("stats") or {}).get("downloadCount") or 0,
                "file": ((v.get("files") or [{}])[0]).get("name"),
                "sizeKB": ((v.get("files") or [{}])[0]).get("sizeKB"),
                "query": q,
            }
            rows.append(row)
            safe_print(
                f"HIT id={mid} dl={row['downloads']} base={row['base']} | {name}"
            )
    rows.sort(key=lambda r: r["downloads"], reverse=True)
    return rows


def pick_flux_version(model: dict, prefer: int | None) -> dict | None:
    vers = model.get("modelVersions") or []
    if prefer:
        for v in vers:
            if v.get("id") == prefer:
                return v
    for v in vers:
        b = (v.get("baseModel") or "").lower()
        if "flux" in b:
            return v
    return vers[0] if vers else None


def download(tok: str, version_id: int, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 5_000_000:
        safe_print(f"HAVE {dest.name} ({dest.stat().st_size // 1_000_000} MB)")
        return True
    url = f"https://civitai.com/api/download/models/{version_id}"
    safe_print(f"GET version={version_id} -> {dest.name}")
    try:
        data = api(url, tok, binary=True)
    except urllib.error.HTTPError as exc:
        safe_print(f"FAIL HTTP {exc.code} {dest.name}")
        return False
    except Exception as exc:
        safe_print(f"FAIL {dest.name}: {exc}")
        return False
    head = data[:200].lstrip().lower()
    if len(data) < 1_000_000 or head.startswith(b"<!doctype") or head.startswith(b"{"):
        safe_print(f"FAIL bad payload size={len(data)} {dest.name}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    safe_print(f"OK {dest.name} ({dest.stat().st_size // 1_000_000} MB)")
    return True


def score(row: dict) -> int:
    name = (row.get("name") or "").lower()
    base = (row.get("base") or "").lower()
    s = 0
    if "flux.1 d" in base:
        s += 60
    elif "kontext" in base:
        s += 50
    elif "flux" in base:
        s += 40
    if any(x in name for x in ("non-face", "non face", "nfa", "identity", "does not change face")):
        s += 50
    if any(x in name for x in ("cum on face", "cumonface", "cumifier", "add cum", "cof")):
        s += 35
    if "tongue" in name:
        s += 25
    if any(x in base for x in ("pony", "sd 1.5", "illustrious", "sdxl", "noobai")):
        s -= 50
    s += min((row.get("downloads") or 0) // 400, 25)
    return s


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tok = token()
    OUT.mkdir(parents=True, exist_ok=True)
    LORAS.mkdir(parents=True, exist_ok=True)

    me = api("https://civitai.com/api/v1/me", tok)
    safe_print(f"AUTH_OK user={me.get('username')}")

    hits = discover(tok)
    ranked = sorted(hits, key=score, reverse=True)
    (OUT / "search_hits.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")
    safe_print("--- ranked fits ---")
    for row in ranked[:15]:
        safe_print(f"score={score(row):3d} dl={row['downloads']} id={row['id']} | {row['name']} | {row['base']}")

    results = []
    for label, mid, prefer, fname, why in KNOWN:
        try:
            model = api(f"https://civitai.com/api/v1/models/{mid}", tok)
        except Exception as exc:
            safe_print(f"META_FAIL {label}: {exc}")
            results.append({"label": label, "ok": False, "error": str(exc)})
            continue
        v = pick_flux_version(model, prefer)
        if not v:
            results.append({"label": label, "ok": False, "error": "no version"})
            continue
        # For pony/sd15 keep preferred even if not flux
        if label in ("pony_tongue", "csenhance") and prefer:
            for cand in model.get("modelVersions") or []:
                if cand.get("id") == prefer:
                    v = cand
                    break
        ok = download(tok, int(v["id"]), LORAS / fname)
        results.append(
            {
                "label": label,
                "ok": ok,
                "file": fname,
                "version": v.get("id"),
                "base": v.get("baseModel"),
                "why": why,
                "model_name": model.get("name"),
            }
        )

    # Download top 3 Flux facial/tongue extras not already known
    known_ids = {k[1] for k in KNOWN}
    extra = 0
    for row in ranked:
        if extra >= 3:
            break
        if row["id"] in known_ids:
            continue
        if score(row) < 60:
            continue
        base = (row.get("base") or "").lower()
        if "flux" not in base:
            continue
        fname = row.get("file") or f"civitai_{row['id']}.safetensors"
        fname = re.sub(r"[^a-zA-Z0-9._-]+", "_", fname)[:120]
        if not fname.endswith(".safetensors"):
            fname += ".safetensors"
        ok = download(tok, int(row["version_id"]), LORAS / fname)
        results.append(
            {
                "label": f"extra_{row['id']}",
                "ok": ok,
                "file": fname,
                "version": row["version_id"],
                "base": row.get("base"),
                "name": row.get("name"),
                "score": score(row),
            }
        )
        if ok:
            extra += 1

    (OUT / "downloaded.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok_n = sum(1 for r in results if r.get("ok"))
    safe_print(f"DONE ok={ok_n}/{len(results)}")
    return 0 if ok_n else 1


if __name__ == "__main__":
    raise SystemExit(main())
