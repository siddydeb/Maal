"""Run act_bj / act_hj / act_titjob on 3 Mongo test inputs (PIN 18000 / utester).

Writes generations with meta.test_run=True + meta.preset_id so they show in
the app Library under the tester PIN.

Usage (repo root, Comfy up):
  python scripts/run_act_presets_batch.py
  python scripts/run_act_presets_batch.py --limit 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_act_presets_v2"
PRESETS = ("act_bj", "act_hj", "act_titjob")
BATCH_TAG = "18000_act_presets_v2"


def _prompt(preset_id: str) -> str:
    from backend.ai_engine.runtime_overlay import load_json

    for p in load_json("presets.json") or []:
        if isinstance(p, dict) and p.get("id") == preset_id:
            return str(p.get("prompt") or "").strip()
    raise SystemExit(f"preset {preset_id} missing from private/presets.json")


def _pick_relevant(items: list[dict], limit: int) -> list[dict]:
    """Prefer recent portrait-ish inputs: image/* and larger files first."""
    scored: list[tuple[int, dict]] = []
    for it in items:
        ct = str(it.get("content_type") or "").lower()
        if not ct.startswith("image/"):
            continue
        size = int(it.get("size_bytes") or 0)
        # Prefer mid/large photos over tiny thumbs
        score = size
        name = str(it.get("filename") or "").lower()
        if any(k in name for k in ("portrait", "face", "person", "woman", "girl")):
            score += 5_000_000
        scored.append((score, it))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [it for _, it in scored[:limit]]
    if len(picked) < limit:
        # fall back to newest remaining
        seen = {p["id"] for p in picked}
        for it in items:
            if it["id"] in seen:
                continue
            if not str(it.get("content_type") or "").lower().startswith("image/"):
                continue
            picked.append(it)
            if len(picked) >= limit:
                break
    return picked[:limit]


async def _one(
    *,
    idx: int,
    total: int,
    item: dict,
    preset_id: str,
    prompt: str,
    owner: str,
    settings,
) -> dict:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest

    iid = item["id"]
    print(f"[{idx}/{total}] {preset_id} input {iid[:8]} …", flush=True)
    got = await db.get_test_input_bytes(iid, owner=owner)
    if not got:
        return {"id": iid, "preset_id": preset_id, "ok": False, "error": "missing_bytes"}
    image_bytes, _ctype, in_name = got

    folder = OUT / f"{idx:02d}_{preset_id}_{iid[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "start.png").write_bytes(image_bytes)

    try:
        result = await ai_engine_run(
            GenerateRequest(
                mode="img",
                prompt=prompt,
                prompt_english=prompt,
                image_bytes=image_bytes,
                negative=None,
                seed=None,
                profile="quality",
                channel="stable",
            ),
            settings=settings,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        (folder / "error.txt").write_text(
            err + "\n" + traceback.format_exc(), encoding="utf-8"
        )
        print(f"  FAIL {err}", flush=True)
        return {"id": iid, "preset_id": preset_id, "ok": False, "error": err}

    out_name = f"wan_img_{iid[:8]}_{preset_id}.png"
    (folder / "out.png").write_bytes(result.data)

    record = await db.store_media(
        data=result.data,
        filename=out_name,
        content_type=result.content_type or "image/png",
        kind="img",
        prompt=prompt,
        model=str(result.model_label or preset_id),
        owner=owner,
        meta={
            "mode": "img",
            "preset_id": preset_id,
            "test_run": True,
            "act_edit": {
                "act_bj": "oral",
                "act_hj": "handjob",
                "act_titjob": "titjob",
            }.get(preset_id),
            "source_input_id": iid,
            "source_input_name": in_name,
            "workflow_ref": result.workflow_ref,
            "task_type": result.plan.task_type,
            "planner_path": result.plan.planner_path,
            "model_label": result.model_label,
            "engine_warnings": result.warnings,
            "batch": BATCH_TAG,
            "batch_at": datetime.now(timezone.utc).isoformat(),
            "recipe": "v2_i2i_undress_strong",
        },
    )
    print(
        f"  OK gen={record['id'][:8]} task={result.plan.task_type} "
        f"label={result.model_label}",
        flush=True,
    )
    return {
        "id": iid,
        "preset_id": preset_id,
        "ok": True,
        "generation_id": record["id"],
        "task_type": result.plan.task_type,
        "model_label": result.model_label,
        "folder": str(folder),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="How many inputs (default 3)")
    ap.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated test_input ids (skip auto-pick)",
    )
    args = ap.parse_args()

    from backend import db
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    prompts = {pid: _prompt(pid) for pid in PRESETS}

    client = ComfyClient(settings)
    if not await client.health(timeout=12.0, retries=3):
        print(f"ComfyUI not reachable at {settings.comfyui_url}", file=sys.stderr)
        return 1

    await db.connect()
    all_items = await db.list_test_inputs(owner=owner, limit=100)
    id_filter = [x.strip() for x in (args.ids or "").split(",") if x.strip()]
    if id_filter:
        by_id = {it["id"]: it for it in all_items}
        items = []
        for iid in id_filter:
            if iid in by_id:
                items.append(by_id[iid])
            else:
                # allow prefix match
                hit = next((it for it in all_items if it["id"].startswith(iid)), None)
                if hit:
                    items.append(hit)
        if not items:
            print("No matching --ids in test_inputs", file=sys.stderr)
            return 1
    else:
        items = _pick_relevant(all_items, max(1, int(args.limit)))

    OUT.mkdir(parents=True, exist_ok=True)
    print(
        f"owner={owner} inputs={len(items)} presets={list(PRESETS)} out={OUT}",
        flush=True,
    )
    for it in items:
        print(
            f"  input {it['id'][:8]} name={it.get('filename')} "
            f"size={it.get('size_bytes')}",
            flush=True,
        )

    results: list[dict] = []
    n = 0
    total = len(items) * len(PRESETS)
    for item in items:
        for preset_id in PRESETS:
            n += 1
            results.append(
                await _one(
                    idx=n,
                    total=total,
                    item=item,
                    preset_id=preset_id,
                    prompt=prompts[preset_id],
                    owner=owner,
                    settings=settings,
                )
            )

    manifest = {
        "presets": list(PRESETS),
        "owner": owner,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [it["id"] for it in items],
        "results": results,
        "ok": sum(1 for r in results if r.get("ok")),
        "fail": sum(1 for r in results if not r.get("ok")),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"DONE ok={manifest['ok']} fail={manifest['fail']} -> {OUT}", flush=True)
    print(
        "App: unlock PIN 18000 -> Library (filter BJ / HJ / Titjob presets).",
        flush=True,
    )
    return 0 if manifest["fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
