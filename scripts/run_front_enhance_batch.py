"""Run enhance_boobs (keep-outfit + Kontext figure reshape) on all PIN-18000 test inputs.

Writes:
  - Mongo generations for owner=utester, meta.preset_id=enhance_boobs, meta.test_run=True
    (visible in the app under PIN 18000 → Boobs / library)
  - Side-by-side local copies under tmp_test/18000_front_enhance/ for approve browsing

Usage (repo root, Comfy up):
  python scripts/run_front_enhance_batch.py
  python scripts/run_front_enhance_batch.py --limit 5
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

OUT = REPO / "tmp_test" / "18000_front_enhance"
PRESET = "enhance_boobs"


def _prompt() -> str:
    from backend.ai_engine.runtime_overlay import load_json

    for p in load_json("presets.json") or []:
        if isinstance(p, dict) and p.get("id") == PRESET:
            return str(p.get("prompt") or "").strip()
    raise SystemExit(f"preset {PRESET} missing from private/presets.json")


async def _one(
    *,
    idx: int,
    total: int,
    item: dict,
    prompt: str,
    owner: str,
    settings,
) -> dict:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest

    iid = item["id"]
    print(f"[{idx}/{total}] input {iid[:8]} …", flush=True)
    got = await db.get_test_input_bytes(iid, owner=owner)
    if not got:
        return {"id": iid, "ok": False, "error": "missing_bytes"}
    image_bytes, _ctype, in_name = got

    folder = OUT / f"{idx:02d}_{iid[:8]}"
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
        (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
        print(f"  FAIL {err}", flush=True)
        return {"id": iid, "ok": False, "error": err}

    out_name = f"wan_img_{iid[:8]}_front.png"
    (folder / "out.png").write_bytes(result.data)

    record = await db.store_media(
        data=result.data,
        filename=out_name,
        content_type=result.content_type or "image/png",
        kind="img",
        prompt=prompt,
        model=str(result.model_label or "keep_outfit"),
        owner=owner,
        meta={
            "mode": "img",
            "preset_id": PRESET,
            "test_run": True,
            "source_input_id": iid,
            "source_input_name": in_name,
            "workflow_ref": result.workflow_ref,
            "task_type": result.plan.task_type,
            "planner_path": result.plan.planner_path,
            "model_label": result.model_label,
            "engine_warnings": result.warnings,
            "batch": "18000_front_enhance",
            "batch_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        f"  OK gen={record['id'][:8]} task={result.plan.task_type} "
        f"label={result.model_label}",
        flush=True,
    )
    return {
        "id": iid,
        "ok": True,
        "generation_id": record["id"],
        "task_type": result.plan.task_type,
        "model_label": result.model_label,
        "folder": str(folder),
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="Max inputs (0 = all)")
    args = ap.parse_args()

    from backend import db
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    prompt = _prompt()

    client = ComfyClient(settings)
    if not await client.health(timeout=12.0, retries=3):
        print(f"ComfyUI not reachable at {settings.comfyui_url}", file=sys.stderr)
        return 1

    await db.connect()
    items = await db.list_test_inputs(owner=owner)
    # oldest first so batch order is stable for approve
    items = list(reversed(items))
    if args.limit and args.limit > 0:
        items = items[: args.limit]

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"owner={owner} inputs={len(items)} out={OUT}", flush=True)
    print(f"lora map check via overlay on first job", flush=True)

    results: list[dict] = []
    for i, item in enumerate(items, 1):
        results.append(
            await _one(
                idx=i,
                total=len(items),
                item=item,
                prompt=prompt,
                owner=owner,
                settings=settings,
            )
        )

    manifest = {
        "preset_id": PRESET,
        "owner": owner,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "ok": sum(1 for r in results if r.get("ok")),
        "fail": sum(1 for r in results if not r.get("ok")),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(f"DONE ok={manifest['ok']} fail={manifest['fail']} -> {OUT}", flush=True)
    print("App: unlock PIN 18000 → Library / Boobs (enhance_boobs) to approve.", flush=True)
    return 0 if manifest["fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
