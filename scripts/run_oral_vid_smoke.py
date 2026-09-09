"""Smoke one Oral (BJ) img2vid with K3NK deepthroat stack on a Mongo test input."""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_oral_vid_smoke"
PRESET = "oral"


def _prompt() -> str:
    from backend.ai_engine.runtime_overlay import load_json

    for p in load_json("presets.json") or []:
        if isinstance(p, dict) and p.get("id") == PRESET:
            return str(p.get("prompt") or "").strip()
    raise SystemExit(f"preset {PRESET} missing")


async def main() -> int:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest
    from backend.ai_engine.workflows.video_i2v.lora_stack import resolve_video_lora_stack
    from backend.ai_engine.workflows.video_i2v.motion import extract_motion_hints
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    prompt = _prompt()
    motion = extract_motion_hints(prompt)

    client = ComfyClient(settings)
    if not await client.health(timeout=12.0, retries=3):
        print(f"ComfyUI not reachable at {settings.comfyui_url}", file=sys.stderr)
        return 1

    available = await client.list_lora_filenames() or set()
    stack = resolve_video_lora_stack(
        settings,
        include_optional=True,
        available_names=available,
        trust_remote=True,
        nsfw=True,
        motion_kinds=list(motion.get("motion_kinds") or []),
    )
    print("motion_kinds=", motion.get("motion_kinds"), flush=True)
    print("applied=", stack.applied_ids, flush=True)
    print("high=", stack.high, flush=True)
    print("low=", stack.low, flush=True)
    if "deepthroat_high" not in stack.applied_ids:
        print("WARN: deepthroat_high not in stack", file=sys.stderr)

    await db.connect()
    items = await db.list_test_inputs(owner=owner, limit=1)
    if not items:
        print("no test inputs", file=sys.stderr)
        return 1
    item = items[0]
    iid = item["id"]
    got = await db.get_test_input_bytes(iid, owner=owner)
    if not got:
        print("missing bytes", file=sys.stderr)
        return 1
    image_bytes, _ctype, in_name = got

    folder = OUT / f"01_{iid[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "start.png").write_bytes(image_bytes)
    print(f"input={iid} name={in_name} out={folder}", flush=True)

    try:
        result = await ai_engine_run(
            GenerateRequest(
                mode="vid",
                prompt=prompt,
                prompt_english=prompt,
                image_bytes=image_bytes,
                negative=None,
                seed=42,
                profile="balanced",
                channel="stable",
                video_seconds=3.0,
            ),
            settings=settings,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
        print(f"FAIL {err}", flush=True)
        return 1

    ext = "mp4" if (result.content_type or "").startswith("video") or result.kind == "vid" else "bin"
    out_path = folder / f"out.{ext}"
    out_path.write_bytes(result.data)
    print(
        f"OK bytes={len(result.data)} kind={result.kind} ct={result.content_type} "
        f"task={result.plan.task_type} label={result.model_label}",
        flush=True,
    )

    record = None
    try:
        record = await db.store_media(
            data=result.data,
            filename=f"wan_vid_{iid[:8]}_oral.mp4",
            content_type=result.content_type or "video/mp4",
            kind="vid",
            prompt=prompt,
            model=str(result.model_label or "oral"),
            owner=owner,
            meta={
                "mode": "vid",
                "preset_id": PRESET,
                "test_run": True,
                "source_input_id": iid,
                "source_input_name": in_name,
                "workflow_ref": result.workflow_ref,
                "task_type": result.plan.task_type,
                "model_label": result.model_label,
                "engine_warnings": result.warnings,
                "batch": "18000_oral_vid_smoke",
                "batch_at": datetime.now(timezone.utc).isoformat(),
                "applied_loras": stack.applied_ids,
            },
        )
        print(f"mongo gen={record['id']}", flush=True)
    except Exception as exc:
        print(f"mongo_store_skip {type(exc).__name__}: {exc}", flush=True)

    manifest = {
        "preset_id": PRESET,
        "owner": owner,
        "input_id": iid,
        "applied_loras": stack.applied_ids,
        "high": stack.high,
        "low": stack.low,
        "model_label": result.model_label,
        "generation_id": (record or {}).get("id"),
        "folder": str(folder),
        "out": str(out_path),
        "bytes": len(result.data),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
