"""Detached 832 oral Blink smoke on orange saree — survives Cursor close."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_oral_vid_smoke_saree_blink_832"
LOG = OUT / "run.log"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "oral"


def _log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def main() -> int:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest
    from backend.ai_engine.runtime_overlay import load_json
    from backend.ai_engine.workflows.video_i2v.lora_stack import resolve_video_lora_stack
    from backend.ai_engine.workflows.video_i2v.motion import extract_motion_hints
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    out_mp4 = OUT / f"01_{IID[:8]}" / "out.mp4"
    if out_mp4.is_file() and out_mp4.stat().st_size > 100_000:
        _log(f"already done: {out_mp4} ({out_mp4.stat().st_size} bytes)")
        return 0

    prompt = next(
        str(p.get("prompt") or "").strip()
        for p in (load_json("presets.json") or [])
        if isinstance(p, dict) and p.get("id") == PRESET
    )
    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    client = ComfyClient(settings)
    if not await client.health(timeout=12.0, retries=3):
        _log(f"FAIL comfy down {settings.comfyui_url}")
        return 1

    available = await client.list_lora_filenames() or set()
    motion = extract_motion_hints(prompt)
    stack = resolve_video_lora_stack(
        settings,
        include_optional=True,
        available_names=available,
        trust_remote=True,
        nsfw=True,
        motion_kinds=list(motion.get("motion_kinds") or []),
    )
    _log(f"high={stack.high}")

    await db.connect()
    got = await db.get_test_input_bytes(IID, owner=owner)
    if not got:
        _log("FAIL missing bytes")
        return 1
    image_bytes, _ct, in_name = got
    folder = OUT / f"01_{IID[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "start.png").write_bytes(image_bytes)
    _log(f"start gen input={IID} name={in_name}")

    try:
        result = await ai_engine_run(
            GenerateRequest(
                mode="vid",
                prompt=prompt,
                prompt_english=prompt,
                image_bytes=image_bytes,
                seed=42,
                profile="balanced",
                channel="stable",
                video_seconds=3.0,
            ),
            settings=settings,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        (folder / "error.txt").write_text(err, encoding="utf-8")
        _log(f"FAIL {exc}")
        return 1

    out_mp4.write_bytes(result.data)
    dims = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(out_mp4),
        ],
        text=True,
    ).strip()
    _log(f"OK bytes={len(result.data)} dims={dims} label={result.model_label}")

    gen = None
    try:
        rec = await db.store_media(
            data=result.data,
            filename=f"wan_vid_{IID[:8]}_oral_blink832.mp4",
            content_type=result.content_type or "video/mp4",
            kind="vid",
            prompt=prompt,
            model=str(result.model_label or "oral"),
            owner=owner,
            meta={
                "mode": "vid",
                "preset_id": PRESET,
                "test_run": True,
                "source_input_id": IID,
                "batch": "18000_oral_vid_smoke_saree_blink_832",
                "dims": dims,
                "batch_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        gen = rec["id"]
        _log(f"mongo {gen}")
    except Exception as exc:
        _log(f"mongo_skip {exc}")

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "out": str(out_mp4),
                "dims": dims,
                "gen": gen,
                "label": result.model_label,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
