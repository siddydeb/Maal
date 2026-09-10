"""Smoke cumshot_clothes (Flux + cof) on orange saree test input.

Always stores the result in Mongo (meta.test_run=True) for the tester PIN library.
"""

from __future__ import annotations

import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_cumshot_img_smoke_saree"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"


def _log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def main() -> int:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest
    from backend.ai_engine.runtime_overlay import load_json
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

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
    cof = "flux_kontext_fluid_v1.safetensors"
    if not any(Path(n).name.lower() == cof.lower() for n in available):
        _log(f"FAIL missing lora {cof}")
        return 1
    _log(f"cof ok ({cof})")

    await db.connect()
    got = await db.get_test_input_bytes(IID, owner=owner)
    if not got:
        _log("FAIL missing bytes")
        return 1
    image_bytes, _ct, in_name = got
    folder = OUT / f"01_{IID[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "start.png").write_bytes(image_bytes)
    _log(f"start gen input={IID} name={in_name} preset={PRESET}")

    try:
        result = await ai_engine_run(
            GenerateRequest(
                mode="img",
                prompt=prompt,
                prompt_english=prompt,
                image_bytes=image_bytes,
                negative=None,
                seed=42,
                profile="quality",
                channel="stable",
            ),
            settings=settings,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        (folder / "error.txt").write_text(err, encoding="utf-8")
        _log(f"FAIL {exc}")
        return 1

    out_png = folder / "out.png"
    out_png.write_bytes(result.data)
    _log(
        f"OK bytes={len(result.data)} task={result.plan.task_type} "
        f"label={result.model_label} warnings={result.warnings}"
    )

    gen = None
    try:
        rec = await db.store_media(
            data=result.data,
            filename=f"wan_img_{IID[:8]}_cumshot_clothes.png",
            content_type=result.content_type or "image/png",
            kind="img",
            prompt=prompt,
            model=str(result.model_label or PRESET),
            owner=owner,
            meta={
                "mode": "img",
                "preset_id": PRESET,
                "test_run": True,
                "source_input_id": IID,
                "source_input_name": in_name,
                "workflow_ref": result.workflow_ref,
                "task_type": result.plan.task_type,
                "planner_path": result.plan.planner_path,
                "model_label": result.model_label,
                "engine_warnings": result.warnings,
                "batch": "18000_cumshot_img_smoke_saree",
                "batch_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        gen = rec["id"]
        _log(f"mongo {gen}")
    except Exception as exc:
        _log(f"mongo_FAIL {exc}")
        return 1

    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "out": str(out_png),
                "gen": gen,
                "task_type": result.plan.task_type,
                "label": result.model_label,
                "warnings": result.warnings,
                "workflow_ref": result.workflow_ref,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
