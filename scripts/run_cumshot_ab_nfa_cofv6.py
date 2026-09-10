"""A/B face-fluid LoRAs on orange saree — store to Mongo (test_run, unopened badge).

Uses FLUID_COF_FILE / FLUID_COF_STRENGTH env overrides (no file rewrite).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_cumshot_ab_nfa_cofv6"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"

CANDIDATES = (
    ("nfa_v2", "flux_facial_fluid_v1.safetensors", "0.90"),
    ("cof_v6", "COF_v6_rollback.safetensors", "1.0"),
)


def _log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


async def _one(
    *,
    tag: str,
    filename: str,
    strength: str,
    prompt: str,
    image_bytes: bytes,
    in_name: str,
    owner: str,
    settings,
    available: set[str],
) -> dict:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest

    if not any(Path(n).name.lower() == filename.lower() for n in available):
        return {"tag": tag, "ok": False, "error": f"missing_lora:{filename}"}

    os.environ["FLUID_COF_FILE"] = filename
    os.environ["FLUID_COF_STRENGTH"] = strength

    folder = OUT / f"{tag}_{IID[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "start.png").write_bytes(image_bytes)

    try:
        result = await ai_engine_run(
            GenerateRequest(
                mode="img",
                prompt=prompt,
                prompt_english=prompt,
                image_bytes=image_bytes,
                seed=42,
                profile="quality",
                channel="stable",
            ),
            settings=settings,
        )
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
        return {"tag": tag, "ok": False, "error": err}

    out_png = folder / "out.png"
    out_png.write_bytes(result.data)
    label = str(result.model_label or "")
    if filename not in label and "loras:" in label:
        return {
            "tag": tag,
            "ok": False,
            "error": f"wrong_lora_in_label:{label}",
            "label": label,
        }

    rec = await db.store_media(
        data=result.data,
        filename=f"wan_img_{IID[:8]}_cumshot_{tag}.png",
        content_type=result.content_type or "image/png",
        kind="img",
        prompt=prompt,
        model=label or f"{PRESET}|{tag}",
        owner=owner,
        meta={
            "mode": "img",
            "preset_id": PRESET,
            "test_run": True,
            "source_input_id": IID,
            "source_input_name": in_name,
            "workflow_ref": result.workflow_ref,
            "task_type": result.plan.task_type,
            "model_label": result.model_label,
            "engine_warnings": result.warnings,
            "batch": "18000_cumshot_ab_nfa_cofv6_v2",
            "variant": tag,
            "cof_file": filename,
            "cof_strength": float(strength),
            "batch_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return {
        "tag": tag,
        "ok": True,
        "generation_id": rec["id"],
        "label": label,
        "warnings": result.warnings,
        "out": str(out_png),
    }


async def main() -> int:
    from backend import db
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
    await db.connect()
    got = await db.get_test_input_bytes(IID, owner=owner)
    if not got:
        _log("FAIL missing bytes")
        return 1
    image_bytes, _ct, in_name = got

    results = []
    for tag, filename, strength in CANDIDATES:
        _log(f"start {tag} file={filename} strength={strength}")
        row = await _one(
            tag=tag,
            filename=filename,
            strength=strength,
            prompt=prompt,
            image_bytes=image_bytes,
            in_name=in_name,
            owner=owner,
            settings=settings,
            available=available,
        )
        results.append(row)
        _log(f"{'OK' if row.get('ok') else 'FAIL'} {tag} {row}")

    os.environ.pop("FLUID_COF_FILE", None)
    os.environ.pop("FLUID_COF_STRENGTH", None)
    (OUT / "manifest_v2.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
