"""Gens with flux_facial_fluid_v1 (NFA) on saree → Mongo."""

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

OUT = REPO / "tmp_test" / "18000_cumshot_nfa_retry"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"
BATCH = "18000_cumshot_nfa_retry"
LORA = "flux_facial_fluid_v1.safetensors"

# Strength / seed pairs — mid→high so gel shows while identity composite holds face
ATTEMPTS = (
    ("nfa_r70", "0.70", 11),
    ("nfa_r80", "0.80", 22),
    ("nfa_r90", "0.90", 33),
    ("nfa_r100", "1.00", 44),
)


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

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    prompt = next(
        str(p.get("prompt") or "").strip()
        for p in (load_json("presets.json") or [])
        if isinstance(p, dict) and p.get("id") == PRESET
    )
    client = ComfyClient(settings)
    assert await client.health(timeout=12, retries=3), "ComfyUI down"
    names = {Path(n).name.lower() for n in (await client.list_lora_filenames() or set())}
    assert LORA.lower() in names, f"Comfy missing {LORA} (repo path D:\\YtAuto\\contrnt\\{LORA} also missing)"

    await db.connect()
    image_bytes, _, in_name = await db.get_test_input_bytes(IID, owner=owner)

    results = []
    for tag, strength, seed in ATTEMPTS:
        os.environ["FLUID_COF_FILE"] = LORA
        os.environ["FLUID_COF_STRENGTH"] = strength
        os.environ["FLUID_SKIP_RECOLOR"] = "1"
        folder = OUT / f"{tag}_{IID[:8]}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        _log(f"start {tag} file={LORA} strength={strength} seed={seed}")
        try:
            result = await ai_engine_run(
                GenerateRequest(
                    mode="img",
                    prompt=prompt,
                    prompt_english=prompt,
                    image_bytes=image_bytes,
                    seed=seed,
                    profile="quality",
                    channel="stable",
                ),
                settings=settings,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
            row = {"tag": tag, "ok": False, "error": err}
            results.append(row)
            _log(f"FAIL {row}")
            continue

        (folder / "out.png").write_bytes(result.data)
        label = str(result.model_label or "")
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
                "batch": BATCH,
                "variant": tag,
                "cof_file": LORA,
                "cof_strength": float(strength),
                "seed": seed,
                "batch_at": datetime.now(timezone.utc).isoformat(),
                "model_label": label,
                "engine_warnings": result.warnings,
                "input_name": in_name,
                "requested_path": r"D:\YtAuto\contrnt\flux_facial_fluid_v1.safetensors",
                "loaded_from": "ComfyUI loras folder",
            },
        )
        row = {
            "tag": tag,
            "ok": True,
            "generation_id": rec["id"],
            "label": label,
            "warnings": result.warnings,
            "out": str(folder / "out.png"),
        }
        results.append(row)
        _log(f"OK {row}")

    for k in ("FLUID_COF_FILE", "FLUID_COF_STRENGTH", "FLUID_SKIP_RECOLOR"):
        os.environ.pop(k, None)
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r.get("ok"))
    _log(f"done ok={ok}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
