"""A/B best-fit Flux LoRAs on orange saree via identity fluid path + tongue edit."""

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

OUT = REPO / "tmp_test" / "18000_civitai_fit_ab"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"
BATCH = "18000_civitai_fit_ab"

# Best Flux facial fits ranked for identity + fluid
FLUID = (
    ("fit_nfa", "flux_facial_fluid_v1.safetensors", "1.00", 71),
    ("fit_cuminator", "flux_cum_on_face_cuminator.safetensors", "0.85", 72),
    ("fit_cum_facial", "flux_cum_facial.safetensors", "0.90", 73),
)

TONGUE_LORA = "tongue-flux-official.safetensors"
TONGUE_PROMPT = (
    "Photorealistic edit of the exact woman in the start image. "
    "Keep her exact face, identity, hair, orange/peach saree, blouse, pose, "
    "framing, lighting, and background. Do not remake her. "
    "Only change this: mouth open, tongue out, natural teeth visible."
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
    assert await client.health(timeout=12, retries=3)
    await db.connect()
    image_bytes, _, in_name = await db.get_test_input_bytes(IID, owner=owner)
    available = {Path(n).name.lower() for n in (await client.list_lora_filenames() or set())}
    results = []

    for tag, filename, strength, seed in FLUID:
        if filename.lower() not in available:
            row = {"tag": tag, "ok": False, "error": f"missing:{filename}"}
            results.append(row)
            _log(f"SKIP {row}")
            continue
        os.environ["FLUID_COF_FILE"] = filename
        os.environ["FLUID_COF_STRENGTH"] = strength
        os.environ["FLUID_SKIP_RECOLOR"] = "1"
        # Prefer Dedistilled if present for heavy coverage
        os.environ.setdefault(
            "FLUID_FLUX_UNET", "Flux1-Dev-DedistilledMixTuned-v3-fp8.safetensors"
        )
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        _log(f"start {tag} lora={filename} strength={strength}")
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
            (folder / "out.png").write_bytes(result.data)
            rec = await db.store_media(
                data=result.data,
                filename=f"wan_img_{IID[:8]}_{tag}.png",
                content_type=result.content_type or "image/png",
                kind="img",
                prompt=prompt,
                model=str(result.model_label or tag),
                owner=owner,
                meta={
                    "mode": "img",
                    "preset_id": PRESET,
                    "test_run": True,
                    "source_input_id": IID,
                    "batch": BATCH,
                    "variant": tag,
                    "lora": filename,
                    "strength": strength,
                    "seed": seed,
                    "input_name": in_name,
                    "batch_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            row = {"tag": tag, "ok": True, "generation_id": rec["id"], "lora": filename}
            results.append(row)
            _log(f"OK {row}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
            row = {"tag": tag, "ok": False, "error": err}
            results.append(row)
            _log(f"FAIL {row}")

    # Tongue on Flux (official) — keep denoise modest for identity
    if TONGUE_LORA.lower() in available:
        tag = "fit_tongue_flux"
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        _log(f"start {tag} lora={TONGUE_LORA}")
        try:
            data, ctype = await client.generate_image(
                image_bytes,
                TONGUE_PROMPT,
                seed=81,
                steps=28,
                denoise=0.42,
                guidance=2.5,
                flux_unet="flux1-dev-fp8.safetensors",
                edit_graph="img2img",
                denoise_cap=0.55,
                loras=[(TONGUE_LORA, 0.70, 0.70)],
            )
            (folder / "out.png").write_bytes(data)
            rec = await db.store_media(
                data=data,
                filename=f"wan_img_{IID[:8]}_{tag}.png",
                content_type=ctype or "image/png",
                kind="img",
                prompt=TONGUE_PROMPT,
                model=f"flux+{TONGUE_LORA}",
                owner=owner,
                meta={
                    "mode": "img",
                    "preset_id": PRESET,
                    "test_run": True,
                    "source_input_id": IID,
                    "batch": BATCH,
                    "variant": tag,
                    "lora": TONGUE_LORA,
                    "seed": 81,
                    "denoise": 0.42,
                    "input_name": in_name,
                    "batch_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            row = {"tag": tag, "ok": True, "generation_id": rec["id"], "lora": TONGUE_LORA}
            results.append(row)
            _log(f"OK {row}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
            row = {"tag": tag, "ok": False, "error": err}
            results.append(row)
            _log(f"FAIL {row}")

    for k in ("FLUID_COF_FILE", "FLUID_COF_STRENGTH", "FLUID_SKIP_RECOLOR"):
        os.environ.pop(k, None)

    (OUT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r.get("ok"))
    _log(f"done ok={ok}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
