"""Clean identity A/B: fresh facial LoRAs + recommended combo on orange saree."""

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

OUT = REPO / "tmp_test" / "18000_fresh_facial_ab"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"
BATCH = "18000_fresh_facial_ab"
DEDIST = "Flux1-Dev-DedistilledMixTuned-v3-fp8.safetensors"

# Singles through production fluid path (face mask + gel composite)
SINGLES = (
    ("s_char_friendly", "flux_char_friendly_facial.safetensors", "1.00", 201),
    ("s_nfa", "flux_facial_fluid_v1.safetensors", "1.05", 202),
    ("s_massive", "flux_facial_cum_massive.safetensors", "0.95", 203),
    ("s_cuminator", "flux_cum_on_face_cuminator.safetensors", "0.85", 204),
    ("s_not_another", "flux_not_another_facial.safetensors", "1.00", 205),
)

# Combos via multi-LoRA img2img + gel highlight composite
COMBOS = (
    (
        "c_friendly_plus_massive",
        [
            ("flux_char_friendly_facial.safetensors", 1.00),
            ("flux_facial_cum_massive.safetensors", 0.40),
        ],
        211,
        "streaks of white translucent semen, face cum glazed, cum on forehead, cum on nose, "
        "thick sticky white gel on face, keep exact identity and orange saree",
    ),
    (
        "c_nfa_plus_massive",
        [
            ("flux_facial_fluid_v1.safetensors", 1.05),
            ("flux_facial_cum_massive.safetensors", 0.35),
        ],
        212,
        "COF, she has clear sticky cum with white reflections over her face, "
        "dripping cum, cum on cheeks, cum on forehead, FCLHGE, keep exact face and clothes",
    ),
    (
        "c_friendly_plus_cuminator",
        [
            ("flux_char_friendly_facial.safetensors", 0.95),
            ("flux_cum_on_face_cuminator.safetensors", 0.40),
        ],
        213,
        "streaks of white translucent semen, Thick Cum Load, Woman with cum on her face, "
        "keep exact identity, orange saree, background unchanged",
    ),
    (
        "c_friendly_massive_cumhere",
        [
            ("flux_char_friendly_facial.safetensors", 0.95),
            ("flux_facial_cum_massive.safetensors", 0.35),
            ("flux_cumhere_v1.safetensors", 0.45),
        ],
        214,
        "streaks of white translucent semen, face cum glazed, cum on clothes and hair, "
        "keep exact face, orange saree identity locked",
    ),
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
    from backend.ai_engine.post.face_lock import composite_fluid_highlights
    from backend.ai_engine.runtime_overlay import load_json
    from backend.ai_engine.schema import GenerateRequest
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
    assert await client.health(timeout=12, retries=3), "Comfy down"
    await db.connect()
    image_bytes, _, in_name = await db.get_test_input_bytes(IID, owner=owner)
    available = {Path(n).name.lower() for n in (await client.list_lora_filenames() or set())}
    results = []

    # --- singles (production fluid identity path) ---
    for tag, filename, strength, seed in SINGLES:
        if filename.lower() not in available:
            row = {"tag": tag, "ok": False, "error": f"missing:{filename}"}
            results.append(row)
            _log(f"SKIP {row}")
            continue
        os.environ["FLUID_COF_FILE"] = filename
        os.environ["FLUID_COF_STRENGTH"] = strength
        os.environ["FLUID_SKIP_RECOLOR"] = "1"
        os.environ["FLUID_FLUX_UNET"] = DEDIST
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        _log(f"start {tag} lora={filename} str={strength}")
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
                    "unet": DEDIST,
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

    # --- combos (multi-LoRA + gel highlight lock) ---
    base_prompt = (
        "Photorealistic edit of the exact woman in the start image. "
        "Keep her exact face, identity, hair, orange peach saree, blouse, pose, "
        "framing, lighting, and background. Do not remake her. "
    )
    for tag, stack, seed, extra in COMBOS:
        missing = [fn for fn, _ in stack if fn.lower() not in available]
        if missing:
            row = {"tag": tag, "ok": False, "error": f"missing:{missing}"}
            results.append(row)
            _log(f"SKIP {row}")
            continue
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        full_prompt = base_prompt + extra
        loras = [(fn, w, w) for fn, w in stack]
        _log(f"start {tag} stack={[(fn, w) for fn, w in stack]}")
        try:
            data, ctype = await client.generate_image(
                image_bytes,
                full_prompt,
                seed=seed,
                steps=28,
                denoise=0.48,
                guidance=2.5,
                flux_unet=DEDIST,
                edit_graph="img2img",
                denoise_cap=0.58,
                loras=loras,
            )
            # Gel-only paste onto start face — identity lock
            try:
                locked = composite_fluid_highlights(image_bytes, data)
                if locked:
                    data = locked
            except Exception as lock_exc:
                _log(f"warn face_lock {tag}: {lock_exc}")
            (folder / "out.png").write_bytes(data)
            rec = await db.store_media(
                data=data,
                filename=f"wan_img_{IID[:8]}_{tag}.png",
                content_type=ctype or "image/png",
                kind="img",
                prompt=full_prompt,
                model=f"combo|{tag}",
                owner=owner,
                meta={
                    "mode": "img",
                    "preset_id": PRESET,
                    "test_run": True,
                    "source_input_id": IID,
                    "batch": BATCH,
                    "variant": tag,
                    "stack": [{"file": fn, "weight": w} for fn, w in stack],
                    "seed": seed,
                    "denoise": 0.48,
                    "unet": DEDIST,
                    "face_lock": "composite_fluid_highlights",
                    "input_name": in_name,
                    "batch_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            row = {"tag": tag, "ok": True, "generation_id": rec["id"], "stack": stack}
            results.append(row)
            _log(f"OK {row}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
            row = {"tag": tag, "ok": False, "error": err}
            results.append(row)
            _log(f"FAIL {row}")

    for k in ("FLUID_COF_FILE", "FLUID_COF_STRENGTH", "FLUID_SKIP_RECOLOR", "FLUID_FLUX_UNET"):
        os.environ.pop(k, None)

    (OUT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r.get("ok"))
    _log(f"done ok={ok}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
