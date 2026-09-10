"""Smoke Dedistilled Flux + NFA v2 (COF) on saree → Mongo."""

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

OUT = REPO / "tmp_test" / "18000_cumshot_dedistilled"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"
BATCH = "18000_cumshot_dedistilled"
UNET = "Flux1-Dev-DedistilledMixTuned-v3-fp8.safetensors"
LORA = "flux_facial_fluid_v1.safetensors"

# Author-style COF prompt for heavy coverage on Dedistilled
COF_PROMPT = (
    "Photorealistic edit of the exact woman in the start image. "
    "CRITICAL IDENTITY LOCK: keep her exact same face geometry, eyes, eyebrows, nose, "
    "lip shape, jaw, skin pores, expression, age, hair, body, pose, framing, lighting, "
    "and background — do not remake or replace her. Keep the SAME orange/peach saree "
    "and blouse. Do not undress. "
    "COF, she has clear sticky cum with white reflections over her face, dripping cum, "
    "cum on forehead, cum on cheeks, cum on chin, cum on eyes, cum on lips, "
    "cum dripping from chin, white sticky cum on face, white sticky semen on face, "
    "face covered with white sticky cum, face covered with white sticky semen. "
    "cumface woman with lots of white, thick, gooey cum all over and covering her face, "
    "cheeks, hair and forehead. The cum coats her face in a thick layer. "
    "A few drips onto the upper neck / blouse edge. Crisp focus."
)

ATTEMPTS = (
    ("dd_cof_90", "0.90", 42, "6.0"),
    ("dd_cof_100", "1.00", 77, "7.0"),
    ("dd_cof_110", "1.10", 101, "8.0"),
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
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    client = ComfyClient(settings)
    assert await client.health(timeout=12, retries=3), "ComfyUI down"

    unet_path = Path(r"E:\Comfy-Desktop\ComfyUI-Shared\models\diffusion_models") / UNET
    assert unet_path.exists() and unet_path.stat().st_size > 1_000_000_000, f"missing {unet_path}"
    names = {Path(n).name.lower() for n in (await client.list_lora_filenames() or set())}
    assert LORA.lower() in names, f"missing lora {LORA}"

    await db.connect()
    image_bytes, _, in_name = await db.get_test_input_bytes(IID, owner=owner)

    results = []
    for tag, strength, seed, guidance in ATTEMPTS:
        os.environ["FLUID_FLUX_UNET"] = UNET
        os.environ["FLUID_COF_FILE"] = LORA
        os.environ["FLUID_COF_STRENGTH"] = strength
        os.environ["FLUID_GUIDANCE"] = guidance
        os.environ["FLUID_SKIP_RECOLOR"] = "1"
        folder = OUT / f"{tag}_{IID[:8]}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        _log(f"start {tag} unet={UNET} lora={LORA} strength={strength} guidance={guidance} seed={seed}")
        try:
            result = await ai_engine_run(
                GenerateRequest(
                    mode="img",
                    prompt=COF_PROMPT,
                    prompt_english=COF_PROMPT,
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
            prompt=COF_PROMPT,
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
                "flux_unet": UNET,
                "guidance": float(guidance),
                "seed": seed,
                "batch_at": datetime.now(timezone.utc).isoformat(),
                "model_label": label,
                "engine_warnings": result.warnings,
                "input_name": in_name,
                "goal": "dedistilled_nfa_ref_match",
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

    for k in (
        "FLUID_FLUX_UNET",
        "FLUID_COF_FILE",
        "FLUID_COF_STRENGTH",
        "FLUID_GUIDANCE",
        "FLUID_SKIP_RECOLOR",
    ):
        os.environ.pop(k, None)
    (OUT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r.get("ok"))
    _log(f"done ok={ok}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
