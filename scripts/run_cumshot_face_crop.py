"""Face-crop fluid iterate: run LoRA on close-up face (ref framing), paste back."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_cumshot_face_crop"
IID = "6a8630fbbd8d7c30d36ddd6b"
PRESET = "cumshot_clothes"
BATCH = "18000_cumshot_face_crop"

ATTEMPTS = (
    ("crop_nfa", "flux_facial_fluid_v1.safetensors", "1.25", 404),
    ("crop_cof6", "COF_v6.safetensors", "1.15", 505),
    ("crop_cumifier", "flux_kontext_fluid_v1.safetensors", "1.10", 606),
)


def _log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _face_crop(image_bytes: bytes) -> tuple[bytes, tuple[int, int, int, int], tuple[int, int]]:
    from backend.ai_engine.post.face_lock import detect_face_box, _portrait_prior

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    box = detect_face_box(img) or _portrait_prior(img.size)
    x, y, w, h = box
    pad_x = int(w * 0.55)
    pad_top = int(h * 0.70)
    pad_bot = int(h * 0.85)
    left = max(0, x - pad_x)
    top = max(0, y - pad_top)
    right = min(img.width, x + w + pad_x)
    bottom = min(img.height, y + h + pad_bot)
    # Square-ish for Flux
    side = max(right - left, bottom - top)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(img.width, left + side)
    bottom = min(img.height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    crop = img.crop((left, top, right, bottom))
    # Upscale small crops so LoRA has room
    if min(crop.size) < 768:
        scale = 768 / min(crop.size)
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)),
            Image.Resampling.LANCZOS,
        )
    buf = BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue(), (left, top, right, bottom), img.size


def _paste_back(full_bytes: bytes, edited_crop: bytes, box: tuple[int, int, int, int]) -> bytes:
    full = Image.open(BytesIO(full_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_crop)).convert("RGB")
    left, top, right, bottom = box
    target = edit.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    out = full.copy()
    out.paste(target, (left, top))
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


async def main() -> int:
    from backend import db
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest
    from backend.ai_engine.runtime_overlay import load_json
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    prompt = (
        "Photorealistic close-up of this exact woman's face. Keep identity identical. "
        "Add heavy realistic semen facial like reference photos: thick glossy opaque "
        "cream-white splatters and sticky drips across forehead, eyelids, nose, cheeks, "
        "lips, and chin with wet specular highlights. Clearly visible gooey clumps. "
        "Do not remake her face. Crisp focus."
    )
    client = ComfyClient(settings)
    assert await client.health(timeout=12, retries=3)
    await db.connect()
    image_bytes, _, in_name = await db.get_test_input_bytes(IID, owner=owner)
    available = {Path(n).name for n in (await client.list_lora_filenames() or set())}
    crop_bytes, box, full_size = _face_crop(image_bytes)
    _log(f"face_box={box} full={full_size}")

    results = []
    for tag, filename, strength, seed in ATTEMPTS:
        if not any(n.lower() == filename.lower() for n in available):
            row = {"tag": tag, "ok": False, "error": f"missing_lora:{filename}"}
            results.append(row)
            _log(f"SKIP {row}")
            continue
        os.environ["FLUID_COF_FILE"] = filename
        os.environ["FLUID_COF_STRENGTH"] = strength
        os.environ["FLUID_SKIP_RECOLOR"] = "1"
        folder = OUT / f"{tag}_{IID[:8]}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "crop_in.png").write_bytes(crop_bytes)
        _log(f"start {tag} file={filename} strength={strength} seed={seed}")
        try:
            result = await ai_engine_run(
                GenerateRequest(
                    mode="img",
                    prompt=prompt,
                    prompt_english=prompt,
                    image_bytes=crop_bytes,
                    seed=seed,
                    profile="quality",
                    channel="stable",
                ),
                settings=settings,
            )
            (folder / "crop_out.png").write_bytes(result.data)
            composed = _paste_back(image_bytes, result.data, box)
            (folder / "out.png").write_bytes(composed)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
            row = {"tag": tag, "ok": False, "error": err}
            results.append(row)
            _log(f"FAIL {row}")
            continue

        label = str(result.model_label or "")
        rec = await db.store_media(
            data=composed,
            filename=f"wan_img_{IID[:8]}_cumshot_{tag}.png",
            content_type="image/png",
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
                "cof_file": filename,
                "cof_strength": float(strength),
                "seed": seed,
                "face_crop_box": list(box),
                "batch_at": datetime.now(timezone.utc).isoformat(),
                "model_label": label,
                "engine_warnings": result.warnings,
                "input_name": in_name,
            },
        )
        row = {"tag": tag, "ok": True, "generation_id": rec["id"], "label": label, "out": str(folder / "out.png")}
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
