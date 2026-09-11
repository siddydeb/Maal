"""Smoke linked tongue + cumshot-enhancer LoRAs on orange saree (3 each).

Note:
- Linked tongue model is Pony (Civitai 721066) — Civitai download auth-walled here.
  Using Flux.1 D sister LoRA from HF Muapi (same concept family) on Flux Dev.
- Cumshot Enhancer (248823) is SD 1.5 anime — run on Realistic Vision V5.1 img2img.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import traceback
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

OUT = REPO / "tmp_test" / "18000_pony_sd15_lora_smoke"
IID = "6a8630fbbd8d7c30d36ddd6b"
BATCH = "18000_pony_sd15_lora_smoke"

TONGUE_LORA = "tongue-flux-v2.1.safetensors"
CS_LORA = "csenhance_v1.safetensors"
SD15_CKPT = "realisticVisionV51_v51VAE.safetensors"

TONGUE_PROMPT = (
    "Photorealistic edit of the exact woman in the start image. "
    "Keep her exact face, identity, hair, orange/peach saree, blouse, pose, "
    "framing, lighting, and background. Do not remake her. "
    "Only change this: mouth open, tongue out, natural teeth visible, "
    "close-up friendly expression, sharp mouth detail. Crisp focus."
)

CS_PROMPT = (
    "photorealistic photo of the same woman, orange peach saree, "
    "1girl, cum on body, cum on face, excessive cum, "
    "keep identity and clothes, detailed skin, sharp photo"
)
CS_NEG = (
    "cartoon, anime, deformed, blurry, lowres, bad anatomy, "
    "extra limbs, watermark, text"
)


def _log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _prep(image_bytes: bytes, max_side: int) -> tuple[bytes, int, int]:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    scale = min(max_side / max(w, h), 1.0)
    nw, nh = int(w * scale) // 8 * 8, int(h * scale) // 8 * 8
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), nw, nh


async def _upload(client: httpx.AsyncClient, base: str, data: bytes, name: str) -> str:
    files = {"image": (name, data, "image/png")}
    r = await client.post(f"{base}/upload/image", files=files, data={"overwrite": "true"})
    r.raise_for_status()
    return (r.json().get("name") or name)


async def _run_graph(client: httpx.AsyncClient, base: str, graph: dict) -> bytes:
    payload = {"prompt": graph, "client_id": str(uuid.uuid4())}
    r = await client.post(f"{base}/prompt", json=payload)
    r.raise_for_status()
    prompt_id = r.json()["prompt_id"]
    for _ in range(600):
        await asyncio.sleep(1.0)
        h = await client.get(f"{base}/history/{prompt_id}")
        h.raise_for_status()
        hist = h.json().get(prompt_id)
        if not hist:
            continue
        outputs = hist.get("outputs") or {}
        for node in outputs.values():
            for img in node.get("images") or []:
                fn = img.get("filename")
                sub = img.get("subfolder") or ""
                t = img.get("type") or "output"
                q = {"filename": fn, "subfolder": sub, "type": t}
                img_r = await client.get(f"{base}/view", params=q)
                img_r.raise_for_status()
                return img_r.content
        if hist.get("status", {}).get("completed") or hist.get("status", {}).get("status_str") == "success":
            break
    raise RuntimeError(f"no output for {prompt_id}")


def _sd15_i2i_graph(
    *,
    image_name: str,
    positive: str,
    negative: str,
    ckpt: str,
    lora: str,
    lora_w: float,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    width: int,
    height: int,
) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": lora,
                "strength_model": lora_w,
                "strength_clip": lora_w,
            },
        },
        "3": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["2", 1]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["2", 1]},
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["3", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": denoise,
                "model": ["2", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "wan_sd15"},
        },
    }


async def main() -> int:
    from backend import db
    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    settings = get_settings()
    owner = (settings.wan_tester_owner or "utester").strip()
    base = settings.comfyui_url.rstrip("/")
    client = ComfyClient(settings)
    assert await client.health(timeout=12, retries=3), "Comfy down"

    loras = {Path(n).name.lower() for n in (await client.list_lora_filenames() or set())}
    assert TONGUE_LORA.lower() in loras, f"missing {TONGUE_LORA}"
    assert CS_LORA.lower() in loras, f"missing {CS_LORA}"

    await db.connect()
    image_bytes, _, in_name = await db.get_test_input_bytes(IID, owner=owner)
    results = []

    # --- 3x tongue (Flux sister LoRA) ---
    for i, seed in enumerate((11, 22, 33), 1):
        tag = f"tongue_flux_{i}"
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "start.png").write_bytes(image_bytes)
        _log(f"start {tag} lora={TONGUE_LORA} seed={seed}")
        try:
            data, ctype = await client.generate_image(
                image_bytes,
                TONGUE_PROMPT,
                seed=seed,
                steps=28,
                denoise=0.55,
                guidance=2.5,
                flux_unet="flux1-dev-fp8.safetensors",
                edit_graph="img2img",
                denoise_cap=0.70,
                loras=[(TONGUE_LORA, 0.75, 0.75)],
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
                    "preset_id": "cumshot_clothes",
                    "test_run": True,
                    "source_input_id": IID,
                    "batch": BATCH,
                    "variant": tag,
                    "lora": TONGUE_LORA,
                    "requested_url": "https://civitai.red/models/721066/female-tongue-mouth-and-teeth-pony",
                    "note": "Pony download auth-walled; used Flux.1 D sister LoRA (Muapi/tongue-flux)",
                    "seed": seed,
                    "input_name": in_name,
                    "batch_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            row = {"tag": tag, "ok": True, "generation_id": rec["id"], "out": str(folder / "out.png")}
            results.append(row)
            _log(f"OK {row}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
            row = {"tag": tag, "ok": False, "error": err}
            results.append(row)
            _log(f"FAIL {row}")

    # --- 3x cumshot enhancer (SD1.5) ---
    prep, w, h = _prep(image_bytes, 768)
    async with httpx.AsyncClient(timeout=120.0) as http:
        img_name = await _upload(http, base, prep, f"wan_in_{uuid.uuid4().hex[:8]}.png")
        for i, seed in enumerate((41, 52, 63), 1):
            tag = f"csenhance_sd15_{i}"
            folder = OUT / tag
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "start.png").write_bytes(image_bytes)
            _log(f"start {tag} ckpt={SD15_CKPT} lora={CS_LORA} seed={seed}")
            try:
                graph = _sd15_i2i_graph(
                    image_name=img_name,
                    positive=CS_PROMPT,
                    negative=CS_NEG,
                    ckpt=SD15_CKPT,
                    lora=CS_LORA,
                    lora_w=0.9,
                    seed=seed,
                    steps=30,
                    cfg=5.5,
                    denoise=0.55,
                    width=w,
                    height=h,
                )
                data = await _run_graph(http, base, graph)
                (folder / "out.png").write_bytes(data)
                rec = await db.store_media(
                    data=data,
                    filename=f"wan_img_{IID[:8]}_{tag}.png",
                    content_type="image/png",
                    kind="img",
                    prompt=CS_PROMPT,
                    model=f"sd15+{CS_LORA}",
                    owner=owner,
                    meta={
                        "mode": "img",
                        "preset_id": "cumshot_clothes",
                        "test_run": True,
                        "source_input_id": IID,
                        "batch": BATCH,
                        "variant": tag,
                        "lora": CS_LORA,
                        "checkpoint": SD15_CKPT,
                        "requested_url": "https://civitai.red/models/248823/cumshot-enhancer-concept-lora",
                        "seed": seed,
                        "input_name": in_name,
                        "batch_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                row = {"tag": tag, "ok": True, "generation_id": rec["id"], "out": str(folder / "out.png")}
                results.append(row)
                _log(f"OK {row}")
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                (folder / "error.txt").write_text(err + "\n" + traceback.format_exc(), encoding="utf-8")
                row = {"tag": tag, "ok": False, "error": err}
                results.append(row)
                _log(f"FAIL {row}")

    (OUT / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(1 for r in results if r.get("ok"))
    _log(f"done ok={ok}/{len(results)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
