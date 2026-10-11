"""Saga studio (PIN 1000): Summertime Saga style stills and animations.

Fully separate from the Wan / Klein pipelines:
- stills: Anima (Cosmos-Predict2 2B) + DarkCookie Summertime Saga LoRA
- motion: Wan 2.2 I2V with the AniSora 3.2 LoRA on the high-noise expert
"""

from __future__ import annotations

import random
import re
from typing import Any, Optional

ANIMA_UNET = "anima-aesthetic-v1.1.safetensors"
ANIMA_CLIP = "qwen_3_06b_base.safetensors"
ANIMA_VAE = "qwen_image_vae.safetensors"
STYLE_LORA = "sts_anima_darkcookie.safetensors"
STYLE_TRIGGER = "D4RKC8K1E"

ANISORA_HIGH = "Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors"
LIGHTX2V_LOW = "Wan2.2_LightX2V_low_n54vv.safetensors"

ASPECTS = {
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "square": (1024, 1024),
}
VIDEO_SECONDS = (3, 5)
VIDEO_FPS = 16

STYLE_PREFIX = f"masterpiece, best quality, score_7, {STYLE_TRIGGER}, adult, "
NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, "
    "text, watermark, signature, child, loli, shota, underage, young child"
)
MOTION_PREFIX = "2D cartoon animation in the same art style, "
MOTION_NEGATIVE = (
    "photorealistic, 3d render, blurry, distorted face, extra limbs, flicker, morphing, "
    "static image, text, watermark"
)

_MINOR_TERMS = re.compile(
    r"\b(child|children|kid|kids|loli|lolita|shota|underage|minor|toddler|infant|baby|"
    r"preteen|pre-teen|teen|teens|teenager|schoolgirl|schoolboy|little girl|little boy|"
    r"young girl|young boy|elementary|middle school|junior high)\b",
    re.IGNORECASE,
)
_AGE = re.compile(r"\b(\d{1,2})\s*(?:-|\s)?(?:years?|yrs?|yo)\b(?:\s*old)?", re.IGNORECASE)


class SagaError(ValueError):
    pass


def check_prompt(prompt: str) -> str:
    """Saga only renders adults; reject anything that reads as a minor."""
    text = (prompt or "").strip()
    if not text:
        raise SagaError("Prompt is required")
    if _MINOR_TERMS.search(text):
        raise SagaError("Saga studio only makes adult characters (18+). Rephrase the prompt.")
    for m in _AGE.finditer(text):
        if int(m.group(1)) < 18:
            raise SagaError("Saga studio only makes adult characters (18+). Rephrase the prompt.")
    return text


def parse_aspect(raw: Optional[str]) -> str:
    key = (raw or "").strip().lower() or "landscape"
    if key not in ASPECTS:
        raise SagaError(f"Unknown aspect '{raw}'")
    return key


def parse_seconds(raw: Optional[Any]) -> int:
    if raw in (None, ""):
        return VIDEO_SECONDS[-1]
    try:
        sec = int(float(raw))
    except (TypeError, ValueError):
        raise SagaError("video_seconds must be a number") from None
    if sec not in VIDEO_SECONDS:
        raise SagaError(f"video_seconds must be one of {VIDEO_SECONDS}")
    return sec


def build_still_graph(prompt: str, *, seed: int, width: int, height: int) -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": ANIMA_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ANIMA_CLIP, "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ANIMA_VAE}},
        "4": {
            "class_type": "LoraLoader",
            "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": STYLE_LORA,
                       "strength_model": 1.0, "strength_clip": 1.0},
        },
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": STYLE_PREFIX + prompt, "clip": ["4", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["4", 1]}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {
            "class_type": "KSampler",
            "inputs": {"model": ["4", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0],
                       "seed": seed, "steps": 30, "cfg": 4.0, "sampler_name": "er_sde",
                       "scheduler": "simple", "denoise": 1.0},
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "wan_saga"}},
    }


async def run_still(client: Any, prompt: str, *, seed: Optional[int], aspect: Optional[str]) -> tuple[bytes, str, dict]:
    text = check_prompt(prompt)
    key = parse_aspect(aspect)
    width, height = ASPECTS[key]
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    data, ctype = await client._run_and_fetch(
        build_still_graph(text, seed=seed, width=width, height=height),
        prefer=("images",),
        input_name=None,
    )
    return data, ctype, {
        "model_label": f"{ANIMA_UNET}+{STYLE_LORA}",
        "seed_used": seed,
        "saga": {"kind": "still", "aspect": key, "width": width, "height": height},
    }


async def run_motion(
    client: Any, image_bytes: bytes, prompt: str, *, seed: Optional[int], seconds: Optional[Any]
) -> tuple[bytes, str, dict]:
    text = check_prompt(prompt)
    sec = parse_seconds(seconds)
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    data, ctype = await client.generate_video(
        image_bytes,
        MOTION_PREFIX + text,
        negative=MOTION_NEGATIVE,
        seed=seed,
        width=832,
        height=832,
        length=sec * VIDEO_FPS + 1,
        steps=8,
        cfg=1.0,
        fps=VIDEO_FPS,
        loras_high=[(ANISORA_HIGH, 1.0)],
        loras_low=[(LIGHTX2V_LOW, 1.0)],
        high_noise_fraction=0.5,
    )
    return data, ctype, {
        "model_label": f"wan2.2_i2v+{ANISORA_HIGH}",
        "seed_used": seed,
        "saga": {"kind": "motion", "seconds": sec},
    }
