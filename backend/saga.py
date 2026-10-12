"""Saga studio (PIN 1000): Summertime Saga style stills and animations.

Fully separate from the Wan / Klein pipelines:
- stills: Anima (Cosmos-Predict2 2B) + DarkCookie Summertime Saga LoRA, optional canon character LoRA,
  optional HD pass (anime upscale + low-denoise resample + face/hand detailers)
- sprite layout: character cut out (isnet-anime) and placed onto a separately generated game background
- motion: full AniSora 3.2 (Wan 2.2 I2V finetune), RIFE frame interpolation, optional 720p, chained second segment
"""

from __future__ import annotations

import copy
import io
import random
import re
from typing import Any, Optional

ANIMA_UNET = "anima-aesthetic-v1.1.safetensors"
ANIMA_CLIP = "qwen_3_06b_base.safetensors"
ANIMA_VAE = "qwen_image_vae.safetensors"
STYLE_LORA = "sts_anima_darkcookie.safetensors"
STYLE_TRIGGER = "D4RKC8K1E"
STYLE_STRENGTH = 1.0
STYLE_STRENGTH_WITH_CHAR = 0.7
CHAR_STRENGTH = 0.8

UPSCALE_MODEL = "RealESRGAN_x4plus_anime_6B.pth"
FACE_DETECTOR = "bbox/face_yolov8m.pt"
HAND_DETECTOR = "bbox/hand_yolov8s.pt"
REMBG_MODEL = "isnet-anime: anime illustrations"
HD_SCALE = 1.5
HIRES_DENOISE = 0.35

# Full AniSora 3.2 experts (D:\ComfyModels, junctioned into diffusion_models\anisora_full on the GPU box).
# They hold the character's face better than base Wan + the AniSora LoRA, and need no speed LoRA at 8 steps.
ANISORA_FULL_HIGH = "anisora_full\\Wan2_2-I2V_AniSoraV3_2_HIGH_14B_fp8_e4m3fn_scaled_KJ.safetensors"
ANISORA_FULL_LOW = "anisora_full\\Wan2_2-I2V_AniSoraV3_2_LOW_14B_fp8_e4m3fn_scaled_KJ.safetensors"
INTERP_MODEL = "rife_v4.26.safetensors"

ASPECTS = {
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "square": (1024, 1024),
}
QUALITIES = ("fast", "hd")
LAYOUTS = ("scene", "sprite")
VIDEO_SECONDS = (3, 5, 8, 10)
VIDEO_FPS = 16
VIDEO_QUALITIES = ("smooth", "hd")
# 720p does not fit the 16 GB card with the 14B experts (~18 min per 5 s), so it stays opt-in and short.
VIDEO_BOX = {"smooth": 832, "hd": 1280}
HD_VIDEO_MAX_SECONDS = 5
SEGMENT_MAX_SECONDS = 5

STYLE_PREFIX = f"masterpiece, best quality, score_7, {STYLE_TRIGGER}, adult, "
NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, "
    "text, watermark, signature, child, loli, shota, underage, young child"
)
SPRITE_SUFFIX = ", solo, full body, standing, simple background, white background, no shadow"
SPRITE_NEGATIVE = NEGATIVE + ", shadow, drop shadow, floor, ground, reflection"
BACKGROUND_PREFIX = "no humans, scenery, detailed background, "
BACKGROUND_NEGATIVE = NEGATIVE + ", 1girl, 1boy, person, people, humans, character"
HAND_PROMPT = "detailed hands, five fingers, well-drawn fingers"
MOTION_PREFIX = "2D cartoon animation in the same art style, "
MOTION_NEGATIVE = (
    "photorealistic, 3d render, blurry, distorted face, extra limbs, flicker, morphing, "
    "static image, text, watermark"
)

# Canon adult cast only (parents, teachers, staff). look = identity tags, outfit = their usual clothes.
CHARACTERS: dict[str, dict[str, str]] = {
    "debbie": {"label": "Debbie", "look": "ssdebbie, mature female, grey eyes, long brown hair, big breasts",
               "outfit": "red low-neck shirt, dark blue pants"},
    "diane": {"label": "Diane", "look": "ssdiane, mature female, grey eyes, short brown hair, purple earrings, big breasts, wide hips",
              "outfit": "denim overalls, white t-shirt"},
    "smith": {"label": "Mrs. Smith", "look": "sssmith, mature female, brown eyes, eyewear, black hair, big breasts, wide hips",
              "outfit": "tight green suit, green skirt, opaque stockings, black heels"},
    "tina": {"label": "Tina Vercetti", "look": "sstina, mature female, green eyes, long red hair, beehive hairdo, big breasts, earrings, pearl necklace",
             "outfit": "charcoal-grey suit, charcoal-grey pants"},
    "helen": {"label": "Helen Potter", "look": "sshelen, mature woman, dark-grey eyes, short brown hair, gold rosary necklace, stud earrings",
              "outfit": "tight short-sleeved purple blouse, white pants"},
    "consuela": {"label": "Consuela", "look": "ssconsuela, mature female, brown eyes, mole under eye, grey-streaked hair, brown wavy hair, ponytail, wide hips",
                 "outfit": "short maid uniform, black shoes"},
    "beth": {"label": "Beth Dinkle", "look": "ssbeth, mature woman, chubby, curvy, red hair, braided hair, freckles, big breasts, purple earrings, hair bow, orange lipstick",
             "outfit": "lilac shirt with rolled up sleeves, burgundy pants, short white apron"},
    "dewitt": {"label": "Melody Dewitt", "look": "ssdewitt, mature woman, curvy, dark-skinned female, big brown eyes, brown cornrow hair, shaved sides, ponytail, big breasts, wide hips",
               "outfit": "purple off-the-shoulder top, light-blue jeans, white sandals"},
    "okita": {"label": "Dr. Okita", "look": "ssokita, adult woman, black eyes, long black hair, hair bun",
              "outfit": "white lab coat, black stockings"},
    "bissette": {"label": "Ms. Bissette", "look": "ssbissette, adult woman, green eyes, brown hair, blue earrings, face freckles",
                 "outfit": "black business suit, white undershirt, black business miniskirt, stockings"},
    "ross": {"label": "Ms. Ross", "look": "ssross, adult woman, green eyes, brown loose afro",
             "outfit": "jungle-green dress, open white shirt with knotted edges and rolled-up sleeves"},
    "angelica": {"label": "Sister Angelica", "look": "ssangelica, adult woman, curvy, pale skin, blonde hair, long braid, huge breasts, wooden rosary",
                 "outfit": "nun's habit, dark-grey nun's tunic, long black skirt"},
    "maria": {"label": "Maria Salieri", "look": "ssmaria, mature woman, voluptuous, brown eyes, mole above lip, brown hair, red flower in hair, green earrings, big breasts",
              "outfit": "short green dress, white apron, black heels"},
    "liu": {"label": "Liu Wang", "look": "ssliu, adult woman, long black hair, hair bun, red hair stick",
            "outfit": "black suit, black skirt, white shirt"},
    "svetlana": {"label": "Svetlana", "look": "sssvetlana, adult woman, brown eyes, light brown hair, loose ponytail, ear piercings, big breasts",
                 "outfit": "blue utility work suit, brown work boots"},
    "bridget": {"label": "Coach Bridget", "look": "ssbridget, adult woman, muscular woman, brown eyes, pink hair, braided rattail",
                "outfit": "grey sweatshirt, grey sweatpants"},
    "irfan": {"label": "Ms. Irfan", "look": "ssirfan, adult woman, brown skin, brown eyes, black eyewear, maroon hijab",
              "outfit": "purple long-sleeved sweater, black pants"},
    "titomi": {"label": "Titomi", "look": "sstitomi, mature japanese woman, curvy, huge breasts, shoulder-length straight dark brown hair, side bangs, dark brown eyes, round face",
               "outfit": "casual blouse, long skirt"},
    "josephine": {"label": "Josephine Sato", "look": "ssjosephine, adult woman, brown eyes, short dark brown bob haircut, side bangs",
                  "outfit": "blue work jacket, white undershirt, blue skirt, black shoes"},
}

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
    _check_adult(text)
    return text


def _check_adult(text: str) -> None:
    if _MINOR_TERMS.search(text):
        raise SagaError("Saga studio only makes adult characters (18+). Rephrase the prompt.")
    for m in _AGE.finditer(text):
        if int(m.group(1)) < 18:
            raise SagaError("Saga studio only makes adult characters (18+). Rephrase the prompt.")


def char_lora(key: str) -> str:
    return f"sts_char_{key}.safetensors"


def _pick(raw: Optional[str], allowed, default: str, what: str) -> str:
    key = (raw or "").strip().lower() or default
    if key not in allowed:
        raise SagaError(f"Unknown {what} '{raw}'")
    return key


def parse_aspect(raw: Optional[str]) -> str:
    return _pick(raw, ASPECTS, "landscape", "aspect")


def parse_seconds(raw: Optional[Any]) -> int:
    if raw in (None, ""):
        return 5
    try:
        sec = int(float(raw))
    except (TypeError, ValueError):
        raise SagaError("video_seconds must be a number") from None
    if sec not in VIDEO_SECONDS:
        raise SagaError(f"video_seconds must be one of {VIDEO_SECONDS}")
    return sec


def _flag(raw: Optional[str], default: bool) -> bool:
    text = (raw or "").strip().lower()
    if not text:
        return default
    return text not in ("0", "false", "off", "no")


def parse_still_options(
    *,
    aspect: Optional[str] = None,
    character: Optional[str] = None,
    outfit: Optional[str] = None,
    quality: Optional[str] = None,
    layout: Optional[str] = None,
    background: Optional[str] = None,
) -> dict[str, Any]:
    char = (character or "").strip().lower() or None
    if char and char not in CHARACTERS:
        raise SagaError(f"Unknown character '{character}'")
    lay = _pick(layout, LAYOUTS, "scene", "layout")
    bg = (background or "").strip() or None
    if lay == "sprite":
        if not bg:
            raise SagaError("Sprite mode needs a background (where she stands)")
        _check_adult(bg)
    return {
        "aspect": parse_aspect(aspect),
        "character": char,
        "outfit": _flag(outfit, True),
        "quality": _pick(quality, QUALITIES, "hd", "quality"),
        "layout": lay,
        "background": bg if lay == "sprite" else None,
    }


def parse_motion_options(*, seconds: Optional[Any] = None, quality: Optional[str] = None) -> dict[str, Any]:
    sec = parse_seconds(seconds)
    q = _pick(quality, VIDEO_QUALITIES, "smooth", "video quality")
    if q == "hd" and sec > HD_VIDEO_MAX_SECONDS:
        raise SagaError(f"720p clips can be at most {HD_VIDEO_MAX_SECONDS} seconds")
    return {"seconds": sec, "quality": q}


def public_config(installed: Optional[set[str]]) -> dict[str, Any]:
    """installed=None (GPU box unreachable) lists every character."""
    chars = [
        {"id": key, "label": c["label"], "outfit": c["outfit"]}
        for key, c in CHARACTERS.items()
        if installed is None or char_lora(key) in installed
    ]
    return {
        "characters": chars,
        "aspects": list(ASPECTS),
        "qualities": list(QUALITIES),
        "layouts": list(LAYOUTS),
        "video_seconds": list(VIDEO_SECONDS),
        "video_qualities": list(VIDEO_QUALITIES),
        "hd_video_max_seconds": HD_VIDEO_MAX_SECONDS,
    }


def character_prompt(prompt: str, character: Optional[str], outfit: bool) -> str:
    if not character:
        return prompt
    c = CHARACTERS[character]
    parts = [c["look"]] + ([c["outfit"]] if outfit else []) + [prompt]
    return ", ".join(parts)


def _snap(v: float, m: int = 16) -> int:
    return max(m, int(round(v / m)) * m)


def build_still_graph(
    prompt: str,
    *,
    seed: int,
    width: int,
    height: int,
    character: Optional[str] = None,
    hd: bool = False,
    negative: str = NEGATIVE,
    cutout: bool = False,
    detail: bool = True,
) -> dict[str, Any]:
    """Anima still. Node "9" is the base decode; hd adds 11-19 (upscale, resample, face + hand detailers)."""
    loras = [(STYLE_LORA, STYLE_STRENGTH_WITH_CHAR if character else STYLE_STRENGTH)]
    if character:
        loras.append((char_lora(character), CHAR_STRENGTH))
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": ANIMA_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": ANIMA_CLIP, "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": ANIMA_VAE}},
    }
    model, clip = ["1", 0], ["2", 0]
    for i, (name, strength) in enumerate(loras):
        nid = str(40 + i)
        g[nid] = {"class_type": "LoraLoader", "inputs": {"model": model, "clip": clip, "lora_name": name,
                                                         "strength_model": strength, "strength_clip": strength}}
        model, clip = [nid, 0], [nid, 1]
    sampler = {"sampler_name": "er_sde", "scheduler": "simple", "cfg": 4.0}
    g.update({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": STYLE_PREFIX + prompt, "clip": clip}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": clip}},
        "7": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["5", 0], "negative": ["6", 0],
                                                   "latent_image": ["7", 0], "seed": seed, "steps": 30,
                                                   "denoise": 1.0, **sampler}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
    })
    image = ["9", 0]
    if hd:
        g.update({
            "11": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
            "12": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["11", 0], "image": image}},
            "13": {"class_type": "ImageScale", "inputs": {"image": ["12", 0], "upscale_method": "lanczos",
                                                          "width": _snap(width * HD_SCALE), "height": _snap(height * HD_SCALE),
                                                          "crop": "disabled"}},
            "14": {"class_type": "VAEEncode", "inputs": {"pixels": ["13", 0], "vae": ["3", 0]}},
            "15": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["5", 0], "negative": ["6", 0],
                                                        "latent_image": ["14", 0], "seed": seed + 1, "steps": 20,
                                                        "denoise": HIRES_DENOISE, **sampler}},
            "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
        })
        image = ["16", 0]
        if detail:
            g["17"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DETECTOR}}
            g["18"] = _detailer(image, model, clip, ["5", 0], ["17", 0], seed + 2, denoise=0.4)
            g["19"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": HAND_DETECTOR}}
            g["20"] = {"class_type": "CLIPTextEncode", "inputs": {"text": STYLE_PREFIX + HAND_PROMPT, "clip": clip}}
            g["21"] = _detailer(["18", 0], model, clip, ["20", 0], ["19", 0], seed + 3, denoise=0.35)
            image = ["21", 0]
    if cutout:
        g.update({
            "30": {"class_type": "RemBGSession+", "inputs": {"model": REMBG_MODEL, "providers": "CPU"}},
            "31": {"class_type": "ImageRemoveBackground+", "inputs": {"rembg_session": ["30", 0], "image": image}},
            "32": {"class_type": "InvertMask", "inputs": {"mask": ["31", 1]}},
            "33": {"class_type": "JoinImageWithAlpha", "inputs": {"image": image, "alpha": ["32", 0]}},
        })
        image = ["33", 0]
    g["10"] = {"class_type": "SaveImage", "inputs": {"images": image, "filename_prefix": "wan_saga"}}
    return g


def _detailer(image, model, clip, positive, detector, seed: int, *, denoise: float) -> dict[str, Any]:
    return {
        "class_type": "FaceDetailer",
        "inputs": {
            "image": image, "model": model, "clip": clip, "vae": ["3", 0],
            "positive": positive, "negative": ["6", 0], "bbox_detector": detector,
            "guide_size": 512, "guide_size_for": True, "max_size": 1024, "seed": seed, "steps": 20,
            "cfg": 4.0, "sampler_name": "er_sde", "scheduler": "simple", "denoise": denoise,
            "feather": 5, "noise_mask": True, "force_inpaint": True, "bbox_threshold": 0.5,
            "bbox_dilation": 10, "bbox_crop_factor": 3.0, "sam_detection_hint": "center-1",
            "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False", "drop_size": 10,
            "wildcard": "", "cycle": 1,
        },
    }


def compose_sprite(background: bytes, sprite: bytes, *, height_frac: float = 0.92) -> bytes:
    """Place a transparent character cut-out bottom-centre on the background, game-sprite style."""
    from PIL import Image

    bg = Image.open(io.BytesIO(background)).convert("RGB")
    ch = Image.open(io.BytesIO(sprite)).convert("RGBA")
    box = ch.getchannel("A").point(lambda a: 255 if a > 16 else 0).getbbox()
    if box:
        ch = ch.crop(box)
    scale = min(bg.height * height_frac / ch.height, bg.width * 0.9 / ch.width)
    ch = ch.resize((max(1, round(ch.width * scale)), max(1, round(ch.height * scale))), Image.Resampling.LANCZOS)
    bg.paste(ch, ((bg.width - ch.width) // 2, bg.height - ch.height), ch)
    buf = io.BytesIO()
    bg.save(buf, format="PNG")
    return buf.getvalue()


async def run_still(
    client: Any, prompt: str, *, seed: Optional[int], aspect: Optional[str] = None,
    options: Optional[dict[str, Any]] = None,
) -> tuple[bytes, str, dict]:
    text = check_prompt(prompt)
    opts = dict(options or {})
    opts.setdefault("aspect", aspect)
    opts = parse_still_options(
        aspect=opts.get("aspect"), character=opts.get("character"),
        outfit=None if opts.get("outfit", True) else "0", quality=opts.get("quality"),
        layout=opts.get("layout"), background=opts.get("background"),
    )
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    hd = opts["quality"] == "hd"
    width, height = ASPECTS[opts["aspect"]]
    subject = character_prompt(text, opts["character"], opts["outfit"])

    if opts["layout"] == "sprite":
        sw, sh = ASPECTS["portrait"]
        sprite, _ = await client._run_and_fetch(
            build_still_graph(subject + SPRITE_SUFFIX, seed=seed, width=sw, height=sh,
                              character=opts["character"], hd=hd, negative=SPRITE_NEGATIVE, cutout=True),
            prefer=("images",), input_name=None,
        )
        back, _ = await client._run_and_fetch(
            build_still_graph(BACKGROUND_PREFIX + opts["background"], seed=seed + 7, width=width, height=height,
                              hd=hd, negative=BACKGROUND_NEGATIVE, detail=False),
            prefer=("images",), input_name=None,
        )
        data, ctype = compose_sprite(back, sprite), "image/png"
    else:
        data, ctype = await client._run_and_fetch(
            build_still_graph(subject, seed=seed, width=width, height=height, character=opts["character"], hd=hd),
            prefer=("images",), input_name=None,
        )
    if hd:
        width, height = _snap(width * HD_SCALE), _snap(height * HD_SCALE)
    label = f"{ANIMA_UNET}+{STYLE_LORA}" + (f"+{char_lora(opts['character'])}" if opts["character"] else "")
    return data, ctype, {
        "model_label": label,
        "seed_used": seed,
        "saga": {"kind": "still", "aspect": opts["aspect"], "width": width, "height": height,
                 "character": opts["character"], "quality": opts["quality"], "layout": opts["layout"]},
    }


def segment_plan(seconds: int) -> tuple[int, int]:
    """(segments, frames per segment) — clips over 5 s are two chained segments sharing one frame."""
    segments = 1 if seconds <= SEGMENT_MAX_SECONDS else 2
    frames = round(seconds * VIDEO_FPS / segments)
    return segments, (frames // 4) * 4 + 1


def extend_video_graph(g: dict[str, Any], *, segments: int, seg_len: int, seed: int, interpolate: int) -> dict[str, Any]:
    """Add a chained second segment (starts on segment 1's last frame) and RIFE interpolation to an i2v graph."""
    frames: list = ["13", 0]
    if segments > 1:
        g["101"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["13", 0], "batch_index": seg_len - 1, "length": 1}}
        w2v = copy.deepcopy(g["10"])
        w2v["inputs"]["start_image"] = ["101", 0]
        g["110"] = w2v
        k_high = copy.deepcopy(g["11"])
        k_high["inputs"].update({"positive": ["110", 0], "negative": ["110", 1], "latent_image": ["110", 2],
                                 "noise_seed": seed + 1})
        g["111"] = k_high
        k_low = copy.deepcopy(g["12"])
        k_low["inputs"].update({"positive": ["110", 0], "negative": ["110", 1], "latent_image": ["111", 0]})
        g["112"] = k_low
        g["113"] = {"class_type": "VAEDecode", "inputs": {"samples": ["112", 0], "vae": ["4", 0]}}
        g["114"] = {"class_type": "ImageFromBatch", "inputs": {"image": ["113", 0], "batch_index": 1, "length": seg_len - 1}}
        g["115"] = {"class_type": "ImageBatch", "inputs": {"image1": ["13", 0], "image2": ["114", 0]}}
        frames = ["115", 0]
    if interpolate > 1:
        g["120"] = {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": INTERP_MODEL}}
        g["121"] = {"class_type": "FrameInterpolate", "inputs": {"interp_model": ["120", 0], "images": frames,
                                                                 "multiplier": interpolate}}
        frames = ["121", 0]
    g["14"]["inputs"]["images"] = frames
    g["14"]["inputs"]["fps"] = VIDEO_FPS * interpolate
    return g


async def run_motion(
    client: Any, image_bytes: bytes, prompt: str, *, seed: Optional[int], seconds: Optional[Any] = None,
    options: Optional[dict[str, Any]] = None,
) -> tuple[bytes, str, dict]:
    text = check_prompt(prompt)
    opts = dict(options or {})
    opts.setdefault("seconds", seconds)
    opts = parse_motion_options(seconds=opts.get("seconds"), quality=opts.get("quality"))
    sec, quality = opts["seconds"], opts["quality"]
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    segments, seg_len = segment_plan(sec)
    interpolate = 2
    box = VIDEO_BOX[quality]
    data, ctype = await client.generate_video(
        image_bytes,
        MOTION_PREFIX + text,
        negative=MOTION_NEGATIVE,
        seed=seed,
        width=box,
        height=box,
        length=seg_len,
        steps=8,
        cfg=1.0,
        fps=VIDEO_FPS,
        unet_high=ANISORA_FULL_HIGH,
        unet_low=ANISORA_FULL_LOW,
        loras_high=[],
        loras_low=[],
        high_noise_fraction=0.5,
        graph_hook=lambda g: extend_video_graph(g, segments=segments, seg_len=seg_len, seed=seed,
                                                interpolate=interpolate),
    )
    return data, ctype, {
        "model_label": "anisora3.2_full+rife",
        "seed_used": seed,
        "saga": {"kind": "motion", "seconds": sec, "quality": quality,
                 "fps": VIDEO_FPS * interpolate, "segments": segments},
    }
