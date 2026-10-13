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
import logging
import random
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

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
LAYOUTS = ("scene", "sprite", "part")
# Photo restyle: how much of the uploaded photo gets redrawn (img2img denoise).
STRENGTHS = {"light": 0.5, "medium": 0.65, "strong": 0.8}
# Without a description the prompt can't pin down the content, so higher denoise invents new scenes.
BLANK_STRENGTHS = {"light": 0.35, "medium": 0.4, "strong": 0.5}
PHOTO_BOX = 1344
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
    "text, watermark, signature, child, loli, shota, underage, young child, fangs, sharp teeth"
)
SPRITE_SUFFIX = ", solo, full body, standing, simple background, white background, no shadow"
SPRITE_NEGATIVE = NEGATIVE + ", shadow, drop shadow, floor, ground, reflection"
BACKGROUND_PREFIX = "no humans, scenery, detailed background, "
PART_SUFFIX = "simple background"
# What a part layer shows; tells the model how to read a lone cut-out (a bare face crop was drawn as a torso).
PART_TYPES: dict[str, str] = {
    "face": "face focus, close-up, portrait, face",
    "hand": "hand focus, close-up, hands",
    "body": "body focus, torso, skin",
    "hair": "hair focus, close-up, hair",
    "other": "",
}
# Margin kept around a part's visible pixels when it is cropped out for redrawing.
PART_MARGIN = 0.15
CHANGES_MAX_CHARS = 500
BACKGROUND_NEGATIVE = NEGATIVE + ", 1girl, 1boy, person, people, humans, character"
HAND_PROMPT = "detailed hands, five fingers, well-drawn fingers"
MOTION_PREFIX = "2D cartoon animation in the same art style, "
MOTION_NEGATIVE = (
    "photorealistic, 3d render, blurry, distorted face, extra limbs, flicker, morphing, "
    "static image, text, watermark"
)

# Plain English -> Danbooru-style tags, rewritten locally by Qwen3 4B (ComfyUI TextGenerate, ~1 s).
REWRITE_MODEL = "qwen_3_4b.safetensors"
REWRITE_MAX_TAGS = 30
REWRITE_INSTRUCTIONS = """Convert the request into comma-separated Danbooru-style tags for an anime image model.
Rules:
- Output only the tags on one line. No sentences, no explanations.
- Include ONLY what the request states or clearly implies. Never add hair colour, eye colour, clothing, places or other details that were not asked for.
- Turn each requested detail into the closest common Danbooru tag (e.g. "in her 40s" -> mature female; "nurse" -> nurse; "darker lipstick" -> dark lipstick).
- People are always adults. Never use child, teen, loli, school or young-looking tags.
- Use each tag once. At most 25 tags.

Examples:
Request: a woman in her 30s with short red hair reading a book in a park
Tags: 1girl, solo, mature female, short hair, red hair, reading, holding book, park, outdoors
Request: make her eyes green and add freckles
Tags: green eyes, freckles
Request: face
Tags: face focus, close-up, portrait"""

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


def check_prompt(prompt: str, *, allow_empty: bool = False) -> str:
    """Saga only renders adults; reject anything that reads as a minor."""
    text = (prompt or "").strip()
    if not text:
        if allow_empty:
            return ""
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
    photo: bool = False,
    strength: Optional[str] = None,
    part: Optional[str] = None,
    changes: Optional[str] = None,
    rewrite: Optional[str] = None,
) -> dict[str, Any]:
    """photo=True: scene restyles the uploaded photo, sprite uses it as the background, part redraws one layer.
    changes: plain-English edits to the uploaded image. rewrite: turn plain English into model tags (default on)."""
    char = (character or "").strip().lower() or None
    if char and char not in CHARACTERS:
        raise SagaError(f"Unknown character '{character}'")
    lay = _pick(layout, LAYOUTS, "scene", "layout")
    if lay == "part" and not photo:
        raise SagaError("Part mode needs the part image (face, hand, body…)")
    bg = (background or "").strip() or None
    if lay == "sprite":
        if not bg and not photo:
            raise SagaError("Sprite mode needs a background (where she stands) or a background photo")
        if bg:
            _check_adult(bg)
    opts = {
        "aspect": parse_aspect(aspect),
        "character": char,
        "outfit": _flag(outfit, True),
        "quality": _pick(quality, QUALITIES, "hd", "quality"),
        "layout": lay,
        "background": bg if lay == "sprite" else None,
    }
    opts["rewrite"] = _flag(rewrite, True)
    if photo:
        opts["photo"] = True
        opts["strength"] = _pick(strength, STRENGTHS, "light" if lay == "part" else "medium", "strength")
        edit = (changes or "").strip()[:CHANGES_MAX_CHARS] or None
        if edit:
            _check_adult(edit)
        opts["changes"] = edit
    if lay == "part":
        opts["part"] = _pick(part, PART_TYPES, "other", "part type")
    return opts


def prepare_photo(image_bytes: bytes, box: int = PHOTO_BOX) -> tuple[bytes, int, int]:
    """RGB PNG fitted inside box x box, cropped to multiples of 16 for the latent."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    scale = min(box / img.width, box / img.height)
    w, h = max(16, round(img.width * scale)), max(16, round(img.height * scale))
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    w16, h16 = w // 16 * 16, h // 16 * 16
    left, top = (w - w16) // 2, (h - h16) // 2
    img = img.crop((left, top, left + w16, top + h16))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w16, h16


def _bleed(rgba: Any) -> Any:
    """Fill transparent pixels by smearing the nearest opaque colours outward (push-pull pyramid).
    A flat fill leaves a hard silhouette edge that the style model outlines with a dark line."""
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.asarray(rgba, dtype=np.float32) / 255.0
    alpha = arr[..., 3:4]
    levels = [(arr[..., :3] * alpha, alpha)]
    while min(levels[-1][1].shape[:2]) > 2:
        c, a = levels[-1]
        size = (max(1, c.shape[1] // 2), max(1, c.shape[0] // 2))
        levels.append((cv2.resize(c, size, interpolation=cv2.INTER_AREA),
                       cv2.resize(a, size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], 1)))
    c, a = levels[-1]
    fill = c / np.maximum(a, 1e-6) * (a > 0) + 0.5 * (a <= 0)
    for c, a in reversed(levels[:-1]):
        up = cv2.resize(fill, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
        fill = c + up * (1.0 - a)
    return Image.fromarray(np.clip(fill * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def prepare_part(image_bytes: bytes, box: int = PHOTO_BOX) -> tuple[bytes, int, int, dict[str, Any]]:
    """A character part (face, hand…) scaled to fill box and padded (never cropped) to multiples of 16,
    with transparent areas and padding filled by _bleed. The plan lets finish_part restore the exact
    original canvas and transparency."""
    from PIL import Image, ImageOps

    full = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGBA")
    alpha = full.getchannel("A")
    region = (0, 0, full.width, full.height)
    visible = alpha.point(lambda a: 255 if a > 8 else 0).getbbox()
    if visible:
        mx = round((visible[2] - visible[0]) * PART_MARGIN) + 4
        my = round((visible[3] - visible[1]) * PART_MARGIN) + 4
        region = (max(0, visible[0] - mx), max(0, visible[1] - my),
                  min(full.width, visible[2] + mx), min(full.height, visible[3] + my))
    rgba = full.crop(region)
    scale = box / max(rgba.size)
    w, h = max(1, round(rgba.width * scale)), max(1, round(rgba.height * scale))
    w16, h16 = max(16, -(-w // 16) * 16), max(16, -(-h // 16) * 16)
    left, top = (w16 - w) // 2, (h16 - h) // 2
    canvas = Image.new("RGBA", (w16, h16), (0, 0, 0, 0))
    canvas.paste(rgba.resize((w, h), Image.Resampling.LANCZOS), (left, top))
    buf = io.BytesIO()
    _bleed(canvas).save(buf, format="PNG")
    plan = {"size": full.size, "region": region, "canvas": (w16, h16), "inner": (left, top, w, h),
            "alpha": alpha if alpha.getextrema()[0] < 255 else None}
    return buf.getvalue(), w16, h16, plan


def finish_part(data: bytes, plan: dict[str, Any]) -> bytes:
    """Undo prepare_part: drop the padding, shrink back into the cropped region of the original canvas,
    re-apply the original alpha."""
    from PIL import Image

    out = Image.open(io.BytesIO(data)).convert("RGB")
    sx, sy = out.width / plan["canvas"][0], out.height / plan["canvas"][1]
    left, top, w, h = plan["inner"]
    out = out.crop((round(left * sx), round(top * sy), round((left + w) * sx), round((top + h) * sy)))
    x0, y0, x1, y1 = plan["region"]
    out = out.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    if (x0, y0, x1, y1) != (0, 0, *plan["size"]):
        canvas = Image.new("RGB", plan["size"], (0, 0, 0))
        canvas.paste(out, (x0, y0))
        out = canvas
    if plan["alpha"] is not None:
        out.putalpha(plan["alpha"])
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


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
        "part_types": list(PART_TYPES),
        "video_seconds": list(VIDEO_SECONDS),
        "video_qualities": list(VIDEO_QUALITIES),
        "hd_video_max_seconds": HD_VIDEO_MAX_SECONDS,
    }


def character_prompt(prompt: str, character: Optional[str], outfit: bool) -> str:
    if not character:
        return prompt
    c = CHARACTERS[character]
    parts = [c["look"]] + ([c["outfit"]] if outfit else []) + [prompt]
    return ", ".join(p for p in parts if p)


def looks_like_tags(text: str) -> bool:
    """Already written as model tags (e.g. '1girl, long hair, smile'), so leave it alone."""
    pieces = [p.strip() for p in text.split(",") if p.strip()]
    return len(pieces) >= 4 and sum(len(p.split()) for p in pieces) / len(pieces) <= 3


def build_rewrite_graph(text: str) -> dict[str, Any]:
    prompt = f"{REWRITE_INSTRUCTIONS}\n\nRequest: {text}\nTags: /no_think"
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": REWRITE_MODEL, "type": "lumina2", "device": "default"}},
        "2": {"class_type": "TextGenerate", "inputs": {"clip": ["1", 0], "prompt": prompt, "max_length": 160,
                                                         "sampling_mode": "off", "use_default_template": True,
                                                         "thinking": False}},
        "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
    }


def clean_tags(raw: str) -> str:
    """Model output -> one deduplicated tag line (drops the empty think block, 'Tags:' and repeat loops)."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S).strip()
    text = text.splitlines()[0] if text else ""
    text = re.sub(r"^\s*tags\s*:\s*", "", text, flags=re.I)
    seen, tags = set(), []
    for tag in (t.strip(" .\"'") for t in text.split(",")):
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    return ", ".join(tags[:REWRITE_MAX_TAGS])


async def rewrite_prompt(client: Any, text: str) -> str:
    """Plain English -> model tags. Falls back to the user's own words on any failure."""
    text = (text or "").strip()
    if not text or looks_like_tags(text):
        return text
    try:
        tags = clean_tags(await client.run_text(build_rewrite_graph(text)))
        _check_adult(tags)
    except Exception as exc:
        logger.warning("Saga prompt rewrite failed, using the original: %s", exc)
        return text
    return tags or text


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
    source_image: Optional[str] = None,
    denoise: float = 1.0,
) -> dict[str, Any]:
    """Anima still. Node "9" is the base decode; hd adds 11-21 (upscale, resample, face + hand detailers).
    source_image (an uploaded ComfyUI input) replaces the empty latent for img2img at the given denoise."""
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
        "8": {"class_type": "KSampler", "inputs": {"model": model, "positive": ["5", 0], "negative": ["6", 0],
                                                   "latent_image": ["7", 0], "seed": seed, "steps": 30,
                                                   "denoise": denoise if source_image else 1.0, **sampler}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
    })
    if source_image:
        g["50"] = {"class_type": "LoadImage", "inputs": {"image": source_image}}
        g["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["50", 0], "vae": ["3", 0]}}
    else:
        g["7"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}}
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


async def _render(client: Any, graph_for, photo: Optional[bytes]) -> bytes:
    """Run one still graph; graph_for(source_name) builds it. Uploads (and scrubs) the photo when given."""
    name = await client._upload_image(photo) if photo else None
    data, _ = await client._run_and_fetch(graph_for(name), prefer=("images",), input_name=name)
    return data


async def run_still(
    client: Any, prompt: str, *, seed: Optional[int], aspect: Optional[str] = None,
    options: Optional[dict[str, Any]] = None, image_bytes: Optional[bytes] = None,
) -> tuple[bytes, str, dict]:
    """Text-to-image, or with image_bytes: restyle the photo (scene), use it as the background (sprite)
    or redraw one character layer (part). Plain English is rewritten into model tags unless rewrite is off."""
    text = check_prompt(prompt, allow_empty=image_bytes is not None)
    opts = dict(options or {})
    opts.setdefault("aspect", aspect)
    opts = parse_still_options(
        aspect=opts.get("aspect"), character=opts.get("character"),
        outfit=None if opts.get("outfit", True) else "0", quality=opts.get("quality"),
        layout=opts.get("layout"), background=opts.get("background"),
        photo=image_bytes is not None, strength=opts.get("strength"), part=opts.get("part"),
        changes=opts.get("changes"), rewrite=None if opts.get("rewrite", True) else "0",
    )
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    hd = opts["quality"] == "hd"
    width, height = ASPECTS[opts["aspect"]]
    photo, plan = None, None
    if opts["layout"] == "part":
        photo, pw, ph, plan = prepare_part(image_bytes)
    elif image_bytes is not None:
        photo, pw, ph = prepare_photo(image_bytes)

    async def adapt(words: Optional[str]) -> str:
        return await rewrite_prompt(client, words or "") if opts["rewrite"] else (words or "")

    desc = await adapt(text)
    edit = await adapt(opts.get("changes")) if photo else ""
    strength = opts.get("strength")
    if edit and strength == "light":
        strength = "medium"

    def denoise_for(described: bool) -> float:
        return (STRENGTHS if described else BLANK_STRENGTHS)[strength] if photo else 1.0

    def join(*parts: str) -> str:
        return ", ".join(p for p in parts if p)

    if plan:
        part_tags = PART_TYPES[opts["part"]]
        subject = join(part_tags, character_prompt(desc, opts["character"], opts["outfit"]), edit, PART_SUFFIX)
        out = await _render(client, lambda n: build_still_graph(
            subject, seed=seed, width=pw, height=ph, character=opts["character"], hd=hd,
            source_image=n, denoise=denoise_for(bool(part_tags or desc or edit))), photo)
        data, ctype = finish_part(out, plan), "image/png"
    elif opts["layout"] == "sprite":
        sw, sh = ASPECTS["portrait"]
        subject = character_prompt(desc or "standing, smiling, looking at viewer", opts["character"], opts["outfit"])
        background = await adapt(opts["background"])
        sprite = await _render(client, lambda _n: build_still_graph(
            subject + SPRITE_SUFFIX, seed=seed, width=sw, height=sh, character=opts["character"], hd=hd,
            negative=SPRITE_NEGATIVE, cutout=True), None)
        if photo:
            width, height = pw, ph
        back = await _render(client, lambda n: build_still_graph(
            BACKGROUND_PREFIX + background, seed=seed + 7, width=width, height=height,
            hd=hd, negative=BACKGROUND_NEGATIVE, detail=False, source_image=n,
            denoise=denoise_for(bool(background))), photo)
        data, ctype = compose_sprite(back, sprite), "image/png"
        subject = join(subject, f"background: {background}" if background else "")
    else:
        if photo:
            width, height = pw, ph
        subject = character_prompt(join(desc, edit), opts["character"], opts["outfit"])
        data = await _render(client, lambda n: build_still_graph(
            subject, seed=seed, width=width, height=height, character=opts["character"], hd=hd,
            source_image=n, denoise=denoise_for(bool(desc or edit))), photo)
        ctype = "image/png"
    if plan:
        width, height = plan["size"]
    elif hd:
        width, height = _snap(width * HD_SCALE), _snap(height * HD_SCALE)
    label = f"{ANIMA_UNET}+{STYLE_LORA}" + (f"+{char_lora(opts['character'])}" if opts["character"] else "")
    meta = {"kind": "still", "aspect": opts["aspect"], "width": width, "height": height,
            "character": opts["character"], "quality": opts["quality"], "layout": opts["layout"],
            "prompt_used": subject}
    if photo:
        meta.update(kind="photo", aspect="photo", strength=strength, changes=opts.get("changes"))
    if plan:
        meta.update(kind="part", part=opts["part"], transparent=plan["alpha"] is not None)
    return data, ctype, {"model_label": label, "seed_used": seed, "saga": meta}


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
