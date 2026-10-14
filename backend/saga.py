"""Saga studio: Summertime Saga style stills and animations.

Fully separate from the Wan / Klein pipelines:
- stills: Anima (Cosmos-Predict2 2B) + DarkCookie Summertime Saga LoRA, optional canon character LoRA,
  optional HD pass (anime upscale + low-denoise resample + face/hand detailers)
- sprite layout: character cut out (isnet-anime) and placed onto a separately generated game background
- motion: full AniSora 3.2 (Wan 2.2 I2V finetune), RIFE frame interpolation, optional 720p, chained second segment
"""

from __future__ import annotations

import copy
import io
import json
import logging
import random
import re
import zipfile
from typing import Any, Callable, Optional

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
STRENGTHS = {"subtle": 0.3, "light": 0.5, "medium": 0.65, "strong": 0.8}
# Without a description the prompt can't pin down the content, so higher denoise invents new scenes.
BLANK_STRENGTHS = {"subtle": 0.25, "light": 0.35, "medium": 0.4, "strong": 0.5}
DENOISE_RANGE = (0.2, 0.8)
MAX_VARIANTS = 4
PHOTO_BOX = 1344
# Game faces are ~120 px; a 1344 box blows them up 11x and the added texture turns to noise when shrunk back.
PART_BOX = 1024
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
NEGATIVE_MAX_CHARS = 300
# The game's flat cartoon colouring; without these, small parts drift to painted skin and soft mouths.
PART_STYLE = "flat color, cel shading, clean dark brown lineart, simple shading"
PART_NEGATIVE = NEGATIVE + (
    ", realistic, photorealistic, skin texture, pores, detailed iris, painterly, 3d, glossy lips"
)
# Repaint mask at render scale: grown past one latent cell (8 px) so the layer's soft edge is fully redrawn.
MASK_GROW_PX = 17
MASK_FEATHER_PX = 4
# Talking face / animation frame: pixels differing from the main layer by more than this are redrawn.
PAIR_DIFF = 24
# keep_lines: original pixels at or below LINE_DARK luminance are copied back fully, fading out by LINE_LIGHT.
LINE_DARK, LINE_LIGHT = 50, 95
# match_skin: pixels within this distance of the dominant skin tone count as skin. Lightness is weighted
# down (shading tones differ mostly in L) so lips, which differ in redness, are left alone.
SKIN_RADIUS = 10.0
SKIN_FADE = 4.0
SKIN_L_WEIGHT = 0.4
SKIN_MIN_PIXELS = 50
BATCH_MAX_ITEMS = 400
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


def _flag(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw if raw is not None else "").strip().lower()
    if not text:
        return default
    return text not in ("0", "false", "off", "no")


def parse_denoise(raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SagaError("denoise must be a number") from None
    lo, hi = DENOISE_RANGE
    if not lo <= value <= hi:
        raise SagaError(f"denoise must be between {lo} and {hi}")
    return round(value, 3)


def parse_negative(raw: Optional[str]) -> Optional[str]:
    """Extra things to keep out of the picture (added to the built-in negative), as written."""
    text = re.sub(r"\s+", " ", raw or "").strip(" ,")
    if len(text) > NEGATIVE_MAX_CHARS:
        raise SagaError(f"Avoid can be at most {NEGATIVE_MAX_CHARS} characters")
    return text or None


def with_negative(base: str, opts: dict[str, Any]) -> str:
    return _join(base, opts.get("negative") or "")


def parse_count(raw: Any) -> int:
    if raw in (None, ""):
        return 1
    try:
        count = int(str(raw).strip())
    except (TypeError, ValueError):
        raise SagaError("count must be a whole number") from None
    if not 1 <= count <= MAX_VARIANTS:
        raise SagaError(f"count must be between 1 and {MAX_VARIANTS}")
    return count


def parse_still_options(
    *,
    aspect: Optional[str] = None,
    character: Optional[str] = None,
    outfit: Any = None,
    quality: Optional[str] = None,
    layout: Optional[str] = None,
    background: Optional[str] = None,
    photo: bool = False,
    strength: Optional[str] = None,
    part: Optional[str] = None,
    changes: Optional[str] = None,
    rewrite: Any = None,
    count: Any = None,
    denoise: Any = None,
    keep_lines: Any = None,
    match_skin: Any = None,
    keep_size: Any = None,
    negative: Optional[str] = None,
    context: bool = False,
    pair: bool = False,
) -> dict[str, Any]:
    """photo=True: scene restyles the uploaded photo, sprite uses it as the background, part redraws one layer.
    changes: plain-English edits to the uploaded image. rewrite: turn plain English into model tags (default on).
    negative: extra words to keep out (e.g. "stubble, beard"), sent as written.
    count: variants on seeds seed..seed+count-1. denoise: exact img2img amount, overrides strength.
    Part only: context (layers beneath), pair (talking face redrawn from the main result), keep_lines,
    match_skin. keep_size: scene/sprite photos come back at the photo's own size."""
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
    opts["count"] = parse_count(count)
    opts["negative"] = parse_negative(negative)
    if (context or pair) and lay != "part":
        raise SagaError("context and pair images only work in Part mode")
    if photo:
        opts["photo"] = True
        opts["strength"] = _pick(strength, STRENGTHS, "light" if lay == "part" else "medium", "strength")
        opts["denoise"] = parse_denoise(denoise)
        edit = (changes or "").strip()[:CHANGES_MAX_CHARS] or None
        if edit:
            _check_adult(edit)
        opts["changes"] = edit
        if lay != "part":
            opts["keep_size"] = _flag(keep_size, False)
    if lay == "part":
        opts["part"] = _pick(part, PART_TYPES, "other", "part type")
        opts["keep_lines"] = _flag(keep_lines, False)
        opts["match_skin"] = _flag(match_skin, False)
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


def open_layer(data: bytes) -> Any:
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGBA")


def _png(img: Any) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _visible(alpha: Any) -> Any:
    return alpha.point(lambda a: 255 if a > 8 else 0)


def part_region(layers: list[Any]) -> tuple[int, int, int, int]:
    """Union of the layers' visible pixels plus a margin; the whole canvas when nothing is transparent."""
    w, h = layers[0].size
    boxes = [b for b in (_visible(im.getchannel("A")).getbbox() for im in layers) if b]
    if not boxes:
        return (0, 0, w, h)
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    mx, my = round((x1 - x0) * PART_MARGIN) + 4, round((y1 - y0) * PART_MARGIN) + 4
    return (max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my))


def prepare_part(
    layer: Any, *, context: Any = None, repaint: Any = None,
    region: Optional[tuple[int, int, int, int]] = None, box: int = PART_BOX,
) -> tuple[bytes, int, int, dict[str, Any]]:
    """Crop the part's region from the layer laid over its context, scale it to fill box and pad (never crop)
    to multiples of 16; empty areas are filled by _bleed. Returns one PNG with the picture on the left and
    the repaint mask (white = redraw; the layer's own pixels unless repaint is given) on the right.
    The plan lets finish_part put the result back on the original canvas."""
    from PIL import Image, ImageFilter

    region = region or part_region([layer])
    picture = Image.alpha_composite(context, layer) if context is not None else layer
    repaint = repaint if repaint is not None else _visible(layer.getchannel("A"))
    crop, mask_crop = picture.crop(region), repaint.crop(region)
    scale = box / max(crop.size)
    w, h = max(1, round(crop.width * scale)), max(1, round(crop.height * scale))
    w16, h16 = max(16, -(-w // 16) * 16), max(16, -(-h // 16) * 16)
    left, top = (w16 - w) // 2, (h16 - h) // 2
    canvas = Image.new("RGBA", (w16, h16), (0, 0, 0, 0))
    canvas.paste(crop.resize((w, h), Image.Resampling.LANCZOS), (left, top))
    mask = Image.new("L", (w16, h16), 0)
    mask.paste(mask_crop.resize((w, h), Image.Resampling.BILINEAR), (left, top))
    mask = mask.filter(ImageFilter.MaxFilter(MASK_GROW_PX)).filter(ImageFilter.GaussianBlur(MASK_FEATHER_PX))
    sheet = Image.new("RGB", (w16 * 2, h16))
    sheet.paste(_bleed(canvas), (0, 0))
    sheet.paste(mask.convert("RGB"), (w16, 0))
    plan = {"size": layer.size, "region": region, "canvas": (w16, h16), "inner": (left, top, w, h)}
    return _png(sheet), w16, h16, plan


def finish_part(data: bytes, plan: dict[str, Any]) -> Any:
    """Undo prepare_part: drop the padding and shrink back into the region of an RGB canvas the original size."""
    from PIL import Image

    out = Image.open(io.BytesIO(data)).convert("RGB")
    sx, sy = out.width / plan["canvas"][0], out.height / plan["canvas"][1]
    left, top, w, h = plan["inner"]
    out = out.crop((round(left * sx), round(top * sy), round((left + w) * sx), round((top + h) * sy)))
    x0, y0, x1, y1 = plan["region"]
    out = out.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", plan["size"], (0, 0, 0))
    canvas.paste(out, (x0, y0))
    return canvas


def with_alpha(rgb: Any, original: Any) -> Any:
    """Cut a result to the original layer's transparency (opaque layers stay RGB)."""
    alpha = original.getchannel("A")
    if alpha.getextrema()[0] == 255:
        return rgb
    out = rgb.copy()
    out.putalpha(alpha)
    return out


def diff_mask(a: Any, b: Any) -> Any:
    """White where two same-size layers differ (premultiplied colour or alpha), grown by 2 px."""
    import numpy as np
    from PIL import Image, ImageFilter

    x, y = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    px, py = x[..., :3] * x[..., 3:] / 255.0, y[..., :3] * y[..., 3:] / 255.0
    d = np.maximum(np.abs(px - py).max(axis=-1), np.abs(x[..., 3] - y[..., 3]))
    mask = Image.fromarray(((d > PAIR_DIFF) * 255).astype(np.uint8), "L")
    return mask.filter(ImageFilter.MaxFilter(5))


def keep_lines(result: Any, original: Any) -> Any:
    """Copy the original's dark outline pixels back over the result (keeps expression and line weight)."""
    import numpy as np
    from PIL import Image

    o = np.asarray(original, dtype=np.float32)
    r = np.asarray(result.convert("RGB"), dtype=np.float32)
    lum = o[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    w = np.clip((LINE_LIGHT - lum) / (LINE_LIGHT - LINE_DARK), 0.0, 1.0) * (o[..., 3] / 255.0)
    out = r * (1.0 - w[..., None]) + o[..., :3] * w[..., None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGB")


def _lab(rgb: Any) -> Any:
    import cv2
    import numpy as np

    return cv2.cvtColor(np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _skin_dist(lab: Any, median: Any) -> Any:
    import numpy as np

    d = lab - median
    return np.sqrt((d[..., 0] * SKIN_L_WEIGHT) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)


def _skin_tone(lab: Any, rgb: Any, alpha: Any) -> Optional[tuple[Any, Any, Any]]:
    """(median, mean, std) Lab of the dominant skin tone among opaque pixels, or None if too few."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(np.asarray(rgb.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    hue, sat, val = hsv[..., 0] * 360.0 / 255.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    cand = ((hue <= 50) | (hue >= 340)) & (sat >= 0.12) & (sat <= 0.8) & (val >= 0.3)
    cand &= np.asarray(alpha) > 200
    if cand.sum() < SKIN_MIN_PIXELS:
        return None
    med = np.median(lab[cand], axis=0)
    sel = cand & (_skin_dist(lab, med) < SKIN_RADIUS)
    if sel.sum() < SKIN_MIN_PIXELS:
        return None
    return med, lab[sel].mean(axis=0), lab[sel].std(axis=0) + 1e-3


def skin_params(result: Any, layer: Any, reference: Any) -> Optional[dict[str, Any]]:
    """How to move the result's skin tones onto the reference's (context, or the original layer)."""
    res = _skin_tone(_lab(result), result, layer.getchannel("A"))
    ref = _skin_tone(_lab(reference), reference, reference.getchannel("A"))
    if ref is None and reference is not layer:
        ref = _skin_tone(_lab(layer), layer, layer.getchannel("A"))
    if res is None or ref is None:
        return None
    return {"median": res[0], "mean": res[1], "std": res[2], "ref_mean": ref[1], "ref_std": ref[2]}


def match_skin(result: Any, alpha: Any, params: Optional[dict[str, Any]]) -> Any:
    """Shift skin-like result pixels so their Lab mean/spread match the reference skin; other colours stay."""
    import cv2
    import numpy as np
    from PIL import Image

    if not params:
        return result
    lab = _lab(result)
    dist = _skin_dist(lab, params["median"])
    weight = np.clip((SKIN_RADIUS + SKIN_FADE - dist) / (2 * SKIN_FADE), 0.0, 1.0) * (np.asarray(alpha) > 0)
    scale = np.clip(params["ref_std"] / params["std"], 0.7, 1.4)
    moved = (lab - params["mean"]) * scale + params["ref_mean"]
    lab = lab + (moved - lab) * weight[..., None]
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")


def pack_inputs(files: dict[str, bytes], manifest: Optional[dict[str, Any]] = None) -> bytes:
    """Several job inputs (layer, context, pair, or a whole batch) stored as one zip."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        if manifest is not None:
            zf.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            zf.writestr(f"files/{name}", data)
    return buf.getvalue()


def is_bundle(data: Optional[bytes]) -> bool:
    return bool(data) and data[:4] == b"PK\x03\x04"


def unpack_inputs(data: bytes) -> tuple[dict[str, bytes], Optional[dict[str, Any]]]:
    """Inverse of pack_inputs; a plain image comes back as {"image": data}."""
    if not is_bundle(data):
        return {"image": data}, None
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        manifest = json.loads(zf.read("manifest.json")) if "manifest.json" in names else None
        files = {n[len("files/"):]: zf.read(n) for n in names if n.startswith("files/")}
    return files, manifest


def bundle_preview(data: bytes) -> bytes:
    """The main image of a job input (for queue thumbnails)."""
    files, manifest = unpack_inputs(data)
    if manifest and manifest.get("items"):
        return files[manifest["items"][0]["image"]]
    return files.get("image") or next(iter(files.values()))


def safe_name(raw: str) -> str:
    """Upload file name -> safe relative path inside the result zip."""
    name = (raw or "").replace("\\", "/").strip().lstrip("/")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        raise SagaError(f"Bad file name '{raw}'")
    return "/".join(parts)


def _as_list(raw: Any) -> list[str]:
    if raw in (None, ""):
        return []
    return [raw] if isinstance(raw, str) else list(raw)


def check_batch(files: dict[str, bytes], manifest: Any) -> dict[str, Any]:
    """Validate a batch manifest against the uploaded files: names exist, images open, sizes match per item.
    Item: {"image": name, "context"?: name, "pair"?: name or [names], "part"?: type}."""
    from PIL import Image

    items = manifest.get("items") if isinstance(manifest, dict) else None
    if not isinstance(items, list) or not items:
        raise SagaError("manifest needs a non-empty 'items' list")
    if len(items) > BATCH_MAX_ITEMS:
        raise SagaError(f"At most {BATCH_MAX_ITEMS} items per batch")
    sizes: dict[str, tuple[int, int]] = {}

    def size_of(name: str) -> tuple[int, int]:
        if name not in files:
            raise SagaError(f"'{name}' is in the manifest but was not uploaded")
        if name not in sizes:
            try:
                sizes[name] = Image.open(io.BytesIO(files[name])).size
            except Exception:
                raise SagaError(f"'{name}' is not an image") from None
        return sizes[name]

    clean = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("image"):
            raise SagaError(f"Item {i + 1} needs an 'image'")
        entry = {"image": safe_name(item["image"]), "pair": [safe_name(p) for p in _as_list(item.get("pair"))]}
        if item.get("context"):
            entry["context"] = safe_name(item["context"])
        if item.get("part"):
            entry["part"] = _pick(item["part"], PART_TYPES, "other", "part type")
        size = size_of(entry["image"])
        for other in entry["pair"] + ([entry["context"]] if "context" in entry else []):
            if size_of(other) != size:
                raise SagaError(f"'{other}' is {size_of(other)}, but '{entry['image']}' is {size}; they must match")
        clean.append(entry)
    return {"items": clean}


def parse_motion_options(
    *, seconds: Optional[Any] = None, quality: Optional[str] = None, negative: Optional[str] = None,
) -> dict[str, Any]:
    sec = parse_seconds(seconds)
    q = _pick(quality, VIDEO_QUALITIES, "smooth", "video quality")
    if q == "hd" and sec > HD_VIDEO_MAX_SECONDS:
        raise SagaError(f"720p clips can be at most {HD_VIDEO_MAX_SECONDS} seconds")
    return {"seconds": sec, "quality": q, "negative": parse_negative(negative)}


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
        "strengths": STRENGTHS,
        "denoise_range": list(DENOISE_RANGE),
        "max_variants": MAX_VARIANTS,
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
    masked: bool = False,
) -> dict[str, Any]:
    """Anima still. Node "9" is the base decode; hd adds 11-21 (upscale, resample, face + hand detailers).
    source_image (an uploaded ComfyUI input) replaces the empty latent for img2img at the given denoise.
    masked: source_image is prepare_part's sheet (picture | mask, each width x height) and only the
    mask is redrawn, in the base and the HD pass."""
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
    masked = masked and bool(source_image)
    if masked:
        crop = {"width": width, "height": height, "y": 0}
        g.update({
            "50": {"class_type": "LoadImage", "inputs": {"image": source_image}},
            "51": {"class_type": "ImageCrop", "inputs": {"image": ["50", 0], "x": 0, **crop}},
            "52": {"class_type": "ImageCrop", "inputs": {"image": ["50", 0], "x": width, **crop}},
            "53": {"class_type": "ImageToMask", "inputs": {"image": ["52", 0], "channel": "red"}},
            "54": {"class_type": "VAEEncode", "inputs": {"pixels": ["51", 0], "vae": ["3", 0]}},
            "7": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["54", 0], "mask": ["53", 0]}},
        })
    elif source_image:
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
        if masked:
            g["55"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["14", 0], "mask": ["53", 0]}}
            g["15"]["inputs"]["latent_image"] = ["55", 0]
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


def _join(*parts: str) -> str:
    return ", ".join(p for p in parts if p)


def _label(character: Optional[str]) -> str:
    return f"{ANIMA_UNET}+{STYLE_LORA}" + (f"+{char_lora(character)}" if character else "")


def new_seed() -> int:
    return random.randint(0, 2**32 - MAX_VARIANTS - 1)


async def _still_setup(
    client: Any, prompt: str, options: dict[str, Any], *, photo: bool, context: bool = False, pair: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate the options and rewrite the words once (shared by variants and batch items)."""
    text = check_prompt(prompt, allow_empty=photo)
    raw = dict(options)
    opts = parse_still_options(
        aspect=raw.get("aspect"), character=raw.get("character"), outfit=raw.get("outfit"),
        quality=raw.get("quality"), layout=raw.get("layout"), background=raw.get("background"),
        photo=photo, strength=raw.get("strength"), part=raw.get("part"), changes=raw.get("changes"),
        rewrite=raw.get("rewrite"), count=raw.get("count"), denoise=raw.get("denoise"),
        keep_lines=raw.get("keep_lines"), match_skin=raw.get("match_skin"), keep_size=raw.get("keep_size"),
        negative=raw.get("negative"), context=context, pair=pair,
    )

    async def adapt(words: Optional[str]) -> str:
        return await rewrite_prompt(client, words or "") if opts["rewrite"] else (words or "")

    words = {"desc": await adapt(text), "edit": await adapt(opts.get("changes")) if photo else ""}
    if opts["layout"] == "sprite":
        words["background"] = await adapt(opts["background"])
    return opts, words


def _denoise(opts: dict[str, Any], described: bool) -> float:
    if not opts.get("photo"):
        return 1.0
    if opts.get("denoise") is not None:
        return opts["denoise"]
    return (STRENGTHS if described else BLANK_STRENGTHS)[opts["strength"]]


def part_subject(opts: dict[str, Any], words: dict[str, str]) -> str:
    return _join(PART_TYPES[opts["part"]], character_prompt(words["desc"], opts["character"], opts["outfit"]),
                 words["edit"], PART_STYLE, PART_SUFFIX)


def _part_denoise(opts: dict[str, Any], words: dict[str, str]) -> float:
    return _denoise(opts, bool(PART_TYPES[opts["part"]] or words["desc"] or words["edit"]))


async def _render_part(
    client: Any, layer: Any, *, context: Any, repaint: Any, region: tuple[int, int, int, int],
    subject: str, seed: int, denoise: float, opts: dict[str, Any],
) -> Any:
    sheet, w, h, plan = prepare_part(layer, context=context, repaint=repaint, region=region)
    out = await _render(client, lambda n: build_still_graph(
        subject, seed=seed, width=w, height=h, character=opts["character"], hd=opts["quality"] == "hd",
        negative=with_negative(PART_NEGATIVE, opts), detail=False, source_image=n, denoise=denoise,
        masked=True), sheet)
    return finish_part(out, plan)


async def render_part_set(
    client: Any, layer: Any, followers: list[Any], *, context: Any, subject: str, seed: int,
    denoise: float, opts: dict[str, Any],
) -> list[Any]:
    """Redraw the layer inside its own outline, then rebuild each follower (talking face, animation frame)
    from that result, redrawing only where its original differs from the layer's original with the same
    seed and prompt, so swapping between them never flickers. Each comes back cut to its own transparency."""
    from PIL import Image, ImageFilter

    region = part_region([layer, *followers])
    kw = {"context": context, "region": region, "subject": subject, "seed": seed, "denoise": denoise, "opts": opts}
    gen = await _render_part(client, layer, repaint=None, **kw)
    skin = skin_params(gen, layer, context if context is not None else layer) if opts.get("match_skin") else None

    def post(img: Any, original: Any) -> Any:
        img = match_skin(img, original.getchannel("A"), skin)
        return keep_lines(img, original) if opts.get("keep_lines") else img

    main = post(gen, layer)
    out = [with_alpha(main, layer)]
    for frame in followers:
        changed = diff_mask(layer, frame)
        if changed.getbbox() is None:
            out.append(with_alpha(main, frame))
            continue
        src = Image.composite(frame.convert("RGB"), main, changed)
        src.putalpha(frame.getchannel("A"))
        redrawn = post(await _render_part(client, src, repaint=changed, **kw), frame)
        blend = changed.filter(ImageFilter.GaussianBlur(1.5))
        out.append(with_alpha(Image.composite(redrawn, main, blend), frame))
    return out


async def run_still_outputs(
    client: Any, prompt: str, *, seed: Optional[int], aspect: Optional[str] = None,
    options: Optional[dict[str, Any]] = None, inputs: Optional[dict[str, bytes]] = None,
) -> list[tuple[bytes, str, dict]]:
    """Text-to-image, or with inputs["image"]: restyle the photo (scene), use it as the background (sprite)
    or redraw one character layer (part; inputs may add "context" and "pair"). One output per variant
    (count, seeds seed..seed+count-1), each followed by its talking face when a pair is given.
    Plain English is rewritten into model tags unless rewrite is off."""
    from PIL import Image

    inputs = {k: v for k, v in (inputs or {}).items() if v}
    image_bytes = inputs.get("image")
    options = dict(options or {})
    if options.get("aspect") is None:
        options["aspect"] = aspect
    opts, words = await _still_setup(client, prompt, options, photo=image_bytes is not None,
                                     context="context" in inputs, pair="pair" in inputs)
    seed = seed if seed is not None else new_seed()
    label, count = _label(opts["character"]), opts["count"]
    base = {"character": opts["character"], "quality": opts["quality"], "layout": opts["layout"]}
    outputs: list[tuple[bytes, str, dict]] = []

    if opts["layout"] == "part":
        layer = open_layer(image_bytes)
        context = open_layer(inputs["context"]) if "context" in inputs else None
        followers = [open_layer(inputs["pair"])] if "pair" in inputs else []
        for other, what in [(context, "context"), *((f, "pair") for f in followers)]:
            if other is not None and other.size != layer.size:
                raise SagaError(f"The {what} image is {other.width}x{other.height} but the layer is "
                                f"{layer.width}x{layer.height}; they must be the same size")
        subject, denoise = part_subject(opts, words), _part_denoise(opts, words)
        for i in range(count):
            images = await render_part_set(client, layer, followers, context=context, subject=subject,
                                           seed=seed + i, denoise=denoise, opts=opts)
            for role, img in zip(("main", "pair"), images):
                meta = {**base, "kind": "part", "aspect": "photo", "width": img.width, "height": img.height,
                        "prompt_used": subject, "strength": opts["strength"], "denoise": denoise,
                        "changes": opts.get("changes"), "part": opts["part"], "transparent": img.mode == "RGBA",
                        "context": context is not None, "keep_lines": opts["keep_lines"],
                        "match_skin": opts["match_skin"], "negative": opts["negative"], "role": role,
                        "variant": i + 1, "variants": count}
                outputs.append((_png(img), "image/png", {"model_label": label, "seed_used": seed + i, "saga": meta}))
        return outputs

    photo, width, height = (prepare_photo(image_bytes) if image_bytes is not None
                            else (None, *ASPECTS[opts["aspect"]]))
    keep = open_layer(image_bytes).size if photo and opts.get("keep_size") else None
    if words["edit"] and opts["layout"] == "scene" and opts.get("strength") == "light":
        opts["strength"] = "medium"
    hd = opts["quality"] == "hd"
    for i in range(count):
        s = seed + i
        if opts["layout"] == "sprite":
            sw, sh = ASPECTS["portrait"]
            subject = character_prompt(words["desc"] or "standing, smiling, looking at viewer",
                                       opts["character"], opts["outfit"])
            background = words["background"]
            denoise = _denoise(opts, bool(background))
            sprite = await _render(client, lambda _n: build_still_graph(
                subject + SPRITE_SUFFIX, seed=s, width=sw, height=sh, character=opts["character"], hd=hd,
                negative=with_negative(SPRITE_NEGATIVE, opts), cutout=True), None)
            back = await _render(client, lambda n: build_still_graph(
                BACKGROUND_PREFIX + background, seed=s + 7, width=width, height=height,
                hd=hd, negative=with_negative(BACKGROUND_NEGATIVE, opts), detail=False, source_image=n,
                denoise=denoise), photo)
            data = compose_sprite(back, sprite)
            subject = _join(subject, f"background: {background}" if background else "")
        else:
            subject = character_prompt(_join(words["desc"], words["edit"]), opts["character"], opts["outfit"])
            denoise = _denoise(opts, bool(words["desc"] or words["edit"]))
            data = await _render(client, lambda n: build_still_graph(
                subject, seed=s, width=width, height=height, character=opts["character"], hd=hd,
                negative=with_negative(NEGATIVE, opts), source_image=n, denoise=denoise), photo)
        out_w, out_h = (_snap(width * HD_SCALE), _snap(height * HD_SCALE)) if hd else (width, height)
        if keep:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            data, (out_w, out_h) = _png(img.resize(keep, Image.Resampling.LANCZOS)), keep
        meta = {**base, "kind": "still", "aspect": opts["aspect"], "width": out_w, "height": out_h,
                "prompt_used": subject, "negative": opts["negative"]}
        if photo:
            meta.update(kind="photo", aspect="photo", strength=opts["strength"], denoise=denoise,
                        changes=opts.get("changes"))
        if count > 1:
            meta.update(variant=i + 1, variants=count)
        outputs.append((data, "image/png", {"model_label": label, "seed_used": s, "saga": meta}))
    return outputs


async def run_still(
    client: Any, prompt: str, *, seed: Optional[int], aspect: Optional[str] = None,
    options: Optional[dict[str, Any]] = None, image_bytes: Optional[bytes] = None,
) -> tuple[bytes, str, dict]:
    """First output of run_still_outputs."""
    outputs = await run_still_outputs(client, prompt, seed=seed, aspect=aspect, options=options,
                                      inputs={"image": image_bytes} if image_bytes else None)
    return outputs[0]


async def run_batch(
    client: Any, prompt: str, *, seed: Optional[int], options: Optional[dict[str, Any]], bundle: bytes,
    progress: Optional[Callable[[int, int], None]] = None,
) -> tuple[bytes, str, dict]:
    """Many part layers in one job with the same prompt, seed and settings (so a character stays the same
    person) -> zip of the results under their upload names plus report.json. After 3 failures in a row
    (GPU or tunnel down) it stops and returns what finished."""
    files, manifest = unpack_inputs(bundle)
    items = check_batch(files, manifest)["items"]
    opts, words = await _still_setup(client, prompt, {**(options or {}), "layout": "part", "count": 1}, photo=True)
    seed = seed if seed is not None else new_seed()
    contexts: dict[str, Any] = {}
    report: list[dict[str, Any]] = []
    stopped, streak = None, 0
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for i, item in enumerate(items):
            item_opts = {**opts, "part": item.get("part") or opts["part"]}
            subject, denoise = part_subject(item_opts, words), _part_denoise(item_opts, words)
            names = [item["image"], *item["pair"]]
            try:
                ctx_name = item.get("context")
                if ctx_name and ctx_name not in contexts:
                    contexts[ctx_name] = open_layer(files[ctx_name])
                images = await render_part_set(
                    client, open_layer(files[item["image"]]), [open_layer(files[n]) for n in item["pair"]],
                    context=contexts.get(ctx_name) if ctx_name else None, subject=subject, seed=seed,
                    denoise=denoise, opts=item_opts)
                for name, img in zip(names, images):
                    zf.writestr(name, _png(img))
                report.append({"image": item["image"], "ok": True, "outputs": names, "prompt_used": subject})
                streak = 0
            except Exception as exc:
                logger.warning("Saga batch item %s failed: %s", item["image"], exc)
                report.append({"image": item["image"], "ok": False, "error": str(exc)})
                streak += 1
                if streak >= 3:
                    stopped = f"Stopped after 3 failures in a row: {exc}"
            if progress:
                progress(i + 1, len(items))
            if stopped:
                break
        done = sum(1 for r in report if r["ok"])
        if not done:
            raise SagaError(stopped or report[-1].get("error") or "Every item failed")
        shared_subject, shared_denoise = part_subject(opts, words), _part_denoise(opts, words)
        summary = {"prompt_used": shared_subject, "seed_used": seed, "denoise": shared_denoise,
                   "strength": opts["strength"], "quality": opts["quality"], "part": opts["part"],
                   "keep_lines": opts["keep_lines"], "match_skin": opts["match_skin"], "negative": opts["negative"],
                   "done": done, "failed": len(report) - done, "total": len(items), "stopped": stopped,
                   "items": report}
        zf.writestr("report.json", json.dumps(summary, indent=1))
    meta = {key: summary[key] for key in ("prompt_used", "denoise", "strength", "quality", "part", "keep_lines",
                                          "match_skin", "negative", "done", "failed", "total", "stopped")}
    meta.update(kind="batch", layout="part", character=opts["character"], changes=opts.get("changes"))
    return buf.getvalue(), "application/zip", {"model_label": _label(opts["character"]), "seed_used": seed,
                                               "saga": meta}


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
    opts = parse_motion_options(seconds=opts.get("seconds"), quality=opts.get("quality"),
                                negative=opts.get("negative"))
    sec, quality = opts["seconds"], opts["quality"]
    seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    segments, seg_len = segment_plan(sec)
    interpolate = 2
    box = VIDEO_BOX[quality]
    data, ctype = await client.generate_video(
        image_bytes,
        MOTION_PREFIX + text,
        negative=with_negative(MOTION_NEGATIVE, opts),
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
                 "fps": VIDEO_FPS * interpolate, "segments": segments, "negative": opts["negative"]},
    }
