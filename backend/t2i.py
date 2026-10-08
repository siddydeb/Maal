"""Text-to-image (no upload) and the Klein keep-face edit.

T2I engines: FLUX.2 Klein 9B (default for new girls) or Flux Dev + realism LoRA stack.
A trained character uses Flux Dev (Flux.1 LoRA) unless its entry says ``"engine": "klein"``.

Knobs, LoRA filenames, characters and saved outfits live in ``private/t2i.json``.
A character entry looks like::

    {"id": "zara", "name": "Zara", "lora": "zara_v1.safetensors",
     "strength": 1.0, "trigger": "zara_v1 woman", "description": "..."}

Klein characters may add ``"loras": [{"file", "strength"}]`` (stacked after the
character LoRA), ``"suffix"`` (appended after the user text), ``"ref_gridfs_id"``
(face photo fed as a ReferenceLatent) and ``"face_fix": true`` (re-render the
largest face at full resolution; fixes small faces in full-body shots).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from backend.ai_engine.runtime_overlay import load_json

_DEFAULTS: dict[str, Any] = {
    "unet": None,
    "steps": 30,
    "guidance": 3.0,
    "sampler": "euler",
    "scheduler": "beta",
    "default_aspect": "portrait",
    "aspects": {
        "portrait": {"label": "Portrait 2:3", "size": [832, 1216]},
        "square": {"label": "Square 1:1", "size": [1024, 1024]},
        "landscape": {"label": "Landscape 3:2", "size": [1216, 832]},
    },
    "prompt_suffix": "",
    "loras": [],
    "characters": [],
    "outfits": [],
    # "klein" | "flux"; characters use their own "engine" (default flux).
    "engine": "klein",
    "klein": {
        "unet": "flux-2-klein-9b-Q8_0.gguf",
        "clip": "qwen_3_8b_fp8mixed.safetensors",
        "vae": "flux2-vae.safetensors",
        "steps": 4,
        "t2i_loras": [],
        "edit_loras": [{"file": "klein_consistency_v2.safetensors", "strength": 0.8}],
        "edit_suffix": "Keep her face, skin tone, hair and identity exactly the same.",
    },
}

ENGINES = {"klein": "Klein 9B", "flux": "Flux Dev"}

FACE_FIX_DEFAULTS = {"denoise": 0.4, "steps": 8}
FACE_FIX_PROMPT = (
    "Close-up photo of the face of {trigger}, the same woman as in the reference image, "
    "natural unretouched skin with visible pores, same lighting and expression."
)
BACKGROUND_KEEP = (
    "Keep the scene of image 1 exactly the same: same place, camera angle, perspective, framing, "
    "objects and colors. She fits naturally into the scene at the correct size and position, lit by "
    "the same light as image 1, with matching shadows, color grading, grain and focus."
)

_LORA_CACHE: dict[str, Any] = {"at": 0.0, "names": None}
_LORA_CACHE_TTL = 60.0


class T2IError(ValueError):
    pass


@dataclass
class T2IPlan:
    prompt: str
    width: int
    height: int
    loras: list[tuple[str, float, float]]
    character_id: Optional[str]
    outfit_id: Optional[str]
    aspect: str
    engine: str = "flux"
    warnings: list[str] = field(default_factory=list)
    ref_gridfs_id: Optional[str] = None
    face_fix: Optional[dict[str, Any]] = None
    background: bool = False


def load_config(*, with_trained: bool = True) -> dict[str, Any]:
    raw = load_json("t2i.json", default=None) or {}
    cfg = dict(_DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if v is not None})
    cfg["klein"] = {**_DEFAULTS["klein"], **(raw.get("klein") or {})}
    if with_trained:
        try:
            from backend.training import list_characters_sync

            trained = list_characters_sync()
        except Exception:
            trained = []
        if trained:
            ids = {c["id"] for c in trained}
            cfg["characters"] = [
                c for c in cfg.get("characters") or [] if c.get("id") not in ids
            ] + trained
    return cfg


def _by_id(items: list[dict[str, Any]], item_id: Optional[str]) -> Optional[dict[str, Any]]:
    if not item_id:
        return None
    for it in items or []:
        if str(it.get("id")) == item_id:
            return it
    return None


async def installed_loras(client: Any, *, refresh: bool = False) -> Optional[set[str]]:
    """Cached LoraLoader menu from Comfy; None when Comfy can't be queried."""
    now = time.monotonic()
    if not refresh and _LORA_CACHE["names"] is not None and now - _LORA_CACHE["at"] < _LORA_CACHE_TTL:
        return _LORA_CACHE["names"]
    try:
        names = await client.lora_names()
    except Exception:
        return _LORA_CACHE["names"]
    _LORA_CACHE.update(at=now, names=names)
    return names


def public_config(cfg: dict[str, Any], installed: Optional[set[str]]) -> dict[str, Any]:
    """Safe subset for the UI (no LoRA filenames)."""

    def has(fn: Optional[str]) -> bool:
        return bool(fn) and (installed is None or fn in installed)

    aspects = [
        {"id": k, "label": v.get("label") or k, "width": v["size"][0], "height": v["size"][1]}
        for k, v in (cfg.get("aspects") or {}).items()
    ]
    return {
        "default_aspect": cfg.get("default_aspect"),
        "aspects": aspects,
        "characters": [
            {
                "id": c.get("id"),
                "name": c.get("name") or c.get("id"),
                "installed": has(c.get("lora")),
                "engine": character_engine(c),
                "face_fix": bool(c.get("face_fix")) and character_engine(c) == "klein",
            }
            for c in cfg.get("characters") or []
        ],
        "outfits": [
            {"id": o.get("id"), "name": o.get("name") or o.get("id")}
            for o in cfg.get("outfits") or []
        ],
        "realism_ready": all(
            has(l.get("file")) for l in cfg.get("loras") or [] if not l.get("when")
        ),
        "engines": [{"id": k, "label": v} for k, v in ENGINES.items()],
        "default_engine": resolve_engine(cfg, None, None),
        "klein_ready": all(has(l.get("file")) for l in cfg["klein"].get("edit_loras") or []),
    }


def character_engine(character: dict[str, Any]) -> str:
    return "klein" if str(character.get("engine") or "").lower() == "klein" else "flux"


def resolve_engine(
    cfg: dict[str, Any], engine: Optional[str], character: Optional[dict[str, Any]]
) -> str:
    if character:
        return character_engine(character)
    choice = (engine or cfg.get("engine") or "flux").strip().lower()
    return choice if choice in ENGINES else "flux"


def plan_t2i(
    prompt: str,
    *,
    cfg: dict[str, Any],
    installed: Optional[set[str]],
    character_id: Optional[str] = None,
    outfit_id: Optional[str] = None,
    aspect: Optional[str] = None,
    engine: Optional[str] = None,
    face_fix: Optional[bool] = None,
    background: bool = False,
) -> T2IPlan:
    """``background``: the user's photo is image 1 and she is placed into it (Klein only);
    the output then takes the background's aspect instead of ``aspect``."""
    warnings: list[str] = []
    character = _by_id(cfg.get("characters") or [], character_id)
    if character_id and not character:
        raise T2IError(f"Unknown character '{character_id}'.")
    engine_id = resolve_engine(cfg, engine, character)
    if background and engine_id != "klein":
        raise T2IError("A background photo needs the Klein 9B model.")
    outfit = _by_id(cfg.get("outfits") or [], outfit_id)
    if outfit_id and not outfit:
        raise T2IError(f"Unknown outfit '{outfit_id}'.")

    aspects = cfg.get("aspects") or {}
    aspect_id = aspect if aspect in aspects else cfg.get("default_aspect")
    if aspect_id not in aspects:
        aspect_id = next(iter(aspects), None)
    if not aspect_id:
        raise T2IError("No aspect sizes configured.")
    width, height = (int(x) for x in aspects[aspect_id]["size"])

    def available(fn: str) -> bool:
        return installed is None or fn in installed

    stack: list[tuple[str, float, float]] = []
    triggers: list[str] = []
    user_text = (prompt or "").strip()
    lora_specs = cfg["klein"].get("t2i_loras") if engine_id == "klein" else cfg.get("loras")
    for spec in lora_specs or []:
        fn = str(spec.get("file") or "")
        if not fn:
            continue
        if spec.get("only_without_character") and character:
            continue
        when = spec.get("when")
        if when and not re.search(when, user_text, re.IGNORECASE):
            continue
        if not available(fn):
            warnings.append(f"LoRA missing on ComfyUI, skipped: {fn}")
            continue
        s = float(spec.get("strength", 0.8))
        stack.append((fn, s, s))
        if spec.get("trigger"):
            triggers.append(str(spec["trigger"]).strip())

    parts: list[str] = []
    if triggers:
        parts.append(", ".join(triggers) + ".")
    if character:
        fn = str(character.get("lora") or "")
        if fn:
            if not available(fn):
                raise T2IError(
                    f"Character LoRA for '{character.get('name') or character_id}' "
                    "is not installed on ComfyUI yet."
                )
            s = float(character.get("strength", 1.0))
            # Character LoRA goes first so realism LoRAs sit on top of identity.
            stack.insert(0, (fn, s, s))
            for i, spec in enumerate(character.get("loras") or []):
                extra = str(spec.get("file") or "")
                if not extra:
                    continue
                if not available(extra):
                    warnings.append(f"LoRA missing on ComfyUI, skipped: {extra}")
                    continue
                es = float(spec.get("strength", 0.6))
                stack.insert(1 + i, (extra, es, es))
    ref_id: Optional[str] = None
    if character and engine_id == "klein":
        ref_id = str(character.get("ref_gridfs_id") or "") or None
    trigger = str((character or {}).get("trigger") or "").strip()
    if background:
        who = (trigger or "the woman") + (" from image 2" if ref_id else "")
        parts.append(f"Place {who} into the scene of image 1.")
    elif trigger:
        parts.append(trigger + ",")
    if character and character.get("description"):
        parts.append(str(character["description"]).strip())
    if user_text and user_text[-1] not in ".!?,;":
        user_text += "."
    parts.append(user_text)
    if outfit and outfit.get("text"):
        parts.append(str(outfit["text"]).strip())
    if background:
        parts.append(BACKGROUND_KEEP)
    if character and character.get("suffix"):
        parts.append(str(character["suffix"]).strip())
    if cfg.get("prompt_suffix"):
        parts.append(str(cfg["prompt_suffix"]).strip())

    fix: Optional[dict[str, Any]] = None
    if character and engine_id == "klein" and character.get("face_fix") and face_fix is not False:
        fix = {**FACE_FIX_DEFAULTS, "prompt": FACE_FIX_PROMPT.format(trigger=trigger or "the woman")}

    return T2IPlan(
        prompt=" ".join(p for p in parts if p),
        width=width,
        height=height,
        loras=stack,
        character_id=character.get("id") if character else None,
        outfit_id=outfit.get("id") if outfit else None,
        aspect="background" if background else aspect_id,
        engine=engine_id,
        warnings=warnings,
        ref_gridfs_id=ref_id,
        face_fix=fix,
        background=background,
    )


def _normalize_photo(data: bytes) -> tuple[bytes, int, int]:
    """Upright RGB PNG at ~1MP keeping the photo's aspect (Klein edit size)."""
    import io

    from PIL import Image, ImageOps

    from backend.workflows_klein import edit_size

    with Image.open(io.BytesIO(data)) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    width, height = edit_size(*img.size)
    buf = io.BytesIO()
    img.resize((width, height), Image.Resampling.LANCZOS).save(buf, format="PNG")
    return buf.getvalue(), width, height


Box = tuple[float, float, float, float]


def parse_place_box(raw: Any) -> Optional[Box]:
    """"x0,y0,x1,y1" as fractions of the photo -> box, or None when empty."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        vals = [float(v) for v in (raw.split(",") if isinstance(raw, str) else raw)]
    except (TypeError, ValueError):
        raise T2IError("place_box must be four numbers: x0,y0,x1,y1.")
    if len(vals) != 4:
        raise T2IError("place_box must be four numbers: x0,y0,x1,y1.")
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in vals)
    if x1 - x0 < 0.04 or y1 - y0 < 0.04:
        raise T2IError("The marked spot is too small.")
    return x0, y0, x1, y1


class _Spot:
    """Crowded photo: she is generated in a crop around the marked box (which is mostly empty,
    so the normal placement works) and pasted back inside the box only, leaving everyone else
    untouched."""

    CONTEXT = 0.35
    MAX_MP = 2.5

    def __init__(self, photo: bytes, box: Box) -> None:
        import io

        from PIL import Image, ImageDraw, ImageFilter, ImageOps

        from backend.workflows_klein import edit_size

        with Image.open(io.BytesIO(photo)) as im:
            base = ImageOps.exif_transpose(im).convert("RGB")
        scale = min(1.0, (self.MAX_MP * 1024 * 1024 / (base.width * base.height)) ** 0.5)
        if scale < 1.0:
            base = base.resize((round(base.width * scale), round(base.height * scale)), Image.Resampling.LANCZOS)
        W, H = base.size
        bx0, by0, bx1, by1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
        mx, my = (bx1 - bx0) * self.CONTEXT, (by1 - by0) * self.CONTEXT
        self.crop = (int(max(0, bx0 - mx)), int(max(0, by0 - my)), int(min(W, bx1 + mx)), int(min(H, by1 + my)))
        cx0, cy0 = self.crop[:2]
        region = base.crop(self.crop)
        self.width, self.height = edit_size(*region.size)
        self.base = base

        paste = Image.new("L", region.size, 0)
        ImageDraw.Draw(paste).rectangle([bx0 - cx0, by0 - cy0, bx1 - cx0, by1 - cy0], fill=255)
        self.paste_mask = paste.filter(ImageFilter.GaussianBlur(radius=max(region.size) * 0.03))

        sx, sy = self.width / region.width, self.height / region.height
        face = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        ImageDraw.Draw(face).rectangle(
            [(bx0 - cx0) * sx, (by0 - cy0) * sy, (bx1 - cx0) * sx, (by1 - cy0) * sy], fill=(255, 255, 255)
        )
        self.region_png = _png(region.resize((self.width, self.height), Image.Resampling.LANCZOS))
        self.face_mask_png = _png(face)

    def paste(self, generated: bytes) -> bytes:
        import io

        from PIL import Image

        cx0, cy0, cx1, cy1 = self.crop
        with Image.open(io.BytesIO(generated)) as im:
            gen = im.convert("RGB").resize((cx1 - cx0, cy1 - cy0), Image.Resampling.LANCZOS)
        out = self.base.copy()
        out.paste(gen, (cx0, cy0), self.paste_mask)
        return _png(out)


def _png(img: Any) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _gridfs_bytes(file_id: str) -> bytes:
    from bson import ObjectId

    from backend import db

    stream = await db.fs().open_download_stream(ObjectId(file_id))
    return await stream.read()


async def run_t2i(
    client: Any,
    prompt: str,
    *,
    seed: Optional[int] = None,
    character_id: Optional[str] = None,
    outfit_id: Optional[str] = None,
    aspect: Optional[str] = None,
    engine: Optional[str] = None,
    face_fix: Optional[bool] = None,
    background_bytes: Optional[bytes] = None,
    place_box: Optional[Box] = None,
) -> tuple[bytes, str, dict[str, Any]]:
    """Returns (image_bytes, content_type, meta). ``background_bytes`` places her into that photo;
    ``place_box`` (fractions x0,y0,x1,y1) puts her only in that spot (crowded photos)."""
    cfg = load_config()
    installed = await installed_loras(client)
    plan = plan_t2i(
        prompt,
        cfg=cfg,
        installed=installed,
        character_id=character_id,
        outfit_id=outfit_id,
        aspect=aspect,
        engine=engine,
        face_fix=face_fix,
        background=bool(background_bytes),
    )
    if plan.engine == "klein":
        kc = cfg["klein"]
        unet = str(kc["unet"])
        refs: list[bytes] = []
        spot: Optional[_Spot] = None
        if background_bytes and place_box:
            spot = _Spot(background_bytes, place_box)
            plan.width, plan.height = spot.width, spot.height
            refs.append(spot.region_png)
        elif background_bytes:
            bg, plan.width, plan.height = _normalize_photo(background_bytes)
            refs.append(bg)
        face_index = len(refs)
        if plan.ref_gridfs_id:
            try:
                refs.append(await _gridfs_bytes(plan.ref_gridfs_id))
            except Exception:
                plan.warnings.append("Character face photo missing; generated from the LoRA only.")
        data, ctype, seed_used = await client.generate_klein(
            plan.prompt,
            width=plan.width,
            height=plan.height,
            seed=seed,
            steps=int(kc.get("steps") or 4),
            unet=unet,
            clip=kc.get("clip"),
            vae=kc.get("vae"),
            loras=[(fn, sm) for fn, sm, _ in plan.loras],
            ref_images=refs,
            face_ref_index=face_index,
            face_fix=plan.face_fix,
            face_mask=spot.face_mask_png if spot else None,
        )
        if spot:
            data, ctype = spot.paste(data), "image/png"
            plan.width, plan.height = spot.base.size
        workflow_ref = "t2i.klein.v1"
    else:
        data, ctype, seed_used = await client.generate_t2i(
            plan.prompt,
            width=plan.width,
            height=plan.height,
            seed=seed,
            steps=int(cfg.get("steps") or 30),
            guidance=float(cfg.get("guidance") or 3.0),
            sampler=str(cfg.get("sampler") or "euler"),
            scheduler=str(cfg.get("scheduler") or "beta"),
            flux_unet=cfg.get("unet") or None,
            loras=plan.loras,
        )
        unet = cfg.get("unet") or client.settings.flux_unet
        workflow_ref = "t2i.v1"
    lora_tags = "|".join(f"{fn}@{sm:g}" for fn, sm, _ in plan.loras)
    meta = {
        "task_type": "image.t2i",
        "workflow_ref": workflow_ref,
        "model_label": f"{unet}|t2i" + (f"|{lora_tags}" if lora_tags else ""),
        "seed_used": seed_used,
        "t2i": {
            "engine": plan.engine,
            "character": plan.character_id,
            "outfit": plan.outfit_id,
            "aspect": plan.aspect,
            "width": plan.width,
            "height": plan.height,
            "final_prompt": plan.prompt,
            "face_ref": bool(plan.ref_gridfs_id),
            "face_fix": plan.face_fix is not None,
            "background": plan.background,
            "place_box": list(place_box) if place_box and plan.background else None,
        },
        "engine_warnings": plan.warnings,
    }
    return data, ctype, meta


def edit_prompt(prompt: str, cfg: dict[str, Any]) -> str:
    text = (prompt or "").strip()
    suffix = str(cfg["klein"].get("edit_suffix") or "").strip()
    if not suffix or re.search(r"\bkeep\b.*\b(face|identity)\b", text, re.IGNORECASE):
        return text
    if text and text[-1] not in ".!?":
        text += "."
    return f"{text} {suffix}".strip()


async def run_klein_edit(
    client: Any,
    image_bytes: bytes,
    prompt: str,
    *,
    seed: Optional[int] = None,
) -> tuple[bytes, str, dict[str, Any]]:
    """Klein 9B reference edit of an uploaded image (keeps the face). Returns (bytes, ctype, meta)."""
    cfg = load_config(with_trained=False)
    kc = cfg["klein"]
    installed = await installed_loras(client)
    warnings: list[str] = []
    loras: list[tuple[str, float]] = []
    for spec in kc.get("edit_loras") or []:
        fn = str(spec.get("file") or "")
        if not fn:
            continue
        if installed is not None and fn not in installed:
            warnings.append(f"LoRA missing on ComfyUI, skipped: {fn}")
            continue
        loras.append((fn, float(spec.get("strength", 0.8))))
    image_bytes, width, height = _normalize_photo(image_bytes)
    final_prompt = edit_prompt(prompt, cfg)
    unet = str(kc["unet"])
    data, ctype, seed_used = await client.generate_klein(
        final_prompt,
        width=width,
        height=height,
        seed=seed,
        steps=int(kc.get("steps") or 4),
        unet=unet,
        clip=kc.get("clip"),
        vae=kc.get("vae"),
        loras=loras,
        image_bytes=image_bytes,
    )
    lora_tags = "|".join(f"{fn}@{sm:g}" for fn, sm in loras)
    meta = {
        "task_type": "image.edit",
        "workflow_ref": "edit.klein.v1",
        "model_label": f"{unet}|edit" + (f"|{lora_tags}" if lora_tags else ""),
        "seed_used": seed_used,
        "edit": {"engine": "klein", "width": width, "height": height, "final_prompt": final_prompt},
        "engine_warnings": warnings,
    }
    return data, ctype, meta
