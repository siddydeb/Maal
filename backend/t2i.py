"""Text-to-image (no upload) and the Klein keep-face edit.

T2I engines: FLUX.2 Klein 9B (default for new girls) or Flux Dev + realism LoRA stack;
a trained character always uses Flux Dev since its LoRA is Flux.1.

Knobs, LoRA filenames, characters and saved outfits live in ``private/t2i.json``.
A character entry looks like::

    {"id": "zara", "name": "Zara", "lora": "zara_v1.safetensors",
     "strength": 1.0, "trigger": "zara_v1 woman", "description": "..."}
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
    # "klein" | "flux"; trained characters always use flux (their LoRAs are Flux.1).
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


def resolve_engine(cfg: dict[str, Any], engine: Optional[str], character_id: Optional[str]) -> str:
    if character_id:
        return "flux"
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
) -> T2IPlan:
    warnings: list[str] = []
    engine_id = resolve_engine(cfg, engine, character_id)
    character = _by_id(cfg.get("characters") or [], character_id)
    if character_id and not character:
        raise T2IError(f"Unknown character '{character_id}'.")
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
        if character.get("trigger"):
            parts.append(str(character["trigger"]).strip() + ",")
        if character.get("description"):
            parts.append(str(character["description"]).strip())
    if user_text and user_text[-1] not in ".!?,;":
        user_text += "."
    parts.append(user_text)
    if outfit and outfit.get("text"):
        parts.append(str(outfit["text"]).strip())
    if cfg.get("prompt_suffix"):
        parts.append(str(cfg["prompt_suffix"]).strip())

    return T2IPlan(
        prompt=" ".join(p for p in parts if p),
        width=width,
        height=height,
        loras=stack,
        character_id=character.get("id") if character else None,
        outfit_id=outfit.get("id") if outfit else None,
        aspect=aspect_id,
        engine=engine_id,
        warnings=warnings,
    )


async def run_t2i(
    client: Any,
    prompt: str,
    *,
    seed: Optional[int] = None,
    character_id: Optional[str] = None,
    outfit_id: Optional[str] = None,
    aspect: Optional[str] = None,
    engine: Optional[str] = None,
) -> tuple[bytes, str, dict[str, Any]]:
    """Returns (image_bytes, content_type, meta)."""
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
    )
    if plan.engine == "klein":
        kc = cfg["klein"]
        unet = str(kc["unet"])
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
        )
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
    import io

    from PIL import Image, ImageOps

    from backend.workflows_klein import edit_size

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
    with Image.open(io.BytesIO(image_bytes)) as im:
        img = ImageOps.exif_transpose(im).convert("RGB")
    width, height = edit_size(*img.size)
    img = img.resize((width, height), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    del img
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
