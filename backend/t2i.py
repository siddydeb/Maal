"""Text-to-image (no upload): Flux Dev + realism LoRA stack + optional character LoRA.

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
}

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
    warnings: list[str] = field(default_factory=list)


def load_config(*, with_trained: bool = True) -> dict[str, Any]:
    raw = load_json("t2i.json", default=None) or {}
    cfg = dict(_DEFAULTS)
    cfg.update({k: v for k, v in raw.items() if v is not None})
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
    }


def plan_t2i(
    prompt: str,
    *,
    cfg: dict[str, Any],
    installed: Optional[set[str]],
    character_id: Optional[str] = None,
    outfit_id: Optional[str] = None,
    aspect: Optional[str] = None,
) -> T2IPlan:
    warnings: list[str] = []
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
    for spec in cfg.get("loras") or []:
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
    )
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
    lora_tags = "|".join(f"{fn}@{sm:g}" for fn, sm, _ in plan.loras)
    meta = {
        "task_type": "image.t2i",
        "workflow_ref": "t2i.v1",
        "model_label": f"{unet}|t2i" + (f"|{lora_tags}" if lora_tags else ""),
        "seed_used": seed_used,
        "t2i": {
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
