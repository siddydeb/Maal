"""
Generation graphs for Wan Studio.

IMG  → Flux Dev fp8 img2img (best prompt adherence / realistic edits on this machine)
VID  → Wan 2.2 I2V 14B dual-stage high+low noise (best image→video fidelity here)

Models expected in ComfyUI:
  diffusion_models/flux1-dev-fp8.safetensors
  clip/clip_l.safetensors + text_encoders|clip/t5xxl_fp8_e4m3fn.safetensors
  vae/ae.safetensors
  diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
  diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
  text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors
  vae/wan_2.1_vae.safetensors
"""

from __future__ import annotations

from typing import Any


# Quality-only defaults — no identity / style / content lockrails that fight the prompt.
DEFAULT_NEGATIVE_WAN = (
    "blurry, low quality, jpeg artifacts, watermark, text, logo, "
    "soft focus, out of focus, gaussian blur, motion blur, smeared details, "
    "blurry face, unrecognizable face, deformed face, melted face, "
    "different person, face swap, changed identity, wrong face, "
    "changed background, different room, scene change, cut, jump cut, "
    "deformed hands, extra fingers, fused fingers, bad anatomy, "
    "flicker, flickering, temporal inconsistency, frame jitter, "
    "morphing face, identity drift, warping, melting, "
    "over-smoothed, plastic skin, muddy details, compression artifacts"
)

DEFAULT_NEGATIVE_FLUX = (
    "blurry, low quality, jpeg artifacts, watermark, text, logo, "
    "deformed, extra fingers, mutated hands"
)

# Logical LoRA ids → filenames. Extra ids merge from gitignored private/lora_files.py.
LORA_FILES: dict[str, str] = {
    "clothes_remover": "clothes_remover_v0.safetensors",
}
LORA_DEFAULT_STRENGTH: dict[str, float] = {
    "clothes_remover": 0.85,
}
try:
    from backend.ai_engine.runtime_overlay import load_private_module

    _lora_overlay = load_private_module("lora_files")
    if _lora_overlay is not None:
        LORA_FILES.update(getattr(_lora_overlay, "LORA_FILES", {}) or {})
        LORA_DEFAULT_STRENGTH.update(
            getattr(_lora_overlay, "LORA_DEFAULT_STRENGTH", {}) or {}
        )
        extra_neg = getattr(_lora_overlay, "WAN_NEGATIVE_EXTRA", "") or ""
        if extra_neg:
            DEFAULT_NEGATIVE_WAN = DEFAULT_NEGATIVE_WAN + ", " + extra_neg
except Exception:
    pass

# Flux Union ControlNet (pose / depth / canny). Prefer fp8 on 16GB.
CONTROLNET_FILES: dict[str, str] = {
    "flux_union_pro": "flux_shakker_labs_union_pro-fp8_e4m3fn.safetensors",
    "flux_union_pro_bf16": "flux_controlnet_union_pro.safetensors",
}
KONTEXT_UNET = "flux1-dev-kontext_fp8_scaled.safetensors"


def _snap(n: int, step: int = 16, minimum: int = 64) -> int:
    n = max(minimum, int(n))
    return max(minimum, (n // step) * step)


def fit_dims(src_w: int, src_h: int, max_w: int, max_h: int, step: int = 16) -> tuple[int, int]:
    """Preserve aspect ratio inside max box; snap to step."""
    max_w, max_h = _snap(max_w, step), _snap(max_h, step)
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    scale = min(max_w / src_w, max_h / src_h, 1.0)
    return _snap(int(src_w * scale), step), _snap(int(src_h * scale), step)


def resolve_lora_stack(
    lora_ids: list[str] | None,
    *,
    strengths: dict[str, float] | None = None,
) -> list[tuple[str, float, float]]:
    """Return [(filename, strength_model, strength_clip), ...] for Comfy LoraLoader."""
    out: list[tuple[str, float, float]] = []
    strengths = strengths or {}
    for lid in lora_ids or []:
        key = str(lid).strip()
        if not key:
            continue
        fn = LORA_FILES.get(key, key if key.endswith(".safetensors") else None)
        if not fn:
            continue
        s = float(strengths.get(key, LORA_DEFAULT_STRENGTH.get(key, 0.8)))
        s = max(0.0, min(1.5, s))
        out.append((fn, s, s))
    return out


def _inject_loras(
    graph: dict[str, Any],
    *,
    model_node: str,
    clip_node: str,
    loras: list[tuple[str, float, float]],
    start_id: int = 20,
) -> tuple[str, str, int]:
    """
    Chain LoraLoader nodes after model/clip loaders.
    Returns (model_ref_id, clip_ref_id, clip_output_slot).
    DualCLIPLoader CLIP is slot 0; LoraLoader CLIP is slot 1.
    """
    model_ref, clip_ref = model_node, clip_node
    clip_slot = 0
    nid = start_id
    for filename, sm, sc in loras:
        key = str(nid)
        graph[key] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": [model_ref, 0],
                "clip": [clip_ref, clip_slot],
                "lora_name": filename,
                "strength_model": float(sm),
                "strength_clip": float(sc),
            },
        }
        model_ref, clip_ref = key, key
        clip_slot = 1  # LoraLoader: MODEL=0, CLIP=1
        nid += 1
    return model_ref, clip_ref, clip_slot


def resolve_controlnet_filename(key: str | None = None) -> str | None:
    """Return first known ControlNet weight name (caller verifies Comfy has it)."""
    if key and key in CONTROLNET_FILES:
        return CONTROLNET_FILES[key]
    for fn in CONTROLNET_FILES.values():
        return fn
    return None


def build_i2i_prompt(
    *,
    image_name: str,
    positive: str,
    negative: str | None,
    flux_unet: str,
    flux_clip_l: str,
    flux_t5: str,
    flux_vae: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    denoise: float,
    denoise_cap: float = 0.85,
    wrap_preserve: bool = False,
    wrap_mode: str | None = None,
    loras: list[tuple[str, float, float]] | None = None,
    controlnet_name: str | None = None,
    control_image_name: str | None = None,
    control_type: str = "openpose",
    control_strength: float = 0.65,
    control_end: float = 0.85,
    mask_image_name: str | None = None,
    pulid_file: str | None = None,
    pulid_weight: float = 0.80,
    pulid_provider: str = "CUDA",
) -> dict[str, Any]:
    """
    Flux Dev img2img — encode source; prompt text goes through as-is unless wrap_preserve.
    CFG stays at 1.0; guidance is applied via FluxGuidance.
    Optional Union ControlNet (pose/depth/…) via control_image_name.
    Optional PuLID on the sampled model (identity pass). wrap_mode fabric keeps
    garment type/color while allowing drape.
    """
    width, height = _snap(width), _snap(height)
    cap = max(0.28, min(0.95, float(denoise_cap)))
    strength = max(0.20, min(cap, float(denoise)))
    edit = (positive or "").strip()
    mode = (wrap_mode or "").strip().lower()
    if not mode and wrap_preserve:
        mode = "identity"
    if mode in ("identity", "true", "1"):
        pos = (
            "Photorealistic photograph. This is an image edit of the provided photo. "
            "Preserve identity, face, pose, camera angle, framing, lighting, "
            "background, the same clothes (same color, cut, neckline, and fabric), "
            "and any object already in her hands. Do not invent a new person or scene. "
            f"Only apply this requested change: {edit}"
        )
    elif mode == "fabric":
        pos = (
            "Photorealistic photograph of the same person. Keep identity, face, hair, "
            "pose, camera, lighting, and background. Keep the same garment type and color. "
            "Fabric may drape and fill as the body shape changes. Do not invent a new "
            "outfit, person, or scene. Request: "
            f"{edit}"
        )
    else:
        pos = edit
    neg = (negative or "").strip() or DEFAULT_NEGATIVE_FLUX

    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": flux_unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": flux_clip_l,
                "clip_name2": flux_t5,
                "type": "flux",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": flux_vae},
        },
    }
    model_ref, clip_ref, clip_slot = _inject_loras(
        graph, model_node="1", clip_node="2", loras=list(loras or [])
    )
    graph.update(
        {
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": pos, "clip": [clip_ref, clip_slot]},
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": neg, "clip": [clip_ref, clip_slot]},
            },
            "6": {
                "class_type": "FluxGuidance",
                "inputs": {"conditioning": ["4", 0], "guidance": float(guidance)},
            },
            "7": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "8": {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["7", 0],
                    "upscale_method": "lanczos",
                    "width": width,
                    "height": height,
                    "crop": "disabled",
                },
            },
            "9": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["8", 0], "vae": ["3", 0]},
            },
            "10": {
                "class_type": "ModelSamplingFlux",
                "inputs": {
                    "model": [model_ref, 0],
                    "max_shift": 1.15,
                    "base_shift": 0.5,
                    "width": width,
                    "height": height,
                },
            },
        }
    )

    pos_ref, neg_ref = "6", "5"
    if controlnet_name and control_image_name:
        # Target pose / structure map (not the source photo — source OpenPose locks old pose).
        graph["14"] = {
            "class_type": "LoadImage",
            "inputs": {"image": control_image_name},
        }
        graph["15"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["14", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        }
        graph["16"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": controlnet_name},
        }
        # Union Pro modes: canny/tile/depth/blur/pose/gray/low_quality
        ctype = (control_type or "openpose").lower().strip()
        union_type = {
            "openpose": "pose",
            "pose": "pose",
            "depth": "depth",
            "canny": "canny",
            "tile": "tile",
        }.get(ctype, "pose")
        graph["17"] = {
            "class_type": "SetUnionControlNetType",
            "inputs": {"control_net": ["16", 0], "type": union_type},
        }
        c_strength = max(0.15, min(1.0, float(control_strength)))
        c_end = max(0.2, min(1.0, float(control_end)))
        graph["18"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["6", 0],
                "negative": ["5", 0],
                "control_net": ["17", 0],
                "image": ["15", 0],
                "strength": c_strength,
                "start_percent": 0.0,
                "end_percent": c_end,
            },
        }
        pos_ref, neg_ref = "18", "18"

    latent_ref = "9"
    if mask_image_name:
        # IDs 90+ — LoraLoader chain starts at 20 and ControlNet uses 14-18.
        graph["90"] = {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image_name},
        }
        graph["91"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["90", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
        }
        graph["92"] = {
            "class_type": "ImageToMask",
            "inputs": {"image": ["91", 0], "channel": "red"},
        }
        graph["93"] = {
            "class_type": "SetLatentNoiseMask",
            "inputs": {"samples": ["9", 0], "mask": ["92", 0]},
        }
        latent_ref = "93"

    model_for_sampler = "10"
    if pulid_file:
        # IDs 70+ — LoRA chain starts at 20; ControlNet 14-18; mask 90+.
        graph["70"] = {
            "class_type": "PulidFluxModelLoader",
            "inputs": {"pulid_file": pulid_file},
        }
        graph["71"] = {
            "class_type": "PulidFluxEvaClipLoader",
            "inputs": {},
        }
        graph["72"] = {
            "class_type": "PulidFluxInsightFaceLoader",
            "inputs": {"provider": pulid_provider if pulid_provider in ("CPU", "CUDA", "ROCM") else "CUDA"},
        }
        graph["73"] = {
            "class_type": "ApplyPulidFlux",
            "inputs": {
                "model": ["10", 0],
                "pulid_flux": ["70", 0],
                "eva_clip": ["71", 0],
                "face_analysis": ["72", 0],
                "image": ["8", 0],
                "weight": float(pulid_weight),
                "start_at": 0.0,
                "end_at": 1.0,
            },
        }
        model_for_sampler = "73"

    graph["11"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [model_for_sampler, 0],
            "seed": seed,
            "steps": int(steps),
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": [pos_ref, 0],
            "negative": [neg_ref, 1 if pos_ref == "18" else 0],
            "latent_image": [latent_ref, 0],
            "denoise": strength,
        },
    }
    graph["12"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
    }
    graph["13"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["12", 0], "filename_prefix": "flux_i2i"},
    }
    return graph


def build_kontext_edit_prompt(
    *,
    image_name: str,
    positive: str,
    negative: str | None,
    flux_unet: str,
    flux_clip_l: str,
    flux_t5: str,
    flux_vae: str,
    steps: int,
    guidance: float,
    seed: int,
    loras: list[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    """
    Flux Kontext instruction edit (Comfy template flux_kontext_dev_basic).

    LoadImage → FluxKontextImageScale → VAEEncode → ReferenceLatent on positive,
    ConditioningZeroOut negative, KSampler denoise=1.0 on encoded latent size.
    Optional LoraLoader chain after UNET/DualCLIP.
    """
    pos = (positive or "").strip()
    neg = (negative or "").strip() or DEFAULT_NEGATIVE_FLUX
    # Official template uses ~2.5; clamp to a sensible Kontext range.
    g = max(1.0, min(5.0, float(guidance) if guidance else 2.5))

    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": flux_unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": flux_clip_l,
                "clip_name2": flux_t5,
                "type": "flux",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": flux_vae},
        },
    }
    model_ref, clip_ref, clip_slot = _inject_loras(
        graph, model_node="1", clip_node="2", loras=list(loras or [])
    )
    graph.update(
        {
            "4": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
            },
            "5": {
                "class_type": "FluxKontextImageScale",
                "inputs": {"image": ["4", 0]},
            },
            "6": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["5", 0], "vae": ["3", 0]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": pos, "clip": [clip_ref, clip_slot]},
            },
            "8": {
                "class_type": "ReferenceLatent",
                "inputs": {"conditioning": ["7", 0], "latent": ["6", 0]},
            },
            "9": {
                "class_type": "FluxGuidance",
                "inputs": {"conditioning": ["8", 0], "guidance": g},
            },
            "10": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["7", 0]},
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "model": [model_ref, 0],
                    "seed": seed,
                    "steps": max(8, int(steps)),
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "positive": ["9", 0],
                    "negative": ["10", 0],
                    "latent_image": ["6", 0],
                    "denoise": 1.0,
                },
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {"images": ["12", 0], "filename_prefix": "flux_kontext"},
            },
        }
    )
    return graph


def build_t2i_prompt(
    *,
    positive: str,
    flux_unet: str,
    flux_clip_l: str,
    flux_t5: str,
    flux_vae: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    sampler: str = "euler",
    scheduler: str = "beta",
    loras: list[tuple[str, float, float]] | None = None,
) -> dict[str, Any]:
    """
    Flux Dev text-to-image: empty latent → KSampler denoise=1.0.

    Flux Dev is guidance-distilled (cfg=1), so the negative is a zeroed
    conditioning. LoRA strengths may be negative (e.g. SameFace Fix).
    """
    width, height = _snap(width), _snap(height)
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": flux_unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": flux_clip_l,
                "clip_name2": flux_t5,
                "type": "flux",
                "device": "default",
            },
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": flux_vae},
        },
    }
    model_ref, clip_ref, clip_slot = _inject_loras(
        graph, model_node="1", clip_node="2", loras=list(loras or [])
    )
    graph.update(
        {
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": (positive or "").strip(), "clip": [clip_ref, clip_slot]},
            },
            "5": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["4", 0]},
            },
            "6": {
                "class_type": "FluxGuidance",
                "inputs": {"conditioning": ["4", 0], "guidance": float(guidance)},
            },
            "7": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "10": {
                "class_type": "ModelSamplingFlux",
                "inputs": {
                    "model": [model_ref, 0],
                    "max_shift": 1.15,
                    "base_shift": 0.5,
                    "width": width,
                    "height": height,
                },
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["10", 0],
                    "seed": seed,
                    "steps": max(8, int(steps)),
                    "cfg": 1.0,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "positive": ["6", 0],
                    "negative": ["5", 0],
                    "latent_image": ["7", 0],
                    "denoise": 1.0,
                },
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
            },
            "13": {
                "class_type": "SaveImage",
                "inputs": {"images": ["12", 0], "filename_prefix": "flux_i2i_t2i"},
            },
        }
    )
    return graph


def _inject_loras_model_only(
    graph: dict[str, Any],
    *,
    model_node: str,
    loras: list[tuple[str, float]],
    start_id: int = 30,
) -> tuple[str, int]:
    """
    Chain LoraLoaderModelOnly after a UNET loader (Wan dual-expert path).
    Returns (final_model_node_id, next_free_id).
    """
    model_ref = model_node
    nid = start_id
    for filename, strength in loras:
        if not filename:
            continue
        key = str(nid)
        graph[key] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [model_ref, 0],
                "lora_name": filename,
                "strength_model": float(strength),
            },
        }
        model_ref = key
        nid += 1
    return model_ref, nid


def build_i2v_prompt(
    *,
    image_name: str,
    positive: str,
    negative: str | None,
    unet_high: str,
    unet_low: str,
    vae: str,
    clip: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    cfg: float,
    seed: int,
    fps: int,
    shift: float = 5.0,
    loras_high: list[tuple[str, float]] | None = None,
    loras_low: list[tuple[str, float]] | None = None,
    high_noise_fraction: float | None = None,
) -> dict[str, Any]:
    """
    Wan 2.2 I2V-A14B — dual KSamplerAdvanced (high noise → low noise).
    Start image is locked via WanImageToVideo (identity-preserving motion).
    Optional dual-stage LoraLoaderModelOnly chains on each expert.
    """
    width, height = _snap(width, 16), _snap(height, 16)
    length = max(5, int(length))
    if (length - 1) % 4 != 0:
        length = ((length - 1) // 4) * 4 + 1

    total_steps = max(8, int(steps))
    # Bias toward low-noise expert (detail / temporal stability). I2V MoE
    # boundary ~0.9 → most of the schedule should refine, not invent motion.
    # Extra motion bumps high-noise share so clothed start frames can invent anatomy.
    frac = 0.4 if high_noise_fraction is None else float(high_noise_fraction)
    frac = max(0.25, min(0.7, frac))
    split = max(1, min(total_steps - 1, int(round(total_steps * frac))))
    pos = (positive or "").strip()
    neg = (negative or "").strip() or DEFAULT_NEGATIVE_WAN

    graph: dict[str, Any] = {
        # High-noise expert
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_high, "weight_dtype": "default"},
        },
        # Low-noise expert
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet_low, "weight_dtype": "default"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip, "type": "wan", "device": "default"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": pos, "clip": ["3", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": neg, "clip": ["3", 0]},
        },
        "7": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
    }

    high_model, next_id = _inject_loras_model_only(
        graph, model_node="1", loras=list(loras_high or []), start_id=30
    )
    low_model, _ = _inject_loras_model_only(
        graph, model_node="2", loras=list(loras_low or []), start_id=next_id
    )

    graph["8"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": [high_model, 0], "shift": float(shift)},
    }
    graph["9"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"model": [low_model, 0], "shift": float(shift)},
    }
    graph["10"] = {
        "class_type": "WanImageToVideo",
        "inputs": {
            "positive": ["5", 0],
            "negative": ["6", 0],
            "vae": ["4", 0],
            "width": width,
            "height": height,
            "length": length,
            "batch_size": 1,
            "start_image": ["7", 0],
        },
    }
    # Stage 1 — high noise
    graph["11"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["8", 0],
            "add_noise": "enable",
            "noise_seed": seed,
            "steps": total_steps,
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["10", 0],
            "negative": ["10", 1],
            "latent_image": ["10", 2],
            "start_at_step": 0,
            "end_at_step": split,
            "return_with_leftover_noise": "enable",
        },
    }
    # Stage 2 — low noise
    graph["12"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            "model": ["9", 0],
            "add_noise": "disable",
            "noise_seed": 0,
            "steps": total_steps,
            "cfg": float(cfg),
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["10", 0],
            "negative": ["10", 1],
            "latent_image": ["11", 0],
            "start_at_step": split,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
        },
    }
    graph["13"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["12", 0], "vae": ["4", 0]},
    }
    graph["14"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["13", 0], "fps": fps},
    }
    graph["15"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["14", 0],
            "filename_prefix": "wan_i2v14",
            "format": "auto",
            "codec": "auto",
        },
    }
    return graph

