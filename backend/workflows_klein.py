"""FLUX.2 Klein 9B (distilled, 4-step) graphs: text-to-image and single-reference edit.

Edit = the input image VAE-encoded into a ReferenceLatent on the positive conditioning,
sampled from an empty latent at the input's aspect (no denoise strength involved).
"""

from __future__ import annotations

from typing import Any, Optional

KLEIN_UNET = "flux-2-klein-9b-Q8_0.gguf"
KLEIN_CLIP = "qwen_3_8b_fp8mixed.safetensors"
KLEIN_VAE = "flux2-vae.safetensors"


def build_klein_prompt(
    *,
    positive: str,
    width: int,
    height: int,
    seed: int,
    steps: int = 4,
    unet: str = KLEIN_UNET,
    clip: str = KLEIN_CLIP,
    vae: str = KLEIN_VAE,
    loras: Optional[list[tuple[str, float]]] = None,
    ref_image_name: Optional[str] = None,
    ref_image_names: Optional[list[str]] = None,
    face_ref_index: int = 0,
    face_fix: Optional[dict[str, Any]] = None,
    face_mask_image_name: Optional[str] = None,
    filename_prefix: str = "wan_klein",
) -> dict[str, Any]:
    """References are chained in order (prompt calls them image 1, image 2, ...; max 4).
    ``face_fix`` ({"prompt", "denoise", "steps"}) re-renders the largest detected face at full
    resolution with the same model and LoRAs, guided by reference ``face_ref_index``.
    ``face_mask_image_name`` (white = her area) limits the face fix to faces inside it."""
    refs = list(ref_image_names or ([ref_image_name] if ref_image_name else []))[:4]
    loader = "UnetLoaderGGUF" if unet.endswith(".gguf") else "UNETLoader"
    unet_inputs: dict[str, Any] = {"unet_name": unet}
    if loader == "UNETLoader":
        unet_inputs["weight_dtype"] = "default"
    g: dict[str, Any] = {
        "1": {"class_type": loader, "inputs": unet_inputs},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
    }
    model: list[Any] = ["1", 0]
    for i, (name, strength) in enumerate(loras or []):
        nid = str(20 + i)
        g[nid] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model, "lora_name": name, "strength_model": float(strength)},
        }
        model = [nid, 0]

    g["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["2", 0]}}
    g["6"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}}
    pos: list[Any] = ["5", 0]
    for k, name in enumerate(refs):
        load, scale, enc, ref = (str(30 + 4 * k + j) for j in range(4))
        g[load] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g[scale] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": [load, 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1},
        }
        g[enc] = {"class_type": "VAEEncode", "inputs": {"pixels": [scale, 0], "vae": ["3", 0]}}
        g[ref] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": [enc, 0]}}
        pos = [ref, 0]

    g.update({
        "7": {"class_type": "CFGGuider", "inputs": {"model": model, "positive": pos, "negative": ["6", 0], "cfg": 1.0}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "Flux2Scheduler", "inputs": {"steps": int(steps), "width": width, "height": height}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["11", 0]},
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": filename_prefix}},
    })
    if face_fix is not None:
        within = None
        if face_mask_image_name:
            g["61"] = {"class_type": "LoadImageMask", "inputs": {"image": face_mask_image_name, "channel": "red"}}
            within = ["61", 0]
        face_latent = [str(32 + 4 * face_ref_index), 0] if 0 <= face_ref_index < len(refs) else None
        _add_face_fix(g, model=model, seed=seed, ref_latent=face_latent, within=within, **face_fix)
    return g


def _add_face_fix(
    g: dict[str, Any],
    *,
    model: list[Any],
    seed: int,
    ref_latent: Optional[list[Any]],
    within: Optional[list[Any]],
    prompt: str,
    denoise: float = 0.5,
    steps: int = 8,
    detector: str = "bbox/face_yolov8m.pt",
) -> None:
    g["50"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": detector}}
    g["51"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}}
    pos: list[Any] = ["51", 0]
    if ref_latent:
        g["52"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": ref_latent}}
        pos = ["52", 0]
    g["53"] = {
        "class_type": "BboxDetectorSEGS",
        "inputs": {
            "bbox_detector": ["50", 0], "image": ["13", 0], "threshold": 0.5,
            "dilation": 16, "crop_factor": 2.2, "drop_size": 24, "labels": "all",
        },
    }
    segs: list[Any] = ["53", 0]
    if within:
        # Crowded scene: only faces inside her area are candidates.
        g["56"] = {"class_type": "ImpactSegsAndMask", "inputs": {"segs": segs, "mask": within}}
        segs = ["56", 0]
    # Only the largest face is hers; background people keep their own faces.
    g["54"] = {
        "class_type": "ImpactSEGSOrderedFilter",
        "inputs": {"segs": segs, "target": "area(=w*h)", "order": True, "take_start": 0, "take_count": 1},
    }
    g["55"] = {
        "class_type": "DetailerForEach",
        "inputs": {
            "image": ["13", 0], "segs": ["54", 0], "model": model, "clip": ["2", 0], "vae": ["3", 0],
            "guide_size": 1024, "guide_size_for": True, "max_size": 1024,
            "seed": seed, "steps": int(steps), "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
            "positive": pos, "negative": ["6", 0], "denoise": float(denoise),
            "feather": 12, "noise_mask": True, "force_inpaint": True, "wildcard": "", "cycle": 1,
        },
    }
    g["14"]["inputs"]["images"] = ["55", 0]


def edit_size(width: int, height: int, megapixels: float = 1.0) -> tuple[int, int]:
    """Output size for an edit: input aspect at ~1MP, multiples of 16."""
    scale = (megapixels * 1024 * 1024 / max(1, width * height)) ** 0.5
    return max(256, int(width * scale) // 16 * 16), max(256, int(height * scale) // 16 * 16)
