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
    filename_prefix: str = "wan_klein",
) -> dict[str, Any]:
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
    if ref_image_name:
        g["30"] = {"class_type": "LoadImage", "inputs": {"image": ref_image_name}}
        g["31"] = {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": ["30", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1},
        }
        g["32"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["31", 0], "vae": ["3", 0]}}
        g["33"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": pos, "latent": ["32", 0]}}
        pos = ["33", 0]

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
    return g


def edit_size(width: int, height: int, megapixels: float = 1.0) -> tuple[int, int]:
    """Output size for an edit: input aspect at ~1MP, multiples of 16."""
    scale = (megapixels * 1024 * 1024 / max(1, width * height)) ** 0.5
    return max(256, int(width * scale) // 16 * 16), max(256, int(height * scale) // 16 * 16)
