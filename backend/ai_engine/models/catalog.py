"""Bootstrap model catalog + planner.default_model slot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.ai_engine.models.manager import ModelRecord, manager

# Skip disk probes for logical/pip/builtin names (not Comfy weight files).
_SKIP_DISK = frozenset(
    {
        "u2net",
        "full_frame",
        "RMBG-2.0",
        "Florence-2-large",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    }
)

# Incomplete HuggingFace downloads must not count as installed.
_MIN_WEIGHT_BYTES = 50 * 1024 * 1024  # 50 MiB
_MIN_LORA_BYTES = 1 * 1024 * 1024  # 1 MiB — unlock LoRAs can be ~19MB


def _comfy_model_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        from backend.config import get_settings

        cdir = (get_settings().comfyui_dir or "").strip()
        if cdir:
            base = Path(cdir)
            roots.append(base / "models")
            yaml_path = base / "extra_model_paths.yaml"
            if yaml_path.is_file():
                text = yaml_path.read_text(encoding="utf-8", errors="ignore")
                for m in re.finditer(r"base_path:\s*(.+)", text):
                    bp = m.group(1).strip().strip("\"'")
                    if bp:
                        roots.append(Path(bp) / "models")
    except Exception:
        pass
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[Path] = []
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _find_weight(filename: str) -> Path | None:
    if not filename or filename in _SKIP_DISK:
        return None
    # Allow subdirectory paths under a model folder (e.g. loras/miscellaneous/foo.safetensors).
    rel = filename.replace("\\", "/")
    if "/" in rel and not rel.startswith("Qwen"):
        for root in _comfy_model_roots():
            for sub in ("loras", "controlnet", "diffusion_models", "unet", ""):
                p = (root / sub / rel) if sub else (root / rel)
                try:
                    min_b = _MIN_LORA_BYTES if sub == "loras" else _MIN_WEIGHT_BYTES
                    if p.is_file() and p.stat().st_size >= min_b:
                        return p
                except OSError:
                    continue
        return None
    subdirs = (
        "diffusion_models",
        "unet",
        "checkpoints",
        "vae",
        "text_encoders",
        "clip",
        "upscale_models",
        "loras",
        "controlnet",
        "pulid",
        "insightface",
        "sams",
        "sam",
        "sam2",
        "groundingdino",
        "",
    )
    for root in _comfy_model_roots():
        for sub in subdirs:
            p = (root / sub / filename) if sub else (root / filename)
            try:
                min_b = _MIN_LORA_BYTES if sub == "loras" else _MIN_WEIGHT_BYTES
                if p.is_file() and p.stat().st_size >= min_b:
                    return p
                # Misc staged LoRAs
                if sub == "loras":
                    misc = root / "loras" / "miscellaneous" / Path(filename).name
                    if misc.is_file() and misc.stat().st_size >= min_b:
                        return misc
            except OSError:
                continue
    return None


def refresh_installed_from_disk() -> None:
    """Mark backbone/video (and other file-backed) models from Comfy model dirs.

    When ``COMFYUI_DIR`` is unset (typical Render → remote tunnel), there are no
    local roots to scan — do **not** flip backbones to missing; availability is
    refreshed from ComfyUI over HTTP instead.
    """
    roots = _comfy_model_roots()
    for mid, rec in list(manager._models.items()):  # noqa: SLF001
        fn = rec.filename
        if not fn or fn in _SKIP_DISK or "/" in fn:
            continue
        if not fn.endswith((".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".bin")):
            continue
        found = _find_weight(fn) if roots else None
        if found:
            rec.status = "installed"
            rec.local_path = str(found)
            try:
                rec.disk_mb = max(1, found.stat().st_size // (1024 * 1024))
            except OSError:
                pass
        elif roots and mid.startswith(("backbone.", "video.", "lora.", "controlnet.")) and rec.role in (
            "backbone",
            "video",
            "video_post",
            "lora",
            "control",
        ):
            # Honest status only when we could actually probe disk
            rec.status = "missing"
            rec.local_path = None


def apply_comfy_filenames(filenames: set[str]) -> int:
    """Mark catalog entries installed when ComfyUI lists their weight filename."""
    if not filenames:
        return 0
    marked = 0
    for rec in list(manager._models.values()):  # noqa: SLF001 — same package
        fn = rec.filename
        if not fn or fn in _SKIP_DISK:
            continue
        if fn in filenames:
            if rec.status != "installed":
                marked += 1
            rec.status = "installed"
    return marked


async def refresh_installed_from_comfy(settings: Any | None = None) -> int:
    """Query remote ComfyUI loader menus and mark matching weights installed."""
    try:
        import httpx
        from backend.config import get_settings

        cfg = settings or get_settings()
        base = (cfg.comfyui_url or "").rstrip("/")
        if not base:
            return 0
        names: set[str] = set()
        async with httpx.AsyncClient(timeout=20.0) as client:
            # Full object_info is large; hit the loaders we care about.
            for node in (
                "UNETLoader",
                "CheckpointLoaderSimple",
                "VAELoader",
                "LoraLoader",
                "CLIPLoader",
                "DualCLIPLoader",
                "ControlNetLoader",
            ):
                try:
                    r = await client.get(f"{base}/object_info/{node}")
                    if r.status_code != 200:
                        continue
                    payload = r.json() or {}
                    node_info = payload.get(node) or payload
                    required = ((node_info.get("input") or {}).get("required")) or {}
                    for _key, spec in required.items():
                        if not isinstance(spec, list) or not spec:
                            continue
                        choices = spec[0]
                        if isinstance(choices, list):
                            for c in choices:
                                if isinstance(c, str) and c.endswith(
                                    (".safetensors", ".gguf", ".ckpt", ".pth", ".pt", ".bin")
                                ):
                                    names.add(c)
                except Exception:
                    continue
        return apply_comfy_filenames(names)
    except Exception:
        return 0


def bootstrap_models() -> None:
    """Register known logical models. Install status is optimistic for Phase 1."""

    def add(
        model_id: str,
        *,
        filename: str | None = None,
        role: str = "misc",
        status: str = "installed",
        replacement_candidates: list[str] | None = None,
        **kwargs,
    ) -> None:
        manager.register_model(
            ModelRecord(
                model_id=model_id,
                filename=filename,
                role=role,
                status=status,  # type: ignore[arg-type]
                replacement_candidates=replacement_candidates or [],
                **kwargs,
            )
        )

    # Backbones (filenames match backend.config defaults / .env.example)
    add(
        "backbone.flux_dev_fp8",
        filename="flux1-dev-fp8.safetensors",
        role="backbone",
        replacement_candidates=["backbone.flux_kontext_dev_fp8"],
    )
    add(
        "backbone.flux_kontext_dev_fp8",
        filename="flux1-dev-kontext_fp8_scaled.safetensors",
        role="backbone",
        status="missing",  # not required for Phase 1 legacy path
        replacement_candidates=["backbone.flux_dev_fp8"],
    )
    add(
        "backbone.flux_fill",
        filename=None,
        role="backbone",
        status="missing",
        replacement_candidates=["backbone.flux_dev_fp8"],
    )
    add(
        "controlnet.flux_union_pro_fp8",
        filename="flux_shakker_labs_union_pro-fp8_e4m3fn.safetensors",
        role="control",
        status="missing",
        download_source="https://huggingface.co/Kijai/flux-fp8",
        replacement_candidates=["controlnet.flux_union_pro"],
    )
    add(
        "controlnet.flux_union_pro",
        filename="flux_controlnet_union_pro.safetensors",
        role="control",
        status="missing",
        download_source="https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro",
    )

    try:
        from backend.ai_engine.runtime_overlay import load_private_module

        extra = load_private_module("catalog_loras")
        if extra is not None and hasattr(extra, "register"):
            extra.register(add)
    except Exception:
        pass

    add(
        "video.wan22_i2v_high_fp8",
        filename="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        role="video",
        replacement_candidates=["video.wan22_ti2v_5b"],
    )
    add(
        "video.wan22_i2v_low_fp8",
        filename="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        role="video",
    )
    add(
        "video.wan22_ti2v_5b",
        filename="wan2.2_ti2v_5B_fp16.safetensors",
        role="video",
        status="installed",
    )

    add(
        "identity.pulid_flux",
        filename="pulid_flux_v0.9.1.safetensors",
        role="identity",
        status="installed",
    )
    add(
        "upscale.ultrasharp_4x",
        filename="4x-UltraSharp.pth",
        role="upscale",
        status="installed",
    )
    add(
        "video.rife",
        filename="rife47.pth",
        role="video_post",
        status="missing",
        download_source="https://github.com/hzwer/Practical-RIFE",
    )
    add(
        "video.lightx2v_lora",
        filename="lightx2v_wan_lora.safetensors",
        role="video",
        status="missing",
        download_source="LightX2V Wan LoRA (optional Draft accel)",
    )

    # Perception stack (Phase 4)
    add(
        "grounding.dino",
        filename="GroundingDINO_SwinT_OGC.pth",
        role="grounding",
        status="missing",
        download_source="https://github.com/IDEA-Research/GroundingDINO",
        replacement_candidates=["grounding.florence2"],
    )
    add(
        "grounding.florence2",
        filename="Florence-2-large",
        role="grounding",
        status="missing",
        download_source="https://huggingface.co/microsoft/Florence-2-large",
    )
    add(
        "seg.sam2",
        filename="sam2_hiera_base_plus.safetensors",
        role="seg",
        status="missing",
        download_source="https://github.com/facebookresearch/sam2",
        replacement_candidates=["seg.sam"],
    )
    add(
        "seg.sam",
        filename="sam_vit_b_01ec64.pth",
        role="seg",
        status="missing",
    )
    add(
        "matting.birefnet",
        filename="BiRefNet-general-epoch_244.pth",
        role="matting",
        status="missing",
        download_source="https://github.com/ZhengPeng7/BiRefNet",
        replacement_candidates=["matting.rembg"],
    )
    add(
        "matting.rmbg2",
        filename="RMBG-2.0",
        role="matting",
        status="missing",
        replacement_candidates=["matting.rembg"],
    )
    add(
        "matting.rembg",
        filename="u2net",
        role="matting",
        status="missing",
        download_source="pip:rembg",
    )
    add(
        "matting.heuristic",
        filename="full_frame",
        role="matting",
        status="installed",
        version="builtin",
    )
    try:
        import rembg  # noqa: F401

        rec = manager.get("matting.rembg")
        if rec:
            rec.status = "installed"
    except ImportError:
        pass

    # Planner VLM — slot indirection (do not hardcode in planner code)
    add(
        "vlm.qwen25_vl_7b",
        filename="Qwen/Qwen2.5-VL-7B-Instruct",
        role="vlm",
        status="missing",
        replacement_candidates=["vlm.qwen25_vl_3b"],
        version="2.5-7b",
        download_source="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    )
    add(
        "vlm.qwen25_vl_3b",
        filename="Qwen/Qwen2.5-VL-3B-Instruct",
        role="vlm",
        status="missing",
        version="2.5-3b",
        download_source="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
    )
    manager.set_slot("planner.default_model", "vlm.qwen25_vl_7b")

    try:
        from backend.config import get_settings

        path = (get_settings().ai_engine_vlm_model_path or "").strip()
        if path:
            rec = manager.get("vlm.qwen25_vl_7b")
            if rec:
                rec.status = "installed"
                rec.local_path = path
                rec.filename = path
    except Exception:
        pass

    refresh_installed_from_disk()
