"""Garment / torso cloth masks for keep-outfit reshape.

Prefer Comfy CLIPSeg when available. Always keep a local fabric-aware fallback
so phone jobs never block on detector downloads.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFilter

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

from backend.ai_engine.perception.image_ops import image_to_png_bytes, load_rgb
from backend.ai_engine.perception.types import MaskResult

log = logging.getLogger(__name__)


def build_clipseg_garment_graph(
    *,
    image_name: str,
    prompt: str = "shirt, clothing, top",
    threshold: float = 0.35,
    smooth: int = 4,
    dilate: int = 2,
    blur: int = 6,
) -> dict[str, Any]:
    """CLIPSeg → MASK → RGB image saved for fetch."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadCLIPSegModels+", "inputs": {}},
        "3": {
            "class_type": "ApplyCLIPSeg+",
            "inputs": {
                "clip_seg": ["2", 0],
                "image": ["1", 0],
                "prompt": prompt,
                "threshold": float(threshold),
                "smooth": int(smooth),
                "dilate": int(dilate),
                "blur": int(blur),
            },
        },
        "4": {
            "class_type": "MaskToImage",
            "inputs": {"mask": ["3", 0]},
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": "garment_mask"},
        },
    }


def detect_garment_mask(
    image_bytes: bytes,
    *,
    settings: Any = None,
    prefer_comfy: bool = True,
) -> MaskResult:
    """White = garment / cloth region."""
    img = load_rgb(image_bytes)
    w, h = img.size
    warnings: list[str] = []

    if prefer_comfy:
        try:
            mask_png = _clipseg_via_comfy(image_bytes, settings=settings)
            if mask_png:
                m = Image.open(BytesIO(mask_png)).convert("L")
                if m.size != (w, h):
                    m = m.resize((w, h), Image.Resampling.BILINEAR)
                cov = _coverage(m)
                if cov >= 0.04:
                    return MaskResult(
                        mask_png=image_to_png_bytes(m),
                        width=w,
                        height=h,
                        source="clipseg",
                        labels=["clothing"],
                        warnings=warnings,
                        meta={"coverage": cov},
                    )
                warnings.append("clipseg_low_coverage")
        except Exception as exc:
            log.warning("CLIPSeg garment mask failed: %s", exc)
            warnings.append(f"clipseg_failed:{exc}")

    local = _local_fabric_mask(img)
    warnings.append("garment_local_fabric")
    return MaskResult(
        mask_png=image_to_png_bytes(local),
        width=w,
        height=h,
        source="local_fabric",
        labels=["clothing"],
        warnings=warnings,
        meta={"coverage": _coverage(local)},
    )


def keep_outfit_edit_mask_png(
    image_bytes: bytes,
    *,
    settings: Any = None,
    garment: Optional[MaskResult] = None,
    prefer_comfy: bool = True,
    region: str = "bust",
) -> tuple[bytes, dict[str, Any]]:
    """
    White RGB PNG = sampler noise_mask for keep-outfit reshape.

    region:
      - bust: chest core ∩ garment (default)
      - hip: lower-torso / hip core ∩ garment
      - curves: max(bust, hip) cores ∩ garment

    Soft edges. Coverage ~12–20% (hip/curves up to ~22%).
    """
    from backend.ai_engine.post.face_lock import (
        _clip_to_bust_bounds,
        _clip_to_hip_bounds,
        _ensure_edit_coverage,
        _portrait_prior,
        chest_keep_mask,
        detect_face_box,
        hip_keep_mask,
    )

    region_key = (region or "bust").strip().lower()
    if region_key in ("ass", "butt", "hips", "glute", "glutes", "thighs"):
        region_key = "hip"
    if region_key not in ("bust", "hip", "curves"):
        region_key = "bust"

    img = load_rgb(image_bytes)
    face = detect_face_box(img) or _portrait_prior(img.size)
    grow = max(5, int(min(img.size) * 0.012))
    if grow % 2 == 0:
        grow += 1

    if region_key == "hip":
        core = hip_keep_mask(img.size, face)
        core = core.filter(ImageFilter.MaxFilter(size=grow))
        core = _clip_to_hip_bounds(core, img.size, face)
        clip_fn = _clip_to_hip_bounds
        cov_hi = 0.22
    elif region_key == "curves":
        from PIL import ImageChops

        bust = chest_keep_mask(img.size, face).filter(ImageFilter.MaxFilter(size=grow))
        bust = _clip_to_bust_bounds(bust, img.size, face)
        hip = hip_keep_mask(img.size, face).filter(ImageFilter.MaxFilter(size=grow))
        hip = _clip_to_hip_bounds(hip, img.size, face)
        if np is not None:
            core = Image.fromarray(
                np.maximum(np.array(bust, dtype=np.uint8), np.array(hip, dtype=np.uint8)),
                mode="L",
            )
        else:
            core = ImageChops.lighter(bust, hip)

        def clip_fn(mask, size, face_box):  # type: ignore[no-redef]
            # Allow both bands; still zero face / extreme edges.
            w, h = size
            arr = np.array(mask, dtype=np.uint8, copy=True) if np is not None else None
            x_lo, x_hi = int(w * 0.12), int(w * 0.88)
            y_lo = int(h * 0.28)
            y_hi = int(h * 0.96)
            if arr is None:
                px = mask.load()
                for yy in range(h):
                    for xx in range(w):
                        if xx < x_lo or xx > x_hi or yy < y_lo or yy > y_hi:
                            px[xx, yy] = 0
                return mask
            arr[:, :x_lo] = 0
            arr[:, x_hi:] = 0
            arr[:y_lo, :] = 0
            arr[y_hi:, :] = 0
            return Image.fromarray(arr, mode="L")

        cov_hi = 0.24
    else:
        core = chest_keep_mask(img.size, face)
        core = core.filter(ImageFilter.MaxFilter(size=grow))
        core = _clip_to_bust_bounds(core, img.size, face)
        clip_fn = _clip_to_bust_bounds
        cov_hi = 0.28

    meta: dict[str, Any] = {"face": list(face), "region": region_key}
    if garment is None:
        garment = detect_garment_mask(
            image_bytes, settings=settings, prefer_comfy=prefer_comfy
        )
    meta["garment_source"] = garment.source
    meta["garment_warnings"] = list(garment.warnings)

    g = Image.open(BytesIO(garment.mask_png)).convert("L")
    if g.size != img.size:
        g = g.resize(img.size, Image.Resampling.BILINEAR)
    g = g.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(img.size) * 0.004))))

    if np is not None:
        c_arr = np.array(core, dtype=np.float32)
        g_arr = np.array(g, dtype=np.float32)
        g_norm = g_arr / 255.0
        if float(g_norm.mean()) < 0.02:
            out_arr = c_arr
            meta["garment_intersect"] = False
        else:
            # Keep most of the bust core even outside original fabric so volume
            # can grow and the same cloth can stretch tighter.
            factor = np.clip(0.70 + 0.30 * g_norm, 0.0, 1.0)
            out_arr = c_arr * factor
            meta["garment_intersect"] = True
        out = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8), mode="L")
    else:
        gate = g.point(lambda p: 255 if p > 40 else 0)
        out = Image.composite(core, Image.new("L", img.size, 0), gate)
        meta["garment_intersect"] = True

    out = clip_fn(out, img.size, face)
    cov_lo = 0.16 if region_key == "bust" else 0.12
    out = _ensure_edit_coverage(out, img.size, face, lo=cov_lo, hi=cov_hi, clip=clip_fn)
    soft = max(3, int(min(img.size) * 0.008))
    out = out.filter(ImageFilter.GaussianBlur(radius=soft))
    out = _ensure_edit_coverage(out, img.size, face, lo=cov_lo, hi=cov_hi, clip=clip_fn)
    meta["coverage"] = _coverage(out)

    rgb = Image.merge("RGB", (out, out, out))
    return image_to_png_bytes(rgb), meta


def soft_restore_outside_edit(
    original_bytes: bytes,
    edited_bytes: bytes,
    edit_mask_rgb_or_l: bytes,
    *,
    garment_mask_png: Optional[bytes] = None,
) -> bytes:
    """Soft composite edit core; feather strap/face restore from the start photo."""
    from backend.ai_engine.post.face_lock import (
        _face_mask,
        _portrait_prior,
        detect_face_box,
        garment_restore_mask,
    )

    orig = Image.open(BytesIO(original_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_bytes)).convert("RGB")
    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)

    keep = Image.open(BytesIO(edit_mask_rgb_or_l)).convert("L")
    if keep.size != orig.size:
        keep = keep.resize(orig.size, Image.Resampling.BILINEAR)
    keep = keep.filter(
        ImageFilter.GaussianBlur(radius=max(2, int(min(orig.size) * 0.004)))
    )
    out = Image.composite(edit, orig, keep)

    face = detect_face_box(orig) or _portrait_prior(orig.size)
    # Light strap/hand restore only — do not paste original chest fabric back
    # (that undoes tighter cloth + volume).
    straps = garment_restore_mask(orig.size, face)
    straps = straps.point(lambda p: int(p * 0.35))
    straps = straps.filter(
        ImageFilter.GaussianBlur(radius=max(2, int(min(orig.size) * 0.005)))
    )
    out = Image.composite(orig, out, straps)
    out = Image.composite(orig, out, _face_mask(orig.size, face))
    return image_to_png_bytes(out)


def _clipseg_via_comfy(image_bytes: bytes, *, settings: Any) -> Optional[bytes]:
    import asyncio

    from backend.comfy_client import ComfyClient
    from backend.config import get_settings

    cfg = settings or get_settings()

    async def _run() -> bytes:
        client = ComfyClient(cfg)
        name = await client._upload_image(image_bytes)
        try:
            graph = build_clipseg_garment_graph(image_name=name)
            data, _ctype = await client._run_and_fetch(
                graph, prefer=("images",), input_name=name
            )
            return data
        finally:
            try:
                await client._full_scrub(input_name=name)
            except Exception:
                pass

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    # Already inside an event loop (edit runner) — isolate in a thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run())).result(timeout=180)


def _local_fabric_mask(img: Image.Image) -> Image.Image:
    """Geometry + non-skin fabric prior in the torso band."""
    from backend.ai_engine.post.face_lock import _portrait_prior, detect_face_box

    w, h = img.size
    face = detect_face_box(img) or _portrait_prior(img.size)
    fx, fy, fw, fh = face
    chin = min(max(fy + fh * 1.0, h * 0.28), h * 0.50)
    mid = min(max(fx + fw / 2.0, w * 0.35), w * 0.65)

    if np is None:
        m = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(m)
        draw.ellipse(
            [mid - w * 0.28, chin, mid + w * 0.28, min(h * 0.78, chin + h * 0.42)],
            fill=255,
        )
        return m.filter(ImageFilter.GaussianBlur(radius=max(4, int(min(w, h) * 0.01))))

    arr = np.array(img, dtype=np.float32)
    x1, y1 = max(0, fx), max(0, fy)
    x2, y2 = min(w, fx + fw), min(h, fy + fh)
    face_pix = arr[y1:y2, x1:x2].reshape(-1, 3)
    if face_pix.size == 0:
        skin = np.array([180.0, 140.0, 120.0], dtype=np.float32)
    else:
        skin = np.median(face_pix, axis=0)

    yy, xx = np.mgrid[0:h, 0:w]
    cy = chin + h * 0.18
    rx = w * 0.30
    ry = h * 0.22
    torso = ((xx - mid) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2 <= 1.0
    torso &= yy >= chin
    torso &= yy <= h * 0.78
    torso &= xx >= w * 0.18
    torso &= xx <= w * 0.82

    dist = np.linalg.norm(arr - skin[None, None, :], axis=2)
    fabric = torso & (dist > 28.0)
    fabric |= torso & (dist > 18.0) & (arr.mean(axis=2) < skin.mean() - 12)

    out = Image.fromarray((fabric.astype(np.uint8) * 255), mode="L")
    out = out.filter(ImageFilter.MaxFilter(5))
    out = out.filter(ImageFilter.MinFilter(5))
    out = out.filter(ImageFilter.GaussianBlur(radius=max(3, int(min(w, h) * 0.008))))
    return out


def _coverage(mask: Image.Image) -> float:
    if np is None:
        hist = mask.histogram()
        return sum(hist[128:]) / float(max(1, mask.size[0] * mask.size[1]))
    return float((np.array(mask) > 127).mean())
