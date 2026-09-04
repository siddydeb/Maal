"""Paste the start-image face back after a full-frame edit.

Region-limited edits: GPU may only change a torso-core region (sampler mask).
Strap / hem / face restore happens in post, not on the noise_mask.
PuLID is applied in the Flux img2img graph when pulid_file is set.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


# Guard band: keep-core starts below this fraction of image height.
_CHEST_GUARD = 0.40


def restore_original_face(original_bytes: bytes, edited_bytes: bytes) -> bytes:
    orig = Image.open(BytesIO(original_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_bytes)).convert("RGB")
    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)

    box = detect_face_box(orig) or _portrait_prior(orig.size)
    mask = _face_mask(orig.size, box)
    out = Image.composite(orig, edit, mask)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def bust_inpaint_mask_png(
    image_bytes: bytes,
    *,
    settings=None,
    prefer_comfy: bool = True,
) -> bytes:
    """White PNG mask covering the inner edit region — for Flux img2img noise_mask.

    Uses garment detection ∩ bust core when available (CLIPSeg / local fabric).
    Target coverage is 12–20%. Strap restore stays in post only.
    """
    try:
        from backend.ai_engine.perception.garment_mask import keep_outfit_edit_mask_png

        mask_png, _meta = keep_outfit_edit_mask_png(
            image_bytes,
            settings=settings,
            prefer_comfy=prefer_comfy,
            region="bust",
        )
        return mask_png
    except Exception:
        pass
    orig = Image.open(BytesIO(image_bytes)).convert("RGB")
    face = detect_face_box(orig) or _portrait_prior(orig.size)
    mask = chest_keep_mask(orig.size, face)
    grow = max(5, int(min(orig.size) * 0.012))
    if grow % 2 == 0:
        grow += 1
    mask = mask.filter(ImageFilter.MaxFilter(size=grow))
    mask = _clip_to_bust_bounds(mask, orig.size, face)
    mask = _ensure_edit_coverage(mask, orig.size, face, lo=0.16, hi=0.28)
    soft = max(3, int(min(orig.size) * 0.008))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=soft))
    mask = _ensure_edit_coverage(mask, orig.size, face, lo=0.16, hi=0.28)
    rgb = Image.merge("RGB", (mask, mask, mask))
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


def hip_inpaint_mask_png(
    image_bytes: bytes,
    *,
    settings=None,
    prefer_comfy: bool = True,
) -> bytes:
    """White PNG mask for lower-body reshape (hips) — Flux img2img noise_mask."""
    try:
        from backend.ai_engine.perception.garment_mask import keep_outfit_edit_mask_png

        mask_png, _meta = keep_outfit_edit_mask_png(
            image_bytes,
            settings=settings,
            prefer_comfy=prefer_comfy,
            region="hip",
        )
        return mask_png
    except Exception:
        pass
    orig = Image.open(BytesIO(image_bytes)).convert("RGB")
    face = detect_face_box(orig) or _portrait_prior(orig.size)
    mask = hip_keep_mask(orig.size, face)
    grow = max(5, int(min(orig.size) * 0.012))
    if grow % 2 == 0:
        grow += 1
    mask = mask.filter(ImageFilter.MaxFilter(size=grow))
    mask = _clip_to_hip_bounds(mask, orig.size, face)
    mask = _ensure_edit_coverage(
        mask, orig.size, face, lo=0.12, hi=0.22, clip=_clip_to_hip_bounds
    )
    soft = max(3, int(min(orig.size) * 0.008))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=soft))
    rgb = Image.merge("RGB", (mask, mask, mask))
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()

def restore_outside_chest(
    original_bytes: bytes,
    edited_bytes: bytes,
    *,
    edit_mask_png: bytes | None = None,
    garment_mask_png: bytes | None = None,
) -> bytes:
    """Original everywhere except the inner GPU edit region (soft edges)."""
    if edit_mask_png:
        try:
            from backend.ai_engine.perception.garment_mask import soft_restore_outside_edit

            return soft_restore_outside_edit(
                original_bytes,
                edited_bytes,
                edit_mask_png,
                garment_mask_png=garment_mask_png,
            )
        except Exception:
            pass
    orig = Image.open(BytesIO(original_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_bytes)).convert("RGB")
    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)

    face = detect_face_box(orig) or _portrait_prior(orig.size)
    keep = chest_keep_mask(orig.size, face)
    keep = keep.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(orig.size) * 0.006))))
    out = Image.composite(edit, orig, keep)
    straps = garment_restore_mask(orig.size, face)
    straps = straps.point(lambda p: int(p * 0.85))
    straps = straps.filter(ImageFilter.GaussianBlur(radius=max(2, int(min(orig.size) * 0.005))))
    out = Image.composite(orig, out, straps)
    out = Image.composite(orig, out, _face_mask(orig.size, face))
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def chest_keep_mask(
    size: tuple[int, int],
    face_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """White = GPU edit core. Black = must stay the start photo."""
    w, h = size
    if face_box is None:
        face_box = _portrait_prior(size)
    fx, fy, fw, fh = face_box
    mid = min(max(fx + fw / 2.0, w * 0.38), w * 0.62)
    chin = min(max(fy + fh * 1.02, h * 0.30), h * 0.48)
    y1 = chin + h * 0.01
    y2 = min(h * 0.84, chin + h * 0.56)
    half = min(fw * 1.85, w * 0.46)
    x1 = max(w * 0.12, mid - half)
    x2 = min(w * 0.88, mid + half)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cy = (y1 + y2) / 2.0
    ry = max(6.0, (y2 - y1) / 2.0)
    rx = max(6.0, (x2 - x1) * 0.52)
    dx = (x2 - x1) * 0.30
    for cx in (mid - dx, mid + dx):
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    radius = max(3, int(min(w, h) * 0.007))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=radius))
    mask = _clip_to_bust_bounds(mask, size, face_box)
    return mask


def _clip_to_bust_bounds(
    mask: Image.Image,
    size: tuple[int, int],
    face_box: tuple[int, int, int, int],
) -> Image.Image:
    """Hard-zero sleeves, outer shirt, face, hem, and background."""
    w, h = size
    fx, fy, fw, fh = face_box
    chin = min(max(fy + fh * 1.00, h * 0.28), h * 0.50)
    arr = np.array(mask, dtype=np.uint8, copy=True) if np is not None else None
    if arr is None:
        px = mask.load()
        x_lo, x_hi = int(w * 0.12), int(w * 0.88)
        y_lo, y_hi = int(chin), int(h * 0.84)
        for yy in range(h):
            for xx in range(w):
                if xx < x_lo or xx > x_hi or yy < y_lo or yy > y_hi:
                    px[xx, yy] = 0
        return mask
    x_lo, x_hi = int(w * 0.12), int(w * 0.88)
    y_lo, y_hi = int(chin), int(h * 0.84)
    arr[:, :x_lo] = 0
    arr[:, x_hi:] = 0
    arr[:y_lo, :] = 0
    arr[y_hi:, :] = 0
    return Image.fromarray(arr, mode="L")


def _clip_to_hip_bounds(
    mask: Image.Image,
    size: tuple[int, int],
    face_box: tuple[int, int, int, int],
) -> Image.Image:
    """Keep only the lower torso / hip band; zero face, chest, and feet edge."""
    w, h = size
    fx, fy, fw, fh = face_box
    # Start below mid-torso so chest reshape LoRAs do not fight hip edits.
    y_lo = int(min(max(fy + fh * 1.55, h * 0.48), h * 0.62))
    y_hi = int(min(h * 0.96, max(y_lo + int(h * 0.22), h * 0.90)))
    x_lo, x_hi = int(w * 0.12), int(w * 0.88)
    arr = np.array(mask, dtype=np.uint8, copy=True) if np is not None else None
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


def hip_keep_mask(
    size: tuple[int, int],
    face_box: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Soft elliptical prior over hips / seat (portrait or 3/4 crop)."""
    w, h = size
    if face_box is None:
        fx, fy, fw, fh = _portrait_prior(size)
    else:
        fx, fy, fw, fh = face_box
    mid = min(max(fx + fw / 2.0, w * 0.35), w * 0.65)
    cy = min(max(fy + fh * 2.05, h * 0.62), h * 0.78)
    rx = min(w * 0.28, max(w * 0.18, fw * 0.55))
    ry = min(h * 0.18, max(h * 0.12, fh * 0.55))
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([mid - rx, cy - ry, mid + rx, cy + ry], fill=255)
    # Slightly wider lower lobe for seat / upper thigh.
    draw.ellipse(
        [mid - rx * 1.05, cy + ry * 0.15, mid + rx * 1.05, cy + ry * 1.35],
        fill=255,
    )
    radius = max(2, int(min(w, h) * 0.008))
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def _mask_coverage(mask: Image.Image) -> float:
    if np is None:
        hist = mask.histogram()
        on = sum(hist[128:])
        return on / float(max(1, mask.size[0] * mask.size[1]))
    arr = np.array(mask, dtype=np.uint8)
    return float((arr > 127).mean())


def _ensure_edit_coverage(
    mask: Image.Image,
    size: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    lo: float = 0.12,
    hi: float = 0.20,
    clip=None,
) -> Image.Image:
    """Grow or shrink the sampler mask into the coverage band."""
    clip_fn = clip or _clip_to_bust_bounds
    cur = mask
    cov = _mask_coverage(cur)
    odd = 5
    while cov < lo and odd <= 41:
        cur = cur.filter(ImageFilter.MaxFilter(size=odd))
        cur = clip_fn(cur, size, face_box)
        cov = _mask_coverage(cur)
        odd += 4
    odd = 3
    while cov > hi and odd <= 21:
        cur = cur.filter(ImageFilter.MinFilter(size=odd))
        cur = clip_fn(cur, size, face_box)
        cov = _mask_coverage(cur)
        odd += 2
    return cur

def _bust_lobes(
    size: tuple[int, int],
    face_box: tuple[int, int, int, int] | None,
) -> list[tuple[float, float, float, float]]:
    w, h = size
    if face_box is None:
        fx, fy, fw, fh = _portrait_prior(size)
    else:
        fx, fy, fw, fh = face_box
    mid = min(max(fx + fw / 2.0, w * 0.35), w * 0.65)
    chin = min(max(fy + fh * 0.98, h * 0.28), h * 0.50)
    cy = min(max(chin + h * 0.14, h * 0.46), h * 0.60)
    rx = min(w * 0.24, max(w * 0.16, fw * 0.46))
    ry = min(h * 0.135, max(h * 0.095, fh * 0.28))
    dx = max(w * 0.115, fw * 0.36)
    return [(mid - dx, cy, rx, ry), (mid + dx, cy, rx, ry)]


def garment_restore_mask(
    size: tuple[int, int],
    face_box: tuple[int, int, int, int] | None = None,
) -> Image.Image:
    """White = stamp original straps, outer shirt, hem, and hands back."""
    w, h = size
    if face_box is None:
        fx, fy, fw, fh = _portrait_prior(size)
    else:
        fx, fy, fw, fh = face_box
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    mid = fx + fw / 2.0
    strap_y = min(max(fy + fh * 1.02, h * 0.36), h * 0.46)
    srx = max(8, int(w * 0.08))
    sry = max(6, int(h * 0.04))
    draw.ellipse(
        [mid - fw * 0.52 - srx, strap_y - sry, mid - fw * 0.52 + srx, strap_y + sry],
        fill=255,
    )
    draw.ellipse(
        [mid + fw * 0.52 - srx, strap_y - sry, mid + fw * 0.52 + srx, strap_y + sry],
        fill=255,
    )
    # Collar sides only — leave the center so a seam line can land.
    neck_y = min(max(fy + fh * 1.06, h * 0.36), h * 0.46)
    nw = max(8, int(fw * 0.22))
    nh = max(6, int(h * 0.028))
    for side in (-1.0, 1.0):
        cx = mid + side * fw * 0.40
        draw.ellipse([cx - nw, neck_y - nh, cx + nw, neck_y + nh], fill=255)
    for cx, cy, rx, ry in (
        (int(w * 0.11), int(h * 0.50), int(w * 0.20), int(h * 0.16)),
        (int(w * 0.89), int(h * 0.50), int(w * 0.20), int(h * 0.16)),
        # Raised hands / upper arms — warp must not smear the sleeve or handheld object.
        (int(w * 0.18), int(h * 0.32), int(w * 0.17), int(h * 0.13)),
        (int(w * 0.82), int(h * 0.32), int(w * 0.17), int(h * 0.13)),
        (int(w * 0.14), int(h * 0.42), int(w * 0.14), int(h * 0.12)),
        (int(w * 0.86), int(h * 0.42), int(w * 0.14), int(h * 0.12)),
        # Draped outer shirts / hanging sleeves on both inputs.
        (int(w * 0.08), int(h * 0.48), int(w * 0.18), int(h * 0.22)),
        (int(w * 0.92), int(h * 0.48), int(w * 0.18), int(h * 0.22)),
    ):
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    hem_y = min(max(fy + fh * 1.48, h * 0.60), h * 0.72)
    draw.ellipse(
        [mid - fw * 0.50, hem_y - int(h * 0.06), mid + fw * 0.50, hem_y + int(h * 0.05)],
        fill=255,
    )
    hx, hy = int(w * 0.48), int(h * 0.88)
    hrx, hry = int(w * 0.14), int(h * 0.07)
    draw.ellipse([hx - hrx, hy - hry, hx + hrx, hy + hry], fill=255)
    radius = max(2, int(min(w, h) * 0.005))
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def _bloat_bust(
    arr: "np.ndarray",
    size: tuple[int, int],
    face_box: tuple[int, int, int, int],
    *,
    strength: float,
) -> "np.ndarray":
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx = xx.copy()
    sy = yy.copy()
    best = np.full((h, w), 9.0, dtype=np.float32)
    lobes = _bust_lobes(size, face_box)
    mid_x = float(0.5 * (lobes[0][0] + lobes[1][0])) if len(lobes) >= 2 else float(w) * 0.5
    for cx, cy, rx, ry in lobes:
        rx = max(float(rx), 1.0)
        ry = max(float(ry), 1.0)
        nx = (xx - cx) / rx
        ny = (yy - cy) / ry
        r2 = nx * nx + ny * ny
        closer = (r2 < 1.0) & (r2 < best)
        t = np.clip(1.0 - r2, 0.0, 1.0)
        fall = t * t
        # Widen the inner region slightly; keep vertical pull off the hem.
        fx = 1.0 - strength * fall
        fy = 1.0 - (strength * 0.52) * fall
        # Mild center-line gather for deeper cleavage under cloth.
        toward = 1.0 - (strength * 0.18) * fall
        sx = np.where(
            closer,
            cx + (xx - cx) * fx * toward + (mid_x - cx) * (1.0 - toward) * 0.35,
            sx,
        )
        sy = np.where(closer, cy + (yy - cy) * fy, sy)
        best = np.where(closer, r2, best)
    return _sample_bilinear(arr, sx, sy)


def _sample_bilinear(arr: "np.ndarray", sx: "np.ndarray", sy: "np.ndarray") -> "np.ndarray":
    h, w = arr.shape[:2]
    sx = np.clip(sx, 0.0, w - 1.001)
    sy = np.clip(sy, 0.0, h - 1.001)
    x0 = np.floor(sx).astype(np.int32)
    y0 = np.floor(sy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (sx - x0)[..., None]
    wy = (sy - y0)[..., None]
    ia = arr[y0, x0]
    ib = arr[y0, x1]
    ic = arr[y1, x0]
    id_ = arr[y1, x1]
    return ia * (1 - wx) * (1 - wy) + ib * wx * (1 - wy) + ic * (1 - wx) * wy + id_ * wx * wy


def detect_face_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) in image pixels, or None."""
    haar = _haar_box(img)
    if haar is not None:
        return haar
    return _skin_box(img)


def _haar_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    cascade = None
    for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"):
        path = getattr(cv2.data, "haarcascades", "") + name
        try:
            c = cv2.CascadeClassifier(path)
            if not c.empty():
                cascade = c
                break
        except Exception:
            continue
    if cascade is None:
        return None
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32)
    )
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def _skin_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    w, h = img.size
    if w < 16 or h < 16:
        return None
    scan_h = max(1, int(h * 0.55))
    crop = img.crop((0, 0, w, scan_h)).resize(
        (max(32, w // 8), max(24, scan_h // 8)),
        Image.Resampling.BILINEAR,
    )
    ycbcr = crop.convert("YCbCr")
    pix = ycbcr.load()
    cw, ch = crop.size
    mask = [[False] * cw for _ in range(ch)]
    for yy in range(ch):
        for xx in range(cw):
            _y, cb, cr = pix[xx, yy]
            if 77 <= cb <= 127 and 133 <= cr <= 173 and 40 <= _y <= 230:
                mask[yy][xx] = True
    visited = [[False] * cw for _ in range(ch)]
    best = 0
    best_box = None
    for yy in range(ch):
        for xx in range(cw):
            if not mask[yy][xx] or visited[yy][xx]:
                continue
            area, box = _flood(mask, visited, xx, yy, cw, ch)
            if area > best:
                best = area
                best_box = box
    if not best_box or best < max(12, (cw * ch) // 50):
        return None
    x1, y1, x2, y2 = best_box
    sx, sy = w / float(cw), scan_h / float(ch)
    return (
        int(x1 * sx),
        int(y1 * sy),
        max(8, int((x2 - x1 + 1) * sx)),
        max(8, int((y2 - y1 + 1) * sy)),
    )


def _flood(
    mask: list[list[bool]],
    visited: list[list[bool]],
    sx: int,
    sy: int,
    w: int,
    h: int,
) -> tuple[int, tuple[int, int, int, int]]:
    stack = [(sx, sy)]
    visited[sy][sx] = True
    n = 0
    x1 = x2 = sx
    y1 = y2 = sy
    while stack:
        x, y = stack.pop()
        n += 1
        x1, x2 = min(x1, x), max(x2, x)
        y1, y2 = min(y1, y), max(y2, y)
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not visited[ny][nx]:
                visited[ny][nx] = True
                stack.append((nx, ny))
    return n, (x1, y1, x2, y2)


def _portrait_prior(size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    return (
        int(w * 0.16),
        int(h * 0.02),
        int(w * 0.68),
        int(h * 0.42),
    )


def _face_mask(size: tuple[int, int], box: tuple[int, int, int, int]) -> Image.Image:
    w, h = size
    x, y, fw, fh = box
    pad_x = int(fw * 0.34)
    pad_top = int(fh * 0.58)
    pad_bot = int(fh * 0.22)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(w, x + fw + pad_x)
    y2 = min(h, y + fh + pad_bot)
    y2 = min(y2, int(h * _CHEST_GUARD))
    if y2 <= y1 + 8:
        y2 = min(h, y1 + max(8, int(h * 0.28)))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([x1, y1, x2, y2], fill=255)
    radius = max(4, int(min(w, h) * 0.018))
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))
