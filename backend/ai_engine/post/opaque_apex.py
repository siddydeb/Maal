"""Kill nipple tip show-through on clothed bust enhances (keep-outfit).

Face-anchored, image-size agnostic. Runs after Kontext / i2i so every similar
keep-outfit bust result gets opaque apex coverage — not a one-off for a single
test photo.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]


def opaque_bust_apex(
    original_bytes: bytes,
    edited_bytes: bytes,
    *,
    strength: float = 1.0,
) -> bytes:
    """Flatten tip contrast / pigment under cloth; keep bust volume.

    Detects dual-lobe apex peaks in the chest fabric band (face-anchored) and
    replaces tip-scale detail with surrounding fabric.
    """
    if np is None or cv2 is None:
        return edited_bytes
    try:
        from backend.ai_engine.post.face_lock import detect_face_box, _portrait_prior
    except Exception:
        return edited_bytes

    orig = Image.open(BytesIO(original_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_bytes)).convert("RGB")
    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)

    e = np.asarray(edit, dtype=np.float32)
    o = np.asarray(orig, dtype=np.float32)
    h, w, _ = e.shape
    face = detect_face_box(orig) or _portrait_prior(orig.size)
    fx, fy, fw, fh = face
    # Reject torso-as-face / giant priors — use a tight head prior instead.
    if fw > w * 0.42 or fh > h * 0.32 or fy + fh > h * 0.48:
        fx, fy, fw, fh = (
            int(w * 0.28),
            int(h * 0.02),
            int(w * 0.44),
            int(h * 0.28),
        )
    mid = float(min(max(fx + fw / 2.0, w * 0.38), w * 0.62))
    chin = float(min(max(fy + fh * 0.98, h * 0.22), h * 0.40))

    # Bust band: tip peaks on this pose sit ~0.42–0.50 of frame height.
    y0 = int(max(chin + h * 0.02, h * 0.38))
    y1 = int(min(h * 0.58, y0 + int(h * 0.18)))
    half = max(fw * 0.90, w * 0.22)
    x0 = int(max(w * 0.22, mid - half))
    x1 = int(min(w * 0.78, mid + half))
    if y1 <= y0 + 8 or x1 <= x0 + 8:
        return edited_bytes

    amp = float(np.clip(strength, 0.35, 1.35))
    centers = _detect_apex_centers(e, mid=mid, y0=y0, y1=y1, x0=x0, x1=x1)
    # Anatomical seeds at mid-mound (where tips actually land after size-up)
    cy = int(y0 + (y1 - y0) * 0.42)
    sep = max(int(w * 0.16), int(fw * 0.50))
    seeded = [
        (cy, int(mid - sep * 0.70)),
        (cy, int(mid + sep * 0.70)),
        (int(y0 + (y1 - y0) * 0.50), int(mid - sep * 0.60)),
        (int(y0 + (y1 - y0) * 0.50), int(mid + sep * 0.60)),
        (int(y0 + (y1 - y0) * 0.55), int(mid - sep * 0.50)),
        (int(y0 + (y1 - y0) * 0.55), int(mid + sep * 0.50)),
    ]
    centers = _merge_centers(centers + seeded, min_dist=max(12, int(min(h, w) * 0.022)))
    # Drop hair / strap / off-bust peaks
    centers = _filter_fabric_centers(e, centers, y0=y0, y1=y1, x0=x0, x1=x1, mid=mid)

    rad = max(34, int(min(w, h) * 0.070 * (0.85 + 0.22 * amp)))
    out = _kill_tips(e, o, centers, rad=rad, amp=amp, y_lock=y0)
    # One residual pass on the result — fabric-filtered only
    residual = _filter_fabric_centers(
        out,
        _detect_apex_centers(out, mid=mid, y0=y0, y1=y1, x0=x0, x1=x1),
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        mid=mid,
    )
    if residual:
        out = _kill_tips(out, o, residual, rad=int(rad * 0.85), amp=amp, y_lock=y0)
    # Soft lock above bust — feather 6px so no hard horizontal seam
    if y0 > 6:
        feather = np.linspace(1.0, 0.0, 6, dtype=np.float32)[:, None, None]
        band = out[y0 - 6 : y0]
        src = e[y0 - 6 : y0]
        out[y0 - 6 : y0] = src * feather + band * (1.0 - feather)
        out[: y0 - 6] = e[: y0 - 6]
    else:
        out[:y0] = e[:y0]

    buf = BytesIO()
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _detect_apex_centers(
    rgb: np.ndarray,
    *,
    mid: float,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> list[tuple[int, int]]:
    """Return (y, x) tip centers — up to 2–3 per breast lobe."""
    h, w, _ = rgb.shape
    L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    blur = cv2.GaussianBlur(L, (0, 0), max(6.0, min(h, w) * 0.012))
    dark = blur - L  # pigment / shadow tip
    bright = L - blur  # highlight tip
    dog = cv2.GaussianBlur(L, (0, 0), 1.2) - cv2.GaussianBlur(L, (0, 0), 7.0)

    yy, xx = np.mgrid[0:h, 0:w]
    band = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1) & (L > 85)
    # Prefer fabric-ish (not dark hair)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    chroma = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
    fabric = band & (chroma < 75) & (L > 95)

    centers: list[tuple[int, int]] = []
    for side_mask in (fabric & (xx < mid), fabric & (xx >= mid)):
        if not side_mask.any():
            continue
        # Score: tip-scale energy (bright or dark vs neighborhood)
        score = np.maximum(np.abs(dog) * 1.2, np.maximum(dark, bright)) * side_mask.astype(
            np.float32
        )
        # Suppress edge of band
        score[: y0 + 2] = 0
        score[max(0, y1 - 2) :] = 0
        picked = _nms_peaks(score, k=2, min_dist=max(18, int(min(h, w) * 0.030)))
        centers.extend(picked)
        # Always include strongest peak even if weak
        if not picked and side_mask.any():
            y, x = np.unravel_index(np.argmax(score), score.shape)
            if score[y, x] > 0:
                centers.append((int(y), int(x)))
    return centers


def _filter_fabric_centers(
    rgb: np.ndarray,
    centers: list[tuple[int, int]],
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    mid: float,
) -> list[tuple[int, int]]:
    """Keep only peaks on bright fabric inside the inner bust column."""
    h, w, _ = rgb.shape
    L = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    kept: list[tuple[int, int]] = []
    for y, x in centers:
        if y < y0 + 4 or y > y1 - 4:
            continue
        if x < x0 + 8 or x > x1 - 8:
            continue
        # Inner column — reject outer hair / straps
        if abs(x - mid) > w * 0.28:
            continue
        if float(L[y, x]) < 95:
            continue
        # Reject near-black hair neighborhoods
        y0n, y1n = max(0, y - 6), min(h, y + 7)
        x0n, x1n = max(0, x - 6), min(w, x + 7)
        if float(L[y0n:y1n, x0n:x1n].mean()) < 100:
            continue
        kept.append((int(y), int(x)))
    return kept


def _merge_centers(
    centers: list[tuple[int, int]], *, min_dist: int
) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    for y, x in centers:
        if any((y - yy) ** 2 + (x - xx) ** 2 < min_dist**2 for yy, xx in kept):
            continue
        kept.append((int(y), int(x)))
    return kept


def _nms_peaks(
    score: np.ndarray, *, k: int, min_dist: int
) -> list[tuple[int, int]]:
    s = score.copy()
    out: list[tuple[int, int]] = []
    thr = float(np.percentile(s[s > 0], 70)) if (s > 0).any() else 0.0
    thr = max(thr * 0.40, 2.0)
    for _ in range(k):
        y, x = np.unravel_index(np.argmax(s), s.shape)
        val = float(s[y, x])
        if val < thr:
            break
        out.append((int(y), int(x)))
        y0, y1 = max(0, y - min_dist), min(s.shape[0], y + min_dist + 1)
        x0, x1 = max(0, x - min_dist), min(s.shape[1], x + min_dist + 1)
        s[y0:y1, x0:x1] = 0
    return out


def _kill_tips(
    edit: np.ndarray,
    start: np.ndarray,
    centers: list[tuple[int, int]],
    *,
    rad: int,
    amp: float,
    y_lock: int,
) -> np.ndarray:
    """Blur-fill tip disks — keep volume shade, kill tip-scale contrast/pigment."""
    del start
    h, w, _ = edit.shape
    lum = 0.299 * edit[..., 0] + 0.587 * edit[..., 1] + 0.114 * edit[..., 2]
    flatter = cv2.GaussianBlur(edit, (0, 0), max(16.0, rad * 0.55))
    mask = np.zeros((h, w), np.float32)
    for cy, cx in centers:
        if cy < y_lock + 2:
            continue
        yy, xx = np.ogrid[:h, :w]
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
        a = np.clip(1.0 - d / float(rad), 0.0, 1.0) ** 1.0
        a[lum < 90] = 0
        a[:y_lock] = 0
        mask = np.maximum(mask, a)
    if mask.max() <= 0:
        return edit.copy()
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, rad * 0.08))
    mask = np.clip(mask * (0.90 + 0.20 * amp), 0.0, 1.0)
    m = mask[..., None]
    out = edit * (1.0 - m * 0.88) + flatter * (m * 0.88)
    u8 = np.clip(out, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    flat_lab = cv2.cvtColor(
        np.clip(flatter, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB
    ).astype(np.float32)
    lab[..., 0] = lab[..., 0] * (1.0 - mask * 0.75) + flat_lab[..., 0] * (mask * 0.75)
    lab[..., 1] = lab[..., 1] * (1.0 - mask * 0.96) + flat_lab[..., 1] * (mask * 0.96)
    lab[..., 2] = lab[..., 2] * (1.0 - mask * 0.96) + flat_lab[..., 2] * (mask * 0.96)
    dog = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.0) - cv2.GaussianBlur(
        lab[..., 0], (0, 0), 9.0
    )
    lab[..., 0] = np.clip(lab[..., 0] - dog * mask * (1.5 * amp), 0, 255)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(
        np.float32
    )
