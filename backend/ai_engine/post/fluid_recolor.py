"""Recolor overlay highlights toward a consistent translucent finish.

Preserve source fabric and hair color. Only correct gray/blue digital artifacts.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, ImageFilter

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


# Thin films stay translucent; thicker beads stay more opaque.
_GEL_THIN = (236.0, 228.0, 216.0)
_GEL_THICK = (250.0, 246.0, 238.0)


def recolor_white_paint_to_gel(
    original_bytes: bytes,
    edited_bytes: bytes,
    *,
    face_only: bool = False,
) -> bytes:
    """face_only: boost face translucency (skin show-through); never wipe clothes."""
    orig = Image.open(BytesIO(original_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_bytes)).convert("RGB")
    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)
    if np is not None:
        rgb = _recolor_numpy(orig, edit, face_only=face_only)
    else:
        rgb = _recolor_pil(orig, edit, face_only=face_only)
    buf = BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue()


def _sparse_hair_drip_mask(h: int, w: int, hair: "np.ndarray") -> "np.ndarray":
    """Soft inconsistent highlight mask — translucent, not hard outlines."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    cols = []
    for frac in (0.40, 0.48, 0.55, 0.62):
        x = int(w * frac)
        band = hair[int(h * 0.06) : int(h * 0.26), max(0, x - 2) : min(w, x + 3)]
        if band.size and float(band.mean()) > 0.15:
            cols.append(x)
    if not cols:
        cols = [int(w * 0.45), int(w * 0.52), int(w * 0.58)]
    y0 = int(h * 0.06)
    for i, x in enumerate(cols[:3]):
        length = int(h * (0.10 + 0.03 * (i % 2)))
        width = 1 + (i % 2)
        y3 = min(h - 1, y0 + length)
        draw.line([(x, y0), (x + (i % 2) - 1, y3)], fill=160, width=width)
        draw.ellipse([x - 1, y3 - 1, x + 2, y3 + 2], fill=190)
    soft = mask.filter(ImageFilter.GaussianBlur(radius=max(2.0, min(w, h) * 0.006)))
    arr = np.asarray(soft, dtype=np.float32) / 255.0
    return arr * hair.astype(np.float32)


def _luma(arr):
    return arr[..., 0] * 0.2126 + arr[..., 1] * 0.7152 + arr[..., 2] * 0.0722


def _face_region_mask(img: Image.Image) -> Image.Image:
    """Soft white = face/upper (boost translucency). Clothes stay editable."""
    w, h = img.size
    box = _detect_face_box(img)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    if box is None:
        # Upper portrait band — face and chin, not the full torso.
        draw.ellipse(
            [int(w * 0.12), int(h * 0.02), int(w * 0.88), int(h * 0.55)],
            fill=255,
        )
    else:
        x, y, fw, fh = box
        pad_x = int(fw * 0.25)
        pad_top = int(fh * 0.40)
        pad_bot = int(fh * 0.55)
        draw.ellipse(
            [
                max(0, x - pad_x),
                max(0, y - pad_top),
                min(w, x + fw + pad_x),
                min(h, y + fh + pad_bot),
            ],
            fill=255,
        )
    return mask.filter(ImageFilter.GaussianBlur(radius=max(3, int(min(w, h) * 0.015))))


def _detect_face_box(img: Image.Image) -> tuple[int, int, int, int] | None:
    try:
        import cv2
        import numpy as np  # noqa: F811
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
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(32, 32))
    if faces is None or len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    return int(x), int(y), int(w), int(h)


def _recolor_numpy(
    orig: Image.Image, edit: Image.Image, *, face_only: bool = False
) -> Image.Image:
    o = np.asarray(orig, dtype=np.float32)
    e = np.asarray(edit, dtype=np.float32)
    oy = _luma(o)
    ey = _luma(e)
    esat = e.max(axis=2) - e.min(axis=2)
    dy = ey - oy

    paint = (dy > 14.0) & (ey > 155.0) & (esat < 70.0)
    chalk = (dy > 20.0) & (ey > 200.0) & (esat < 45.0)
    paint = paint | chalk
    gray_glitch = (
        (dy > 8.0)
        & (ey > 85.0)
        & (ey < 200.0)
        & (esat < 36.0)
        & (np.abs(e[..., 0] - e[..., 1]) < 18.0)
        & (np.abs(e[..., 1] - e[..., 2]) < 18.0)
        & ~paint
    )
    blue_wisp = (
        (dy > 4.0)
        & (e[..., 2] > e[..., 0] + 6.0)
        & (e[..., 2] > e[..., 1] + 3.0)
        & (ey > 90.0)
        & (esat > 8.0)
        & (esat < 120.0)
        & ~paint  # don't erase actual gel/chalk fills
    )
    paint_img = Image.fromarray((paint.astype(np.uint8) * 255), mode="L")
    fat_paint = np.asarray(paint_img.filter(ImageFilter.MinFilter(size=5))) > 0
    rim = (np.asarray(paint_img.filter(ImageFilter.MaxFilter(size=5))) > 0) & ~(
        np.asarray(paint_img.filter(ImageFilter.MinFilter(size=3))) > 0
    )
    # Only strip rims of fat sticker blobs — thin strokes are all "rim" morphologically.
    cartoon_rim = rim & fat_paint & (dy > 8.0) & (ey > 180.0)

    h, _w = o.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    osat = o.max(axis=2) - o.min(axis=2)
    skinish = (
        (o[..., 0] > o[..., 1] + 10.0)
        & (o[..., 0] > o[..., 2] + 8.0)
        & (oy > 72.0)
    )
    hair_core = (oy < 155.0) & (osat < 120.0) & (~skinish) & (yy < 0.55)
    hair_img = Image.fromarray((hair_core.astype(np.uint8) * 255), mode="L")
    hair = np.asarray(hair_img.filter(ImageFilter.MaxFilter(size=9))) > 0
    hair = hair & (yy < 0.55) & (~skinish)

    out = e.copy()
    # Wipe blue/gray wisps broadly; keep cartoon rim thin so the fill survives.
    wisp = (gray_glitch | blue_wisp) & ~hair
    if np.any(wisp):
        wisp_img = Image.fromarray((wisp.astype(np.uint8) * 255), mode="L")
        wisp = np.asarray(wisp_img.filter(ImageFilter.MaxFilter(size=5))) > 0
        out = np.where(wisp[..., None], o, out)
    rim_only = cartoon_rim & ~hair & ~wisp
    if np.any(rim_only):
        rim_img = Image.fromarray((rim_only.astype(np.uint8) * 255), mode="L")
        rim_only = np.asarray(rim_img.filter(ImageFilter.MaxFilter(size=3))) > 0
        out = np.where(rim_only[..., None], o, out)
    bad = wisp | rim_only

    face_m = np.asarray(_face_region_mask(orig), dtype=np.float32) / 255.0
    face_paint = paint & (face_m > 0.15) & ~bad & ~hair
    if np.any(face_paint):
        # Thickness from how hard the edit painted — thick more opaque, thin more translucent.
        thick = np.clip(dy / 85.0, 0.0, 1.0)
        thin = np.array(_GEL_THIN, dtype=np.float32)
        thick_c = np.array(_GEL_THICK, dtype=np.float32)
        gel = thin + (thick_c - thin) * thick[..., None]
        # Prefer ref-like opaque beads; thin edges still show a little skin.
        paint_w = (0.62 + 0.35 * thick) * np.clip(face_m + 0.15, 0.0, 1.0)
        stained = o * (1.0 - paint_w[..., None]) + gel * paint_w[..., None]
        # Specular on raised beads only (not a cartoon outline).
        hi = np.clip((ey - 190.0) / 50.0, 0.0, 1.0) * thick * face_paint.astype(np.float32)
        stained = stained * (1.0 - 0.22 * hi[..., None]) + thick_c * (0.22 * hi[..., None])
        stained = np.clip(stained, 0.0, 255.0)
        out = np.where(face_paint[..., None], stained, out)

    delta_rgb = np.abs(e - o).mean(axis=2)
    hair_changed = hair & (
        (dy > 6.0)
        | (delta_rgb > 10.0)
        | ((osat > 8.0) & (esat + 4.0 < osat))
        | paint
        | gray_glitch
        | blue_wisp
    )
    hair_fix = hair_changed
    drip_m = _sparse_hair_drip_mask(h, o.shape[1], hair)
    if np.any(hair_fix) or np.any(drip_m > 0.05):
        restored = o.copy()
        if np.any(hair_fix):
            wet = o * np.array([0.97, 0.95, 0.93], dtype=np.float32)
            restored = np.where(hair_fix[..., None], wet, restored)
        if np.any(drip_m > 0.05):
            gel_c = np.array(_GEL_THICK, dtype=np.float32)
            mix = drip_m * 0.48
            mixed = restored * (1.0 - mix[..., None]) + gel_c * mix[..., None]
            # Small highlight on mask peaks
            mixed = mixed * 0.90 + gel_c * (0.10 * drip_m[..., None])
            restored = np.where((drip_m > 0.06)[..., None], mixed, restored)
        restored = np.clip(restored, 0.0, 255.0)
        apply = hair_fix | (drip_m > 0.06)
        out = np.where(apply[..., None], restored, out)
        hair_fix = apply

    if face_only:
        # Re-gel body even if a nearby rim wipe touched the pixel.
        clothes_paint = paint & (face_m < 0.40) & ~hair & (ey > 155.0)
        if np.any(clothes_paint):
            thick_c = np.clip((ey - 155.0) / 65.0, 0.0, 1.0)
            thin = np.array(_GEL_THIN, dtype=np.float32)
            thick = np.array(_GEL_THICK, dtype=np.float32)
            gel_c = thin + (thick - thin) * thick_c[..., None]
            # Dark fabric: thin films translucent, thick blobs more opaque.
            w_gel = 0.38 + 0.42 * thick_c
            soft = o * (1.0 - w_gel[..., None]) + gel_c * w_gel[..., None]
            soft = soft * (1.0 - 0.16 * thick_c[..., None]) + thick * (0.16 * thick_c[..., None])
            soft = np.clip(soft, 0.0, 255.0)
            out = np.where(clothes_paint[..., None], soft, out)

    changed = bad | face_paint | hair_fix
    if face_only:
        changed = changed | (paint & (face_m < 0.40) & (ey > 155.0) & ~hair)
    if not np.any(changed):
        return edit

    soft = (
        np.asarray(
            Image.fromarray(np.where(changed, 255, 0).astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(radius=2.0)
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    # Keep full strength on detected gel pixels; only soft-edge the fringe into the photo.
    soft = np.maximum(soft, changed.astype(np.float32))
    blended = out * soft[..., None] + o * (1.0 - soft[..., None])
    result = np.where(soft[..., None] > 0.02, blended, e)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")


def _recolor_pil(
    orig: Image.Image, edit: Image.Image, *, face_only: bool = False
) -> Image.Image:
    w, h = orig.size
    ox = orig.load()
    ex = edit.load()
    out_img = Image.new("RGB", (w, h))
    px = out_img.load()
    mask = Image.new("L", (w, h), 0)
    mx = mask.load()
    face = _face_region_mask(orig)
    fx = face.load()
    for y in range(h):
        for x in range(w):
            or_, og, ob = ox[x, y]
            er, eg, eb = ex[x, y]
            oy = (or_ * 54 + og * 183 + ob * 19) >> 8
            ey = (er * 54 + eg * 183 + eb * 19) >> 8
            emax = max(er, eg, eb)
            emin = min(er, eg, eb)
            esat = emax - emin
            dy = ey - oy
            fv = fx[x, y]
            gray_glitch = (
                dy > 8
                and 85 < ey < 210
                and esat < 36
                and abs(er - eg) < 18
                and abs(eg - eb) < 18
            )
            blue_wisp = (
                dy > 5
                and eb > er + 8
                and eb > eg + 4
                and ey > 95
                and 10 < esat < 110
            )
            paint = dy > 14 and ey > 155 and esat < 70
            omax = max(or_, og, ob)
            omin = min(or_, og, ob)
            osat = omax - omin
            skinish = or_ > og + 10 and or_ > ob + 8 and oy > 72
            hair = oy < 155 and osat < 120 and not skinish and (y / max(h, 1)) < 0.55
            delta = (abs(er - or_) + abs(eg - og) + abs(eb - ob)) / 3.0
            hair_changed = hair and (
                dy > 6
                or delta > 10
                or (osat > 8 and esat + 4 < osat)
                or paint
                or gray_glitch
                or blue_wisp
            )
            if hair_changed:
                if dy > 28 and 155 < ey < 245 and esat < 55:
                    mix = 0.12 + 0.14 * min(max((ey - 155.0) / 70.0, 0.0), 1.0)
                    r = or_ * 0.97 * (1 - mix) + _GEL_THIN[0] * mix
                    g = og * 0.95 * (1 - mix) + _GEL_THIN[1] * mix
                    b = ob * 0.93 * (1 - mix) + _GEL_THIN[2] * mix
                    px[x, y] = (
                        int(max(0, min(255, r))),
                        int(max(0, min(255, g))),
                        int(max(0, min(255, b))),
                    )
                else:
                    px[x, y] = (int(or_ * 0.97), int(og * 0.95), int(ob * 0.93))
                mx[x, y] = 255
            elif gray_glitch or blue_wisp:
                px[x, y] = (or_, og, ob)
                mx[x, y] = 255
            elif paint and fv > 40:
                thick = min(max(dy / 85.0, 0.0), 1.0)
                tr = _GEL_THIN[0] * (1 - thick) + _GEL_THICK[0] * thick
                tg = _GEL_THIN[1] * (1 - thick) + _GEL_THICK[1] * thick
                tb = _GEL_THIN[2] * (1 - thick) + _GEL_THICK[2] * thick
                pw = (0.22 + 0.48 * thick) * min(1.0, fv / 255.0 + 0.15)
                r = or_ * (1 - pw) + tr * pw
                g = og * (1 - pw) + tg * pw
                b = ob * (1 - pw) + tb * pw
                hi = min(max((ey - 190.0) / 50.0, 0.0), 1.0) * thick
                r = r * (1 - 0.22 * hi) + _GEL_THICK[0] * 0.22 * hi
                g = g * (1 - 0.22 * hi) + _GEL_THICK[1] * 0.22 * hi
                b = b * (1 - 0.22 * hi) + _GEL_THICK[2] * 0.22 * hi
                px[x, y] = (
                    int(max(0, min(255, r))),
                    int(max(0, min(255, g))),
                    int(max(0, min(255, b))),
                )
                mx[x, y] = 240
            elif face_only and paint and fv < 100 and ey > 155:
                mix = 0.38 + 0.42 * min(max((ey - 155.0) / 65.0, 0.0), 1.0)
                px[x, y] = (
                    int(or_ * (1 - mix) + _GEL_THICK[0] * mix),
                    int(og * (1 - mix) + _GEL_THICK[1] * mix),
                    int(ob * (1 - mix) + _GEL_THICK[2] * mix),
                )
                mx[x, y] = 200
            else:
                px[x, y] = (er, eg, eb)
    soft = mask.filter(ImageFilter.GaussianBlur(radius=2.0))
    return Image.composite(out_img, edit, soft)
