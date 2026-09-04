"""Post Processor — color_match (outside-mask preserve) + deferred hooks."""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Optional

from PIL import Image

from backend.ai_engine.registry.workflows import WorkflowRecord
from backend.ai_engine.schema import EngineResult, ExecutionPlan


def preserve_outside_mask(
    original_bytes: bytes,
    edited_bytes: bytes,
    mask_png: bytes,
) -> bytes:
    """
    White in mask = edited region kept from `edited`; black = restore from `original`.
    Soft feather on the seam to reduce two-tone blemishes.
    """
    orig = Image.open(BytesIO(original_bytes)).convert("RGB")
    edit = Image.open(BytesIO(edited_bytes)).convert("RGB")
    mask = Image.open(BytesIO(mask_png)).convert("L")

    if edit.size != orig.size:
        edit = edit.resize(orig.size, Image.Resampling.LANCZOS)
    if mask.size != orig.size:
        mask = mask.resize(orig.size, Image.Resampling.BILINEAR)

    from PIL import ImageFilter

    soft = max(2, int(min(orig.size) * 0.005))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=soft))
    out = Image.composite(edit, orig, mask)
    buf = BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


async def run_post(
    result: EngineResult,
    workflow: WorkflowRecord,
    plan: ExecutionPlan,
    *,
    original_bytes: Optional[bytes] = None,
    perception: Any = None,
) -> EngineResult:
    profile = workflow.quality_profiles.get(plan.profile)
    stages = list(profile.post_processing) if profile else []
    stages.extend(x for x in plan.post_hints if x not in stages)
    deps_post = (workflow.dependencies or {}).get("post_default") or []
    for s in deps_post:
        if s not in stages:
            stages.append(s)

    applied: list[str] = []
    deferred: list[str] = []

    for stage in stages:
        key = stage.lower().strip()
        if key in ("color_match", "outside_mask_preserve"):
            mask = getattr(perception, "mask", None) if perception is not None else None
            pw = list(getattr(perception, "warnings", None) or []) if perception else []
            # Heuristic / failed masks are rectangular priors — compositing them
            # onto Kontext full-frame edits punches a hard box over the shirt.
            mask_source = str(getattr(mask, "source", "") or "").lower() if mask else ""
            # clipseg / local_fabric / sam are safe soft masks for outside-preserve.
            skip_composite = (
                "mask_failed" in pw
                or mask_source in ("heuristic", "center", "full_frame", "")
                or (
                    "perception_degraded" in pw
                    and mask_source not in ("clipseg", "local_fabric", "garment")
                )
                or "kontext" in str(result.model_label or "").lower()
            )
            if (
                original_bytes
                and mask is not None
                and getattr(mask, "mask_png", None)
                and result.kind == "img"
                and result.data
                and not skip_composite
            ):
                try:
                    result.data = preserve_outside_mask(
                        original_bytes, result.data, mask.mask_png
                    )
                    result.content_type = "image/png"
                    applied.append("color_match")
                except Exception as exc:
                    result.warnings.append(f"color_match_failed:{exc}")
                    deferred.append("color_match")
            else:
                if skip_composite and mask is not None:
                    result.warnings.append("color_match_skipped:unsafe_mask")
                deferred.append("color_match")
        elif key in ("rife_16_to_24", "rife", "interpolate"):
            # RIFE Comfy graph when video.rife installed; else annotate target fps
            from backend.ai_engine.models.manager import manager as model_manager

            hints = plan.params_hints or {}
            gen_fps = int(hints.get("gen_fps") or 16)
            target = 24
            if model_manager.is_available("video.rife") and result.kind == "vid":
                deferred.append("rife_comfy_not_wired")
                result.warnings.append(
                    f"rife_pending:target_fps={target}:source_fps={gen_fps}"
                )
            else:
                deferred.append("rife_16_to_24")
                result.warnings.append(
                    f"interp_deferred:install_video.rife:target_fps={target}:source_fps={gen_fps}"
                )
            # Always record intent for metrics / clients
            plan.params_hints = {
                **hints,
                "interp_target_fps": target,
                "interp_source_fps": gen_fps,
            }
        elif key in ("frame_upscale", "video_upscale"):
            from backend.ai_engine.models.manager import manager as model_manager

            if model_manager.is_available("upscale.ultrasharp_4x") and result.kind == "vid":
                deferred.append("frame_upscale_comfy_not_wired")
                result.warnings.append("frame_upscale_pending:ultrasharp")
            else:
                deferred.append("frame_upscale")
                result.warnings.append("frame_upscale_deferred:missing_upscaler")
        elif key in ("fluid_recolor", "gel_recolor"):
            hints = plan.params_hints or {}
            fluidish = bool(hints.get("fluid_edit"))
            if (
                original_bytes
                and result.kind == "img"
                and result.data
                and fluidish
            ):
                try:
                    from backend.ai_engine.post.fluid_recolor import (
                        recolor_white_paint_to_gel,
                    )

                    face_only = not bool(hints.get("overlay_body"))
                    result.data = recolor_white_paint_to_gel(
                        original_bytes,
                        result.data,
                        face_only=face_only,
                    )
                    result.content_type = "image/png"
                    applied.append("fluid_recolor")
                    if face_only:
                        applied.append("fluid_face_clamp")
                except Exception as exc:
                    result.warnings.append(f"fluid_recolor_failed:{exc}")
                    deferred.append("fluid_recolor")
            else:
                deferred.append("fluid_recolor")
        elif key in ("face_detailer", "face_lock", "hair_polish", "edge_refine"):
            hints = plan.params_hints or {}
            can_lock = (
                key in ("face_detailer", "face_lock")
                and (
                    bool(hints.get("clothed_enhance"))
                    or str(getattr(plan, "task_type", "") or "")
                    == "edit.keep_outfit_reshape"
                )
                and not bool(hints.get("pose_edit"))
                and not bool(hints.get("fluid_edit"))
                and original_bytes
                and result.kind == "img"
                and result.data
            )
            if can_lock:
                try:
                    task_type = str(getattr(plan, "task_type", "") or "")
                    # keep_outfit runner already runs face/hand/garment post — do not re-lock.
                    if task_type == "edit.keep_outfit_reshape":
                        deferred.append("face_lock")
                    else:
                        from backend.ai_engine.post.face_lock import (
                            restore_original_face,
                            restore_outside_chest,
                        )

                        labels = {
                            str(t.get("label") or "")
                            for t in (plan.targets or [])
                            if isinstance(t, dict)
                        }
                        # Inner region from GPU; start photo everywhere else.
                        if hints.get("clothed_enhance") or labels & {
                            "chest",
                            "bust",
                        }:
                            result.data = restore_outside_chest(
                                original_bytes, result.data
                            )
                            applied.append("chest_lock")
                        else:
                            result.data = restore_original_face(
                                original_bytes, result.data
                            )
                        result.content_type = "image/png"
                        applied.append("face_lock")
                except Exception as exc:
                    result.warnings.append(f"face_lock_failed:{exc}")
                    deferred.append("face_lock")
            else:
                deferred.append(key)
        else:
            deferred.append(key)

    if applied:
        result.warnings.append(f"post_applied:{','.join(applied)}")
    if deferred:
        result.warnings.append(f"post_deferred:{','.join(deferred)}")
    return result
