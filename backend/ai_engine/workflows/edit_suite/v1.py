"""P1 Image Editing Engine — register all Phase 5 edit workflows (v1)."""

from __future__ import annotations

from backend.ai_engine.workflows._shared.factory import register_edit_workflow

# Fallback chain targets (must be registered in this module too)
_INSTRUCTION = "instruction_edit.v1"
_OBJECT = "object_replace.v1"
_IMG2IMG = "image_img2img.v0_legacy"


def register() -> None:
    # --- instruction / general ---
    register_edit_workflow(
        id="instruction_edit",
        task_types=["edit.general_instruction"],
        task_kind="instruction",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["identity.pulid_flux"],
        perception=[],
        fallback_workflow_ref=_IMG2IMG,
        post_balanced=["color_match"],
        post_quality=["color_match", "face_detailer"],
        post_ultra=["color_match", "face_detailer"],
    )

    # Keep-outfit reshape — Flux Kontext ReferenceLatent + figure reshape LoRA.
    # Face / outside-bust restored in edit_runner post.
    register_edit_workflow(
        id="keep_outfit_reshape",
        task_types=["edit.keep_outfit_reshape"],
        task_kind="keep_outfit",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["identity.pulid_flux", "seg.sam", "seg.sam2", "grounding.dino"],
        perception=["garment"],
        denoise_override=0.80,
        fallback_workflow_ref=_IMG2IMG,
        post_balanced=["color_match", "face_detailer"],
        post_quality=["color_match", "face_detailer"],
        post_ultra=["color_match", "face_detailer"],
    )

    # --- clothing ---
    register_edit_workflow(
        id="clothing_replace",
        task_types=["edit.clothing_replace"],
        task_kind="clothing",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_fill", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=[
            "identity.pulid_flux",
            "grounding.dino",
            "seg.sam2",
        ],
        perception=["grounding", "sam2"],
        fallback_workflow_ref=_OBJECT,
        post_balanced=["color_match"],
        post_quality=["color_match", "face_detailer"],
        post_ultra=["color_match", "face_detailer"],
    )

    # --- objects / inpaint family ---
    register_edit_workflow(
        id="object_replace",
        task_types=["edit.object_replace"],
        task_kind="object",
        preferred_models=["backbone.flux_fill"],
        compatible_models=["backbone.flux_kontext_dev_fp8", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["grounding.dino", "seg.sam2", "identity.pulid_flux"],
        perception=["grounding", "sam2"],
        fallback_workflow_ref=_INSTRUCTION,
    )

    register_edit_workflow(
        id="object_remove",
        task_types=["edit.remove_object"],
        task_kind="remove",
        preferred_models=["backbone.flux_fill"],
        compatible_models=["backbone.flux_kontext_dev_fp8", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["grounding.dino", "seg.sam2"],
        perception=["grounding", "sam2"],
        fallback_workflow_ref=_INSTRUCTION,
    )

    register_edit_workflow(
        id="object_add",
        task_types=["edit.add_object"],
        task_kind="add",
        preferred_models=["backbone.flux_fill"],
        compatible_models=["backbone.flux_kontext_dev_fp8", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["grounding.dino", "seg.sam2"],
        perception=["grounding", "sam2"],
        fallback_workflow_ref=_INSTRUCTION,
    )

    register_edit_workflow(
        id="inpaint",
        task_types=["edit.inpaint"],
        task_kind="inpaint",
        preferred_models=["backbone.flux_fill"],
        compatible_models=["backbone.flux_kontext_dev_fp8", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["grounding.dino", "seg.sam2"],
        perception=["sam2"],
        fallback_workflow_ref=_INSTRUCTION,
    )

    register_edit_workflow(
        id="outpaint",
        task_types=["edit.outpaint"],
        task_kind="outpaint",
        preferred_models=["backbone.flux_fill"],
        compatible_models=["backbone.flux_kontext_dev_fp8", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        perception=[],
        preprocess="outpaint_pad",
        denoise_override=0.55,
        fallback_workflow_ref=_INSTRUCTION,
        post_balanced=["color_match"],
        post_quality=["color_match"],
        post_ultra=["color_match"],
    )

    # --- face / hair / swap ---
    register_edit_workflow(
        id="face_edit",
        task_types=["edit.face"],
        task_kind="face",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["identity.pulid_flux"],
        perception=["face_detect"],
        fallback_workflow_ref=_INSTRUCTION,
        post_balanced=["face_detailer"],
        post_quality=["face_detailer"],
        post_ultra=["face_detailer"],
    )

    register_edit_workflow(
        id="hair_edit",
        task_types=["edit.hair"],
        task_kind="hair",
        preferred_models=["backbone.flux_fill"],
        compatible_models=["backbone.flux_kontext_dev_fp8", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["identity.pulid_flux", "grounding.dino", "seg.sam2"],
        perception=["grounding", "sam2"],
        fallback_workflow_ref=_INSTRUCTION,
    )

    register_edit_workflow(
        id="face_swap",
        task_types=["edit.face_swap"],
        task_kind="face_swap",
        preferred_models=["backbone.flux_dev_fp8"],
        compatible_models=["backbone.flux_kontext_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["identity.pulid_flux"],
        perception=["face_detect"],
        fallback_workflow_ref=_INSTRUCTION,
        beta=True,  # ReActor path not wired; beta until dedicated graph
        post_balanced=["face_detailer"],
        post_quality=["face_detailer"],
        post_ultra=["face_detailer"],
    )

    # --- background / product / style / identity ---
    register_edit_workflow(
        id="background_replace",
        task_types=["edit.background"],
        task_kind="background",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_fill", "backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["matting.birefnet", "matting.rembg"],
        perception=["matting"],
        fallback_workflow_ref=_INSTRUCTION,
        post_balanced=["color_match"],
        post_quality=["color_match"],
        post_ultra=["color_match"],
    )

    register_edit_workflow(
        id="product_edit",
        task_types=["edit.product"],
        task_kind="product",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["matting.birefnet", "matting.rembg"],
        perception=["matting"],
        fallback_workflow_ref=_INSTRUCTION,
    )

    register_edit_workflow(
        id="style_transfer",
        task_types=["edit.style_transfer"],
        task_kind="style",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        perception=[],
        fallback_workflow_ref=_INSTRUCTION,
        denoise_override=0.50,
    )

    register_edit_workflow(
        id="character_consistency",
        task_types=["edit.character_consistency"],
        task_kind="identity",
        preferred_models=["backbone.flux_kontext_dev_fp8"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        optional_models=["identity.pulid_flux"],
        perception=["face_detect"],
        fallback_workflow_ref=_INSTRUCTION,
        post_balanced=["face_detailer"],
        post_quality=["face_detailer"],
        post_ultra=["face_detailer"],
    )

    register_edit_workflow(
        id="image_restore",
        task_types=["edit.restore"],
        task_kind="restore",
        preferred_models=["backbone.flux_dev_fp8"],
        compatible_models=["backbone.flux_kontext_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        perception=[],
        denoise_override=0.30,
        fallback_workflow_ref=_IMG2IMG,
        post_balanced=["face_detailer"],
        post_quality=["face_detailer"],
        post_ultra=["face_detailer"],
    )

    register_edit_workflow(
        id="image_upscale",
        task_types=["image.upscale"],
        task_kind="upscale",
        preferred_models=["upscale.ultrasharp_4x"],
        compatible_models=["backbone.flux_dev_fp8"],
        minimum_models=["backbone.flux_dev_fp8"],
        perception=[],
        preprocess="upscale_pre",
        denoise_override=0.28,
        fallback_workflow_ref=_IMG2IMG,
        post_balanced=[],
        post_quality=[],
        post_ultra=[],
    )
