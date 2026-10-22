# AI Engine — Models

**Companion:** [AI_ENGINE.md](AI_ENGINE.md) · [WORKFLOWS.md](WORKFLOWS.md) · [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

**Hardware constraint:** RTX 5060 Ti **16GB**. Prefer fp8/GGUF/AWQ where quality loss is acceptable; never assume currently installed files are optimal.

**Recommendation review date:** 2026-08-05

---

## 1. Model Registry schema

Every model record **must** store:

| Field | Description |
|-------|-------------|
| `model_id` | Stable logical id, e.g. `backbone.flux_kontext_dev_fp8` |
| `version` | Upstream / pack version string |
| `filename` | On-disk filename expected by Comfy |
| `download_source` | Hugging Face / official URL(s) |
| `vram_mb` | Approximate load VRAM |
| `disk_mb` | Approximate disk footprint |
| `license` | SPDX or named license + commercial caveat |
| `compatible_workflows` | List of `workflow_ref` or logical workflow ids |
| `replacement_candidates` | Ordered `model_id`s to try on load failure / upgrades |
| `benchmark_score` | 0–100 engine rubric (see §2) |
| `role` | backbone / vlm / grounding / seg / matting / identity / upscale / vae / text_encoder / control / video / misc |
| `quantization` | fp16 / fp8 / gguf-Q5 / awq / etc. |
| `status` | `installed` \| `missing` \| `outdated` \| `disabled` |

Model Manager uses this registry for ensure/load/unload/health. Workflow Registry and Planner reference **logical slots / `model_id`s**, never raw filenames in planner code.

### Logical slots (indirection)

Some roles are referenced by **slot id** so architecture code stays stable when vendors change:

| Slot | Resolved by | Initial binding (changeable in registry only) |
|------|-------------|-----------------------------------------------|
| `planner.default_model` | Model Manager at VLM plan time | e.g. concrete `vlm.qwen25_vl_7b` record → swap to Qwen3-VL later without editing Planner |
| Workflow `preferred_models` / `compatible_models` / `minimum_models` | Model Manager at execute bind | See WORKFLOWS.md |

Planner source must call `ModelManager.resolve("planner.default_model")`, not embed `vlm.qwen25_vl_7b`.

---

## 2. Benchmark score rubric (engine-defined)

`benchmark_score` is **not** a marketing rating. Compute as weighted sum on internal fixtures (0–100):

| Component | Weight | Notes |
|-----------|--------|-------|
| Task fitness | 30 | How well it serves its role on fixtures |
| Stability on 16GB | 25 | OOM rate, peak VRAM |
| Speed | 15 | Relative to role baseline |
| License fit | 10 | Commercial clarity for product |
| Ecosystem/Comfy support | 10 | Native nodes vs fragile forks |
| Quality delta vs prior | 10 | Measured metric improvement |

Scores below are **initial estimates** until fixtures run; mark `estimated: true` in implementation.

---

## 3. Comparison tables by capability

Format required for each capability: Best · Second-best · VRAM · Speed · Quality · License · Advantages · Limitations · Why selected.

### 3.1 Planner VLM (bound via slot `planner.default_model`)

Architecture code never hardcodes a vendor VLM id. The registry maps:

```yaml
slot: planner.default_model
target_model_id: vlm.qwen25_vl_7b   # change this one field to adopt Qwen3-VL etc.
```

| Field | Best (initial target_model_id) | Second-best (replacement_candidates) |
|-------|--------------------------------|--------------------------------------|
| Model | **Qwen2.5-VL-7B-Instruct** (AWQ/GPTQ or bf16+offload) | Qwen2.5-VL-3B / InternVL3-8B / future Qwen3-VL |
| VRAM | ~8–12GB bf16; ~5–7GB quant | ~3–6GB |
| Speed | Medium (seconds–low tens) | Faster |
| Quality | Strong structured plans + grounding language | Weaker multi-step plans |
| License | Qwen / Tongyi terms (verify commercial) | Varies (InternVL often more permissive) |
| Advantages | JSON planning, spatial language, 16GB-feasible sequentially | Lower VRAM, faster bypass path |
| Limitations | Must unload before diffusion; not a segmenter | More planner errors |
| **Why selected (initial)** | Best plan quality that still fits sequential 16GB workflow; **swap via slot**, not code | |

Florence-2 is **not** the planner; it is perception inside graphs.

### 3.2 Fast caption / auxiliary perception

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **Florence-2 Large** | Florence-2 Base |
| VRAM | ~2–4GB | ~1–2GB |
| Speed | Fast | Fastest |
| Quality | Good captions/grounding tokens | Lower accuracy |
| License | Microsoft Florence license | Same family |
| Advantages | Cheap in-graph assist | Tiny |
| Limitations | Weak as sole router | Weaker phrases |
| **Why selected** | Complements VLM; good box/caption assist | |

### 3.3 Text grounding (boxes)

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **GroundingDINO** (SwinT/Base) | Florence-2 phrase grounding |
| VRAM | ~1–3GB | ~2–4GB |
| Speed | Fast–medium | Medium |
| Quality | Strong open-vocab boxes | Better complex phrases sometimes |
| License | Apache-2.0 (typical GDINO releases) | Florence terms |
| Advantages | Standard Grounded-SAM stack | Fewer moving parts |
| Limitations | Needs SAM for masks; fails on vague phrases | Slower/heavier for detect-only |
| **Why selected** | Best default for mask pipelines on Comfy | |

### 3.4 Segmentation

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **SAM 2.1** (Large if VRAM allows else Base) | SAM 2.1 Base / emerging SAM3 when Comfy-stable |
| VRAM | Base ~1–2GB; Large higher | Lower |
| Speed | Fast given boxes | Faster |
| Quality | Excellent boundaries | Slightly weaker |
| License | SAM license (Meta) | Same family |
| Advantages | Pixel masks from boxes/points | Lighter |
| Limitations | Needs detector/points; not matting | Hair edges weaker than BiRefNet |
| **Why selected** | Standard quality masker for surgical edits | |

### 3.5 Matting / background removal

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **BiRefNet** | RMBG 2.0 |
| VRAM | ~2–5GB | Similar/lower |
| Speed | Fast | Fast |
| Quality | Strong hair/edge matting | Good general cutout |
| License | Check per weight release | Check per weight |
| Advantages | Better than SAM alone for cutouts | Simple |
| Limitations | Not open-vocab multi-object replace | Weaker complex scenes |
| **Why selected** | Best local cutout quality for BG workflows | |

### 3.6 Instruction image edit backbone

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **FLUX.1 Kontext Dev** fp8 scaled | FLUX.1 Dev img2img (current install) |
| VRAM | ~12–16GB class (fp8 fits 16GB sequentially) | Similar |
| Speed | Medium | Medium |
| Quality | Strong instruction edits / consistency | Weaker localized instruction following |
| License | **Non-commercial** (Dev) — product must enforce | Non-commercial Flux Dev terms |
| Advantages | Built for edit instructions | Already installed |
| Limitations | License; multi-turn degradation | Global denoise bleed |
| **Why selected** | Best local instruction-edit backbone in 2026 Comfy ecosystem | |

**Reject as optimal sole editor:** current global Flux Dev img2img-only pipeline.

### 3.7 Surgical inpaint / masked edit

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **FLUX.1 Fill / Tools Fill** | Masked Flux Dev / Kontext+mask hybrid |
| VRAM | ~12–16GB fp8 class | Similar |
| Speed | Medium | Medium |
| Quality | Best local masked preserve | More bleed risk |
| License | Flux Tools / BFL terms | Flux Dev |
| Advantages | Outside-mask preservation | Fewer new downloads if Fill missing |
| Limitations | Needs good masks | Weaker structure lock without ControlNet |
| **Why selected** | Correct tool for object/clothing/remove/add | |

### 3.8 Identity preservation

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **PuLID-Flux v0.9.1** (already on disk) | IPAdapter Face Plus + clip_vision |
| VRAM | Moderate add-on | Moderate + clip_vision weights |
| Speed | Mild overhead | Mild overhead |
| Quality | Strong Flux face ID | Flexible style/ID |
| License | PuLID terms | IPAdapter terms |
| Advantages | Installed; matches Flux stack | Good when PuLID fails |
| Limitations | Not a swap model; needs face | Needs clip_vision download (folder empty today) |
| **Why selected** | Best available on-machine identity for Flux edits | |

### 3.9 Face swap (only for `edit.face_swap`)

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **ReActor + InsightFace inswapper_128** (installed) | PuLID-only reidentity (not true swap) |
| VRAM | Low–moderate | — |
| Speed | Fast | — |
| Quality | True swap | Identity edit ≠ swap |
| License | InsightFace / ReActor terms (often non-commercial / restrictive) | — |
| Advantages | Already installed | — |
| Limitations | Ethics/license; artifacts | Wrong tool for “edit expression” |
| **Why selected** | Only when planner task is explicit face swap | |

### 3.10 Face restore

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **CodeFormer** (installed) | GFPGANv1.4 (installed) |
| VRAM | Low | Low |
| Speed | Fast | Fast |
| Quality | Identity-friendlier | Smoother skin |
| License | Respective open licenses | — |
| Advantages | On disk; Impact FaceDetailer integration | On disk |
| Limitations | Can plasticize if overused | Same |
| **Why selected** | Post chain on Quality/Ultra face workflows | |

### 3.11 Depth / pose (structure lock)

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **Depth Anything V2** + OpenPose via controlnet_aux; Flux ControlNet weights (**download**) | ZoeDepth |
| VRAM | Low preprocess + ControlNet add | Similar |
| Speed | Fast preprocess | Fast |
| Quality | Good structure lock | Older depth |
| License | Per model | — |
| Advantages | aux pack already installed | — |
| Limitations | **ControlNet weight folder empty today** | Weaker |
| **Why selected** | Best structure stack once weights downloaded | |

### 3.12 Upscale

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **4x-UltraSharp** + UltimateSDUpscale (installed pack) | SeedVR2 (evaluate) / SUPIR (heavy) |
| VRAM | Low–medium tiles | SeedVR2 higher; SUPIR often too heavy for 16GB |
| Speed | Fast–medium | Slower |
| Quality | Strong practical upscale | Potentially higher with SeedVR2 |
| License | UltraSharp community; USDU pack | Per model |
| Advantages | On disk; fits 16GB tiled | Modern |
| Limitations | Not creative “Magnific” alone | VRAM/time |
| **Why selected** | Best installed default; SeedVR2 as Quality/Ultra candidate | |

### 3.13 Image-to-video backbone

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **Wan 2.2 I2V-A14B** fp8 dual high/low (installed) | Wan 2.2 TI2V-5B (installed) |
| VRAM | Tight on 16GB; fp8 + sequential + modest res | Easier 720p attempts |
| Speed | Slow on 16GB | Faster |
| Quality | Higher motion/detail (quality profiles) | Good speed/quality trade |
| License | **Apache-2.0** | Apache-2.0 |
| Advantages | Already installed; correct dual-expert pattern | Faster Draft |
| Limitations | Runtime/timeouts; need VAE A/B | Lower ceiling |
| **Why selected** | Best local silent I2V quality path; TI2V for Draft/fallback | |

**VAE:** A/B `wan2.2_vae` vs current `wan_2.1_vae`; winner becomes registry default.

**Speed path:** LightX2V 4-step LoRAs (installed) for Draft only — do not mix into Ultra full-step unintentionally.

### 3.14 Video + audio (future optional)

| Field | Best | Second-best |
|-------|------|-------------|
| Model | LTX-2.3 (weights present) | — |
| VRAM | High (often 24GB+ class comfortable) | — |
| Speed | Varies | — |
| Quality | Audio+video | — |
| License | LTX terms | — |
| Advantages | Already downloaded | — |
| Limitations | Poor fit as default on 16GB | — |
| **Why selected** | **Not** default I2V; optional future task when VRAM strategy allows | |

### 3.15 Frame interpolation

| Field | Best | Second-best |
|-------|------|-------------|
| Model | **RIFE 4.x** | FILM |
| VRAM | Low | Low–mid |
| Speed | Fast | Medium |
| Quality | Good 16→24/30 | High quality, heavier |
| License | Open | Open |
| Advantages | Standard post | Quality |
| Limitations | Cannot fix bad motion | Slower |
| **Why selected** | Best speed/quality for post-I2V on 16GB | |

---

## 4. Currently installed vs selected

| Role | Installed today | Selected for engine | Action |
|------|-----------------|---------------------|--------|
| Edit backbone | flux1-dev-fp8 | Kontext Dev fp8 | **Download** Kontext |
| Inpaint | (none dedicated) | Flux Fill | **Download** |
| I2V | Wan I2V 14B fp8 | Wan I2V 14B fp8 | Keep; rebuild workflow/profiles |
| VAE Wan | wan_2.1_vae + wan2.2_vae | TBD A/B | Benchmark |
| Identity | pulid_flux_v0.9.1 | PuLID | Wire |
| Upscale | 4x-UltraSharp | UltraSharp (+ SeedVR2 eval) | Wire |
| Grounding/SAM | missing | GDINO + SAM2 | **Download** |
| Matting | missing | BiRefNet | **Download** |
| clip_vision | empty | needed for IPAdapter fallback | **Download** if using IPAdapter |
| ControlNet weights | empty | Flux depth/pose | **Download** |
| VLM planner | missing | Bind slot `planner.default_model` → Qwen2.5-VL-7B initially | **Download** + set slot |
| Style LoRAs Instagirl* | present | **Disabled** for identity defaults | Exclude |

---

## 5. Example registry rows (illustrative)

```yaml
model_id: backbone.flux_kontext_dev_fp8
version: "kontext-dev-fp8-scaled-2025"
filename: flux1-dev-kontext_fp8_scaled.safetensors
download_source: "https://huggingface.co/Comfy-Org/..."  # pin exact in impl
vram_mb: 12000
disk_mb: 12000
license: "FLUX.1 Non-Commercial"
compatible_workflows: [instruction_edit, style_transfer, character_consistency, background_replace]
replacement_candidates: [backbone.flux_dev_fp8]
benchmark_score: 78
# estimated: true until fixtures run
```

```yaml
model_id: video.wan22_i2v_high_fp8
version: "2.2-i2v-a14b-fp8-scaled"
filename: wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
download_source: "https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B"
vram_mb: 13000  # peak with pair + activations; stage carefully
disk_mb: 14000
license: "Apache-2.0"
compatible_workflows: [video_i2v]
replacement_candidates: [video.wan22_ti2v_5b]
benchmark_score: 80
```

---

## 6. Replacement policy

When Model Manager load fails:

1. Try `replacement_candidates` in order  
2. If replacement changes quality class, attach `meta.model_degraded=true`  
3. If all fail → error with `download_source` and disk/VRAM needs  
4. Upgrades (Kontext → future successor): add new `model_id`, point workflows’ required_models to new id, keep old as candidate — **no API/UI change**

---

## 7. Disk / VRAM planning (16GB machine)

Rough additional downloads for P1 (order of magnitude):

- Kontext fp8 ~10–12GB  
- Flux Fill ~10–12GB (if separate)  
- Qwen2.5-VL-7B quant ~5–8GB  
- SAM2 + GroundingDINO ~1–3GB  
- BiRefNet ~1GB  
- ControlNet + clip_vision ~1–5GB  

Operators should plan **≥100GB free** beyond current shared model store for comfortable P1+P2.

Peak runtime VRAM must stay within sequential policy; Ultra profiles that exceed ~14GB estimated peak are invalid registrations.

Live totals are exposed at `/api/health` → `ai_engine.health.disk` (Phase 7).

---

## 8. Phase 7 scorecard (estimated)

All scores below are **`benchmark_estimated: true`** until GPU fixtures promote them. Source of truth in code: `backend/ai_engine/ops/scorecard.py`.

### Models (selected)

| model_id | score | disk_mb est | status intent |
|----------|------:|------------:|---------------|
| backbone.flux_dev_fp8 | 74 | 12000 | installed (runtime) |
| backbone.flux_kontext_dev_fp8 | 78 | 12000 | missing (preferred edit) |
| video.wan22_i2v_high_fp8 | 80 | 14000 | installed |
| video.wan22_ti2v_5b | 68 | 10000 | installed (fallback) |
| vlm.qwen25_vl_7b | 82 | 8000 | missing (slot target) |
| matting.birefnet | 78 | 1000 | missing |
| seg.sam2 | 75 | 900 | missing |
| upscale.ultrasharp_4x | 70 | 70 | installed |

### Stable workflows (P1/P2)

| workflow_ref | score |
|--------------|------:|
| video_i2v.v1 | 76 |
| clothing_replace.v1 | 74 |
| object_replace.v1 / inpaint.v1 | 73 |
| instruction_edit.v1 | 72 |
| background_remove.v1 | 68 |
| video_i2v.v0_legacy | 65 |
| face_swap.v1 (beta) | 55 |
| video_v2v.experimental / video_extend.experimental | 0 |

Promotion rule: after fixture run, set `benchmark_estimated=false` on the record and update this table.

---

## 9. Anima LLLite (Saga part structure control)

ControlNet-LLLite for Anima, used only by Saga **part** redraws (`control=` on `/api/saga`, `/api/jobs`, `/api/saga/batch`).

| Item | Value |
|------|-------|
| Node pack | `custom_nodes/ComfyUI-Anima-LLLite` (kohya-ss), node `AnimaLLLiteApply_sdscripts` (renamed upstream so it does not collide with ComfyUI core's own `AnimaLLLiteApply` / `ModelPatchLoader`; keep the pack updated) |
| Weights (`models/controlnet`) | `anima-lllite-inpainting-v2.safetensors` (control `inpaint`), `anima-lllite-any-test-like-v2.safetensors` (`lines`, `gray`), `anima-lllite-lineart-1.safetensors` (`lines` with `control_model=lineart1`) |
| Base model | Works on `anima-aesthetic-v1.1` (our UNET, not swapped); trained on anima-base v1.0, so quality may be slightly lower than on base |
| License | CircleStone Labs Non-Commercial |

Polarity and inputs:

- **lines**: black lines on a white background (lineart-1 needs exactly this; any-test-like v2 was trained with lines inverted half the time, so either works there). Derived lines are pixels at luminance ≤ 70 fully dark, fading out by 110; transparent pixels are white.
- **gray**: plain luminance of the picture (keeps shading and folds, lets the colour change).
- **inpaint**: 4-channel (picture + mask); the node binarises the mask at 0.5 and blanks the RGB inside it, so inside the layer it only gives edge/colour context. Structure inside the redraw comes from `lines` / `gray`.

Settings: the model card samples use strength 1.0, start 0, end 1.0, and the inpaint sample uses img2img at denoise 0.8 with `SetLatentNoiseMask` (er_sde / simple / cfg 4 / 30 steps — same as Saga). Use img2img with a mask, otherwise colours may shift slightly. API defaults: strength 1.0 (0–2, per control allowed), `control_start` 0, `control_end` 1.0. Clients should use `control_end` 0.8–1.0 and denoise 0.7–0.9; with control, denoise may go up to **0.95** (0.8 without).

Resolution: v2 trained on 512²–1536² buckets (step 16). The node resizes the control image and mask to the latent size, so the same sheet panels serve the base pass and the 1.5× HD pass.

Measured on the RTX 5060 Ti 16 GB (part layer 832×1216, HD, same seed): no control at denoise 0.5 took 70 s with a 14.7 GB VRAM peak; inpaint at 0.75 took 82 s with a 13.6 GB peak. Unmasked pixels drifted by a mean of 2.4/255 in both cases.
