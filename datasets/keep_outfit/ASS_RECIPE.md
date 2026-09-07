# Ass enhance — FROZEN product recipe (2026-09-07)

## Status
**LOCKED.** Do not churn `enhance_ass` knobs. Custom LoRA **not needed**.  
Same public reshape LoRA as bust (`flux_kontext_figure_reshape_v1`).

Gate: `tmp_test/18000_ass_gate/` — **9/9** user gold.  
Keepers: `datasets/keep_outfit/gold/055`–`063` · index `gold/ASS_GATE_PROMOTE.json`

**Policy: never replace a gold `*_target.png`.** Append a new id if a start gets another keeper out.

Boobs remain frozen separately — see [`FROZEN_RECIPE.md`](FROZEN_RECIPE.md) (v2).

## Recipe (do not churn)
| Knob | Value |
|------|--------|
| Preset | `enhance_ass` |
| Graph | Flux Kontext ReferenceLatent (`keep_outfit_kontext`) |
| LoRA | `flux_kontext_figure_reshape_v1.safetensors` via `ass_enhance` |
| Strength | **`1.35`** (`KEEP_OUTFIT_ASS_STRENGTH`) |
| Unlock | **`0.55`** clothed (`aidmaNSFWunlock`) |
| Guidance | **`4.2`** |
| Seed | **`99`** primary (when caller omits seed) |
| Cloth retry | **off** for hip jobs (bust-only salvage) |
| Post | face lock only |
| Prompt | same opaque outfit; larger ass/hips under cloth; do not change breasts |

## Do not
- Touch ass strength / guidance / unlock without a new failure class
- Stack more public hip/bust LoRAs on this path
- Re-open FORCE_I2I / offline warp as product
- Churn bust frozen v2 to fix an ass miss
- Train custom LoRA unless holdout regresses hard
