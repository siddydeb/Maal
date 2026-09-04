# Keep-outfit / Boobs — FROZEN product recipe (2026-09-04)

## Status
Frozen for product. Custom LoRA **not needed** while this holds.
Ass enhance is a separate track (black saree still undresses — not ready).

Gate evidence: `scores/kontext2_gate.txt`, `inputs_kontext2_gate.txt`, freeze + remaining gates.

## Recipe (do not churn)
| Knob | Value |
|------|--------|
| Preset | `enhance_boobs` |
| Workflow | `keep_outfit_reshape.v1` |
| Graph | Flux Kontext ReferenceLatent (`keep_outfit_kontext`) |
| LoRA | `flux_kontext_figure_reshape_v1.safetensors` (Civitai 1802814) |
| Strength | `0.82` (`breast_enhance` / `bust_enhance`) |
| Post | face lock only (no torso paste) |
| Prompt | opaque same outfit, tighter fit, massive bust under cloth |
| Custom LoRA | **not needed** while this holds |

## Gold look target
`tmp_test/18000_volume_kontext2/9.png` / gold pairs `003–015`

## Do not
- Stack more public bust LoRAs
- Re-open denoise/strength loops without a new failure class
- Train custom LoRA unless holdout regresses hard
