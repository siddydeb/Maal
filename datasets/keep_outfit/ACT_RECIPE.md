# Img acts — recipe gate (public Flux LoRAs)

## Status
Boobs + ass **frozen**. Wet **parked** (LoRA swap pending). Acts are **open**.

**LoRA:** oral slot → `flux_pov_a_v1.safetensors` (getphat `bl0j0`, 584MB). Tiny 19MB Flux LoRAs do not drive the act.  
Install: [`GPU_LORA_SWAP_WET_ACT.md`](GPU_LORA_SWAP_WET_ACT.md)

First BJ gate (old getphat POV): `tmp_test/18000_act_bj_gate/review/` — identity weak; re-gate after swap + face lock.

## Stack (edit_runner defaults)
| Knob | BJ | HJ | Titjob |
|------|----|----|--------|
| Preset | `act_bj` | `act_hj` | `act_titjob` |
| Graph | Flux Dev **img2img** (`act_i2i_dev`) | same | same |
| Denoise | **≥0.94** | ≥0.94 | ≥0.94 |
| Remover | clothes_remover | clothes_remover | clothes_remover |
| Unlock | nsfw_unlock | nsfw_unlock | nsfw_unlock |
| Act LoRA | `flux_pov_a_v1` (getphat bl0j0) | `flux_hands_detail_v1` | breast reshape + anatomy |
| Anatomy | `flux_anatomy_m_v1` | same | same |
| Post | face lock | face lock | face lock |

## Gate rule
Pass if: **same face/identity**, act readable (composition correct), anatomy not melted, no random outfit leftover fighting the act.

## Starts
Prefer **face-forward / clear face** crops.  
`tmp_test/act_probe/starts/` · outs `tmp_test/18000_act_bj_gate/`

## Do not
- Churn bust/ass frozen recipes for acts
- Re-open wet until a stronger LoRA lands
