# Keep-outfit / Boobs — FROZEN product recipe v2 (2026-09-05)

## Status
**LOCKED.** Do not churn boobs / keep-outfit bust. Custom LoRA **not needed**.

Ass enhance is also **LOCKED** — see [`ASS_RECIPE.md`](ASS_RECIPE.md) (strength 1.35 / seed 99 / g 4.2; gold `055`–`063`).

Gate: `tmp_test/18000_recipe_lock_v2/` — 31/31; all gold except **#30** (leave be).

Gold pairs saved: `datasets/keep_outfit/gold/`
- **002–030** — lock_v2 outs (latest gate; #30 leave-be `21865` not included)
- **031–052** — prior golds kept (pre-lock_v2 targets; same starts, older outs — do not drop)
- **053–054** — hard_final #6/#8 salvage golds (kept alongside)
- **055–063** — ass gate keepers (append-only; do not churn for bust)
- Holdout ref only: `holdout/h03_target_lock_v2.png`
- Index: `gold/KEEP_PRIOR_GOLDS.json`, `gold/LOCK_V2_PROMOTE.json`, `gold/ASS_GATE_PROMOTE.json`

**Policy: never replace a gold `*_target.png`.** Append a new id if a start gets another keeper out.

## Recipe (do not churn)
| Knob | Value |
|------|--------|
| Preset | `enhance_boobs` |
| Workflow | `keep_outfit_reshape.v1` |
| Graph | Flux Kontext ReferenceLatent (`keep_outfit_kontext`) |
| LoRA | `flux_kontext_figure_reshape_v1.safetensors` (Civitai 1802814) |
| Strength | `0.82` (`breast_enhance` / `bust_enhance`) |
| Unlock | **`0.55`** clothed (`aidmaNSFWunlock`) |
| Guidance | `3.8` primary; cloth-retry salvage may use `4.2` |
| Cloth retry | if undress/tear or odd volume → seeds `7/99/21`, optional breast `0.95` |
| Post | face lock only (no torso paste) |
| Prompt | opaque same outfit, tighter fit, massive bust under cloth; **no sheer / no see-through** |
| Seed | `42` primary |

## Same as early golds?
**Mostly same stack — not identical knobs.**

| | Early golds (v1) | Locked now (v2) |
|--|------------------|-----------------|
| Kontext + reshape LoRA @ 0.82 | yes | yes |
| face lock | yes | yes |
| clothed unlock | **0.95** | **0.55** |
| cloth seed retry | no | **yes** |
| “no sheer / no see-through” | no | **yes** |

So: same product path (Kontext reshape @ 0.82 + face lock). New golds differ slightly from early golds because unlock is lower and hard cases can auto-retry — both are gold-quality; **v2 is the locked product recipe**.

## Known leave-be
- **#30** (`inputs_secret_photo_21865` / old hard #15) — tip/sheer; do not churn recipe to chase it.

## Backup
- v1 snapshot (unlock 0.95, no retry): `backups/gold_recipe_v1_2026-09-04/` · `GOLD_RECIPE_v1_BACKUP.md`
- Probe notes: `RECIPE_v2_PROBE.md` (now promoted)

## Do not
- Touch boobs / keep-outfit bust knobs without a new failure class
- Stack more public bust LoRAs
- Re-open denoise/strength loops
- Train custom LoRA unless holdout regresses hard
- Revert unlock to 0.95 (undresses hard saree/crop)
