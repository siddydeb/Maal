# Gold recipe v1 — BACKUP (do not delete)

Canonical snapshot: `datasets/keep_outfit/backups/gold_recipe_v1_2026-09-04/`

| Knob | Value |
|------|--------|
| Preset | `enhance_boobs` |
| Graph | Flux Kontext `keep_outfit_kontext` |
| LoRA | `flux_kontext_figure_reshape_v1` @ **0.82** |
| Unlock | `aidmaNSFWunlock` @ **0.95** (clothed) |
| Guidance | **3.8** |
| Post | face lock only |

Restore: copy `presets_enhance_boobs.json` slice + `lora_snapshot.json` strengths + `edit_runner.py` from that folder if v2 regresses golds.

**Superseded:** product defaults are now **v2** (`FROZEN_RECIPE.md` — unlock 0.55 + cloth retry). Keep this file as rollback only.
