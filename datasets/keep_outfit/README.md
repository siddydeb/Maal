# Keep-outfit training lab (local)

Paired start→target data for a custom Flux reshape LoRA.  
Protocol: [`../../KEEP_OUTFIT_LORA_TRAIN.md`](../../KEEP_OUTFIT_LORA_TRAIN.md) · current path: [`PATH_B_THEN_A.md`](PATH_B_THEN_A.md)

PNGs are gitignored (`*.png`). README / captions / notes are tracked.

## Interim reshape weight (before custom train)

**Replace** (do not stack) `Huge_natural_breasts_for_FLUX_v2` with:

- **[Bigger breasts and butts — Flux Kontext LoRA](https://civitai.com/models/1802814/bigger-breasts-and-butts-flux-kontext-lora)**
- File: `flux_clothed_figure_volume_v1.safetensors`
- Map: `breast_enhance` / `bust_enhance` → that file in `private/lora_files.py`
- Strength: **0.55–0.70**
- Skip other public bust LoRAs for keep-outfit for now

## Layout

| Folder | Role |
|--------|------|
| `gold/` | Best clothed pairs (`###_start.png` + `###_target.png` + `###.txt`) |
| `hard/` | Same format; darker / straps / hands |
| `holdout/` | Starts only — never train |
| `reg/` | Near-identity pairs |
| `scores/` | `scorecard.csv` |
| `hand_gold/` | Inbox starts for semi-gold / future pairs |
| `refs_qa/` | Look-only refs — **not** train targets |
| `staging/rejected_notes/` | Why past evals were not gold |

**Recipes / GPU install:** [`FROZEN_RECIPE.md`](FROZEN_RECIPE.md) · [`ASS_RECIPE.md`](ASS_RECIPE.md) · [`GPU_LORA_SWAPS.md`](GPU_LORA_SWAPS.md)

**Never overwrite gold targets.** If a start gets a new keeper, append the next free `###` id (keep prior pairs).

## Caption (every gold/hard pair)

```text
photo of a woman, keep same clothes, volume under cloth, fabric may drape, photoreal
```

## Never put in gold/hard targets

- Ref-gallery bodies/outfits
- Nude / identity-swap / nipple-poke junk
- Rectangle paste / melted neckline
