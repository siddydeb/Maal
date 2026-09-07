# GPU LoRA swap — wet + acts (2026-09-07)

Do this on the **GPU / Comfy PC**, then pull this repo on any control PC.

Boobs/ass stay on `flux_kontext_figure_reshape_v1` — **do not remove**.

## Why
Wet (Lurulf) + act POV (old getphat `bl0j0`) failed keep-identity gates.  
Replace those two slots with stronger public Flux.1 D LoRAs.

## 1. Download into Comfy loras folder

Target folder (adjust if your share path differs):

```text
E:\Comfy-Desktop\ComfyUI-Shared\models\loras\
```

| Slot | New on-disk name (required) | Source | Notes |
|------|----------------------------|--------|--------|
| **wet_shirt** | `WetshirtForFlux-1.4.safetensors` | [Civitai 724562](https://civitai.com/models/724562) — Flux v1.4 | Invisidude wet T-shirt; best on light tops |
| **oral_pov** | `flux_pov_bj_v2.safetensors` | [HF Keltezaa/blowjob-pov-flux-lora](https://huggingface.co/Keltezaa/blowjob-pov-flux-lora) | Rename whatever file HF ships → this name |
| see_through (keep) | `See_through_clothes_FLUX.safetensors` | [Civitai 1028424](https://civitai.com/models/1028424) v2 | Optional sheer; use lower strength if stacked |
| male_anatomy (keep) | `flux_anatomy_m_v1.safetensors` | [Civitai 824972](https://civitai.com/models/824972) Dynamic Penis v2 | Keep |
| hands (keep) | `flux_hands_detail_v1.safetensors` | [Civitai 891074](https://civitai.com/models/891074) | Keep for HJ |
| reshape (keep) | `flux_kontext_figure_reshape_v1.safetensors` | [Civitai 1802814](https://civitai.com/models/1802814) | Boobs/ass frozen |

### Optional download helpers (from repo root, on GPU PC)

```powershell
cd "D:\Resume projects\Maal"   # or your clone path
git pull origin main
python private/download_wet_sheer_loras.py
python private/download_flux_act_loras.py
```

If HF remote names differ, download manually in browser and **rename** to the exact names in the table.

### Rollback (leave on disk, unused)

| Old file | Was used for |
|----------|----------------|
| `Wet_ClothesHair_FLUX.safetensors` | wet_shirt (Lurulf) |
| `flux_pov_a_v1.safetensors` | oral_pov (getphat bl0j0) |

## 2. Pull code (this repo already wires the new names)

```powershell
git pull origin main
```

Wired in:

- `private/lora_files.py` — `wet_shirt` / `oral_pov` filenames  
- `private/catalog_loras.py` — download sources  
- `private/download_wet_sheer_loras.py` / `private/download_flux_act_loras.py`

## 3. Restart Comfy

Restart ComfyUI (or use Admin Controls → Restart Comfy) so it rescans `models/loras/`.

## 4. Refresh API overlay (control PC)

From a clone with `tokens&cmd`:

```powershell
python scripts/sync_runtime_overlay.py
```

Confirms `/api/health` shows overlay mounted.

## 5. Smoke check

| Preset | Expect in job label / LoRA list |
|--------|----------------------------------|
| Wet Shirt | `WetshirtForFlux-1.4.safetensors` (+ optional see-through) |
| BJ | `flux_pov_bj_v2.safetensors` + anatomy + remover |

Then re-gate:

- Wet: face-forward light tops  
- BJ: face-forward starts under `tmp_test/act_probe/starts/`

## Do not

- Replace reshape / bust / ass weights  
- Train custom LoRAs for this step  
- Expect Kontext-like identity lock from these Dev LoRAs — still img2img + face lock
