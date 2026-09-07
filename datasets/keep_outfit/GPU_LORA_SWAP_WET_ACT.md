# GPU LoRA swap — wet + acts (2026-09-07)

Do this on the **GPU / Comfy PC**, then pull this repo on any control PC.

Boobs/ass stay on `flux_kontext_figure_reshape_v1` — **do not remove**.

## Why
Public 18–37MB Flux “concept” LoRAs (Invisidude / Keltezaa) do not punch hard enough on img2img.  
Use the large Flux.1 D concept weights: Lurulf wet (292MB) + getphat POV BJ (584MB). Identity still needs face lock — nothing public does both perfectly.

## 1. Download into Comfy loras folder

Target folder (adjust if your share path differs):

```text
E:\Comfy-Desktop\ComfyUI-Shared\models\loras\
```

| Slot | New on-disk name (required) | Source | Notes |
|------|----------------------------|--------|--------|
| **wet_shirt** | `Wet_ClothesHair_FLUX.safetensors` | [Civitai 1459149](https://civitai.com/models/1459149) Lurulf V1 | **292MB** — only large Flux.1 D wet-clothes LoRA. 36MB Invisidude is white-shirt only. |
| **oral_pov** | `flux_pov_a_v1.safetensors` | [Civitai 678730](https://civitai.com/models/678730) getphat POV BJ | **584MB** `bl0j0`. 19MB Keltezaa / mrkakapopoloch rank-8 files do not drive img2img. |
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

| Weak file (do not use) | Why |
|------------------------|-----|
| `WetshirtForFlux-1.4.safetensors` | 36MB Invisidude — white tops only |
| `flux_pov_bj_v2.safetensors` | 19MB Keltezaa rank-8 — does not drive the act |

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
| Wet Shirt | `Wet_ClothesHair_FLUX.safetensors` (+ optional see-through) |
| BJ | `flux_pov_a_v1.safetensors` + anatomy + remover |

Then re-gate:

- Wet: face-forward light tops  
- BJ: face-forward starts under `tmp_test/act_probe/starts/`

## Do not

- Replace reshape / bust / ass weights  
- Train custom LoRAs for this step  
- Expect Kontext-like identity lock from these Dev LoRAs — still img2img + face lock
