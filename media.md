# Media locations

Where images and videos for **this project** (Wan Studio / Maal) live.

Last checked on this PC: **2026-08-18**.

**Lasting library is not on this PC.** It is MongoDB Atlas (GridFS). Local folders are leftovers, review copies, Comfy working dirs, or static app assets.

## Wipe local media (keep Mongo)

From the repo root:

```bash
npm run wipe
```

Dry run (print paths, delete nothing):

```bash
npm run wipe -- --dry-run
```

This deletes local copies only. **MongoDB Atlas / GridFS is never touched.**

| Wiped | Left alone |
|-------|------------|
| `tmp_test/train/` images, thumbs, job JSON | Atlas library |
| `tmp_test/review/` | App icons, pose helper PNG |
| repo `outputs/`, `temp/`, `tmp/`, `temptest_assets/` | LoRAs / models / ControlNets |
| Comfy `input/`, `output/`, `temp/` | `tmp_test` tunnel logs and helper scripts |
| `%TEMP%\wan_thumb_*` | `D:\YtAuto\Newyttest` (other project) |
| `Downloads\wan_img_*` / `wan_vid_*` / `wan_out_*` | Pictures / Desktop unless named `wan_*` |

---

## 1. Lasting library (cloud — not this PC)

| Where | What |
|-------|------|
| **MongoDB Atlas**, database `wan_studio`, GridFS bucket `media` | All generated images/videos, thumbs, test refs, test inputs |
| Render API `https://wan-studio-api.onrender.com` | Serves that media; does not keep files on this machine |
| Frontend `https://frontend-six-chi-37.vercel.app` | Displays media from the API; no server-side media store |

There is **no local MongoDB** on this PC (`C:\data\db` and typical Mongo install paths do not exist).

SPA download filenames (if you save from the UI): `wan_{kind}_{id}.{png|mp4|…}` → usually `C:\Users\User\Downloads\`.

---

## 2. On this PC — generated / review images (real files)

### `D:\YtAuto\contrnt\tmp_test\train\`

Only folder that currently has project gens on disk.

- **35 files** when last scanned (png + jpg thumbs)
- Local test/review copies pulled from Atlas, **not** the source of truth
- Examples: `job_*.png`, `*_thumb.jpg`, `ref_*_thumb.jpg`, `input_*_thumb.jpg`
- Also job JSON next to them (`job_*.json`, `last_job.json`)

Gitignored (`tmp_test/` in `.gitignore`).

### `D:\YtAuto\contrnt\tmp_test\review\`

- Written by `scripts/review_test.py` (latest test runs + refs)
- **Does not exist right now** (nothing downloaded there)
- Wipe with: `python scripts/review_test.py --wipe`

---

## 3. On this PC — Comfy working folders (empty after a successful scrub)

Canonical Comfy root (from `.env` `COMFYUI_DIR`):

`E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI`

| Path | Role | Last scan |
|------|------|-----------|
| `E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI\input` | Uploaded start images (`wan_in_…`) | **0 files** |
| `E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI\output` | Generated img/vid (`flux_*`, `wan_i2v14*`, …) | **0 files** |
| `E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI\temp` | Mid-run frames / intermediates | **0 files** |

These fill during a job, then get wiped (`ZERO_RESIDUE=true`). They can refill if a run crashes before scrub, or if the tunnel/scrub route is down.

Remote scrub: API → Cloudflare tunnel → Comfy `POST /wan_studio/scrub`.  
Manual: `POST /api/scrub` on the API.

---

## 4. On this PC — not generations

| Path | What |
|------|------|
| `D:\YtAuto\contrnt\backend\assets\pose_all_fours_openpose.png` | Pose helper image in the repo |
| `D:\YtAuto\contrnt\frontend\public\icons\` | App icons (`icon-192.png`, `icon-512.png`) |
| `D:\YtAuto\contrnt\frontend\public\favicon.svg` | Favicon |
| `D:\YtAuto\contrnt\frontend\src\assets\` | Vite leftovers (`hero.png`, `react.svg`, `vite.svg`) — not pipeline media |
| `D:\YtAuto\contrnt\frontend\dist\` | Built copies of the public icons (after `npm run build`) |
| `E:\Comfy-Desktop\ComfyUI-Shared\models\` | Weights / LoRAs / ControlNets — **not** outputs |
| `E:\Comfy-Desktop\ComfyUI-Shared\models\loras\` | Image/video LoRA weights |
| `E:\Comfy-Desktop\ComfyUI-Shared\models\controlnet\` | ControlNet weights |
| `E:\Comfy-Desktop\ComfyUI-Cache\download-cache` | Model download cache — **no img/vid** last scan |
| `E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI\custom_nodes\` | Sample images shipped with nodes, not your gens |
| `%TEMP%\wan_thumb_*` | Short-lived ffmpeg thumb dirs during API work; **none sitting there** last scan |

Gitignore also allows these repo folders (they were **not present** last scan):

- `D:\YtAuto\contrnt\outputs\`
- `D:\YtAuto\contrnt\temp\`
- `D:\YtAuto\contrnt\tmp\`
- `D:\YtAuto\contrnt\temptest_assets\`

---

## 5. Places that can get copies, but had none last scan

| Path | When it would have files |
|------|--------------------------|
| `C:\Users\User\Downloads\` as `wan_img_….png` / `wan_vid_….mp4` | Only if you hit download in the app. **No `wan_*` files there** last scan |
| `C:\Users\User\Pictures` | If you save there manually |
| `C:\Users\User\Desktop` | If you save there manually |
| `C:\Users\User\Videos` | If you save there manually |
| Browser profile cache | Only if you open the SPA on this PC |
| `C:\Users\User\.cursor\projects\d-YtAuto-contrnt\agent-transcripts\` | Chat text (prompts), not binary media |

---

## 6. Not media, but next to Comfy / the repo

| Path | What |
|------|------|
| `E:\Comfy-Desktop\ComfyUI-Installs\Khelukhiladi\ComfyUI\user\` | Comfy logs + UI DB (`comfyui*.log`, `comfyui.db`) — not gens |
| `D:\YtAuto\svc\tunnel_8188.url` (and `.log`) | Cloudflare tunnel URL/log for Comfy `:8188` (hidden runtime folder; plus `svc.log`, `trainer_worker.log`) |
| `D:\YtAuto\contrnt\.env` | Config + secrets — **not** gen media |
| `D:\YtAuto\contrnt\tokens&cmd` | Deploy tokens — **not** gen media |

---

## 7. Not this project

`D:\YtAuto\Newyttest\` is a **different** app (YouTube test). It has its own `data\assets\`, `uploads\`, `temp\` media. Do not treat those as Wan Studio files.

---

## Quick map

```
Cloud (source of truth)
  MongoDB Atlas  wan_studio / GridFS bucket "media"
       ▲
       │  served by Render API
       │
This PC
  D:\YtAuto\contrnt\tmp_test\train\     ← review copies (HAS files)
  D:\YtAuto\contrnt\tmp_test\review\    ← review script (empty / missing)
  E:\…\ComfyUI\input                    ← job start images (empty)
  E:\…\ComfyUI\output                   ← job outputs (empty)
  E:\…\ComfyUI\temp                     ← job temps (empty)
  C:\Users\User\Downloads\wan_*         ← only if you downloaded (none)
```

**“Zero residue”** = Comfy disk after a finished job. It does **not** mean nothing is stored in Atlas.
