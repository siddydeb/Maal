# Production URLs (canonical)

Use these links for every push / deploy. Do not create new Vercel or Render projects unless the user explicitly asks.

| Service | URL | Notes |
|--------|-----|--------|
| Frontend (Vercel) | https://frontend-six-chi-37.vercel.app | Project: `frontend` on team `wellswenger-1737s-projects` |
| API (Render) | https://wan-studio-api.onrender.com | Service: `wan-studio-api` (`srv-d9ot8spt0dsc73bqjv0g`) |
| GitHub | https://github.com/wellswenger-svg/Maal | Branch: `main` |

## Env wiring

- Vercel `VITE_API_URL` = `https://wan-studio-api.onrender.com`
- Render `CORS_ORIGINS` = `https://frontend-six-chi-37.vercel.app`
- Render `COMFYUI_URL` = current Cloudflare tunnel to local ComfyUI `:8188` (changes when tunnel restarts)

### Admin Controls UI (admin owner only)

PINs are **not** stored in this repo. Set `WAN_PINS` (bcryptjs hashes only) + `WAN_AUTH_SECRET` on Render only.
If those env vars are empty, unlock is disabled and nobody can open the app.

After unlock, the admin owner sees **Controls** in the footer:

| Button | Needs on Render |
|--------|-----------------|
| Refresh health | nothing |
| Restart API | `RENDER_API_KEY` (= `render=` from `tokens&cmd`) |
| Set tunnel + deploy | same `RENDER_API_KEY` |
| Restart Comfy | `GPU_AGENT_URL` + `GPU_AGENT_SECRET` (gpu_agent running on GPU PC) |
| Scrub residues | Comfy reachable |

Optional: `RENDER_SERVICE_ID` (defaults to `srv-d9ot8spt0dsc73bqjv0g`).

If the API is completely unreachable, Controls cannot run — use `python scripts/restart_wan.py` from a clone.

## Remote restart (any PC with this clone)

Needs gitignored `tokens&cmd` with `render=<Render API key>` (same file as deploys).

```bash
# Bounce the API + wait until health responds
python scripts/restart_wan.py

# Only wake / poll (no restart)
python scripts/restart_wan.py --wake

# After you restart cloudflared on the GPU PC — push the new URL to Render
python scripts/restart_wan.py --tunnel https://YOUR-NEW-SUBDOMAIN.trycloudflare.com
```

### Optional: restart Comfy from another PC

On the **GPU PC** (keep running):

```bash
set GPU_AGENT_SECRET=pick-a-long-secret
set GPU_COMFY_CMD=…your usual Comfy launch command…
python scripts/gpu_agent.py
# second terminal: cloudflared tunnel --url http://127.0.0.1:8799
```

Add to `tokens&cmd` on every clone:

```
gpu_agent=https://….trycloudflare.com
gpu_agent_secret=pick-a-long-secret
```

Then from the other PC:

```bash
python scripts/restart_wan.py --gpu-restart
python scripts/restart_wan.py              # also restart Render API
```

### Leave this PC / recover from elsewhere

Your GPU **must stay on this machine**. Remote Controls cannot invent a GPU on another laptop.

| Failure | What to do away from this PC |
|--------|------------------------------|
| API asleep / crashed | Controls → **Restart server**, or `python scripts/restart_wan.py` |
| Comfy stuck / VRAM full | Controls → **Restart GPU** (needs watchdog / gpu_agent) |
| Tunnel URL changed | Watchdog auto-updates Render `COMFYUI_URL` |
| This PC slept / powered off | Nothing remote can help until it wakes — disable sleep while away |

On the **GPU PC**, install once so the stack starts **hidden at every login**:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_mobile_autostart.ps1
```

That keeps Comfy + Cloudflare tunnels + gpu_agent alive in the background, and pushes new tunnel URLs to Render so phone / other-PC gens keep working.

```powershell
# status only
powershell -ExecutionPolicy Bypass -File scripts/install_mobile_autostart.ps1 -Status
# remove login shortcut
powershell -ExecutionPolicy Bypass -File scripts/install_mobile_autostart.ps1 -Uninstall
```

Runtime: hidden `D:\YtAuto\svc\` (override with `WAN_RUNTIME_DIR`) holds copies of the watchdog (`svc_main.py`) and gpu_agent (`svc_agent.py`), a DPAPI-encrypted subset of `tokens&cmd` (`cfg.bin`, no github/vercel keys), `start.vbs`, logs (`svc.log`, `trainer_worker.log`) and tunnel state (`tunnel_*.url/.log`). Scheduled task `UserSvcKeepalive` (sign-in + every 5 min) runs `start.vbs`, so the stack starts, heals and restarts with the repo folder locked. The watchdog re-syncs the copies on each start while the repo is readable; character-LoRA training still needs the repo unlocked. Leave this PC on (disable sleep).

## Removed / do not use

- ~~https://frontend-lake-five-90.vercel.app~~ — duplicate under AtomikAudio `atomik1`; deleted
