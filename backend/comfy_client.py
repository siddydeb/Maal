"""
ComfyUI HTTP/WS client with zero-residue scrubbing.

After every job:
  1. Pull media bytes into RAM
  2. Secure-wipe ComfyUI input + all job output files on disk
  3. Delete that prompt from ComfyUI history
  4. Sweep leftover wan_* artifacts under input/output/temp
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

import httpx
import websockets

from backend.config import Settings
from backend import scrub
from backend.workflows_wan import (
    build_i2i_prompt,
    build_i2v_prompt,
    build_kontext_edit_prompt,
    fit_dims,
)


class ComfyUIError(RuntimeError):
    pass


# --- ComfyClient helpers live on the class below ---


def _prep_edit_image(image_bytes: bytes, max_w: int, max_h: int) -> tuple[bytes, int, int]:
    """RGB PNG, aspect-fit inside max box (no center crop)."""
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    w, h = fit_dims(img.width, img.height, max_w, max_h)
    if img.size != (w, h):
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), w, h


class ComfyClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        base = settings.comfyui_url.rstrip("/")
        self.base = base
        parsed = urlparse(base)
        self.ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.host = parsed.netloc
        self.timeout = settings.comfyui_timeout_sec
        self._comfy_root: Optional[Path] = (
            Path(settings.comfyui_dir) if settings.comfyui_dir else None
        )

    async def health(self, *, timeout: float = 5.0, retries: int = 1) -> bool:
        """Reachability probe. Use retries/timeout on generate paths (tunnel blips)."""
        attempts = max(1, int(retries))
        for i in range(attempts):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    r = await client.get(f"{self.base}/system_stats")
                    if r.status_code != 200:
                        raise ComfyUIError(f"system_stats HTTP {r.status_code}")
                    if self._comfy_root is None:
                        self._comfy_root = await self._detect_root(r.json())
                    return True
            except Exception:
                if i + 1 >= attempts:
                    return False
                await asyncio.sleep(0.6 * (i + 1))
        return False

    async def list_lora_filenames(self) -> Optional[set[str]]:
        """Basenames Comfy can load from /models/loras (None if unreachable)."""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(f"{self.base}/models/loras")
                if r.status_code != 200:
                    return None
                names: set[str] = set()
                for item in r.json() or []:
                    if not isinstance(item, str) or not item:
                        continue
                    # Comfy may return relative paths under loras/
                    names.add(Path(item).name)
                    names.add(item.replace("\\", "/"))
                return names
        except Exception:
            return None

    async def ensure_root(self) -> Optional[Path]:
        if self._comfy_root and self._comfy_root.is_dir():
            return self._comfy_root
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base}/system_stats")
                if r.status_code == 200:
                    self._comfy_root = await self._detect_root(r.json())
        except Exception:
            pass
        return self._comfy_root

    async def _detect_root(self, stats: dict[str, Any]) -> Optional[Path]:
        if self.settings.comfyui_dir:
            p = Path(self.settings.comfyui_dir)
            return p if p.is_dir() else None
        argv = (stats.get("system") or {}).get("argv") or []
        if argv:
            return scrub.detect_comfy_root_from_argv(argv[0])
        return None

    async def generate_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        negative: Optional[str] = None,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        denoise: Optional[float] = None,
        guidance: Optional[float] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        flux_unet: Optional[str] = None,
        edit_graph: str = "img2img",
        denoise_cap: float = 0.85,
        wrap_preserve: bool = False,
        wrap_mode: Optional[str] = None,
        loras: Optional[list] = None,
        controlnet_name: Optional[str] = None,
        control_image_bytes: Optional[bytes] = None,
        control_type: str = "openpose",
        control_strength: float = 0.65,
        control_end: float = 0.85,
        mask_bytes: Optional[bytes] = None,
        pulid_file: Optional[str] = None,
        pulid_weight: float = 0.80,
        pulid_provider: str = "CUDA",
    ) -> tuple[bytes, str]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        image_bytes, width, height = _prep_edit_image(
            image_bytes,
            max_width or self.settings.image_width,
            max_height or self.settings.image_height,
        )
        image_name = await self._upload_image(image_bytes)
        # Drop caller's buffer after upload so fewer copies linger
        del image_bytes
        control_name: Optional[str] = None
        mask_name: Optional[str] = None
        try:
            unet = flux_unet or self.settings.flux_unet
            clip_l = self.settings.flux_clip_l
            t5 = self.settings.flux_t5
            vae = self.settings.flux_vae
            n_steps = steps if steps is not None else self.settings.sampler_steps
            lora_stack = list(loras or [])
            if edit_graph == "kontext":
                # Official Kontext template guidance is ~2.5; raw mode may pass higher
                g = guidance if guidance is not None else (
                    3.0 if getattr(self.settings, "raw_prompt", True) else 2.5
                )
                workflow = build_kontext_edit_prompt(
                    image_name=image_name,
                    positive=prompt,
                    negative=negative,
                    flux_unet=unet,
                    flux_clip_l=clip_l,
                    flux_t5=t5,
                    flux_vae=vae,
                    steps=n_steps,
                    guidance=g,
                    seed=seed,
                    loras=lora_stack,
                )
            else:
                cap = denoise_cap
                if getattr(self.settings, "raw_prompt", True):
                    cap = max(
                        cap,
                        float(getattr(self.settings, "image_denoise_cap", 0.85) or 0.85),
                    )
                if controlnet_name and control_image_bytes:
                    control_name = await self._upload_image(control_image_bytes)
                if mask_bytes:
                    from io import BytesIO

                    from PIL import Image

                    m = Image.open(BytesIO(mask_bytes)).convert("L").resize(
                        (width, height), Image.Resampling.BILINEAR
                    )
                    rgb = Image.merge("RGB", (m, m, m))
                    buf = BytesIO()
                    rgb.save(buf, format="PNG")
                    mask_name = await self._upload_image(buf.getvalue())
                workflow = build_i2i_prompt(
                    image_name=image_name,
                    positive=prompt,
                    negative=negative,
                    flux_unet=unet,
                    flux_clip_l=clip_l,
                    flux_t5=t5,
                    flux_vae=vae,
                    width=width,
                    height=height,
                    steps=n_steps,
                    guidance=guidance
                    if guidance is not None
                    else self.settings.flux_guidance,
                    seed=seed,
                    denoise=denoise
                    if denoise is not None
                    else self.settings.image_denoise,
                    denoise_cap=cap,
                    wrap_preserve=bool(wrap_preserve),
                    wrap_mode=wrap_mode,
                    loras=lora_stack,
                    controlnet_name=controlnet_name if control_name else None,
                    control_image_name=control_name,
                    control_type=control_type,
                    control_strength=control_strength,
                    control_end=control_end,
                    mask_image_name=mask_name,
                    pulid_file=pulid_file,
                    pulid_weight=pulid_weight,
                    pulid_provider=pulid_provider,
                )
            return await self._run_and_fetch(
                workflow,
                prefer=("images", "gifs", "videos"),
                input_name=image_name,
            )
        except Exception:
            await self._full_scrub(input_name=image_name)
            if control_name:
                try:
                    await self._full_scrub(input_name=control_name)
                except Exception:
                    pass
            if mask_name:
                try:
                    await self._full_scrub(input_name=mask_name)
                except Exception:
                    pass
            raise

    async def generate_video(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        negative: Optional[str] = None,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        length: Optional[int] = None,
        steps: Optional[int] = None,
        cfg: Optional[float] = None,
        fps: Optional[int] = None,
        shift: Optional[float] = None,
        unet_high: Optional[str] = None,
        unet_low: Optional[str] = None,
        vae: Optional[str] = None,
        loras_high: Optional[list] = None,
        loras_low: Optional[list] = None,
        high_noise_fraction: Optional[float] = None,
        wait_timeout: Optional[float] = None,
    ) -> tuple[bytes, str]:
        seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        max_w = width or self.settings.video_width
        max_h = height or self.settings.video_height
        image_bytes, out_w, out_h = _prep_edit_image(image_bytes, max_w, max_h)
        image_name = await self._upload_image(image_bytes)
        del image_bytes
        # Oral Wan I2V routinely exceeds the image default; never wait less than 2h.
        wait = float(wait_timeout) if wait_timeout is not None else float(self.timeout)
        wait = max(wait, 7200.0)
        try:
            workflow = build_i2v_prompt(
                image_name=image_name,
                positive=prompt,
                negative=negative,
                unet_high=unet_high or self.settings.wan_unet_high,
                unet_low=unet_low or self.settings.wan_unet_low,
                vae=vae or self.settings.wan_vae,
                clip=self.settings.wan_clip,
                width=out_w,
                height=out_h,
                length=length if length is not None else self.settings.video_length,
                steps=steps if steps is not None else self.settings.video_steps,
                cfg=cfg if cfg is not None else self.settings.video_cfg,
                seed=seed,
                fps=fps if fps is not None else self.settings.video_fps,
                shift=shift if shift is not None else self.settings.wan_shift,
                loras_high=list(loras_high or []),
                loras_low=list(loras_low or []),
                high_noise_fraction=high_noise_fraction,
            )
            return await self._run_and_fetch(
                workflow,
                prefer=("videos", "gifs", "images"),
                input_name=image_name,
                wait_timeout=wait,
            )
        except Exception:
            await self._full_scrub(input_name=image_name)
            raise

    async def interrupt(self) -> None:
        """Ask ComfyUI to stop the current prompt (best-effort)."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"{self.base}/interrupt")
            if r.status_code not in (200, 204):
                raise ComfyUIError(f"Interrupt failed: {r.status_code} {r.text}")

    async def clear_queue(self) -> None:
        """Interrupt current work and drop pending prompts (orphan cleanup)."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                await client.post(f"{self.base}/interrupt")
            except Exception:
                pass
            try:
                await client.post(f"{self.base}/queue", json={"clear": True})
            except Exception:
                pass
            # Second interrupt helps when a sampler ignores the first stop.
            try:
                await asyncio.sleep(0.4)
                await client.post(f"{self.base}/interrupt")
            except Exception:
                pass

    async def _upload_image(self, image_bytes: bytes) -> str:
        name = f"wan_in_{uuid.uuid4().hex}.png"
        files = {"image": (name, image_bytes, "image/png")}
        # temp folder when supported; falls back to input
        data = {"overwrite": "true", "type": "input"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base}/upload/image", files=files, data=data)
            if r.status_code != 200:
                raise ComfyUIError(f"Upload failed: {r.status_code} {r.text}")
            payload = r.json()
            return payload.get("name") or name

    async def _queue(self, prompt: dict[str, Any], client_id: str) -> str:
        body = {"prompt": prompt, "client_id": client_id}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self.base}/prompt", json=body)
            if r.status_code != 200:
                raise ComfyUIError(f"Queue failed: {r.status_code} {r.text}")
            data = r.json()
            if "error" in data:
                raise ComfyUIError(str(data["error"]))
            if "node_errors" in data and data["node_errors"]:
                raise ComfyUIError(f"Node errors: {data['node_errors']}")
            return data["prompt_id"]

    async def _wait_ws(
        self,
        client_id: str,
        prompt_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """Wait for Comfy completion via WS, reconnecting through tunnel drops.

        Cloudflare tunnels often close long-lived sockets while sampling continues.
        On disconnect we poll /history before opening a new WS.
        """
        uri = f"{self.ws_scheme}://{self.host}/ws?clientId={client_id}"
        limit = float(timeout) if timeout is not None else float(self.timeout)
        deadline = asyncio.get_event_loop().time() + max(60.0, limit)
        last_err: Optional[BaseException] = None

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise ComfyUIError("ComfyUI generation timed out")

            # Already finished while we were disconnected?
            if await self._prompt_finished(prompt_id):
                return

            try:
                try:
                    ws_cm = websockets.connect(
                        uri,
                        max_size=32 * 1024 * 1024,
                        open_timeout=30,
                        ping_interval=20,
                        ping_timeout=30,
                        close_timeout=10,
                    )
                except TypeError:
                    ws_cm = websockets.connect(uri, max_size=32 * 1024 * 1024)

                async with ws_cm as ws:
                    while True:
                        remaining = deadline - asyncio.get_event_loop().time()
                        if remaining <= 0:
                            raise ComfyUIError("ComfyUI generation timed out")
                        try:
                            raw = await asyncio.wait_for(
                                ws.recv(), timeout=min(remaining, 20.0)
                            )
                        except asyncio.TimeoutError:
                            if await self._prompt_finished(prompt_id):
                                return
                            if not await self.health():
                                raise ComfyUIError(
                                    "Lost ComfyUI while waiting (tunnel or Comfy down)."
                                )
                            continue
                        if isinstance(raw, bytes):
                            continue
                        msg = json.loads(raw)
                        mtype = msg.get("type")
                        data = msg.get("data") or {}
                        if (
                            mtype == "execution_error"
                            and data.get("prompt_id") == prompt_id
                        ):
                            exc_msg = (
                                (data.get("exception_message") or "").strip()
                                or (data.get("exception_type") or "").strip()
                                or "ComfyUI execution error"
                            )
                            node = data.get("node_type") or data.get("node_id") or "?"
                            raise ComfyUIError(f"{node}: {exc_msg}")
                        if mtype == "executing":
                            if (
                                data.get("prompt_id") == prompt_id
                                and data.get("node") is None
                            ):
                                return
            except ComfyUIError:
                raise
            except Exception as exc:
                last_err = exc
                name = type(exc).__name__
                if "ConnectionClosed" not in name and "ConnectionReset" not in name:
                    # Unknown hard failure — still try history once before giving up.
                    if await self._prompt_finished(prompt_id):
                        return
                    raise ComfyUIError(f"ComfyUI wait failed: {name}: {exc}") from exc
                # Tunnel/WS drop: brief pause then reconnect / poll history.
                await asyncio.sleep(2.0)
                continue

        if last_err:
            raise ComfyUIError(
                f"ComfyUI wait failed: {type(last_err).__name__}: {last_err}"
            ) from last_err
        raise ComfyUIError("ComfyUI generation timed out")

    async def _prompt_finished(self, prompt_id: str) -> bool:
        """True when history shows success (or raises on recorded execution error)."""
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                r = await client.get(f"{self.base}/history/{prompt_id}")
            if r.status_code != 200:
                return False
            hist = r.json()
            if prompt_id not in hist:
                return False
            entry = hist[prompt_id]
            status = entry.get("status") or {}
            if status.get("status_str") == "error" or status.get("completed") is False:
                msgs = status.get("messages") or []
                detail = ""
                for m in msgs:
                    if isinstance(m, (list, tuple)) and len(m) >= 2 and m[0] == "execution_error":
                        detail = str(m[1])
                        break
                raise ComfyUIError(detail or "ComfyUI execution error")
            outputs = entry.get("outputs") or {}
            if outputs:
                return True
            if status.get("completed") is True or status.get("status_str") == "success":
                return True
        except ComfyUIError:
            raise
        except Exception:
            return False
        return False

    async def _history(self, prompt_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(f"{self.base}/history/{prompt_id}")
            if r.status_code != 200:
                raise ComfyUIError(f"History failed: {r.status_code}")
            hist = r.json()
            if prompt_id not in hist:
                raise ComfyUIError("Prompt not found in history")
            return hist[prompt_id]

    async def _delete_history(self, prompt_id: str) -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                await client.post(f"{self.base}/history", json={"delete": [prompt_id]})
            except Exception:
                pass
            try:
                # Extra: clear entire history so earlier runs don't linger either
                if self.settings.scrub_clear_all_history:
                    await client.post(f"{self.base}/history", json={"clear": True})
            except Exception:
                pass

    async def _view(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        qs = urlencode(
            {"filename": filename, "subfolder": subfolder or "", "type": folder_type}
        )
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.get(f"{self.base}/view?{qs}")
            if r.status_code != 200:
                raise ComfyUIError(f"View failed for {filename}: {r.status_code}")
            return r.content

    def _collect_descriptors(self, outputs: dict[str, Any]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for node_out in outputs.values():
            for key in ("images", "gifs", "videos"):
                for item in node_out.get(key) or []:
                    found.append(
                        {
                            "filename": item.get("filename"),
                            "subfolder": item.get("subfolder", ""),
                            "type": item.get("type", "output"),
                        }
                    )
        return found

    async def _run_and_fetch(
        self,
        workflow: dict[str, Any],
        prefer: tuple[str, ...],
        input_name: str,
        wait_timeout: Optional[float] = None,
    ) -> tuple[bytes, str]:
        await self.ensure_root()
        client_id = uuid.uuid4().hex
        prompt_id = await self._queue(workflow, client_id)
        await self._wait_ws(client_id, prompt_id, timeout=wait_timeout)
        history = await self._history(prompt_id)
        outputs = history.get("outputs") or {}
        descriptors = self._collect_descriptors(outputs)

        buckets: dict[str, list[dict[str, Any]]] = {
            "images": [],
            "gifs": [],
            "videos": [],
        }
        for node_out in outputs.values():
            for key in buckets:
                if key in node_out:
                    buckets[key].extend(node_out[key])

        chosen: Optional[dict[str, Any]] = None
        chosen_kind = "images"
        for kind in prefer:
            if buckets.get(kind):
                chosen = buckets[kind][-1]
                chosen_kind = kind
                break
        if not chosen:
            await self._full_scrub(
                prompt_id=prompt_id,
                input_name=input_name,
                descriptors=descriptors,
            )
            raise ComfyUIError(
                f"No media in ComfyUI outputs. Keys seen: "
                f"{[k for n in outputs.values() for k in n.keys()]}"
            )

        data = await self._view(
            chosen["filename"],
            chosen.get("subfolder", ""),
            chosen.get("type", "output"),
        )
        ctype = _guess_content_type(chosen["filename"], chosen_kind)

        # Critical: wipe every local artifact now that bytes are in RAM
        await self._full_scrub(
            prompt_id=prompt_id,
            input_name=input_name,
            descriptors=descriptors,
        )
        return data, ctype

    async def _remote_scrub(
        self,
        *,
        files: list[dict[str, Any]],
        wipe_prefixes: bool = True,
    ) -> dict[str, Any]:
        """Ask the GPU box (Comfy custom node) to delete files. Works over the tunnel."""
        body = {"files": files, "wipe_prefixes": wipe_prefixes}
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self.base}/wan_studio/scrub", json=body)
            if r.status_code == 404:
                return {
                    "ok": False,
                    "error": (
                        "Remote scrub endpoint missing — restart ComfyUI after "
                        "installing/updating custom_nodes/wan_studio_i2i"
                    ),
                }
            if r.status_code != 200:
                return {
                    "ok": False,
                    "error": f"Remote scrub HTTP {r.status_code}: {r.text[:200]}",
                }
            data = r.json()
            data["ok"] = True
            return data

    async def remote_scrub_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base}/wan_studio/scrub_ping")
                return r.status_code == 200
        except Exception:
            return False

    async def _full_scrub(
        self,
        *,
        prompt_id: Optional[str] = None,
        input_name: Optional[str] = None,
        descriptors: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        if not self.settings.zero_residue:
            return {"enabled": False}

        root = await self.ensure_root()
        passes = max(1, self.settings.scrub_passes)
        report: dict[str, Any] = {"enabled": True, "files_wiped": 0}

        # History first so UI /history can't re-point at files
        if prompt_id:
            await self._delete_history(prompt_id)
            report["history_cleared"] = True

        files: list[dict[str, Any]] = []
        if descriptors:
            for d in descriptors:
                files.append(
                    {
                        "filename": d.get("filename") or "",
                        "subfolder": d.get("subfolder") or "",
                        "type": d.get("type") or "output",
                    }
                )
        if input_name:
            files.append({"filename": input_name, "subfolder": "", "type": "input"})
            files.append({"filename": input_name, "subfolder": "", "type": "temp"})

        # Prefer remote wipe (Render → tunnel → GPU box). Local Path only works
        # when the API process shares a disk with ComfyUI.
        remote = await self._remote_scrub(files=files, wipe_prefixes=True)
        report["remote_scrub"] = remote
        if remote.get("ok"):
            report["files_wiped"] += int(remote.get("wiped") or 0)
        else:
            report["remote_scrub_error"] = remote.get("error")

        if root and root.is_dir():
            if descriptors:
                report["files_wiped"] += scrub.wipe_listed_files(
                    root, descriptors, passes=passes
                )
            if input_name:
                if scrub.wipe_media_descriptor(
                    root, input_name, "", "input", passes=passes
                ):
                    report["files_wiped"] += 1
                if scrub.wipe_media_descriptor(
                    root, input_name, "", "temp", passes=passes
                ):
                    report["files_wiped"] += 1
            report["files_wiped"] += scrub.wipe_our_artifacts(root, passes=passes)
            report["comfy_root"] = str(root)
        elif not remote.get("ok"):
            report["warning"] = (
                "Filesystem wipe skipped — Comfy root not on this host and "
                "remote /wan_studio/scrub unavailable. Restart ComfyUI with "
                "updated wan_studio_i2i custom node."
            )

        return report


def _guess_content_type(filename: str, kind: str) -> str:
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webm"):
        return "video/webm"
    if lower.endswith(".mp4"):
        return "video/mp4"
    if kind == "videos":
        return "video/mp4"
    return "application/octet-stream"
