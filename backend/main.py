"""
Wan Studio — image↔video generation via ComfyUI (Wan 2.2).
All outputs stored only in MongoDB GridFS. Local ComfyUI artifacts are scrubbed.
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

from backend import db
from backend import scrub as scrub_mod
from backend.comfy_client import ComfyClient, ComfyUIError
from backend.config import get_settings
from backend.owners import (
    is_admin_owner,
    is_tester_owner,
    make_token,
    owner_for_pin,
    owner_from_token,
    unlock_enabled,
)
from backend.test_mode import REVIEW_BINS, normalize_preset_id
from backend import ops_remote


class GenerationUpdate(BaseModel):
    prompt: Optional[str] = Field(None, min_length=1)
    meta: Optional[dict] = None


class UnlockBody(BaseModel):
    pin: str = Field(..., min_length=1, max_length=16)


class ReviewAssign(BaseModel):
    id: str = Field(..., min_length=1)
    bin: Optional[str] = None


class TunnelBody(BaseModel):
    url: str = Field(..., min_length=8, max_length=500)


def _extract_token(
    authorization: Optional[str] = None,
    x_wan_token: Optional[str] = None,
    t: Optional[str] = None,
) -> Optional[str]:
    return x_wan_token or t or authorization


def require_owner(
    authorization: Optional[str] = Header(None),
    x_wan_token: Optional[str] = Header(None, alias="X-Wan-Token"),
    t: Optional[str] = Query(None, description="Access token for media URLs"),
) -> str:
    owner = owner_from_token(_extract_token(authorization, x_wan_token, t))
    if not owner:
        raise HTTPException(401, "Unlock with your PIN first")
    return owner


def require_admin(owner: str = Depends(require_owner)) -> str:
    if not is_admin_owner(owner):
        raise HTTPException(403, "Admin PIN required")
    return owner


def require_tester(owner: str = Depends(require_owner)) -> str:
    if not is_tester_owner(owner):
        raise HTTPException(403, "Not allowed")
    return owner

NO_STORE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, private",
    "Pragma": "no-cache",
    "Expires": "0",
}

DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

# Keep strong refs so asyncio tasks aren't GC'd mid-run
_background_jobs: set[asyncio.Task] = set()
_job_tasks: dict[str, asyncio.Task] = {}


def _track_job_task(job_id: str, task: asyncio.Task) -> None:
    _background_jobs.add(task)
    _job_tasks[job_id] = task

    def _done(_t: asyncio.Task) -> None:
        _background_jobs.discard(_t)
        if _job_tasks.get(job_id) is _t:
            _job_tasks.pop(job_id, None)

    task.add_done_callback(_done)


def _json(data: object, status: int = 200) -> Response:
    return Response(
        content=json.dumps(data, default=str),
        status_code=status,
        media_type="application/json",
        headers=NO_STORE,
    )


def _cors_origins() -> list[str]:
    raw = (get_settings().cors_origins or "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await db.connect()
    try:
        jobs = await db.reclaim_active_jobs_on_startup()
        if jobs:
            print(f"[wan] resuming {len(jobs)} job(s) after worker start")
            for job in jobs:
                task = asyncio.create_task(_resume_persisted_job(job))
                _track_job_task(job["id"], task)
    except Exception as exc:
        print(f"[wan] job resume on startup failed: {exc}")
    settings = get_settings()
    if settings.zero_residue and settings.comfyui_dir:
        try:
            scrub_mod.wipe_our_artifacts(
                settings.comfyui_dir, passes=max(1, settings.scrub_passes)
            )
        except Exception:
            pass
    yield
    await db.close()


app = FastAPI(title="Wan Studio", version="1.0.0", lifespan=lifespan)
_origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _comfy() -> ComfyClient:
    return ComfyClient(get_settings())


def _ai_engine_health_blob() -> dict:
    try:
        from backend.ai_engine import engine_health

        return {"health": engine_health(channel="stable")}
    except Exception as exc:
        return {"health_error": str(exc)}


async def _read_image_bytes(file: UploadFile) -> bytes:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty image upload")
    try:
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out = buf.getvalue()
        del raw, img, buf
        return out
    except Exception as exc:
        raise HTTPException(400, f"Invalid image: {exc}") from exc


@app.get("/api/health")
async def health():
    """Liveness for Render. Keep under a few seconds."""
    import asyncio

    settings = get_settings()
    try:
        await asyncio.wait_for(db.db().command("ping"), timeout=2.0)
        mongo_ok = True
    except Exception:
        mongo_ok = False
    comfy_ok = False
    try:
        comfy_ok = await asyncio.wait_for(_comfy().health(), timeout=1.2)
    except Exception:
        pass
    return _json(
        {
            "ok": mongo_ok,
            "mongodb": mongo_ok,
            "comfyui": comfy_ok,
            "comfyui_url": settings.comfyui_url,
            "zero_residue": settings.zero_residue,
            "overlay": __import__(
                "backend.ai_engine.runtime_overlay", fromlist=["overlay_status"]
            ).overlay_status(),
            "model": {
                "img": {"denoise": settings.image_denoise, "unet": settings.flux_unet},
                "vid": {
                    "unet_high": settings.wan_unet_high,
                    "unet_low": settings.wan_unet_low,
                },
            },
        }
    )


@app.post("/api/scrub")
async def scrub_now(_admin: str = Depends(require_admin)):
    """Wipe Wan Studio artifacts under ComfyUI input/output/temp (no Mongo data)."""
    settings = get_settings()
    client = _comfy()
    report: dict = {"enabled": True}

    # Remote wipe on the GPU box (works when API is on Render)
    remote = await client._remote_scrub(files=[], wipe_prefixes=True)
    report["remote_scrub"] = remote

    # Local wipe if this process can see COMFYUI_DIR (dev machine)
    root = await client.ensure_root()
    local_n = 0
    if root and root.is_dir():
        local_n = scrub_mod.wipe_our_artifacts(
            root, passes=max(1, settings.scrub_passes)
        )
        report["comfy_root"] = str(root)
    report["local_wiped"] = local_n

    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            await http.post(
                f"{settings.comfyui_url.rstrip('/')}/history", json={"clear": True}
            )
        report["history_cleared"] = True
    except Exception:
        report["history_cleared"] = False

    wiped = int(remote.get("wiped") or 0) + local_n
    if not remote.get("ok") and local_n == 0:
        raise HTTPException(
            503,
            remote.get("error")
            or "Scrub failed — restart ComfyUI with updated wan_studio_i2i node.",
        )
    return _json({"wiped_files": wiped, **report})


def _parse_video_seconds(raw: Optional[str], mode: str) -> Optional[float]:
    """Accept 2–5s duration for vid mode; ignore otherwise."""
    if mode != "vid" or raw is None or str(raw).strip() == "":
        return None
    try:
        sec = float(str(raw).strip())
    except (TypeError, ValueError):
        raise HTTPException(400, "video_seconds must be a number") from None
    if sec < 2 or sec > 5:
        raise HTTPException(400, "video_seconds must be between 2 and 5")
    return sec


@app.get("/api/auth/status")
async def auth_status():
    return _json({"ok": True, "unlock_enabled": unlock_enabled()})


@app.post("/api/auth/unlock")
async def auth_unlock(body: UnlockBody):
    if not unlock_enabled():
        raise HTTPException(403, "Unlock is disabled")
    owner = owner_for_pin(body.pin)
    if not owner:
        raise HTTPException(401, "Wrong PIN")
    try:
        token = make_token(owner)
    except ValueError:
        raise HTTPException(403, "Unlock is disabled") from None
    return _json(
        {
            "ok": True,
            "owner": owner,
            "admin": is_admin_owner(owner),
            "tester": is_tester_owner(owner),
            "token": token,
            "expires_hours": 24,
        }
    )


@app.get("/api/auth/me")
async def auth_me(owner: str = Depends(require_owner)):
    return _json(
        {
            "ok": True,
            "owner": owner,
            "admin": is_admin_owner(owner),
            "tester": is_tester_owner(owner),
            "unlock_enabled": unlock_enabled(),
        }
    )


@app.get("/api/ops/capabilities")
async def ops_capabilities(_admin: str = Depends(require_admin)):
    return _json({"ok": True, **ops_remote.ops_capabilities(get_settings())})


@app.post("/api/ops/restart-api")
async def ops_restart_api(_admin: str = Depends(require_admin)):
    """Restart the Render web service. Client should poll /api/health afterward."""
    try:
        result = await ops_remote.restart_render_service(get_settings())
    except ops_remote.OpsError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return _json(
        {
            **result,
            "message": "Server restart started. Wait about a minute, then try again.",
        }
    )


@app.post("/api/ops/set-tunnel")
async def ops_set_tunnel(body: TunnelBody, _admin: str = Depends(require_admin)):
    try:
        result = await ops_remote.set_comfyui_url(get_settings(), body.url)
    except ops_remote.OpsError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return _json(
        {
            **result,
            "message": "Tunnel saved. Server is updating — wait about a minute.",
        }
    )


@app.post("/api/ops/restart-comfy")
async def ops_restart_comfy(_admin: str = Depends(require_admin)):
    try:
        result = await ops_remote.restart_gpu_comfy(get_settings())
    except ops_remote.OpsError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return _json(result)


@app.post("/api/generate")
async def generate(
    mode: Literal["img", "vid"] = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
    negative: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    video_seconds: Optional[str] = Form(None),
    owner: str = Depends(require_owner),
):
    """Legacy sync endpoint (long request). Prefer POST /api/jobs for mobile."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required")

    from backend.prompt_fix import normalize_prompt

    resolved = normalize_prompt(prompt, mode=mode)
    prompt_en = resolved["english"] or prompt
    if negative:
        neg_resolved = normalize_prompt(negative, mode=mode, frame=False)
        negative = neg_resolved["english"] or negative

    vid_sec = _parse_video_seconds(video_seconds, mode)
    image_bytes = await _read_image_bytes(image)
    try:
        payload = await _execute_generation(
            mode=mode,
            prompt=prompt,
            prompt_en=prompt_en,
            image_bytes=image_bytes,
            negative=negative,
            seed=seed,
            resolved=resolved,
            video_seconds=vid_sec,
            owner=owner,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Unexpected error: {exc}") from exc
    return _json(payload)


@app.post("/api/jobs")
async def start_job(
    mode: Literal["img", "vid"] = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
    negative: Optional[str] = Form(None),
    seed: Optional[int] = Form(None),
    video_seconds: Optional[str] = Form(None),
    preset_id: Optional[str] = Form(None),
    test_run: Optional[str] = Form(None),
    owner: str = Depends(require_owner),
):
    """
    Start generation as a background job. Returns immediately with job id.
    Poll GET /api/jobs/{id} — survives phone sleep / app switch / fetch abort.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required")

    from backend.prompt_fix import normalize_prompt

    resolved = normalize_prompt(prompt, mode=mode)
    prompt_en = resolved["english"] or prompt
    if negative:
        neg_resolved = normalize_prompt(negative, mode=mode, frame=False)
        negative = neg_resolved["english"] or negative

    vid_sec = _parse_video_seconds(video_seconds, mode)
    try:
        preset = normalize_preset_id(preset_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    is_test = is_tester_owner(owner) or str(test_run or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if is_tester_owner(owner):
        is_test = True
    image_bytes = await _read_image_bytes(image)
    client = _comfy()
    settings = get_settings()
    if not await client.health(timeout=12.0, retries=3):
        raise HTTPException(
            503,
            "ComfyUI is not reachable. Start ComfyUI with Wan 2.2 models loaded "
            f"and set COMFYUI_URL (current: {settings.comfyui_url}).",
        )

    job = await db.create_job(
        mode=mode,
        prompt=prompt,
        prompt_english=prompt_en,
        negative=negative,
        seed=seed,
        owner=owner,
        image_bytes=image_bytes,
        video_seconds=vid_sec,
        preset_id=preset,
        test_run=is_test,
    )
    task = asyncio.create_task(
        _run_job(
            job_id=job["id"],
            mode=mode,
            prompt=prompt,
            prompt_en=prompt_en,
            image_bytes=image_bytes,
            negative=negative,
            seed=seed,
            resolved=resolved,
            video_seconds=vid_sec,
            owner=owner,
            extra_meta={"preset_id": preset, "test_run": is_test},
        )
    )
    _track_job_task(job["id"], task)
    return _json({"id": job["id"], "status": "queued", "mode": mode})


@app.get("/api/jobs")
async def list_jobs(
    active: int = 1,
    limit: int = 30,
    owner: str = Depends(require_owner),
):
    """List jobs for this account. active=1 → queued/running only."""
    limit = min(max(1, int(limit)), 100)
    if int(active or 0):
        items = await db.list_active_jobs(owner=owner, limit=limit)
    else:
        items = await db.list_recent_jobs(owner=owner, limit=limit)
    return _json({"items": items, "total": len(items)})


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, owner: str = Depends(require_owner)):
    job = await db.get_job(job_id, owner=owner)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") not in ("queued", "running"):
        return _json(job)

    updated = await db.finish_job_if_active(
        job_id,
        status="cancelled",
        error="Cancelled by you.",
        finished_at=datetime.now(timezone.utc),
    )
    task = _job_tasks.get(job_id)
    if task and not task.done():
        task.cancel()
    # Stop GPU work if this (or a sibling) prompt is on Comfy right now
    try:
        await _comfy().interrupt()
    except Exception:
        pass
    return _json(updated or job)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, owner: str = Depends(require_owner)):
    job = await db.get_job(job_id, owner=owner)
    if not job:
        raise HTTPException(404, "Job not found")
    return _json(job)


async def _execute_generation(
    *,
    mode: str,
    prompt: str,
    prompt_en: str,
    image_bytes: bytes,
    negative: Optional[str],
    seed: Optional[int],
    resolved: dict,
    video_seconds: Optional[float] = None,
    owner: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> dict:
    from backend.ai_engine import run as ai_engine_run
    from backend.ai_engine.schema import GenerateRequest

    client = _comfy()
    settings = get_settings()

    if not await client.health(timeout=12.0, retries=3):
        raise HTTPException(
            503,
            "ComfyUI is not reachable. Start ComfyUI with Wan 2.2 models loaded "
            f"and set COMFYUI_URL (current: {settings.comfyui_url}).",
        )

    data: Optional[bytes] = None
    content_type = "application/octet-stream"
    kind: Literal["img", "vid"] = "img"
    filename = "out.bin"
    engine_meta: dict = {}

    try:
        result = await ai_engine_run(
            GenerateRequest(
                mode=mode,
                prompt=prompt,
                prompt_english=prompt_en,
                image_bytes=image_bytes,
                negative=negative,
                seed=seed,
                profile="quality",
                channel="stable",
                video_seconds=video_seconds if mode == "vid" else None,
            ),
            settings=settings,
        )
        data = result.data
        content_type = result.content_type
        kind = result.kind
        engine_meta = {
            "workflow_ref": result.workflow_ref,
            "task_type": result.plan.task_type,
            "planner_path": result.plan.planner_path,
            "profile": result.plan.profile,
            "backbone_model_id": result.backbone.model_id if result.backbone else None,
            "backbone_tier": result.backbone.tier if result.backbone else None,
            "model_label": result.model_label,
            "recovery_events": result.recovery_events,
            "engine_warnings": result.warnings,
            "perception": result.perception_meta,
            "plan": result.plan.to_meta(),
        }
        if kind == "img":
            filename = f"wan_img_{uuid.uuid4().hex}.png"
            if "jpeg" in content_type or "jpg" in content_type:
                filename = filename.replace(".png", ".jpg")
            elif "webp" in content_type:
                filename = filename.replace(".png", ".webp")
        else:
            ext = "mp4"
            if "webm" in content_type:
                ext = "webm"
            elif "gif" in content_type:
                ext = "gif"
            filename = f"wan_vid_{uuid.uuid4().hex}.{ext}"
    except ComfyUIError as exc:
        raise HTTPException(502, f"Generation failed: {exc}") from exc
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        del image_bytes
        gc.collect()

    assert data is not None
    model_label = engine_meta.get("model_label") or (
        settings.flux_unet
        if mode == "img"
        else f"{settings.wan_unet_high}+{settings.wan_unet_low}"
    )

    record = await db.store_media(
        data=data,
        filename=filename,
        content_type=content_type,
        kind=kind,
        prompt=prompt,
        model=str(model_label),
        owner=owner,
        meta={
            "negative": negative,
            "seed": seed,
            "mode": mode,
            "prompt_original": resolved["original"],
            "prompt_english": prompt_en,
            "video_seconds": video_seconds,
            **{
                k: v
                for k, v in (extra_meta or {}).items()
                if v not in (None, False, "")
            },
            **engine_meta,
        },
    )
    del data
    gc.collect()

    if settings.zero_residue:
        root = await client.ensure_root()
        if root:
            scrub_mod.wipe_our_artifacts(root, passes=max(1, settings.scrub_passes))

    return {
        "id": record["id"],
        "kind": record["kind"],
        "prompt": record["prompt"],
        "prompt_english": prompt_en,
        "model": record["model"],
        "content_type": record["content_type"],
        "size_bytes": record["size_bytes"],
        "created_at": record["created_at"],
        "media_url": f"/api/media/{record['id']}",
        "thumb_url": f"/api/media/{record['id']}/thumb?w=240",
        "local_residue": False,
        "workflow_ref": engine_meta.get("workflow_ref"),
        "task_type": engine_meta.get("task_type"),
    }


async def _resume_persisted_job(job: dict) -> None:
    """Re-run a queued/running job after API worker restart using stored start image."""
    job_id = job["id"]
    image_bytes = await db.get_job_input_bytes(job_id)
    if not image_bytes:
        await db.finish_job_if_active(
            job_id,
            status="failed",
            error="Missing start image after API restart. Try again.",
            finished_at=datetime.now(timezone.utc),
        )
        return

    from backend.prompt_fix import normalize_prompt

    mode = job.get("mode") or "img"
    prompt = (job.get("prompt") or "").strip()
    prompt_en = (job.get("prompt_english") or "").strip() or prompt
    negative = job.get("negative")
    resolved = normalize_prompt(prompt, mode=mode)
    if not prompt_en:
        prompt_en = resolved.get("english") or prompt
    if negative:
        neg_resolved = normalize_prompt(negative, mode=mode, frame=False)
        negative = neg_resolved.get("english") or negative

    resume_n = int(job.get("resume_count") or 0)
    print(f"[wan] resume job={job_id} mode={mode} attempt={resume_n}")
    await _run_job(
        job_id=job_id,
        mode=mode,
        prompt=prompt,
        prompt_en=prompt_en,
        image_bytes=image_bytes,
        negative=negative,
        seed=job.get("seed"),
        resolved=resolved,
        video_seconds=job.get("video_seconds"),
        owner=job.get("owner"),
        extra_meta={
            "preset_id": job.get("preset_id"),
            "test_run": bool(job.get("test_run")),
        },
    )


async def _run_job(
    *,
    job_id: str,
    mode: str,
    prompt: str,
    prompt_en: str,
    image_bytes: bytes,
    negative: Optional[str],
    seed: Optional[int],
    resolved: dict,
    video_seconds: Optional[float] = None,
    owner: Optional[str] = None,
    extra_meta: Optional[dict] = None,
) -> None:
    await db.update_job(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    settings = get_settings()
    # Hard ceiling so a hung WS/tunnel cannot leave the job "running" forever.
    hard_limit = max(180, int(settings.comfyui_timeout_sec) + 90)
    try:
        payload = await asyncio.wait_for(
            _execute_generation(
                mode=mode,
                prompt=prompt,
                prompt_en=prompt_en,
                image_bytes=image_bytes,
                negative=negative,
                seed=seed,
                resolved=resolved,
                video_seconds=video_seconds,
                owner=owner,
                extra_meta=extra_meta,
            ),
            timeout=hard_limit,
        )
        await db.finish_job_if_active(
            job_id,
            status="done",
            result=payload,
            error=None,
            finished_at=datetime.now(timezone.utc),
        )
    except asyncio.CancelledError:
        await db.finish_job_if_active(
            job_id,
            status="cancelled",
            error="Cancelled by you.",
            finished_at=datetime.now(timezone.utc),
        )
        raise
    except asyncio.TimeoutError:
        await db.finish_job_if_active(
            job_id,
            status="failed",
            error=(
                f"Generation timed out after {hard_limit // 60} min. "
                "Check ComfyUI / the Cloudflare tunnel, then try again."
            ),
            finished_at=datetime.now(timezone.utc),
        )
    except HTTPException as exc:
        detail = exc.detail
        if not isinstance(detail, str):
            detail = str(detail)
        await db.finish_job_if_active(
            job_id,
            status="failed",
            error=detail,
            finished_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        await db.finish_job_if_active(
            job_id,
            status="failed",
            error=f"Unexpected error: {exc}",
            finished_at=datetime.now(timezone.utc),
        )


@app.get("/api/generations")
async def generations(
    limit: int = 30,
    skip: int = 0,
    test_run: Optional[int] = None,
    preset_id: Optional[str] = None,
    owner: str = Depends(require_owner),
):
    limit = min(max(1, int(limit)), 100)
    skip = max(0, int(skip))
    try:
        preset = normalize_preset_id(preset_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    test_flag = None
    if test_run is not None:
        test_flag = bool(int(test_run))
    total = await db.count_generations(
        owner=owner, test_run=test_flag, preset_id=preset
    )
    items = await db.list_generations(
        limit=limit, skip=skip, owner=owner, test_run=test_flag, preset_id=preset
    )
    return _json(
        {
            "items": items,
            "total": total,
            "limit": limit,
            "skip": skip,
        }
    )


@app.get("/api/generations/{gen_id}")
async def generation_meta(gen_id: str, owner: str = Depends(require_owner)):
    doc = await db.get_generation(gen_id, owner=owner)
    if not doc:
        raise HTTPException(404, "Not found")
    return _json(doc)


@app.patch("/api/generations/{gen_id}")
async def generation_update(
    gen_id: str, body: GenerationUpdate, owner: str = Depends(require_owner)
):
    if body.prompt is None and body.meta is None:
        raise HTTPException(400, "Provide prompt and/or meta to update")
    try:
        doc = await db.update_generation(
            gen_id,
            prompt=body.prompt,
            meta=body.meta,
            owner=owner,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not doc:
        raise HTTPException(404, "Not found")
    return _json(doc)


@app.post("/api/generations/{gen_id}/opened")
async def generation_opened(gen_id: str, owner: str = Depends(require_owner)):
    """Mark a library item as opened so the 'Recently generated' badge clears."""
    doc = await db.mark_generation_opened(gen_id, owner=owner)
    if not doc:
        raise HTTPException(404, "Not found")
    return _json(doc)


@app.delete("/api/generations/{gen_id}")
async def generation_delete(gen_id: str, owner: str = Depends(require_owner)):
    ok = await db.delete_generation(gen_id, owner=owner)
    if not ok:
        raise HTTPException(404, "Not found")
    return _json({"ok": True, "id": gen_id, "deleted": True})


@app.get("/api/presets")
async def list_presets(_owner: str = Depends(require_owner)):
    from backend.ai_engine.runtime_overlay import load_json

    presets = load_json("presets.json") or []
    if not isinstance(presets, list):
        presets = []
    return _json({"presets": presets})


@app.get("/api/test/review-bins")
async def list_review_bins(_tester: str = Depends(require_tester)):
    from backend.ai_engine.runtime_overlay import load_json

    bins = load_json("review_bins.json") or []
    if not isinstance(bins, list):
        bins = []
    return _json({"bins": bins})


@app.get("/api/test/summary")
async def test_summary(_tester: str = Depends(require_tester)):
    refs = await db.count_test_refs_by_preset(owner=_tester)
    runs = await db.count_generations_by_preset(owner=_tester, test_run=True)
    inputs = await db.count_test_inputs(owner=_tester)
    review = await db.count_review_bins(owner=_tester)
    return _json({"ok": True, "refs": refs, "runs": runs, "inputs": inputs, "review": review})


@app.get("/api/test/refs")
async def test_refs_list(
    preset_id: Optional[str] = None,
    _tester: str = Depends(require_tester),
):
    try:
        preset = normalize_preset_id(preset_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    items = await db.list_test_refs(owner=_tester, preset_id=preset)
    return _json({"items": items, "total": len(items)})


@app.post("/api/test/refs")
async def test_refs_create(
    image: UploadFile = File(...),
    preset_id: str = Form(...),
    _tester: str = Depends(require_tester),
):
    try:
        preset = normalize_preset_id(preset_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not preset:
        raise HTTPException(400, "preset_id is required")
    data = await _read_image_bytes(image)
    name = f"ref_{preset}_{uuid.uuid4().hex[:10]}.png"
    record = await db.store_test_ref(
        data=data,
        filename=name,
        content_type="image/png",
        preset_id=preset,
        owner=_tester,
    )
    return _json(record)


@app.delete("/api/test/refs/{ref_id}")
async def test_refs_delete(ref_id: str, _tester: str = Depends(require_tester)):
    ok = await db.delete_test_ref(ref_id, owner=_tester)
    if not ok:
        raise HTTPException(404, "Not found")
    return _json({"ok": True, "id": ref_id, "deleted": True})


@app.get("/api/test/refs/{ref_id}/media")
async def test_ref_media(ref_id: str, _tester: str = Depends(require_tester)):
    result = await db.get_test_ref_bytes(ref_id, owner=_tester)
    if not result:
        raise HTTPException(404, "Not found")
    data, content_type, filename = result
    headers = {
        **NO_STORE,
        "Content-Disposition": f'inline; filename="{filename}"',
    }
    return StreamingResponse(io.BytesIO(data), media_type=content_type, headers=headers)


@app.get("/api/test/refs/{ref_id}/thumb")
async def test_ref_thumb(
    ref_id: str,
    w: int = 240,
    _tester: str = Depends(require_tester),
):
    result = await db.get_test_ref_bytes(ref_id, owner=_tester)
    if not result:
        raise HTTPException(404, "Not found")
    data, _ctype, _name = result
    max_w = max(64, min(int(w), 720))
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_w, max_w * 2), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72, optimize=True)
    headers = {
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": f'inline; filename="{ref_id}_thumb.jpg"',
    }
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/jpeg", headers=headers)


@app.get("/api/test/inputs")
async def test_inputs_list(_tester: str = Depends(require_tester)):
    items = await db.list_test_inputs(owner=_tester)
    return _json({"items": items, "total": len(items)})


@app.post("/api/test/inputs")
async def test_inputs_create(
    image: UploadFile = File(...),
    _tester: str = Depends(require_tester),
):
    data = await _read_image_bytes(image)
    name = f"input_{uuid.uuid4().hex[:10]}.png"
    record = await db.store_test_input(
        data=data,
        filename=name,
        content_type="image/png",
        owner=_tester,
    )
    return _json(record)


@app.delete("/api/test/inputs/{item_id}")
async def test_inputs_delete(item_id: str, _tester: str = Depends(require_tester)):
    ok = await db.delete_test_input(item_id, owner=_tester)
    if not ok:
        raise HTTPException(404, "Not found")
    return _json({"ok": True, "id": item_id, "deleted": True})


@app.get("/api/test/inputs/{item_id}/media")
async def test_input_media(item_id: str, _tester: str = Depends(require_tester)):
    result = await db.get_test_input_bytes(item_id, owner=_tester)
    if not result:
        raise HTTPException(404, "Not found")
    data, content_type, filename = result
    headers = {
        **NO_STORE,
        "Content-Disposition": f'inline; filename="{filename}"',
    }
    return StreamingResponse(io.BytesIO(data), media_type=content_type, headers=headers)


@app.get("/api/test/inputs/{item_id}/thumb")
async def test_input_thumb(
    item_id: str,
    w: int = 240,
    _tester: str = Depends(require_tester),
):
    result = await db.get_test_input_bytes(item_id, owner=_tester)
    if not result:
        raise HTTPException(404, "Not found")
    data, _ctype, _name = result
    max_w = max(64, min(int(w), 720))
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((max_w, max_w * 2), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72, optimize=True)
    headers = {
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": f'inline; filename="{item_id}_thumb.jpg"',
    }
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/jpeg", headers=headers)


@app.get("/api/test/runs")
async def test_runs(
    preset_id: Optional[str] = None,
    limit: int = 20,
    _tester: str = Depends(require_tester),
):
    try:
        preset = normalize_preset_id(preset_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    limit = min(max(1, int(limit)), 50)
    items = await db.list_generations(
        limit=limit, skip=0, owner=_tester, test_run=True, preset_id=preset
    )
    refs = await db.list_test_refs(owner=_tester, preset_id=preset) if preset else []
    return _json({"items": items, "refs": refs, "total": len(items)})


@app.post("/api/test/review")
async def test_review_assign(
    body: ReviewAssign,
    _tester: str = Depends(require_tester),
):
    bin_id = (body.bin or "").strip() or None
    if bin_id and bin_id not in REVIEW_BINS:
        raise HTTPException(400, "Unknown review folder")
    doc = await db.get_generation(body.id, owner=_tester)
    meta = doc.get("meta") if doc else None
    if not doc or not (meta or {}).get("test_run"):
        raise HTTPException(404, "Not found")
    if bin_id:
        expected = REVIEW_BINS[bin_id]
        if (meta or {}).get("preset_id") != expected:
            raise HTTPException(400, "Folder does not match this generation")
    updated = await db.set_review_bin(body.id, owner=_tester, bin_id=bin_id)
    if not updated:
        raise HTTPException(404, "Not found")
    return _json(updated)


@app.get("/api/media/{gen_id}")
async def media(
    gen_id: str,
    download: int = 0,
    owner: str = Depends(require_owner),
):
    result = await db.get_media_bytes(gen_id, owner=owner)
    if not result:
        raise HTTPException(404, "Not found")
    data, content_type, filename = result
    disposition = "attachment" if int(download or 0) else "inline"
    headers = {
        **NO_STORE,
        "Content-Disposition": f'{disposition}; filename="{filename}"',
    }
    return StreamingResponse(io.BytesIO(data), media_type=content_type, headers=headers)


@app.get("/api/media/{gen_id}/thumb")
async def media_thumb(
    gen_id: str,
    w: int = 360,
    owner: str = Depends(require_owner),
):
    """Small JPEG preview for library (cached in GridFS after first build)."""
    max_w = max(64, min(int(w), 720))
    try:
        cached = await db.get_or_create_image_thumb(gen_id, owner=owner, max_w=max_w)
    except Exception as exc:
        raise HTTPException(500, f"Thumb failed: {exc}") from exc
    if cached is None:
        raise HTTPException(404, "Not found")
    thumb, content_type = cached
    headers = {
        "Cache-Control": "public, max-age=604800, immutable",
        "Content-Disposition": f'inline; filename="{gen_id}_thumb.jpg"',
    }
    return StreamingResponse(io.BytesIO(thumb), media_type=content_type, headers=headers)


# Optional: serve Vite build from this process (local / single-host only).
# Production: React on Vercel, API on Render (SERVE_FRONTEND=false).
if get_settings().serve_frontend and DIST_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST_DIR / "assets"), name="vite_assets")

    @app.get("/")
    async def spa_index():
        return FileResponse(DIST_DIR / "index.html")

    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        candidate = DIST_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST_DIR / "index.html")


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        # Avoid access log lines that could retain request metadata on disk via redirects
        access_log=False,
    )


if __name__ == "__main__":
    run()
