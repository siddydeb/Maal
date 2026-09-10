"""MongoDB-only persistence. Binary media in GridFS; metadata in collections."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket

from backend.config import get_settings

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None
_fs: Optional[AsyncIOMotorGridFSBucket] = None


async def connect() -> None:
    global _client, _db, _fs
    settings = get_settings()
    _client = AsyncIOMotorClient(settings.mongodb_uri)
    _db = _client[settings.mongodb_db]
    _fs = AsyncIOMotorGridFSBucket(_db, bucket_name="media")
    # Touch connection early so startup fails if URI is bad
    await _client.admin.command("ping")


async def close() -> None:
    global _client, _db, _fs
    if _client is not None:
        _client.close()
    _client = None
    _db = None
    _fs = None


def db() -> AsyncIOMotorDatabase:
    if _db is None:
        raise RuntimeError("Database not connected")
    return _db


def fs() -> AsyncIOMotorGridFSBucket:
    if _fs is None:
        raise RuntimeError("GridFS not connected")
    return _fs


async def store_media(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    kind: str,
    prompt: str,
    model: str,
    meta: Optional[dict[str, Any]] = None,
    owner: Optional[str] = None,
) -> dict[str, Any]:
    """Write bytes into MongoDB GridFS + generations doc; mirror to Cloudinary when configured."""
    file_id = await fs().upload_from_stream(
        filename,
        data,
        metadata={
            "content_type": content_type,
            "kind": kind,
            "prompt": prompt,
            "model": model,
            "owner": owner,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    doc: dict[str, Any] = {
        "kind": kind,  # "img" | "vid"
        "prompt": prompt,
        "model": model,
        "content_type": content_type,
        "filename": filename,
        "gridfs_id": file_id,
        "size_bytes": len(data),
        "meta": meta or {},
        "owner": owner,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db().generations.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


def _generation_query(
    *,
    owner: Optional[str] = None,
    test_run: Optional[bool] = None,
    preset_id: Optional[str] = None,
) -> dict[str, Any]:
    query: dict[str, Any] = {}
    if owner:
        query["owner"] = owner
    if test_run is True:
        query["meta.test_run"] = True
    elif test_run is False:
        query["meta.test_run"] = {"$ne": True}
    if preset_id:
        query["meta.preset_id"] = preset_id
    return query


async def count_generations(
    *,
    owner: Optional[str] = None,
    test_run: Optional[bool] = None,
    preset_id: Optional[str] = None,
) -> int:
    query = _generation_query(owner=owner, test_run=test_run, preset_id=preset_id)
    return int(await db().generations.count_documents(query))


async def list_generations(
    limit: int = 50,
    skip: int = 0,
    *,
    owner: Optional[str] = None,
    test_run: Optional[bool] = None,
    preset_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    query = _generation_query(owner=owner, test_run=test_run, preset_id=preset_id)
    cursor = (
        db()
        .generations.find(query)
        .sort("created_at", -1)
        .skip(max(0, int(skip)))
        .limit(limit)
    )
    return [_serialize(doc) async for doc in cursor]


async def mark_generation_opened(
    gen_id: str, *, owner: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Stamp meta.opened_at without wiping other meta (library 'recently generated' badge)."""
    try:
        oid = ObjectId(gen_id)
    except Exception:
        return None
    query: dict[str, Any] = {"_id": oid}
    if owner:
        query["owner"] = owner
    now = datetime.now(timezone.utc)
    result = await db().generations.update_one(
        query,
        {"$set": {"meta.opened_at": now, "updated_at": now}},
    )
    if result.matched_count == 0:
        return None
    return await get_generation(gen_id, owner=owner)


async def set_review_bin(
    gen_id: str, *, owner: str, bin_id: Optional[str]
) -> Optional[dict[str, Any]]:
    """Set or clear meta.review_bin without wiping other generation meta."""
    try:
        oid = ObjectId(gen_id)
    except Exception:
        return None
    query = {"_id": oid, "owner": owner, "meta.test_run": True}
    now = datetime.now(timezone.utc)
    if bin_id:
        result = await db().generations.update_one(
            query,
            {"$set": {"meta.review_bin": bin_id, "updated_at": now}},
        )
    else:
        result = await db().generations.update_one(
            query,
            {"$unset": {"meta.review_bin": ""}, "$set": {"updated_at": now}},
        )
    if result.matched_count == 0:
        return None
    return await get_generation(gen_id, owner=owner)


async def count_review_bins(*, owner: str) -> dict[str, int]:
    pipeline = [
        {"$match": {"owner": owner, "meta.test_run": True}},
        {"$group": {"_id": "$meta.review_bin", "n": {"$sum": 1}}},
    ]
    out: dict[str, int] = {}
    async for row in db().generations.aggregate(pipeline):
        key = row.get("_id")
        if key:
            out[str(key)] = int(row.get("n") or 0)
        else:
            out["unfiled"] = int(row.get("n") or 0)
    return out


async def count_generations_by_preset(*, owner: str, test_run: bool = True) -> dict[str, int]:
    match: dict[str, Any] = {"owner": owner}
    if test_run:
        match["meta.test_run"] = True
    pipeline = [
        {"$match": match},
        {"$group": {"_id": "$meta.preset_id", "n": {"$sum": 1}}},
    ]
    out: dict[str, int] = {}
    async for row in db().generations.aggregate(pipeline):
        key = row.get("_id")
        if key:
            out[str(key)] = int(row.get("n") or 0)
    return out


async def get_generation(
    gen_id: str, *, owner: Optional[str] = None
) -> Optional[dict[str, Any]]:
    try:
        oid = ObjectId(gen_id)
    except Exception:
        return None
    query: dict[str, Any] = {"_id": oid}
    if owner:
        query["owner"] = owner
    doc = await db().generations.find_one(query)
    return _serialize(doc) if doc else None


async def get_media_bytes(
    gen_id: str, *, owner: Optional[str] = None
) -> Optional[tuple[bytes, str, str]]:
    doc = await get_generation(gen_id, owner=owner)
    if not doc:
        return None
    grid_id = ObjectId(doc["gridfs_id"])
    stream = await fs().open_download_stream(grid_id)
    data = await stream.read()
    return data, doc["content_type"], doc["filename"]


def _placeholder_thumb_jpeg(max_w: int) -> bytes:
    from io import BytesIO

    from PIL import Image, ImageDraw

    w = max(64, min(int(max_w), 720))
    h = max(64, int(w * 9 / 16))
    img = Image.new("RGB", (w, h), (18, 22, 20))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    s = max(12, min(w, h) // 6)
    draw.polygon(
        [(cx - s, cy - s), (cx - s, cy + s), (cx + int(s * 1.2), cy)],
        fill=(196, 240, 77),
    )
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=70, optimize=True)
    return buf.getvalue()


def _ffmpeg_first_frame_jpeg(video: bytes, max_w: int) -> Optional[bytes]:
    import shutil
    import subprocess
    import tempfile
    from io import BytesIO
    from pathlib import Path

    from PIL import Image

    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    max_w = max(64, min(int(max_w), 720))
    with tempfile.TemporaryDirectory(prefix="wan_thumb_") as tmp:
        src = Path(tmp) / "in.bin"
        dst = Path(tmp) / "out.jpg"
        src.write_bytes(video)
        proc = subprocess.run(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-frames:v",
                "1",
                "-vf",
                f"scale={max_w}:-1",
                "-q:v",
                "5",
                str(dst),
            ],
            capture_output=True,
            timeout=20,
            check=False,
        )
        if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 32:
            return None
        img = Image.open(BytesIO(dst.read_bytes())).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=72, optimize=True)
        return buf.getvalue()


async def get_or_create_image_thumb(
    gen_id: str,
    *,
    owner: Optional[str] = None,
    max_w: int = 360,
) -> Optional[tuple[bytes, str]]:
    """Return JPEG thumb bytes, generating + caching in GridFS on first hit."""
    from io import BytesIO

    from PIL import Image

    doc = await get_generation(gen_id, owner=owner)
    if not doc:
        return None
    ctype = (doc.get("content_type") or "").lower()
    kind = (doc.get("kind") or "").lower()
    is_vid = kind == "vid" or ctype.startswith("video/")

    max_w = max(64, min(int(max_w), 720))
    cached_id = doc.get("thumb_gridfs_id")
    cached_w = int(doc.get("thumb_width") or 0)
    if cached_id and cached_w >= max_w:
        try:
            stream = await fs().open_download_stream(ObjectId(str(cached_id)))
            return await stream.read(), "image/jpeg"
        except Exception:
            pass

    grid_id = ObjectId(doc["gridfs_id"])
    stream = await fs().open_download_stream(grid_id)
    data = await stream.read()
    if is_vid:
        thumb = _ffmpeg_first_frame_jpeg(data, max_w) or _placeholder_thumb_jpeg(max_w)
    else:
        img = Image.open(BytesIO(data)).convert("RGB")
        if img.width > max_w:
            nh = max(1, int(img.height * (max_w / img.width)))
            img = img.resize((max_w, nh), Image.Resampling.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=72, optimize=True)
        thumb = buf.getvalue()

    # Replace previous cached thumb if any
    if cached_id:
        try:
            await fs().delete(ObjectId(str(cached_id)))
        except Exception:
            pass
    thumb_oid = await fs().upload_from_stream(
        f"{gen_id}_thumb_{max_w}.jpg",
        BytesIO(thumb),
        metadata={"kind": "thumb", "gen_id": gen_id, "width": max_w},
    )
    try:
        oid = ObjectId(gen_id)
        await db().generations.update_one(
            {"_id": oid},
            {
                "$set": {
                    "thumb_gridfs_id": str(thumb_oid),
                    "thumb_width": max_w,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )
    except Exception:
        pass
    return thumb, "image/jpeg"


async def update_generation(
    gen_id: str,
    *,
    prompt: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
    owner: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Update metadata only (prompt / meta). Media bytes stay in GridFS."""
    try:
        oid = ObjectId(gen_id)
    except Exception:
        return None

    query: dict[str, Any] = {"_id": oid}
    if owner:
        query["owner"] = owner

    updates: dict[str, Any] = {}
    if prompt is not None:
        text = prompt.strip()
        if not text:
            raise ValueError("Prompt cannot be empty")
        updates["prompt"] = text
    if meta is not None:
        updates["meta"] = meta

    if not updates:
        return await get_generation(gen_id, owner=owner)

    updates["updated_at"] = datetime.now(timezone.utc)
    result = await db().generations.update_one(query, {"$set": updates})
    if result.matched_count == 0:
        return None
    return await get_generation(gen_id, owner=owner)


async def delete_generation(gen_id: str, *, owner: Optional[str] = None) -> bool:
    """Delete generation doc and its GridFS media file."""
    try:
        oid = ObjectId(gen_id)
    except Exception:
        return False

    query: dict[str, Any] = {"_id": oid}
    if owner:
        query["owner"] = owner

    doc = await db().generations.find_one(query)
    if not doc:
        return False

    grid_id = doc.get("gridfs_id")
    if grid_id is not None:
        try:
            await fs().delete(ObjectId(grid_id) if not isinstance(grid_id, ObjectId) else grid_id)
        except Exception:
            # Doc still deleted even if GridFS file is already gone
            pass

    thumb_id = doc.get("thumb_gridfs_id")
    if thumb_id is not None:
        try:
            await fs().delete(
                ObjectId(thumb_id) if not isinstance(thumb_id, ObjectId) else thumb_id
            )
        except Exception:
            pass

    result = await db().generations.delete_one({"_id": oid})
    return result.deleted_count > 0


# --- Async generation jobs (survive client disconnect / phone sleep) ---

async def create_job(
    *,
    mode: str,
    prompt: str,
    prompt_english: str,
    negative: Optional[str] = None,
    seed: Optional[int] = None,
    owner: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    video_seconds: Optional[float] = None,
    preset_id: Optional[str] = None,
    test_run: bool = False,
) -> dict[str, Any]:
    import io
    import uuid as _uuid

    now = datetime.now(timezone.utc)
    input_gridfs_id = None
    if image_bytes:
        input_gridfs_id = await fs().upload_from_stream(
            f"job_in_{_uuid.uuid4().hex[:12]}.bin",
            io.BytesIO(image_bytes),
            metadata={"kind": "job_input", "owner": owner, "mode": mode},
        )
    doc = {
        "status": "queued",
        "mode": mode,
        "prompt": prompt,
        "prompt_english": prompt_english,
        "negative": negative,
        "seed": seed,
        "owner": owner,
        "video_seconds": video_seconds,
        "input_gridfs_id": input_gridfs_id,
        "preset_id": preset_id,
        "test_run": bool(test_run),
        "error": None,
        "result": None,
        "created_at": now,
        "updated_at": now,
        "started_at": None,
        "finished_at": None,
        "resume_count": 0,
    }
    res = await db().jobs.insert_one(doc)
    doc["_id"] = res.inserted_id
    return _serialize_job(doc)


async def get_job_input_bytes(job_id: str) -> Optional[bytes]:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    doc = await db().jobs.find_one({"_id": oid}, {"input_gridfs_id": 1})
    if not doc or not doc.get("input_gridfs_id"):
        return None
    try:
        stream = await fs().open_download_stream(doc["input_gridfs_id"])
        return await stream.read()
    except Exception:
        return None


async def delete_job_input(job_id: str) -> None:
    """Best-effort wipe of persisted start image after job finishes."""
    try:
        oid = ObjectId(job_id)
    except Exception:
        return
    doc = await db().jobs.find_one({"_id": oid}, {"input_gridfs_id": 1})
    if not doc:
        return
    gid = doc.get("input_gridfs_id")
    if gid is not None:
        try:
            await fs().delete(gid)
        except Exception:
            pass
    await db().jobs.update_one(
        {"_id": oid},
        {"$unset": {"input_gridfs_id": ""}, "$set": {"updated_at": datetime.now(timezone.utc)}},
    )


async def list_active_jobs(
    *, owner: Optional[str] = None, limit: int = 30
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"status": {"$in": ["queued", "running"]}}
    if owner:
        query["owner"] = owner
    cursor = (
        db()
        .jobs.find(query)
        .sort("created_at", -1)
        .limit(min(max(1, int(limit)), 100))
    )
    out: list[dict[str, Any]] = []
    async for doc in cursor:
        doc = await _maybe_fail_stale_job(doc)
        if doc.get("status") in ("queued", "running"):
            out.append(_serialize_job(doc))
    return out


async def list_recent_jobs(
    *, owner: Optional[str] = None, limit: int = 30
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {}
    if owner:
        query["owner"] = owner
    cursor = (
        db()
        .jobs.find(query)
        .sort("created_at", -1)
        .limit(min(max(1, int(limit)), 100))
    )
    return [_serialize_job(doc) async for doc in cursor]


async def get_job(
    job_id: str, *, owner: Optional[str] = None
) -> Optional[dict[str, Any]]:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    query: dict[str, Any] = {"_id": oid}
    if owner:
        query["owner"] = owner
    doc = await db().jobs.find_one(query)
    if not doc:
        return None
    doc = await _maybe_fail_stale_job(doc)
    return _serialize_job(doc)


async def update_job(job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    fields["updated_at"] = datetime.now(timezone.utc)
    await db().jobs.update_one({"_id": oid}, {"$set": fields})
    doc = await db().jobs.find_one({"_id": oid})
    return _serialize_job(doc) if doc else None


async def finish_job_if_active(
    job_id: str,
    *,
    status: str,
    **fields: Any,
) -> Optional[dict[str, Any]]:
    """Only finalize if still queued/running (avoids racing past timeout/orphan fails)."""
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    fields = {
        **fields,
        "status": status,
        "updated_at": datetime.now(timezone.utc),
    }
    await db().jobs.update_one(
        {"_id": oid, "status": {"$in": ["queued", "running"]}},
        {"$set": fields},
    )
    doc = await db().jobs.find_one({"_id": oid})
    if doc and status in ("done", "failed"):
        await delete_job_input(job_id)
        doc = await db().jobs.find_one({"_id": oid}) or doc
    return _serialize_job(doc) if doc else None


_ORPHAN_NO_PAYLOAD_MSG = (
    "Generation stopped — API restarted before this job could be saved for resume. Try again."
)
_STALE_MSG = (
    "Generation timed out on the server. Comfy may still be busy; wait a minute, then try again."
)


def _as_utc(dt: Any) -> Optional[datetime]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def reclaim_active_jobs_on_startup() -> list[dict[str, Any]]:
    """
    After a worker restart, resume jobs that still have a persisted start image.
    Legacy jobs without input_gridfs_id cannot be resumed and are failed.
    """
    now = datetime.now(timezone.utc)
    settings = get_settings()
    max_age = max(120, int(settings.comfyui_timeout_sec) + 120)
    reclaimable: list[dict[str, Any]] = []

    cursor = db().jobs.find({"status": {"$in": ["queued", "running"]}})
    async for doc in cursor:
        oid = doc["_id"]
        anchor = _as_utc(doc.get("started_at")) or _as_utc(doc.get("created_at"))
        if anchor is not None:
            age = (now - anchor).total_seconds()
            if age >= max_age:
                await db().jobs.update_one(
                    {"_id": oid, "status": {"$in": ["queued", "running"]}},
                    {
                        "$set": {
                            "status": "failed",
                            "error": _STALE_MSG,
                            "finished_at": now,
                            "updated_at": now,
                        }
                    },
                )
                await delete_job_input(str(oid))
                continue

        if not doc.get("input_gridfs_id"):
            await db().jobs.update_one(
                {"_id": oid, "status": {"$in": ["queued", "running"]}},
                {
                    "$set": {
                        "status": "failed",
                        "error": _ORPHAN_NO_PAYLOAD_MSG,
                        "finished_at": now,
                        "updated_at": now,
                    }
                },
            )
            continue

        if int(doc.get("resume_count") or 0) >= 5:
            await db().jobs.update_one(
                {"_id": oid, "status": {"$in": ["queued", "running"]}},
                {
                    "$set": {
                        "status": "failed",
                        "error": (
                            "Job abandoned after repeated API restarts. "
                            "Try generating again."
                        ),
                        "finished_at": now,
                        "updated_at": now,
                    }
                },
            )
            await delete_job_input(str(oid))
            continue

        await db().jobs.update_one(
            {"_id": oid},
            {
                "$set": {
                    "status": "queued",
                    "error": None,
                    "started_at": None,
                    "updated_at": now,
                },
                "$inc": {"resume_count": 1},
            },
        )
        refreshed = await db().jobs.find_one({"_id": oid})
        if refreshed:
            reclaimable.append(_serialize_job(refreshed))

    return reclaimable


async def fail_active_jobs_on_startup() -> int:
    """Deprecated path — prefer reclaim_active_jobs_on_startup."""
    jobs = await reclaim_active_jobs_on_startup()
    return len(jobs)


async def _maybe_fail_stale_job(doc: dict[str, Any]) -> dict[str, Any]:
    """If a job has been queued/running longer than the Comfy timeout, fail it."""
    status = doc.get("status")
    if status not in ("queued", "running"):
        return doc
    settings = get_settings()
    max_age = max(120, int(settings.comfyui_timeout_sec) + 120)
    anchor = _as_utc(doc.get("started_at")) or _as_utc(doc.get("created_at"))
    if anchor is None:
        return doc
    age = (datetime.now(timezone.utc) - anchor).total_seconds()
    if age < max_age:
        return doc
    now = datetime.now(timezone.utc)
    await db().jobs.update_one(
        {"_id": doc["_id"], "status": {"$in": ["queued", "running"]}},
        {
            "$set": {
                "status": "failed",
                "error": _STALE_MSG,
                "finished_at": now,
                "updated_at": now,
            }
        },
    )
    await delete_job_input(str(doc["_id"]))
    refreshed = await db().jobs.find_one({"_id": doc["_id"]})
    return refreshed or doc


def _serialize_job(doc: dict[str, Any]) -> dict[str, Any]:
    out = _serialize(doc)
    for key in ("started_at", "finished_at"):
        if key in out and hasattr(out[key], "isoformat"):
            out[key] = out[key].isoformat()
    # Never leak GridFS ObjectIds oddly; stringify if present
    if "input_gridfs_id" in out and out["input_gridfs_id"] is not None:
        out["input_gridfs_id"] = str(out["input_gridfs_id"])
        out["resumable"] = True
    else:
        out["resumable"] = False
    return out


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    if "gridfs_id" in out and not isinstance(out["gridfs_id"], str):
        out["gridfs_id"] = str(out["gridfs_id"])
    if "created_at" in out and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = out["created_at"].isoformat()
    if "updated_at" in out and hasattr(out["updated_at"], "isoformat"):
        out["updated_at"] = out["updated_at"].isoformat()
    meta = out.get("meta")
    if isinstance(meta, dict):
        cleaned = dict(meta)
        for key, val in list(cleaned.items()):
            if hasattr(val, "isoformat"):
                cleaned[key] = val.isoformat()
        out["meta"] = cleaned
    gid = out.get("id")
    if gid and out.get("gridfs_id"):
        out["media_url"] = f"/api/media/{gid}"
        out["thumb_url"] = f"/api/media/{gid}/thumb?w=240"
        out.pop("cdn_url", None)
        out.pop("poster_url", None)
        out.pop("cloudinary_public_id", None)
        out.pop("cloudinary_resource_type", None)
    return out


def _serialize_ref(doc: dict[str, Any]) -> dict[str, Any]:
    out = _serialize(doc)
    gid = out.get("id")
    if gid and out.get("gridfs_id"):
        out["media_url"] = f"/api/test/refs/{gid}/media"
        out["thumb_url"] = f"/api/test/refs/{gid}/thumb?w=240"
    return out


async def store_test_ref(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    preset_id: str,
    owner: str,
) -> dict[str, Any]:
    file_id = await fs().upload_from_stream(
        filename,
        data,
        metadata={
            "content_type": content_type,
            "kind": "test_ref",
            "preset_id": preset_id,
            "owner": owner,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    doc: dict[str, Any] = {
        "kind": "ref",
        "preset_id": preset_id,
        "filename": filename,
        "content_type": content_type,
        "gridfs_id": file_id,
        "size_bytes": len(data),
        "owner": owner,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db().test_refs.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize_ref(doc)


async def list_test_refs(
    *,
    owner: str,
    preset_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query: dict[str, Any] = {"owner": owner}
    if preset_id:
        query["preset_id"] = preset_id
    cursor = (
        db()
        .test_refs.find(query)
        .sort("created_at", -1)
        .limit(max(1, min(int(limit), 200)))
    )
    return [_serialize_ref(doc) async for doc in cursor]


async def count_test_refs_by_preset(*, owner: str) -> dict[str, int]:
    pipeline = [
        {"$match": {"owner": owner}},
        {"$group": {"_id": "$preset_id", "n": {"$sum": 1}}},
    ]
    out: dict[str, int] = {}
    async for row in db().test_refs.aggregate(pipeline):
        key = row.get("_id")
        if key:
            out[str(key)] = int(row.get("n") or 0)
    return out


async def get_test_ref(ref_id: str, *, owner: str) -> Optional[dict[str, Any]]:
    try:
        oid = ObjectId(ref_id)
    except Exception:
        return None
    doc = await db().test_refs.find_one({"_id": oid, "owner": owner})
    return _serialize_ref(doc) if doc else None


async def get_test_ref_bytes(
    ref_id: str, *, owner: str
) -> Optional[tuple[bytes, str, str]]:
    doc = await get_test_ref(ref_id, owner=owner)
    if not doc:
        return None
    stream = await fs().open_download_stream(ObjectId(doc["gridfs_id"]))
    data = await stream.read()
    return data, doc["content_type"], doc["filename"]


async def delete_test_ref(ref_id: str, *, owner: str) -> bool:
    try:
        oid = ObjectId(ref_id)
    except Exception:
        return False
    doc = await db().test_refs.find_one({"_id": oid, "owner": owner})
    if not doc:
        return False
    grid_id = doc.get("gridfs_id")
    if grid_id is not None:
        try:
            await fs().delete(
                ObjectId(grid_id) if not isinstance(grid_id, ObjectId) else grid_id
            )
        except Exception:
            pass
    result = await db().test_refs.delete_one({"_id": oid})
    return result.deleted_count > 0


def _serialize_input(doc: dict[str, Any]) -> dict[str, Any]:
    out = _serialize(doc)
    gid = out.get("id")
    if gid and out.get("gridfs_id"):
        out["media_url"] = f"/api/test/inputs/{gid}/media"
        out["thumb_url"] = f"/api/test/inputs/{gid}/thumb?w=240"
    return out


async def store_test_input(
    *,
    data: bytes,
    filename: str,
    content_type: str,
    owner: str,
) -> dict[str, Any]:
    file_id = await fs().upload_from_stream(
        filename,
        data,
        metadata={
            "content_type": content_type,
            "kind": "test_input",
            "owner": owner,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    doc: dict[str, Any] = {
        "kind": "input",
        "filename": filename,
        "content_type": content_type,
        "gridfs_id": file_id,
        "size_bytes": len(data),
        "owner": owner,
        "created_at": datetime.now(timezone.utc),
    }
    result = await db().test_inputs.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize_input(doc)


async def list_test_inputs(*, owner: str, limit: int = 100) -> list[dict[str, Any]]:
    cursor = (
        db()
        .test_inputs.find({"owner": owner})
        .sort("created_at", -1)
        .limit(max(1, min(int(limit), 200)))
    )
    return [_serialize_input(doc) async for doc in cursor]


async def count_test_inputs(*, owner: str) -> int:
    return int(await db().test_inputs.count_documents({"owner": owner}))


async def get_test_input(item_id: str, *, owner: str) -> Optional[dict[str, Any]]:
    try:
        oid = ObjectId(item_id)
    except Exception:
        return None
    doc = await db().test_inputs.find_one({"_id": oid, "owner": owner})
    return _serialize_input(doc) if doc else None


async def get_test_input_bytes(
    item_id: str, *, owner: str
) -> Optional[tuple[bytes, str, str]]:
    doc = await get_test_input(item_id, owner=owner)
    if not doc:
        return None
    stream = await fs().open_download_stream(ObjectId(doc["gridfs_id"]))
    data = await stream.read()
    return data, doc["content_type"], doc["filename"]


async def delete_test_input(item_id: str, *, owner: str) -> bool:
    try:
        oid = ObjectId(item_id)
    except Exception:
        return False
    doc = await db().test_inputs.find_one({"_id": oid, "owner": owner})
    if not doc:
        return False
    grid_id = doc.get("gridfs_id")
    if grid_id is not None:
        try:
            await fs().delete(
                ObjectId(grid_id) if not isinstance(grid_id, ObjectId) else grid_id
            )
        except Exception:
            pass
    result = await db().test_inputs.delete_one({"_id": oid})
    return result.deleted_count > 0
