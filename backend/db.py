"""MongoDB-only persistence. Binary media in GridFS; metadata in collections."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import re
import threading

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket
from pymongo import MongoClient

from backend.config import get_settings

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None
_fs: Optional[AsyncIOMotorGridFSBucket] = None
_sync_client: Optional[MongoClient] = None
_tls = threading.local()


def _sync_db():
    """Sync Mongo client shared across job-worker threads."""
    global _sync_client
    settings = get_settings()
    if _sync_client is None:
        _sync_client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=8000)
    return _sync_client[settings.mongodb_db]


def _sync_jobs():
    """Sync jobs collection (keeps the API event loop free for healthz)."""
    return _sync_db()["jobs"]


def update_job_sync(job_id: str, **fields: Any) -> bool:
    """Apply fields. Returns False if a running claim was dropped (already terminal)."""
    try:
        oid = ObjectId(job_id)
    except Exception:
        return False
    fields = {**fields, "updated_at": datetime.now(timezone.utc)}
    # Never resurrect a cancelled/failed/done job (cancel races with worker start).
    if fields.get("status") == "running":
        result = _sync_jobs().update_one(
            {"_id": oid, "status": {"$in": ["queued", "running"]}},
            {"$set": fields},
        )
        if result.matched_count == 0:
            print(
                f"[wan] update_job_sync: job={job_id} running DROPPED "
                "(already finalized)"
            )
            return False
        return True
    _sync_jobs().update_one({"_id": oid}, {"$set": fields})
    return True


def try_mark_job_running_sync(job_id: str) -> bool:
    """Atomically claim queued → running. False if cancelled/already terminal."""
    return update_job_sync(
        job_id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )


def peek_next_queued_job_sync() -> Optional[dict[str, Any]]:
    """FCFS: oldest queued job that still has a start image."""
    doc = _sync_jobs().find_one(
        {"status": "queued", "input_gridfs_id": {"$ne": None}},
        sort=[("created_at", 1)],
    )
    return _serialize_job(doc) if doc else None


def job_is_active_sync(job_id: str) -> bool:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return False
    doc = _sync_jobs().find_one({"_id": oid}, {"status": 1})
    return bool(doc and doc.get("status") in ("queued", "running"))


def finish_job_if_active_sync(
    job_id: str,
    *,
    status: str,
    **fields: Any,
) -> None:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return
    fields = {
        **fields,
        "status": status,
        "updated_at": datetime.now(timezone.utc),
    }
    # If FE polled stale-fail while the worker was still generating, still accept
    # a late "done" so the video is saved instead of discarded.
    status_filter: dict[str, Any]
    if status == "done":
        status_filter = {
            "$or": [
                {"status": {"$in": ["queued", "running"]}},
                {"status": "failed", "error": _STALE_MSG},
            ]
        }
    else:
        status_filter = {"status": {"$in": ["queued", "running"]}}
    result = _sync_jobs().update_one({"_id": oid, **status_filter}, {"$set": fields})
    if result.matched_count == 0:
        print(
            f"[wan] finish_job_if_active_sync: job={job_id} status={status} "
            "DROPPED (already finalized)"
        )
    if status in ("done", "failed", "cancelled"):
        # Best-effort wipe of start image (async GridFS path may not be available here).
        try:
            doc = _sync_jobs().find_one({"_id": oid}, {"input_gridfs_id": 1})
            gid = doc.get("input_gridfs_id") if doc else None
            if gid is not None:
                from gridfs import GridFSBucket

                GridFSBucket(_sync_db(), bucket_name="media").delete(gid)
                _sync_jobs().update_one({"_id": oid}, {"$unset": {"input_gridfs_id": ""}})
        except Exception:
            pass


def get_job_input_bytes_sync(job_id: str) -> Optional[bytes]:
    try:
        oid = ObjectId(job_id)
    except Exception:
        return None
    doc = _sync_jobs().find_one({"_id": oid}, {"input_gridfs_id": 1})
    if not doc or not doc.get("input_gridfs_id"):
        return None
    try:
        from gridfs import GridFSBucket

        return GridFSBucket(_sync_db(), bucket_name="media").open_download_stream(
            doc["input_gridfs_id"]
        ).read()
    except Exception:
        return None


async def connect() -> None:
    """Bind Motor to the current thread's event loop (main API or job worker)."""
    settings = get_settings()
    client = AsyncIOMotorClient(settings.mongodb_uri)
    database = client[settings.mongodb_db]
    bucket = AsyncIOMotorGridFSBucket(database, bucket_name="media")
    await client.admin.command("ping")
    if threading.current_thread() is threading.main_thread():
        global _client, _db, _fs
        _client = client
        _db = database
        _fs = bucket
    else:
        _tls.client = client
        _tls.db = database
        _tls.fs = bucket


async def close() -> None:
    global _client, _db, _fs, _sync_client
    if threading.current_thread() is threading.main_thread():
        if _client is not None:
            _client.close()
        _client = None
        _db = None
        _fs = None
        if _sync_client is not None:
            _sync_client.close()
            _sync_client = None
    else:
        client = getattr(_tls, "client", None)
        if client is not None:
            client.close()
        _tls.client = None
        _tls.db = None
        _tls.fs = None


def db() -> AsyncIOMotorDatabase:
    local = getattr(_tls, "db", None)
    if local is not None:
        return local
    if _db is None:
        raise RuntimeError("Database not connected")
    return _db


def fs() -> AsyncIOMotorGridFSBucket:
    local = getattr(_tls, "fs", None)
    if local is not None:
        return local
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


async def open_media_stream(
    gen_id: str, *, owner: Optional[str] = None
) -> Optional[tuple[Any, int, str, str]]:
    """GridFS handle for true chunked serving — never buffers the whole file.

    Videos can run tens of MB; reading the full body into RAM per request
    scales badly under concurrent viewers on a memory-capped instance.
    """
    doc = await get_generation(gen_id, owner=owner)
    if not doc:
        return None
    grid_id = ObjectId(doc["gridfs_id"])
    stream = await fs().open_download_stream(grid_id)
    return stream, stream.length, doc["content_type"], doc["filename"]


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

_CLIENT_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


def normalize_client_key(raw: Optional[str]) -> Optional[str]:
    """Stable client idempotency key for POST /api/jobs retries."""
    key = str(raw or "").strip()
    if not key or not _CLIENT_KEY_RE.match(key):
        return None
    return key


async def find_job_by_client_key(
    owner: str, client_key: str
) -> Optional[dict[str, Any]]:
    """Return an existing job for this owner+client_key (retry / reload dedupe)."""
    if not owner or not client_key:
        return None
    doc = await db().jobs.find_one(
        {"owner": owner, "client_key": client_key},
        sort=[("created_at", -1)],
    )
    return _serialize_job(doc) if doc else None


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
    client_key: Optional[str] = None,
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
        "client_key": client_key,
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
    ser = _serialize_job(doc)
    try:
        older = await db().jobs.count_documents(
            {
                "status": "queued",
                "created_at": {"$lt": now},
            }
        )
        ser["queue_position"] = int(older) + 1
    except Exception:
        ser["queue_position"] = None
    return ser


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
    """Queued/running jobs in FCFS order, plus recent failures so they don't vanish."""
    limit = min(max(1, int(limit)), 100)
    query: dict[str, Any] = {"status": {"$in": ["queued", "running"]}}
    if owner:
        query["owner"] = owner
    # FCFS: oldest first (running first among equals via status, then created_at)
    cursor = (
        db()
        .jobs.find(query)
        .sort([("status", -1), ("created_at", 1)])
        .limit(limit)
    )
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    queue_pos = 0
    async for doc in cursor:
        doc = await _maybe_fail_stale_job(doc)
        if doc.get("status") in ("queued", "running"):
            sid = str(doc.get("_id") or doc.get("id") or "")
            seen.add(sid)
            ser = _serialize_job(doc)
            if doc.get("status") == "running":
                ser["queue_position"] = 0
            else:
                queue_pos += 1
                ser["queue_position"] = queue_pos
            out.append(ser)

    # Keep failed/cancelled visible briefly so Ongoing doesn't look abandoned.
    recent_cut = datetime.now(timezone.utc) - timedelta(minutes=45)
    recent_q: dict[str, Any] = {
        "status": {"$in": ["failed", "cancelled"]},
        "finished_at": {"$gte": recent_cut},
    }
    if owner:
        recent_q["owner"] = owner
    recent = (
        db()
        .jobs.find(recent_q)
        .sort("finished_at", -1)
        .limit(min(12, limit))
    )
    async for doc in recent:
        sid = str(doc.get("_id") or "")
        if sid in seen:
            continue
        seen.add(sid)
        ser = _serialize_job(doc)
        ser["queue_position"] = None
        out.append(ser)
    return out


async def cancel_all_active_jobs(
    *,
    owner: str,
    include_running: bool = True,
) -> dict[str, Any]:
    """Clear the owner's queue (queued, and optionally the running job)."""
    now = datetime.now(timezone.utc)
    statuses = ["queued", "running"] if include_running else ["queued"]
    filt = {"owner": owner, "status": {"$in": statuses}}
    ids: list[str] = []
    async for doc in db().jobs.find(filt, {"_id": 1}):
        ids.append(str(doc["_id"]))
    cancelled = 0
    for jid in ids:
        updated = await finish_job_if_active(
            jid,
            status="cancelled",
            error="Queue cleared.",
            finished_at=now,
        )
        if updated and updated.get("status") == "cancelled":
            cancelled += 1
            await delete_job_input(jid)
    return {"cancelled": cancelled, "ids": ids}

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


async def get_avg_duration_sec(mode: str, *, sample: int = 20) -> Optional[float]:
    """Median wall-clock duration (started_at -> finished_at) of the most
    recent completed jobs for this mode, used to power ETA estimates."""
    cursor = (
        db()
        .jobs.find(
            {
                "mode": mode,
                "status": "done",
                "started_at": {"$ne": None},
                "finished_at": {"$ne": None},
            }
        )
        .sort("finished_at", -1)
        .limit(max(1, min(int(sample), 100)))
    )
    durations: list[float] = []
    async for doc in cursor:
        started = _as_utc(doc.get("started_at"))
        finished = _as_utc(doc.get("finished_at"))
        if started is None or finished is None:
            continue
        secs = (finished - started).total_seconds()
        if secs > 0:
            durations.append(secs)
    if not durations:
        return None
    durations.sort()
    mid = len(durations) // 2
    if len(durations) % 2:
        return durations[mid]
    return (durations[mid - 1] + durations[mid]) / 2


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
    if status == "done":
        filt: dict[str, Any] = {
            "_id": oid,
            "$or": [
                {"status": {"$in": ["queued", "running"]}},
                {"status": "failed", "error": _STALE_MSG},
            ],
        }
    else:
        filt = {"_id": oid, "status": {"$in": ["queued", "running"]}}
    result = await db().jobs.update_one(filt, {"$set": fields})
    if result.matched_count == 0:
        print(
            f"[wan] finish_job_if_active: job={job_id} status={status} "
            "DROPPED (already finalized)"
        )
    doc = await db().jobs.find_one({"_id": oid})
    if doc and status in ("done", "failed", "cancelled"):
        await delete_job_input(job_id)
        doc = await db().jobs.find_one({"_id": oid}) or doc
    return _serialize_job(doc) if doc else None


_ORPHAN_NO_PAYLOAD_MSG = (
    "Generation stopped — API restarted before this job could be saved for resume. Try again."
)
_STALE_MSG = (
    "Generation timed out on the server. Comfy may still be busy; wait a minute, then try again."
)


def _job_age_limits(doc: Optional[dict[str, Any]] = None) -> tuple[int, int]:
    """Return (max_age_from_start, max_wall_from_create) in seconds."""
    settings = get_settings()
    max_age = max(120, int(settings.comfyui_timeout_sec) + 120)
    # Wan oral I2V can run 60–90+ min; FE/stale sweeps must not kill early.
    if (doc or {}).get("mode") == "vid":
        max_age = max(max_age, 7200)
    max_wall = max(7200, max_age + 900)
    return max_age, max_wall


def _as_utc(dt: Any) -> Optional[datetime]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


async def reclaim_active_jobs_on_startup() -> list[dict[str, Any]]:
    """
    After a worker restart, re-queue jobs that still have a persisted start image.
    FCFS dispatcher will start exactly one at a time. Legacy jobs without
    input_gridfs_id cannot be resumed and are failed.
    """
    now = datetime.now(timezone.utc)
    reclaimable: list[dict[str, Any]] = []

    cursor = db().jobs.find({"status": {"$in": ["queued", "running"]}}).sort(
        "created_at", 1
    )
    async for doc in cursor:
        oid = doc["_id"]
        max_age, max_wall = _job_age_limits(doc)
        created = _as_utc(doc.get("created_at"))
        started = _as_utc(doc.get("started_at"))
        was_running = doc.get("status") == "running"

        # Absolute abandon (queued can wait longer than a single Comfy run).
        wall_limit = max(max_wall, 86400) if not was_running else max_wall
        if created is not None and (now - created).total_seconds() >= wall_limit:
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

        # Only apply GPU timeout to jobs that were actually running.
        if was_running and started is not None:
            if (now - started).total_seconds() >= max_age:
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

        if int(doc.get("resume_count") or 0) >= 8:
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
    """Fail stuck running jobs; queued jobs may wait in FCFS without GPU timeout."""
    status = doc.get("status")
    if status not in ("queued", "running"):
        return doc
    max_age, max_wall = _job_age_limits(doc)
    now = datetime.now(timezone.utc)
    created = _as_utc(doc.get("created_at"))

    # Queued = waiting for the single GPU slot. Only abandon after a long wall clock.
    if status == "queued":
        max_queue_wait = max(max_wall, 86400)  # at least 24h in queue
        if created is not None and (now - created).total_seconds() >= max_queue_wait:
            await db().jobs.update_one(
                {"_id": doc["_id"], "status": "queued"},
                {
                    "$set": {
                        "status": "failed",
                        "error": (
                            "Job sat in queue too long and was dropped. "
                            "Clear the queue or try again."
                        ),
                        "finished_at": now,
                        "updated_at": now,
                    }
                },
            )
            await delete_job_input(str(doc["_id"]))
            refreshed = await db().jobs.find_one({"_id": doc["_id"]})
            return _serialize_job(refreshed) if refreshed else doc
        return doc

    if created is not None and (now - created).total_seconds() >= max_wall:
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
        return _serialize_job(refreshed) if refreshed else doc
    anchor = _as_utc(doc.get("started_at")) or created
    if anchor is None:
        return doc
    age = (now - anchor).total_seconds()
    if age < max_age:
        return doc
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
    return _serialize_job(refreshed) if refreshed else doc


def _serialize_job(doc: dict[str, Any]) -> dict[str, Any]:
    out = _serialize(doc)
    for key in ("started_at", "finished_at"):
        if key in out and hasattr(out[key], "isoformat"):
            out[key] = _as_utc(out[key]).isoformat()
    # Never leak GridFS ObjectIds oddly; stringify if present
    jid = out.get("id")
    if "input_gridfs_id" in out and out["input_gridfs_id"] is not None:
        out["input_gridfs_id"] = str(out["input_gridfs_id"])
        out["resumable"] = True
        if jid:
            out["input_thumb_url"] = f"/api/jobs/{jid}/input-thumb?w=240"
    else:
        out["resumable"] = False
        out["input_thumb_url"] = None
    return out


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    if "_id" in out:
        out["id"] = str(out.pop("_id"))
    if "gridfs_id" in out and not isinstance(out["gridfs_id"], str):
        out["gridfs_id"] = str(out["gridfs_id"])
    if "created_at" in out and hasattr(out["created_at"], "isoformat"):
        out["created_at"] = _as_utc(out["created_at"]).isoformat()
    if "updated_at" in out and hasattr(out["updated_at"], "isoformat"):
        out["updated_at"] = _as_utc(out["updated_at"]).isoformat()
    meta = out.get("meta")
    if isinstance(meta, dict):
        cleaned = dict(meta)
        for key, val in list(cleaned.items()):
            if hasattr(val, "isoformat"):
                cleaned[key] = _as_utc(val).isoformat()
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
