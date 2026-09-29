"""Character-LoRA trainings stored in Mongo.

The API only records requests and serves the dataset; the GPU PC runs
``scripts/character/trainer_worker.py``, which builds the dataset (Kontext via
local ComfyUI → GridFS) and trains (GridFS → temp dir → sd-scripts → ComfyUI loras).

Status flow:
  dataset_queued → dataset_building → review → train_queued → training → done
  (any step → failed | cancelled)
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bson import ObjectId

from backend import db

WORKER_ONLINE_SEC = 90
GPU_LOCK_STALE_SEC = 300
MIN_IMAGES = 10
REPEATS = 10
EPOCHS = 6
SEC_PER_STEP = 7.0
IMG_KIND = "train_img"

_BUSY = ("dataset_building", "training")
_OPEN = ("dataset_queued", "dataset_building", "review", "train_queued", "training")


class TrainingError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except Exception as exc:
        raise TrainingError("Not found") from exc


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "", (name or "").lower())[:16]
    if not slug or not slug[0].isalpha():
        raise TrainingError("Name must start with a letter (letters and numbers only).")
    return slug


def estimate_train_minutes(n_images: int) -> int:
    return int(round(n_images * REPEATS * EPOCHS * SEC_PER_STEP / 60 + 5))


def _iso(v: Any) -> Any:
    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        return v.isoformat()
    return v


def _ser_training(doc: dict[str, Any], n_images: Optional[int] = None) -> dict[str, Any]:
    out = {k: _iso(v) for k, v in doc.items() if k != "_id"}
    out["id"] = str(doc["_id"])
    out.pop("hero_gridfs_id", None)
    if n_images is not None:
        out["image_count"] = n_images
        out["train_minutes_estimate"] = estimate_train_minutes(n_images)
    return out


def _ser_image(doc: dict[str, Any]) -> dict[str, Any]:
    iid = str(doc["_id"])
    return {
        "id": iid,
        "idx": doc.get("idx"),
        "kind": doc.get("kind"),
        "caption": doc.get("caption"),
        "thumb_url": f"/api/train/images/{iid}?w=320",
        "media_url": f"/api/train/images/{iid}",
    }


async def _owned(owner: str, training_id: str) -> dict[str, Any]:
    doc = await db.db().trainings.find_one({"_id": _oid(training_id), "owner": owner})
    if not doc:
        raise TrainingError("Not found")
    return doc


async def hero_candidates(owner: str, limit: int = 48) -> list[dict[str, Any]]:
    """Recent text-to-image results (fall back to any image) to pick a hero from."""
    base = {"owner": owner, "kind": "img"}
    cur = db.db().generations.find({**base, "meta.mode": "t2i"}).sort("created_at", -1).limit(limit)
    docs = await cur.to_list(length=limit)
    if not docs:
        cur = db.db().generations.find(base).sort("created_at", -1).limit(limit)
        docs = await cur.to_list(length=limit)
    return [
        {
            "id": str(d["_id"]),
            "prompt": d.get("prompt"),
            "created_at": _iso(d.get("created_at")),
            "thumb_url": f"/api/media/{d['_id']}/thumb?w=240",
        }
        for d in docs
    ]


async def create_training(owner: str, name: str, hero_generation_id: str) -> dict[str, Any]:
    display = (name or "").strip()[:32]
    slug = slugify(display)
    if await db.db().characters.find_one({"slug": slug}):
        raise TrainingError(f"A girl named '{slug}' already exists — pick another name.")
    if await db.db().trainings.find_one({"slug": slug, "status": {"$in": list(_OPEN) + ["done"]}}):
        raise TrainingError(f"'{slug}' is already being trained — pick another name.")

    gen = await db.get_generation(hero_generation_id, owner=owner)
    if not gen or gen.get("kind") != "img":
        raise TrainingError("Pick an image from your library as the hero.")
    stream = await db.fs().open_download_stream(ObjectId(gen["gridfs_id"]))
    hero_bytes = await stream.read()

    now = _now()
    trigger = f"{slug}_v1"
    doc = {
        "owner": owner,
        "name": display,
        "slug": slug,
        "trigger": trigger,
        "hero_generation_id": hero_generation_id,
        "status": "dataset_queued",
        "progress": {"phase": "dataset", "done": 0, "total": 0, "message": "Waiting for the trainer PC…"},
        "cancel_requested": False,
        "error": None,
        "warnings": [],
        "created_at": now,
        "updated_at": now,
    }
    res = await db.db().trainings.insert_one(doc)
    tid = str(res.inserted_id)
    gid = await db.fs().upload_from_stream(
        f"train_{tid}_000.png",
        io.BytesIO(hero_bytes),
        metadata={"kind": IMG_KIND, "training_id": tid, "owner": owner},
    )
    await db.db().training_images.insert_one(
        {
            "training_id": tid,
            "idx": 0,
            "kind": "hero",
            "caption": f"{trigger} woman, photo",
            "gridfs_id": gid,
            "content_type": gen.get("content_type") or "image/png",
            "created_at": now,
        }
    )
    doc["_id"] = res.inserted_id
    return _ser_training(doc, 1)


async def _image_count(training_id: str) -> int:
    return await db.db().training_images.count_documents({"training_id": training_id})


async def list_trainings(owner: str) -> list[dict[str, Any]]:
    docs = await db.db().trainings.find({"owner": owner}).sort("created_at", -1).to_list(length=50)
    return [_ser_training(d, await _image_count(str(d["_id"]))) for d in docs]


async def get_training(owner: str, training_id: str) -> dict[str, Any]:
    doc = await _owned(owner, training_id)
    imgs = await db.db().training_images.find({"training_id": training_id}).sort("idx", 1).to_list(length=200)
    out = _ser_training(doc, len(imgs))
    out["images"] = [_ser_image(i) for i in imgs]
    return out


async def read_image(owner: str, image_id: str) -> tuple[bytes, str]:
    img = await db.db().training_images.find_one({"_id": _oid(image_id)})
    if not img:
        raise TrainingError("Not found")
    await _owned(owner, img["training_id"])
    stream = await db.fs().open_download_stream(img["gridfs_id"])
    return await stream.read(), img.get("content_type") or "image/jpeg"


async def _delete_image_doc(img: dict[str, Any]) -> None:
    try:
        await db.fs().delete(img["gridfs_id"])
    except Exception:
        pass
    await db.db().training_images.delete_one({"_id": img["_id"]})


async def delete_image(owner: str, image_id: str) -> None:
    img = await db.db().training_images.find_one({"_id": _oid(image_id)})
    if not img:
        raise TrainingError("Not found")
    t = await _owned(owner, img["training_id"])
    if t["status"] != "review":
        raise TrainingError("Shots can only be removed while reviewing.")
    await _delete_image_doc(img)


async def start_training(owner: str, training_id: str) -> dict[str, Any]:
    t = await _owned(owner, training_id)
    if t["status"] != "review":
        raise TrainingError("Build and review the dataset first.")
    n = await _image_count(training_id)
    if n < MIN_IMAGES:
        raise TrainingError(f"Need at least {MIN_IMAGES} shots (have {n}).")
    await db.db().trainings.update_one(
        {"_id": t["_id"], "status": "review"},
        {
            "$set": {
                "status": "train_queued",
                "progress": {"phase": "train", "done": 0, "total": n * REPEATS * EPOCHS, "message": "Waiting for the trainer PC…"},
                "updated_at": _now(),
            }
        },
    )
    return await get_training(owner, training_id)


async def cancel_training(owner: str, training_id: str) -> dict[str, Any]:
    t = await _owned(owner, training_id)
    now = _now()
    if t["status"] in _BUSY:
        await db.db().trainings.update_one(
            {"_id": t["_id"]}, {"$set": {"cancel_requested": True, "updated_at": now}}
        )
    elif t["status"] in _OPEN:
        await db.db().trainings.update_one(
            {"_id": t["_id"], "status": t["status"]},
            {"$set": {"status": "cancelled", "finished_at": now, "updated_at": now}},
        )
    return await get_training(owner, training_id)


async def delete_training(owner: str, training_id: str) -> None:
    """Remove the record and its dataset from Mongo. A finished girl stays usable."""
    t = await _owned(owner, training_id)
    if t["status"] in _BUSY:
        raise TrainingError("Cancel it first.")
    async for img in db.db().training_images.find({"training_id": training_id}):
        await _delete_image_doc(img)
    await db.db().trainings.delete_one({"_id": t["_id"]})


async def worker_status() -> dict[str, Any]:
    doc = await db.db().worker_status.find_one({"_id": "trainer"}) or {}
    hb = doc.get("heartbeat_at")
    if isinstance(hb, datetime) and hb.tzinfo is None:
        hb = hb.replace(tzinfo=timezone.utc)
    online = bool(hb and _now() - hb < timedelta(seconds=WORKER_ONLINE_SEC))
    return {"online": online, "heartbeat_at": _iso(hb), "busy": bool(doc.get("busy")) and online}


def gpu_held_by_training_sync() -> bool:
    """True while a live training owns the GPU (app queue must not start jobs)."""
    cutoff = _now() - timedelta(seconds=GPU_LOCK_STALE_SEC)
    return (
        db._sync_db().trainings.find_one(
            {"status": "training", "heartbeat_at": {"$gt": cutoff}}, {"_id": 1}
        )
        is not None
    )


def list_characters_sync() -> list[dict[str, Any]]:
    """Trained girls in t2i character format."""
    out = []
    for c in db._sync_db().characters.find({}).sort("created_at", 1):
        out.append(
            {
                "id": c["slug"],
                "name": c.get("name") or c["slug"],
                "lora": c["lora"],
                "trigger": c.get("trigger") or f"{c['slug']}_v1 woman",
                "strength": float(c.get("strength", 1.0)),
            }
        )
    return out
