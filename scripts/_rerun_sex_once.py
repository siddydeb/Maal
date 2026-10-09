"""Clear orphan Sex jobs/Comfy, then re-submit exactly one Sex job with same image."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from bson import ObjectId
from gridfs import GridFSBucket
from pymongo import MongoClient

API = "https://wan-studio-api.onrender.com"
JOB_RUNNING = "6ab7b9a47666cb948b6c671b"
JOB_FAILED = "6ab7b5e47666cb948b6c670e"


def load_env():
    env = {}
    for line in Path(".env").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("\"'")
    toks = {}
    for line in Path("tokens&cmd").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        toks[k.strip().lower()] = v.strip()
    return env, toks


def http_json(method, url, *, headers=None, data=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            return int(r.status), json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            payload = raw
        return int(e.code), payload


def clear_comfy(base: str):
    base = base.rstrip("/")
    for path, body in (
        ("/interrupt", None),
        ("/queue", json.dumps({"clear": True}).encode()),
    ):
        req = urllib.request.Request(
            base + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                print(path, r.status)
        except Exception as e:
            print(path, "ERR", e)
    time.sleep(0.5)
    try:
        urllib.request.urlopen(
            urllib.request.Request(base + "/interrupt", method="POST"), timeout=15
        ).read()
    except Exception:
        pass
    q = json.loads(urllib.request.urlopen(base + "/queue", timeout=20).read().decode())
    print(
        "queue after clear",
        "running",
        len(q.get("queue_running") or []),
        "pending",
        len(q.get("queue_pending") or []),
    )


def main():
    env, toks = load_env()
    client = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=8000)
    db = client[env.get("MONGODB_DB", "wan_studio")]

    # 1) health + clear comfy orphans
    code, health = http_json("GET", f"{API}/api/health")
    assert code == 200 and health.get("comfyui"), health
    comfy = health["comfyui_url"]
    print("comfy", comfy)
    clear_comfy(comfy)

    # 2) finalize any active sex jobs so only the new one can run
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for jid in (JOB_RUNNING, JOB_FAILED):
        oid = ObjectId(jid)
        before = db.jobs.find_one({"_id": oid}, {"status": 1, "error": 1})
        print("before", jid, before)
    res = db.jobs.update_many(
        {
            "owner": "utester",
            "preset_id": "sex",
            "status": {"$in": ["queued", "running"]},
        },
        {
            "$set": {
                "status": "cancelled",
                "error": "Cleared orphan before single re-run.",
                "finished_at": now,
                "updated_at": now,
            }
        },
    )
    print("finalized active sex jobs", res.modified_count)

    # Prefer input from the latest attempt (has gridfs)
    src = db.jobs.find_one({"_id": ObjectId(JOB_RUNNING)})
    if not src or not src.get("input_gridfs_id"):
        raise SystemExit("missing start image on latest sex job")
    bucket = GridFSBucket(db, bucket_name="media")
    grid_out = bucket.open_download_stream(src["input_gridfs_id"])
    image_bytes = grid_out.read()
    print("image bytes", len(image_bytes))

    prompt = (src.get("prompt") or "").strip()
    video_seconds = src.get("video_seconds") or 5
    print("prompt[:120]", prompt[:120])

    # 3) unlock
    pin = toks.get("prowler_pin")
    if not pin:
        raise SystemExit("missing prowler_pin")
    code, unlock = http_json(
        "POST",
        f"{API}/api/auth/unlock",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"pin": pin}).encode(),
    )
    if code != 200 or not isinstance(unlock, dict) or not unlock.get("token"):
        raise SystemExit(f"unlock failed {code} {unlock}")
    token = unlock["token"]
    print("unlocked owner", unlock.get("owner"))

    # 4) ensure no other active jobs for this owner
    active = list(
        db.jobs.find(
            {"owner": "utester", "status": {"$in": ["queued", "running"]}},
            {"_id": 1, "preset_id": 1, "status": 1},
        )
    )
    print("active before submit", [(str(a["_id"]), a.get("preset_id"), a.get("status")) for a in active])
    if active:
        db.jobs.update_many(
            {"_id": {"$in": [a["_id"] for a in active]}},
            {
                "$set": {
                    "status": "cancelled",
                    "error": "Cleared before single re-run.",
                    "finished_at": now,
                    "updated_at": now,
                }
            },
        )

    # 5) multipart submit one job
    import uuid

    boundary = "----WanBoundary" + uuid.uuid4().hex
    client_key = f"rerun-sex-{uuid.uuid4()}"
    fields = {
        "mode": "vid",
        "prompt": prompt,
        "video_seconds": str(float(video_seconds)),
        "preset_id": "sex",
        "test_run": "1",
        "client_key": client_key,
    }
    body = b""
    for k, v in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        body += f"{v}\r\n".encode()
    body += f"--{boundary}\r\n".encode()
    body += b'Content-Disposition: form-data; name="image"; filename="start.png"\r\n'
    body += b"Content-Type: image/png\r\n\r\n"
    body += image_bytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    code, job = http_json(
        "POST",
        f"{API}/api/jobs",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-Wan-Token": token,
            "Accept": "application/json",
        },
        data=body,
        timeout=120,
    )
    print("submit", code, job)
    if code not in (200, 201) or not isinstance(job, dict) or not job.get("id"):
        raise SystemExit(f"submit failed {code} {job}")

    new_id = job["id"]
    # 6) verify only one active
    active2 = list(
        db.jobs.find(
            {"owner": "utester", "status": {"$in": ["queued", "running"]}},
            {"_id": 1, "preset_id": 1, "status": 1, "client_key": 1},
        )
    )
    print(
        "active after submit",
        [(str(a["_id"]), a.get("preset_id"), a.get("status"), a.get("client_key")) for a in active2],
    )
    if len(active2) != 1 or str(active2[0]["_id"]) != new_id:
        raise SystemExit(f"expected exactly 1 active job={new_id}, got {active2}")

    q = json.loads(urllib.request.urlopen(comfy.rstrip("/") + "/queue", timeout=20).read().decode())
    print(
        "comfy queue",
        "running",
        len(q.get("queue_running") or []),
        "pending",
        len(q.get("queue_pending") or []),
    )
    print("OK single job", new_id, "client_key", client_key)


if __name__ == "__main__":
    main()
