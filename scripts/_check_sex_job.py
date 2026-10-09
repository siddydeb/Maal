from pathlib import Path
import json, urllib.request
from pymongo import MongoClient

root = Path(r"d:/YtAuto/contrnt")
env = {}
for line in (root / ".env").read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip().strip("\"'")

toks = {}
for line in (root / "tokens&cmd").read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    toks[k.strip().lower()] = v.strip()

api = "https://wan-studio-api.onrender.com"
jid = "6ab7bad37666cb948b6c6727"
pin = toks.get("prowler_pin")
req = urllib.request.Request(
    f"{api}/api/auth/unlock",
    data=json.dumps({"pin": pin}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
unlock = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
token = unlock["token"]

req = urllib.request.Request(
    f"{api}/api/jobs/{jid}",
    headers={"X-Wan-Token": token, "Accept": "application/json"},
)
j = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
keys = ("id", "status", "preset_id", "mode", "error", "progress", "client_key", "created_at", "started_at")
print("job", {k: j.get(k) for k in keys})

req2 = urllib.request.Request(
    f"{api}/api/comfy/status",
    headers={"X-Wan-Token": token, "Accept": "application/json"},
)
print("comfy", json.loads(urllib.request.urlopen(req2, timeout=60).read().decode()))

cli = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=20000)
db = cli[env.get("MONGODB_DB", "wan_studio")]
active = list(
    db.jobs.find(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 1, "preset_id": 1, "status": 1, "client_key": 1, "owner": 1, "error": 1},
    )
)
print("active_count", len(active))
print("active", [(str(x["_id"]), x.get("owner"), x.get("preset_id"), x.get("status"), x.get("client_key")) for x in active])
