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

cli = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=20000)
db = cli[env.get("MONGODB_DB", "wan_studio")]
active = list(
    db.jobs.find(
        {"status": {"$in": ["queued", "running"]}},
        {"_id": 1, "preset_id": 1, "status": 1, "client_key": 1, "owner": 1},
    )
)
print("active_count", len(active))
for x in active:
    print(str(x["_id"]), x.get("owner"), x.get("preset_id"), x.get("status"), x.get("client_key"))

health = json.loads(urllib.request.urlopen("https://wan-studio-api.onrender.com/api/health", timeout=60).read().decode())
comfy = health.get("comfyui_url", "").rstrip("/")
print("comfy_url", comfy)
q = json.loads(urllib.request.urlopen(comfy + "/queue", timeout=20).read().decode())
print("queue_running", len(q.get("queue_running") or []), "queue_pending", len(q.get("queue_pending") or []))
hist = json.loads(urllib.request.urlopen(comfy + "/history?max_items=3", timeout=20).read().decode())
print("history_keys", list(hist.keys())[:5] if isinstance(hist, dict) else type(hist))
