from pathlib import Path
from datetime import timezone
import json
from pymongo import MongoClient
from bson import ObjectId
from gridfs import GridFSBucket

env = {}
for line in Path(".env").read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    env[k.strip()] = v.strip().strip("\"'")

client = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=8000)
db = client[env.get("MONGODB_DB", "wan_studio")]

for jid in ("6ab7b9a47666cb948b6c671b", "6ab7b5e47666cb948b6c670e"):
    j = db.jobs.find_one({"_id": ObjectId(jid)})
    print("====", jid, "====")
    for k in sorted(j.keys()):
        if k in ("prompt", "prompt_english", "error", "negative"):
            print(k, ":", (str(j.get(k)) or "")[:400])
        elif k == "result":
            r = j.get("result")
            print("result keys", list(r.keys()) if isinstance(r, dict) else type(r))
        elif k == "input_gridfs_id":
            print("input_gridfs_id", j.get(k))
        else:
            print(k, ":", j.get(k))
    print()

# health / comfy
import urllib.request
h = json.loads(urllib.request.urlopen("https://wan-studio-api.onrender.com/api/health", timeout=40).read().decode())
base = h["comfyui_url"].rstrip("/")
q = json.loads(urllib.request.urlopen(base + "/queue", timeout=20).read().decode())
print("comfy queue_running", len(q.get("queue_running") or []), "pending", len(q.get("queue_pending") or []))
# peek running prompt loras if any
for item in (q.get("queue_running") or [])[:1]:
    prompt = item[2] if len(item) > 2 else {}
    loras = []
    for nid, node in prompt.items():
        if node.get("class_type") == "LoraLoaderModelOnly":
            loras.append(node.get("inputs", {}).get("lora_name"))
    print("running loras", loras)
    print("running prompt_id", item[1] if len(item) > 1 else None)
