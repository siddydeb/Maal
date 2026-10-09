"""Show LoRA stack / sampling meta for recent Sex jobs (read-only)."""

from __future__ import annotations

import json
from pathlib import Path

from bson import ObjectId
from pymongo import MongoClient

REPO = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (REPO / ".env").read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip("\"'")
    return env


def main() -> int:
    env = load_env()
    db = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=15000)[env.get("MONGODB_DB", "wan_studio")]
    jobs = list(db.jobs.find({"preset_id": "sex"}).sort("created_at", -1).limit(6))
    for j in jobs:
        print("=" * 100)
        print(j["_id"], j.get("status"), j.get("created_at"), "err=", (j.get("error") or "")[:160])
        res = j.get("result") or {}
        gid = res.get("id")
        meta = res.get("meta") or {}
        if gid and not meta:
            try:
                g = db.generations.find_one({"_id": ObjectId(gid)}, {"meta": 1})
                meta = (g or {}).get("meta") or {}
            except Exception:
                pass
        keys = [k for k in meta if any(s in k.lower() for s in ("lora", "motion", "step", "cfg", "shift", "hnf", "fps", "width", "height", "frames", "kind", "prompt"))]
        for k in sorted(keys):
            v = meta[k]
            s = json.dumps(v, default=str)
            print(f"  {k}: {s[:700]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
