"""GPU-PC worker for character-LoRA training requested from the app (Train tab, tester PIN).

Nothing stays on this PC between steps:
  - dataset: hero from GridFS → Kontext on local ComfyUI → each shot uploaded to GridFS
  - training: shots downloaded from GridFS to a temp dir → sd-scripts → LoRA copied into
    ComfyUI loras → temp dir (images, caches, checkpoints) deleted
While training, the app's job queue pauses (backend.training.gpu_held_by_training_sync).

Started and kept alive by scripts/wan_stack_watchdog.py; safe to run by hand:
  python scripts/character/trainer_worker.py
"""

from __future__ import annotations

import asyncio
import io
import msvcrt
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from bson import ObjectId
from gridfs import GridFSBucket
from PIL import Image
from pymongo import MongoClient, ReturnDocument

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend import training as T  # noqa: E402
from backend.comfy_client import ComfyClient  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.workflows_wan import KONTEXT_UNET  # noqa: E402

ROOT = Path(r"E:\LoraTraining")
SD_SCRIPTS = ROOT / "sd-scripts"
MODELS = ROOT / "models"
WORK = ROOT / "work"
HEARTBEAT_FILE = ROOT / "trainer_worker.heartbeat"
LOCK_FILE = ROOT / "trainer_worker.lock"
COMFY_LORAS = Path(r"E:\Comfy-Desktop\ComfyUI-Shared\models\loras")

DIM = 16
LEARNING_RATE = 4e-4
RESOLUTION = 768
BLOCKS_TO_SWAP = 10

KEEP = (
    " Keep her exact same face, facial features, eye color, skin tone, body shape, "
    "hairstyle and hair color. Photorealistic, natural skin texture."
)
# (Kontext instruction, caption of what changed). Captions name only what varies,
# so the trigger word absorbs face and body.
VARIATIONS: list[tuple[str, str]] = [
    ("Make it a close-up headshot of her face looking straight at the camera with a neutral expression, plain light grey wall behind her.", "close-up headshot, neutral expression, looking at viewer, plain grey background"),
    ("Show her face in left side profile view, close-up, soft window light.", "close-up, side profile view, soft window light"),
    ("Show her face in right three-quarter view, looking slightly away from the camera, close-up.", "close-up, three-quarter view, looking away"),
    ("Make her laugh with her mouth open and eyes squinting, close-up portrait outdoors in a park.", "close-up portrait, laughing, outdoors, park"),
    ("Make her smile softly with closed lips, head tilted, waist-up shot in a bright cafe.", "waist-up, soft smile, head tilted, cafe"),
    ("Make her look serious and confident with her arms crossed, waist-up, in an office wearing a navy blazer over a white shirt.", "waist-up, serious expression, arms crossed, navy blazer, white shirt, office"),
    ("Show her full body standing on a city sidewalk wearing a black leather jacket, grey t-shirt and blue jeans, daytime.", "full body, standing, black leather jacket, grey t-shirt, blue jeans, city street, daytime"),
    ("Show her full body walking on a beach at sunset wearing a flowing white linen dress.", "full body, walking, white linen dress, beach, sunset"),
    ("Show her sitting cross-legged on a bed in a cozy bedroom wearing an oversized beige knit sweater, warm lamp light.", "sitting on bed, oversized beige sweater, bedroom, warm lamp light"),
    ("Show her in the kitchen holding a coffee mug, wearing a pink pajama set, morning light.", "holding coffee mug, pink pajamas, kitchen, morning light"),
    ("Show her at the gym wearing a black sports bra and black leggings, mirror selfie with a phone.", "mirror selfie, black sports bra, black leggings, gym"),
    ("Show her in a red evening gown at a restaurant table at night, candle light.", "red evening gown, restaurant, night, candle light"),
    ("Show her wearing a yellow bikini standing by a swimming pool, bright sunlight.", "yellow bikini, standing, swimming pool, bright sunlight"),
    ("Take the photo from a high angle looking down at her while she looks up at the camera, indoors.", "high angle, looking up at viewer, indoors"),
    ("Take the photo from a low angle looking up at her, outdoors with blue sky behind her.", "low angle, blue sky background, outdoors"),
    ("Show her from behind looking back over her shoulder at the camera, wearing a white tank top.", "from behind, looking over shoulder, white tank top"),
    ("Make it a night photo with direct camera flash, she is at a party wearing a silver sequin top.", "night, camera flash, party, silver sequin top"),
    ("Make it golden hour backlight in a wheat field, she wears a denim jacket, hair blowing in the wind.", "golden hour, backlit, wheat field, denim jacket, wind in hair"),
    ("Show her lying on a sofa reading a book, wearing a grey hoodie, relaxed.", "lying on sofa, reading, grey hoodie"),
    ("Show her in a car driver seat wearing sunglasses pushed up on her head, daylight, selfie angle.", "car interior, selfie, sunglasses on head, daylight"),
    ("Show her with a surprised expression, eyebrows raised, close-up, studio lighting against a dark background.", "close-up, surprised expression, studio lighting, dark background"),
    ("Show her winking playfully at the camera, close-up, outdoors on a rooftop at dusk.", "close-up, winking, rooftop, dusk"),
    ("Show her wearing a green sundress sitting on a park bench, medium shot, overcast day.", "medium shot, green sundress, sitting on bench, park, overcast"),
    ("Show her in black lace lingerie standing in a hotel room, soft window light, medium shot.", "medium shot, black lace lingerie, hotel room, window light"),
]

STEP_RE = re.compile(r"steps:\s+\d+%.*?\|\s*(\d+)/(\d+)")


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def now() -> datetime:
    return datetime.now(timezone.utc)


class Worker:
    def __init__(self) -> None:
        self.settings = get_settings()
        client = MongoClient(self.settings.mongodb_uri, serverSelectionTimeoutMS=15000)
        self.db = client[self.settings.mongodb_db]
        self.fs = GridFSBucket(self.db, bucket_name="media")
        self.current: Optional[ObjectId] = None
        self.stop = threading.Event()

    # ---------- bookkeeping ----------

    def heartbeat_loop(self) -> None:
        while not self.stop.is_set():
            try:
                t = now()
                self.db.worker_status.update_one(
                    {"_id": "trainer"},
                    {"$set": {"heartbeat_at": t, "host": socket.gethostname(), "busy": self.current is not None}},
                    upsert=True,
                )
                if self.current is not None:
                    self.db.trainings.update_one(
                        {"_id": self.current, "status": {"$in": ["dataset_building", "training"]}},
                        {"$set": {"heartbeat_at": t}},
                    )
                HEARTBEAT_FILE.write_text(t.isoformat(), encoding="utf-8")
            except Exception as exc:
                log(f"heartbeat failed: {exc}")
            self.stop.wait(15)

    def update(self, tid: ObjectId, **fields: Any) -> None:
        self.db.trainings.update_one({"_id": tid}, {"$set": {**fields, "updated_at": now()}})

    def progress(self, tid: ObjectId, phase: str, done: int, total: int, message: str) -> None:
        self.update(tid, progress={"phase": phase, "done": done, "total": total, "message": message})

    def cancel_requested(self, tid: ObjectId) -> bool:
        doc = self.db.trainings.find_one({"_id": tid}, {"cancel_requested": 1})
        return bool(doc and doc.get("cancel_requested"))

    def finish(self, tid: ObjectId, status: str, message: str, error: Optional[str] = None) -> None:
        doc = self.db.trainings.find_one({"_id": tid}, {"progress": 1}) or {}
        prog = {**(doc.get("progress") or {}), "message": message}
        self.update(tid, status=status, error=error, finished_at=now(), progress=prog)
        log(f"training {tid} -> {status}: {message}")

    def claim(self, src: str, dst: str) -> Optional[dict[str, Any]]:
        return self.db.trainings.find_one_and_update(
            {"status": src},
            {"$set": {"status": dst, "started_at": now(), "heartbeat_at": now(), "updated_at": now()}},
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
        )

    def recover_after_crash(self) -> None:
        """Only one worker exists, so anything 'busy' at startup was interrupted."""
        for src, dst in (("dataset_building", "dataset_queued"), ("training", "train_queued")):
            res = self.db.trainings.update_many({"status": src}, {"$set": {"status": dst, "updated_at": now()}})
            if res.modified_count:
                log(f"re-queued {res.modified_count} interrupted {src}")
        if WORK.is_dir():
            shutil.rmtree(WORK, ignore_errors=True)

    # ---------- dataset ----------

    def build_dataset(self, t: dict[str, Any]) -> None:
        tid = t["_id"]
        self.current = tid
        try:
            asyncio.run(self._build_dataset(t))
        except Exception as exc:
            self.finish(tid, "failed", "Dataset build failed.", error=str(exc)[:500])
        finally:
            self.current = None

    async def _build_dataset(self, t: dict[str, Any]) -> None:
        tid = t["_id"]
        stid = str(tid)
        client = ComfyClient(self.settings)
        if not await client.health(timeout=10.0, retries=3):
            self.update(tid, status="dataset_queued")
            self.progress(tid, "dataset", 0, len(VARIATIONS), "ComfyUI is down on the trainer PC — will retry.")
            time.sleep(60)
            return
        hero = self.db.training_images.find_one({"training_id": stid, "kind": "hero"})
        if not hero:
            self.finish(tid, "failed", "Hero image missing.", error="hero missing")
            return
        hero_bytes = self.fs.open_download_stream(hero["gridfs_id"]).read()
        have = {d["idx"] for d in self.db.training_images.find({"training_id": stid}, {"idx": 1})}
        warnings: list[str] = list(t.get("warnings") or [])
        total = len(VARIATIONS)
        for i, (instruction, caption) in enumerate(VARIATIONS, start=1):
            if self.cancel_requested(tid):
                self.finish(tid, "cancelled", "Cancelled.")
                return
            if i in have:
                continue
            self.progress(tid, "dataset", i - 1, total, f"Making shot {i} of {total}: {caption}")
            data = None
            for attempt in (1, 2):
                try:
                    data, _ = await client.generate_image(
                        hero_bytes,
                        instruction + KEEP,
                        seed=1000 + i,
                        steps=28,
                        guidance=2.5,
                        flux_unet=KONTEXT_UNET,
                        edit_graph="kontext",
                    )
                    break
                except Exception as exc:
                    log(f"shot {i} attempt {attempt} failed: {exc}")
                    await asyncio.sleep(5)
            if data is None:
                warnings.append(f"shot {i} ({caption}) failed")
                continue
            buf = io.BytesIO()
            Image.open(io.BytesIO(data)).convert("RGB").save(buf, format="JPEG", quality=95)
            gid = self.fs.upload_from_stream(
                f"train_{stid}_{i:03d}.jpg",
                io.BytesIO(buf.getvalue()),
                metadata={"kind": T.IMG_KIND, "training_id": stid, "owner": t.get("owner")},
            )
            self.db.training_images.insert_one(
                {
                    "training_id": stid,
                    "idx": i,
                    "kind": "variation",
                    "caption": f"{t['trigger']} woman, {caption}",
                    "gridfs_id": gid,
                    "content_type": "image/jpeg",
                    "created_at": now(),
                }
            )
            self.update(tid, warnings=warnings)
        n = self.db.training_images.count_documents({"training_id": stid})
        self.update(tid, status="review", warnings=warnings)
        self.progress(tid, "dataset", total, total, f"{n} shots ready — remove any where her face drifted, then start training.")
        log(f"dataset {tid} ready with {n} shots")

    # ---------- training ----------

    def gpu_busy_with_app(self) -> bool:
        if self.db.jobs.find_one({"status": "running"}, {"_id": 1}):
            return True
        try:
            q = httpx.get(f"{self.settings.comfyui_url.rstrip('/')}/queue", timeout=10).json()
            return bool(q.get("queue_running") or q.get("queue_pending"))
        except Exception:
            return False

    def train(self, t: dict[str, Any]) -> None:
        tid = t["_id"]
        self.current = tid
        work = WORK / str(tid)
        try:
            self._train(t, work)
        except Exception as exc:
            self.finish(tid, "failed", "Training failed.", error=str(exc)[:800])
        finally:
            shutil.rmtree(work, ignore_errors=True)
            self.current = None

    def _train(self, t: dict[str, Any], work: Path) -> None:
        tid = t["_id"]
        stid = str(tid)
        slug, trigger = t["slug"], t["trigger"]

        while self.gpu_busy_with_app():
            if self.cancel_requested(tid):
                self.finish(tid, "cancelled", "Cancelled.")
                return
            self.progress(tid, "train", 0, 0, "Waiting for the current app job to finish (new app jobs are paused)…")
            time.sleep(15)
        try:
            httpx.post(
                f"{self.settings.comfyui_url.rstrip('/')}/free",
                json={"unload_models": True, "free_memory": True},
                timeout=15,
            )
        except Exception:
            pass

        img_dir = work / "img" / f"{T.REPEATS}_{trigger} woman"
        img_dir.mkdir(parents=True, exist_ok=True)
        images = list(self.db.training_images.find({"training_id": stid}).sort("idx", 1))
        for img in images:
            ext = ".png" if "png" in (img.get("content_type") or "") else ".jpg"
            (img_dir / f"{img['idx']:03d}{ext}").write_bytes(self.fs.open_download_stream(img["gridfs_id"]).read())
            (img_dir / f"{img['idx']:03d}.txt").write_text(img["caption"], encoding="utf-8")
        total = len(images) * T.REPEATS * T.EPOCHS
        self.progress(tid, "train", 0, total, "Loading Flux for training…")

        toml = work / "dataset.toml"
        toml.write_text(
            "\n".join(
                [
                    "[general]",
                    'caption_extension = ".txt"',
                    "keep_tokens = 1",
                    "",
                    "[[datasets]]",
                    f"resolution = {RESOLUTION}",
                    "batch_size = 1",
                    "enable_bucket = true",
                    "min_bucket_reso = 512",
                    "max_bucket_reso = 1024",
                    "",
                    "  [[datasets.subsets]]",
                    f"  image_dir = '{img_dir.as_posix()}'",
                    f"  num_repeats = {T.REPEATS}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        out_dir = work / "out"
        out_name = f"char_{slug}_v1"
        cmd = [
            str(SD_SCRIPTS / "venv" / "Scripts" / "accelerate.exe"), "launch",
            "--num_processes", "1", "--num_machines", "1", "--mixed_precision", "bf16",
            "--dynamo_backend", "no", "--num_cpu_threads_per_process", "1",
            "flux_train_network.py",
            "--pretrained_model_name_or_path", str(MODELS / "flux1-dev.safetensors"),
            "--clip_l", str(MODELS / "clip_l.safetensors"),
            "--t5xxl", str(MODELS / "t5xxl_fp16.safetensors"),
            "--ae", str(MODELS / "ae.safetensors"),
            "--dataset_config", str(toml),
            "--output_dir", str(out_dir), "--output_name", out_name,
            "--save_model_as", "safetensors", "--save_precision", "bf16",
            "--network_module", "networks.lora_flux",
            "--network_dim", str(DIM), "--network_alpha", str(DIM),
            "--network_train_unet_only",
            "--optimizer_type", "adafactor",
            "--optimizer_args", "relative_step=False", "scale_parameter=False", "warmup_init=False",
            "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "100", "--max_grad_norm", "0.0",
            "--learning_rate", str(LEARNING_RATE),
            "--max_train_epochs", str(T.EPOCHS), "--save_every_n_epochs", str(T.EPOCHS),
            "--mixed_precision", "bf16", "--fp8_base", "--sdpa", "--gradient_checkpointing",
            "--blocks_to_swap", str(BLOCKS_TO_SWAP),
            "--cache_latents", "--cache_latents_to_disk",
            "--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
            "--guidance_scale", "1.0", "--timestep_sampling", "shift", "--discrete_flow_shift", "3.1582",
            "--model_prediction_type", "raw", "--loss_type", "l2",
            "--max_data_loader_n_workers", "1", "--persistent_data_loader_workers",
            "--seed", "42",
        ]
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        log(f"training {tid} ({slug}): {len(images)} shots, {total} steps")
        proc = subprocess.Popen(
            cmd,
            cwd=str(SD_SCRIPTS),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        tail: deque[str] = deque(maxlen=30)
        last_report = 0.0
        latest: Optional[tuple[int, int]] = None
        started = time.time()
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail.append(line)
            m = STEP_RE.search(line)
            if m:
                latest = (int(m.group(1)), int(m.group(2)))
            if time.time() - last_report < 20:
                continue
            last_report = time.time()
            if self.cancel_requested(tid):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
                proc.wait()
                self.finish(tid, "cancelled", "Cancelled.")
                return
            if latest:
                step, steps = latest
                eta = ""
                if step > 0:
                    left = (time.time() - started) / step * (steps - step)
                    eta = f" · about {int(left // 3600)}h {int(left % 3600 // 60)}m left"
                self.progress(tid, "train", step, steps, f"Training step {step} of {steps}{eta}")
        rc = proc.wait()
        lora = out_dir / f"{out_name}.safetensors"
        if rc != 0 or not lora.is_file():
            self.finish(tid, "failed", "Training failed.", error="\n".join(list(tail)[-12:])[:1500])
            return

        dest = COMFY_LORAS / lora.name
        shutil.copy2(lora, dest)
        self.db.characters.update_one(
            {"slug": slug},
            {
                "$set": {
                    "slug": slug,
                    "name": t.get("name") or slug,
                    "lora": dest.name,
                    "trigger": f"{trigger} woman",
                    "strength": 1.0,
                    "owner": t.get("owner"),
                    "training_id": stid,
                    "created_at": now(),
                }
            },
            upsert=True,
        )
        self.update(tid, lora_file=dest.name)
        self.progress(tid, "train", total, total, "Done")
        self.finish(tid, "done", f"Ready — pick {t.get('name') or slug} under Girl in Text to Image.")

    # ---------- loop ----------

    def run(self) -> None:
        self.recover_after_crash()
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        log("trainer worker up")
        while True:
            try:
                t = self.claim("dataset_queued", "dataset_building")
                if t:
                    log(f"building dataset for {t['slug']} ({t['_id']})")
                    self.build_dataset(t)
                    continue
                t = self.claim("train_queued", "training")
                if t:
                    self.train(t)
                    continue
            except Exception as exc:
                log(f"loop error: {exc}")
            time.sleep(10)


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK_FILE, "a+")
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        log("another trainer worker is running — exit")
        return 0
    Worker().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
