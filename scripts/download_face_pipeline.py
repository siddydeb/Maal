"""Download the face pipeline: Z-Image Turbo (T2I) + FLUX.2 Klein 9B (edit) and their LoRAs.

Hugging Face files are ungated (Comfy-Org / unsloth). Civitai LoRAs use civitai= from tokens&cmd
(never printed); the version matching each base model is picked automatically.
Resumable: partial files continue from where they stopped.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = Path(r"E:\Comfy-Desktop\ComfyUI-Shared\models")
UA = "Mozilla/5.0 (compatible; WanStudioFacePipeline/1.0)"
HF = "https://huggingface.co/{repo}/resolve/main/{path}"

HF_FILES = [
    ("Comfy-Org/z_image_turbo", "split_files/diffusion_models/z_image_turbo_bf16.safetensors", "diffusion_models"),
    ("Comfy-Org/z_image_turbo", "split_files/text_encoders/qwen_3_4b.safetensors", "text_encoders"),
    ("unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q8_0.gguf", "unet"),
    ("Comfy-Org/flux2-klein-9B", "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors", "text_encoders"),
    ("Comfy-Org/flux2-klein-9B", "split_files/vae/flux2-vae.safetensors", "vae"),
    ("dx8152/Flux2-Klein-9B-Consistency", "Flux2-Klein-9B-consistency-V2.safetensors", "loras/klein_consistency_v2.safetensors"),
]

# (civitai model id, accepted base models in preference order, local file name)
CIVITAI = [
    (2268008, ["ZImageTurbo"], "zit_realistic_snapshot.safetensors"),
    (2088956, ["ZImageTurbo"], "zit_famegrid.safetensors"),
    (580857, ["ZImageTurbo"], "zit_skin_texture.safetensors"),
    (562884, ["ZImageTurbo"], "zit_skin_tone_glamour.safetensors"),
    (2395852, ["ZImageTurbo"], "zit_radiant_realism.safetensors"),
    (2234266, ["ZImageTurbo"], "zit_detail_slider.safetensors"),
    (1939453, ["Flux.2 Klein 9B-base", "Flux.2 Klein 9B"], "klein_consistence_edit.safetensors"),
    (2462105, ["Flux.2 Klein 9B"], "klein_ultrareal.safetensors"),
    (2374977, ["Flux.2 Klein 9B"], "klein_realism_engine.safetensors"),
    (2367983, ["Flux.2 Klein 9B-base", "Flux.2 Klein 9B"], "klein_outfit_tryon.safetensors"),
    (2380153, ["Flux.2 Klein 9B-base", "Flux.2 Klein 9B"], "klein_copy_pose.safetensors"),
]


def civitai_token() -> str:
    txt = (REPO / "tokens&cmd").read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?im)^\s*civitai=(.+)$", txt)
    if not m:
        raise SystemExit("missing civitai= in tokens&cmd")
    return m.group(1).strip().strip('"').strip("'")


def fetch(url: str, dest: Path, headers: dict[str, str] | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 6):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                total = have + int(r.headers.get("Content-Length") or 0)
                mode = "ab" if have and r.status == 206 else "wb"
                done = have if mode == "ab" else 0
                last = time.time()
                with open(part, mode) as f:
                    while chunk := r.read(8 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 30:
                            print(f"   {dest.name}: {done / 2**30:.2f} / {total / 2**30:.2f} GB", flush=True)
                            last = time.time()
            part.replace(dest)
            print(f"OK   {dest.name} ({dest.stat().st_size / 2**30:.2f} GB)", flush=True)
            return
        except Exception as exc:
            print(f"   retry {attempt} {dest.name}: {exc}", flush=True)
            time.sleep(10 * attempt)
    raise SystemExit(f"failed: {dest.name}")


def pick_version(model_id: int, bases: list[str], tok: str) -> dict | None:
    req = urllib.request.Request(
        f"https://civitai.com/api/v1/models/{model_id}", headers={"User-Agent": UA, "Authorization": f"Bearer {tok}"}
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        versions = json.loads(r.read()).get("modelVersions") or []
    for base in bases:
        for v in versions:
            if v.get("baseModel") == base:
                return v
    return None


def main() -> int:
    only = set(sys.argv[1:])
    for repo, path, where in HF_FILES:
        dest = MODELS / where if where.endswith(".safetensors") else MODELS / where / Path(path).name
        if only and dest.name not in only:
            continue
        if dest.exists():
            print(f"HAVE {dest.name}", flush=True)
            continue
        print(f"GET  {repo}/{path}", flush=True)
        fetch(HF.format(repo=repo, path=path), dest)

    tok = civitai_token()
    for model_id, bases, name in CIVITAI:
        dest = MODELS / "loras" / name
        if only and name not in only:
            continue
        if dest.exists():
            print(f"HAVE {name}", flush=True)
            continue
        v = pick_version(model_id, bases, tok)
        if not v:
            print(f"SKIP {name}: no version for {bases}", flush=True)
            continue
        words = ", ".join(v.get("trainedWords") or [])[:80]
        print(f"GET  civitai {model_id} v{v['id']} ({v.get('baseModel')}) -> {name}  trigger: {words or '-'}", flush=True)
        fetch(f"https://civitai.com/api/download/models/{v['id']}", dest, {"Authorization": f"Bearer {tok}"})
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
