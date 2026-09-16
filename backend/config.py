from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    cors_origins: str = "*"
    serve_frontend: bool = False

    mongodb_uri: str
    mongodb_db: str = "wan_studio"

    comfyui_url: str = "http://127.0.0.1:8188"
    # Wan oral I2V on 16GB often needs 35–50 min (52 steps, dual LoRA stage).
    comfyui_timeout_sec: int = 3600
    comfyui_dir: Optional[str] = None

    zero_residue: bool = True
    scrub_passes: int = 1
    scrub_clear_all_history: bool = True

    # --- IMG: Flux Dev ---
    flux_unet: str = "flux1-dev-fp8.safetensors"
    flux_clip_l: str = "clip_l.safetensors"
    flux_t5: str = "t5xxl_fp8_e4m3fn.safetensors"
    flux_vae: str = "ae.safetensors"
    flux_guidance: float = 4.0
    image_width: int = 1024
    image_height: int = 1024
    # Higher = stronger prompt adherence on Dev img2img; Kontext uses denoise 1.0
    image_denoise: float = 0.55
    sampler_steps: int = 28
    # Raw mode: no edit/motion wrappers, quality-only negatives, user text first
    raw_prompt: bool = True
    # Soft default denoise ceiling for Dev img2img (raised in raw mode by runners)
    image_denoise_cap: float = 0.85

    # --- VID: Wan 2.2 I2V 14B dual-stage ---
    wan_unet_high: str = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    wan_unet_low: str = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
    wan_vae: str = "wan_2.1_vae.safetensors"
    wan_clip: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    wan_shift: float = 5.0
    video_width: int = 640
    video_height: int = 640
    video_length: int = 49
    video_fps: int = 16
    video_steps: int = 32
    video_cfg: float = 3.5

    # legacy aliases (ignored by new workflows; kept for old .env keys)
    wan_unet: str = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    cfg: float = 3.5

    # AI Engine
    ai_engine_dev_mode: bool = False
    # Conditional VLM planner (slot planner.default_model). Set path to enable.
    ai_engine_vlm_enabled: bool = True
    ai_engine_vlm_model_path: Optional[str] = None

    # Auth — bcrypt hashes only, never plaintext PINs in git.
    # Format: $2a$12$…:ownerId,$2a$12$…:ownerId
    # Generate with: node frontend/scripts/hash_pin.js <pin>  (bcryptjs)
    wan_pins: str = ""
    wan_auth_secret: str = ""
    wan_admin_owner: str = "u9977"
    # Owner id only (never a PIN). Override with WAN_TESTER_OWNER.
    wan_tester_owner: str = "utester"

    # Ops UI (admin owner only) — restart Render / update tunnel / bounce GPU Comfy
    render_api_key: Optional[str] = None  # RENDER_API_KEY
    render_service_id: str = "srv-d9ot8spt0dsc73bqjv0g"
    gpu_agent_url: Optional[str] = None  # GPU_AGENT_URL
    gpu_agent_secret: Optional[str] = None  # GPU_AGENT_SECRET


@lru_cache
def get_settings() -> Settings:
    return Settings()
