"""Paths and generation presets for the Qwen-Image-2.1 chat pipeline.

Machine-specific values live in `local.json` at the repository root (git-ignored).
See `local.example.json`. Every key is optional; defaults point inside the repo.
"""
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

_LOCAL_FILE = ROOT_DIR / "local.json"
LOCAL: dict = {}
if _LOCAL_FILE.exists():
    LOCAL = json.loads(_LOCAL_FILE.read_text(encoding="utf-8-sig"))


def _get(key: str, default):
    env = os.environ.get("QIC_" + key.upper())
    if env is not None:
        return env
    return LOCAL.get(key, default)


DATA_DIR = Path(_get("data_dir", ROOT_DIR / "data"))
CONV_DIR = DATA_DIR / "conversations"
IMG_DIR = DATA_DIR / "images"
for _d in (CONV_DIR, IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MODEL_ROOT = Path(_get("model_root", ROOT_DIR / "models" / "qwen-image-2.1"))
SDCPP_DIR = Path(_get("sdcpp_dir", ROOT_DIR / "bin" / "sdcpp"))

SD_HOST = _get("sd_host", "127.0.0.1")
SD_PORT = int(_get("sd_port", 1234))
SD_URL = f"http://{SD_HOST}:{SD_PORT}"
APP_PORT = int(_get("app_port", 7860))
# Largest output size for edits (reference-image jobs); see server.pick_size.
EDIT_MAX_PIXELS = int(_get("edit_max_pixels", 1440 * 1440))
# Load all model weights right after start-up with a tiny dummy job (see server.warm_up).
WARMUP = str(_get("warmup", True)).lower() not in ("0", "false", "no", "off")

OLLAMA_URL = _get("ollama_url", "http://127.0.0.1:11434")
# Official Qwen-Image-2.1 prompt enhancers imported into Ollama by setup.ps1.
PE_T2I_MODEL = _get("pe_t2i_model", "qwen-image-pe-t2i")
PE_I2I_MODEL = _get("pe_i2i_model", "qwen-image-pe-i2i")
PE_DIR = MODEL_ROOT / "prompt_enhancer"

TELEGRAM_TOKEN = str(_get("telegram_token", "") or "")
_allowed = _get("telegram_allowed_users", [])
if isinstance(_allowed, str):
    _allowed = [x for x in _allowed.replace(" ", "").split(",") if x]
TELEGRAM_ALLOWED = {int(x) for x in _allowed}
# Who gets a "the bot is up" message when the app starts (e.g. after the PC rebooted).
# Default: the first allowed user (the owner). [] turns the notice off.
_notify = _get("telegram_notify_users", [int(_allowed[0])] if _allowed else [])
if isinstance(_notify, str):
    _notify = [x for x in _notify.replace(" ", "").split(",") if x]
TELEGRAM_NOTIFY = [int(x) for x in _notify]

# Component files (relative to MODEL_ROOT). These IDs double as sd-server
# model-file IDs because MODEL_ROOT is the first --model-dir root.
MODELS = {
    "turbo": "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
    "base": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
}
RECIPES = {"turbo": "qwen-image-21-turbo", "base": "qwen-image-21"}
VAE = "vae/qwen_image_2.1_vae_bf16.safetensors"
LLM = "text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
LLM_VISION = "text_encoders/mmproj-Qwen3VL-8B-Instruct-F16.gguf"

# Turbo's distilled schedule (same as sd.cpp's `--sigmas qwen-image-2.1-turbo`).
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0]

SAMPLING = {
    "turbo": {"steps": 8, "cfg": 1.0, "sigmas": TURBO_SIGMAS},
    "base": {"steps": 30, "cfg": 4.0, "sigmas": []},
}

# Target pixel counts per quality preset (width*height).
QUALITY = {
    "fast": 1024 * 1024,
    "standard": 1440 * 1440,
    "high": 2048 * 2048,
}

ASPECTS = {
    "1:1": (1, 1), "4:3": (4, 3), "3:4": (3, 4), "3:2": (3, 2),
    "2:3": (2, 3), "16:9": (16, 9), "9:16": (9, 16),
}

SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_SETTINGS = {
    "model": "turbo",
    "quality": "fast",
    "use_llm": True,
    "preview": True,
}


def load_settings() -> dict:
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(s: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
