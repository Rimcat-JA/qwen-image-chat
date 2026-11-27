"""Paths and generation presets for the Qwen-Image-2.1 chat pipeline.

Machine-specific values live in `local.json` at the repository root (git-ignored).
See `local.example.json`. Every key is optional; defaults point inside the repo.
"""
import json
import os
import sys
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


def detect_vram_gb() -> float | None:
    """Dedicated memory of the largest GPU, in GiB (None if unknown).

    Windows: the display-adapter class key in the registry (works for AMD, NVIDIA and
    Intel; WMI's AdapterRAM is capped at 4 GB). Linux: nvidia-smi, or amdgpu's sysfs.
    """
    sizes: list[int] = []
    if sys.platform == "win32":
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
                for i in range(64):
                    try:
                        sub = winreg.EnumKey(k, i)
                    except OSError:
                        break
                    try:
                        with winreg.OpenKey(k, sub) as s:
                            v, _ = winreg.QueryValueEx(s, "HardwareInformation.qwMemorySize")
                            sizes.append(int.from_bytes(v, "little") if isinstance(v, bytes) else int(v))
                    except OSError:
                        pass
        except OSError:
            pass
    else:
        import glob
        import subprocess
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            sizes += [int(float(x)) * 1024 ** 2 for x in out.split() if x.strip()]
        except Exception:  # noqa: BLE001
            pass
        for p in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
            try:
                sizes.append(int(Path(p).read_text().strip()))
            except Exception:  # noqa: BLE001
                pass
    return round(max(sizes) / 1024 ** 3, 1) if sizes else None


# Output-size limits by VRAM. Edits attend over the output *and* the reference
# tokens, so they need much more memory than text-to-image at the same size: on a
# 16 GB RX 9070 XT a 4 MP edit needs a single 3.2 GB buffer, above the driver's
# per-allocation limit, while a 4 MP text-to-image and a ~2 MP edit work.
# Only the 16 GB tier is measured; the others are conservative estimates.
# Override with "vram_gb", "max_pixels" or "edit_max_pixels" in local.json.
VRAM_TIERS = [  # (min GiB, text-to-image max pixels, edit max pixels)
    (22, 2048 * 2048, 1440 * 1440),
    (15, 2048 * 2048, 1440 * 1440),   # measured: RX 9070 XT 16 GB
    (11, 1440 * 1440, 1024 * 1024),
    (0, 1024 * 1024, 1024 * 1024),
]
_vram_cfg = _get("vram_gb", None)
VRAM_GB = float(_vram_cfg) if _vram_cfg not in (None, "", 0) else detect_vram_gb()
_tier = next(t for t in VRAM_TIERS if (VRAM_GB if VRAM_GB is not None else 16) >= t[0])
GEN_MAX_PIXELS = int(_get("max_pixels", _tier[1]))
EDIT_MAX_PIXELS = int(_get("edit_max_pixels", _tier[2]))
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

# Conversation (text chat) through NanoGPT's OpenAI-compatible API. Without a key the
# bot stays image-only. /api/v1 bills subscription-included models to the
# subscription and anything else to the balance (unless the key is "Subscription only").
NANOGPT_API_KEY = str(_get("nanogpt_api_key", "") or "")
NANOGPT_BASE_URL = str(_get("nanogpt_base_url", "https://api.nano-gpt.com/api/v1")).rstrip("/")
NANOGPT_MODEL = str(_get("nanogpt_model", "z-ai/glm-5.3-flash-uncensored"))
# Shown by /llm in Telegram: short name -> (model id, label). All have vision and tool
# calling. Ones not included in the NanoGPT subscription are billed to the balance.
NANOGPT_MODEL_CHOICES = {
    "glm": ("z-ai/glm-5.3-flash-uncensored", "GLM 5.3 Flash Uncensored（サブスク）"),
    "heretic": ("pqhaz/DeepSeek-V4.1-Flash-1017-heretic", "DeepSeek V4.1 Flash Heretic（サブスク）"),
    "deepseek": ("deepseek/deepseek-v4.1-flash-1017-uncensored", "DeepSeek V4.1 Flash 1017 Uncensored（従量課金）"),
}
_extra = _get("nanogpt_model_choices", {})  # {"name": "model/id"} in local.json adds more
if isinstance(_extra, dict):
    NANOGPT_MODEL_CHOICES.update({k: (v, v) for k, v in _extra.items()})
CHAT_HISTORY_TURNS = int(_get("chat_history_turns", 20))
# Replaces the assistant's persona (who it is, how it answers); see chat.PERSONA.
# The rules for when to make an image are always added after it.
CHAT_PERSONA = str(_get("chat_persona", "") or "")

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
