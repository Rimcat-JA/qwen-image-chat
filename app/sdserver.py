"""Start, watch and restart stable-diffusion.cpp's sd-server.

The app owns the image engine's lifecycle: it starts sd-server if nothing is
listening on SD_PORT, restarts it if it dies (e.g. a Vulkan out-of-memory abort),
and stops it on shutdown when it was the one that started it.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys
import time

import config as C

log = logging.getLogger("sdserver")
_proc: subprocess.Popen | None = None
current_model = "turbo"
_lock = asyncio.Lock()


def port_open() -> bool:
    try:
        with socket.create_connection((C.SD_HOST, C.SD_PORT), timeout=1):
            return True
    except OSError:
        return False


def owned() -> bool:
    return _proc is not None and _proc.poll() is None


def command(model: str = "turbo") -> list[str]:
    m = C.MODEL_ROOT
    return [
        str(C.SDCPP_DIR / ("sd-server.exe" if sys.platform == "win32" else "sd-server")),
        "--model-dir", str(m),
        "--diffusion-model", str(m / C.MODELS[model]),
        "--vae", str(m / C.VAE),
        "--llm", str(m / C.LLM),
        "--llm_vision", str(m / C.LLM_VISION),
        # Weights live in RAM and are copied to VRAM per job. A staging TTL of 0
        # hands the VRAM back right after each job, so the prompt enhancer (Ollama)
        # can use the GPU in between.
        "--offload-to-cpu", "--weight-staging-ttl", "0",
        # Text encoder (Qwen3-VL 8B) on the CPU: conditioning gets ~10 s slower, but the
        # freed VRAM lets the DiT keep its prefix KV cache, which makes edits ~25 s faster.
        "--backend", "te=cpu",
        "--vae-tiling",
        "--mmap",
        "--listen-ip", C.SD_HOST, "--listen-port", str(C.SD_PORT),
    ]


def prefetch(model: str) -> float:
    """Read the weight files once, sequentially, so they sit in the OS file cache.

    sd-server reads tensors in many small pieces. From a cold cache that ran at only
    20-70 MB/s even on an NVMe SSD (5 min to load ~14 GB), while one big sequential
    read does 2 GB/s. Reading the files first makes sd-server's own load hit the cache.
    """
    t = time.time()
    m = C.MODEL_ROOT
    for rel in (C.MODELS[model], C.LLM, C.LLM_VISION, C.VAE):
        try:
            with open(m / rel, "rb", buffering=0) as f:
                while f.read(64 * 1024 * 1024):
                    pass
        except OSError as e:
            log.warning("prefetch %s: %s", rel, e)
    return time.time() - t


async def ensure_running(model: str | None = None, timeout: float = 90) -> bool:
    """Start sd-server if it isn't reachable, or restart it with another diffusion model.

    Switching Turbo <-> Base inside a running sd-server keeps the old weights in VRAM
    and fails with out-of-memory on 16 GB cards, so a model switch is a clean restart
    (a few seconds; the weights are loaded lazily from the OS file cache).
    Returns True once it accepts connections.
    """
    global _proc, current_model
    async with _lock:
        want = model or current_model
        if port_open():
            if want == current_model or not owned():
                return True
            log.warning("switching sd-server model %s -> %s", current_model, want)
            _proc.terminate()
            try:
                _proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _proc.kill()
            for _ in range(20):
                if not port_open():
                    break
                await asyncio.sleep(0.25)
        current_model = want
        exe = C.SDCPP_DIR / ("sd-server.exe" if sys.platform == "win32" else "sd-server")
        if not exe.exists():
            log.error("sd-server not found at %s (run setup)", exe)
            return False
        logs = C.DATA_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = open(logs / "sd-server.log", "ab")
        out.write(f"\n===== start {time.ctime()} =====\n".encode())
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        log.warning("prefetched %s weights in %.0f s", want, await asyncio.to_thread(prefetch, want))
        log.warning("starting sd-server")
        _proc = subprocess.Popen(command(want), cwd=str(C.SDCPP_DIR), stdout=out, stderr=subprocess.STDOUT,
                                 creationflags=flags)
        t = time.time()
        while time.time() - t < timeout:
            if _proc.poll() is not None:
                log.error("sd-server exited with code %s", _proc.returncode)
                return False
            if port_open():
                return True
            await asyncio.sleep(0.5)
        return False


async def watchdog(is_busy) -> None:
    """Restart sd-server when it disappears (checked every 10 s while idle)."""
    while True:
        await asyncio.sleep(10)
        try:
            if not is_busy() and not port_open():
                log.warning("sd-server is down; restarting")
                await ensure_running()
        except Exception as e:  # noqa: BLE001
            log.warning("watchdog error: %s", e)


def stop_if_owned() -> None:
    if _proc and _proc.poll() is None:
        _proc.terminate()
