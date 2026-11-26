"""Optional prompt rewriting ("prompt enhancement") with Qwen's official enhancers.

Qwen ships two fine-tuned Qwen3.5-VL 9B models for Qwen-Image-2.1:
  * PE-T2I: short request (any language) -> long English description + aspect ratio
  * PE-I2I: edit instruction + input image(s) -> precise edit instruction
They run in Ollama (imported from GGUF by setup.ps1). We call them with thinking
off (about 10 s on the GPU instead of 25+ s) and keep_alive=0, so the
enhancer leaves VRAM as soon as it has answered and the image model gets the
whole GPU back.

The mode (generate/edit) was already decided by router.py; the enhancer only
rewrites the prompt.
"""
from __future__ import annotations

import base64
import json
import re
from functools import lru_cache

import httpx

import config as C


@lru_cache(maxsize=2)
def _system_prompt(kind: str) -> str:
    p = C.PE_DIR / f"sys_{kind}.txt"
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


async def available_models() -> set[str]:
    try:
        async with httpx.AsyncClient(timeout=2) as c:
            r = await c.get(f"{C.OLLAMA_URL}/api/tags")
            return {m.get("name", "").removesuffix(":latest") for m in r.json().get("models", [])}
    except Exception:
        return set()


async def ollama_available() -> bool:
    models = await available_models()
    return C.PE_T2I_MODEL in models


def _parse_json(text: str) -> dict | None:
    text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


async def _chat(model: str, system: str, user: str, images: list[str] | None = None) -> dict | None:
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = images
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {"temperature": 0.7, "top_p": 0.95, "top_k": 20, "num_ctx": 12288},
        "messages": [{"role": "system", "content": system}, msg],
    }
    async with httpx.AsyncClient(timeout=240) as c:
        try:
            r = await c.post(f"{C.OLLAMA_URL}/api/chat", json=body)
            r.raise_for_status()
            return _parse_json(r.json()["message"]["content"])
        finally:
            await _wait_unloaded(c, model)


async def _wait_unloaded(c: httpx.AsyncClient, model: str, timeout: float = 30) -> None:
    """keep_alive=0 unloads *after* the reply is sent. Wait until the VRAM is really
    free, otherwise the image model's text encoder can hit out-of-memory."""
    import asyncio
    import time
    t = time.time()
    while time.time() - t < timeout:
        try:
            r = await c.get(f"{C.OLLAMA_URL}/api/ps")
            names = {m.get("name", "").removesuffix(":latest") for m in r.json().get("models", [])}
            if model not in names:
                await asyncio.sleep(0.5)  # let the driver actually release the memory
                return
        except Exception:
            return
        await asyncio.sleep(0.4)


def _small_b64(path) -> str:
    """Downscale reference images for the enhancer: it only needs to see what is in them."""
    from PIL import Image
    import io
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


QUOTED = re.compile(r"[「『\"“]([^」』\"”]{1,80})[」』\"”]")


def missing_literals(user_text: str, prompt: str) -> list[str]:
    """Text the user quoted (「ハッピーハロウィン」) that the rewrite dropped or translated."""
    return [q for q in QUOTED.findall(user_text) if q not in prompt]


async def enhance(mode: str, text: str, history: list[str], refs: list[str]) -> dict | None:
    """Return {"prompt", "aspect", "enhancer"} or None (caller keeps the raw text)."""
    res = await _enhance(mode, text, history, refs)
    if not res:
        return None
    missing = missing_literals(text, res["prompt"])
    if missing:
        if mode == "edit":
            # A rewrite that says "Happy Halloween" when the user asked for 「ハッピーハロウィン」
            # is wrong; the raw Japanese instruction works well for edits, so keep that.
            return None
        res["prompt"] += " " + " ".join(
            f'The text must read exactly "{q}" (keep this exact wording and script).' for q in missing)
    return res


async def _enhance(mode: str, text: str, history: list[str], refs: list[str]) -> dict | None:
    if not text.strip():
        return None
    models = await available_models()
    try:
        if mode == "generate" and C.PE_T2I_MODEL in models:
            user = text
            if history:
                # Give the enhancer the earlier requests so "same style, but a dog" resolves.
                user = "Earlier requests in this chat (context only):\n- " + "\n- ".join(history[-3:]) + \
                       "\n\nCurrent request:\n" + text
            data = await _chat(C.PE_T2I_MODEL, _system_prompt("T2I"), user)
            if data and len(str(data.get("rewritten_prompt", ""))) > 20:
                ar = str(data.get("wh_ratio") or "")
                return {"prompt": data["rewritten_prompt"].strip(),
                        "aspect": ar if ar in C.ASPECTS else None, "enhancer": "Qwen-Image-2.1-PE-T2I"}
        if mode == "edit" and C.PE_I2I_MODEL in models and refs:
            imgs = [_small_b64(C.IMG_DIR / n) for n in refs[:4]]
            instr = text
            if re.search(r"[぀-ヿ]", text):
                # PE-I2I writes Chinese prose for Chinese input and English otherwise, but
                # kanji-heavy Japanese gets mistaken for Chinese. Say which language it is.
                instr += ("\n\n(The instruction above is written in Japanese, not Chinese. Any text it asks to "
                          "show in the image must stay exactly as written, in Japanese.)")
            data = await _chat(C.PE_I2I_MODEL, _system_prompt("I2I"), instr, imgs)
            if data and len(str(data.get("rewritten_prompt", ""))) > 5:
                ar = str(data.get("wh_ratio") or "")
                return {"prompt": data["rewritten_prompt"].strip(),
                        "aspect": ar if ar in C.ASPECTS else None, "enhancer": "Qwen-Image-2.1-PE-I2I"}
    except Exception:
        return None
    return None
