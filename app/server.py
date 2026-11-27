"""ChatGPT-style chat front-end for Qwen-Image-2.1 running on stable-diffusion.cpp's sd-server."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import random
import time
import sys
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

import config as C
import sdserver
from enhancer import enhance, ollama_available
from router import heuristic_plan

app = FastAPI(title="Qwen-Image Chat")

# In-memory live state of running jobs: message id -> dict
LIVE: dict[str, dict] = {}
_locks: dict[str, asyncio.Lock] = {}
# One GPU: the prompt enhancer (Ollama) and the image model must not share VRAM at
# the same time, so a whole job (enhance + generate) holds this lock.
GPU_LOCK = asyncio.Lock()


# ---------------------------------------------------------------- storage
def _conv_path(cid: str) -> Path:
    if not cid.replace("-", "").isalnum():
        raise HTTPException(400, "bad id")
    return C.CONV_DIR / f"{cid}.json"


def load_conv(cid: str) -> dict:
    p = _conv_path(cid)
    if not p.exists():
        raise HTTPException(404, "conversation not found")
    return json.loads(p.read_text(encoding="utf-8"))


def save_conv(conv: dict) -> None:
    conv["updated"] = time.time()
    tmp = _conv_path(conv["id"]).with_suffix(".tmp")
    tmp.write_text(json.dumps(conv, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(_conv_path(conv["id"]))


def lock_for(cid: str) -> asyncio.Lock:
    return _locks.setdefault(cid, asyncio.Lock())


def save_image_bytes(data: bytes, max_side: int = 2752) -> str:
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    name = f"{uuid.uuid4().hex}.png"
    im.save(C.IMG_DIR / name, "PNG")
    return name


def image_data_url(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((C.IMG_DIR / name).read_bytes()).decode()


# ---------------------------------------------------------------- sizing
def size_for(aspect: tuple[float, float], pixels: int) -> tuple[int, int]:
    a = aspect[0] / aspect[1]
    w = math.sqrt(pixels * a)
    h = w / a
    r = lambda v: max(256, int(round(v / 32)) * 32)
    return r(w), r(h)


def pick_size(plan: dict, ref_names: list[str], quality: str, aspect_override: str) -> tuple[int, int]:
    pixels = C.QUALITY.get(quality, C.QUALITY["fast"])
    if ref_names:
        # Edits attend over the output *and* the reference tokens. At 4 MP that needs a
        # single 3.2 GB GPU buffer, above what the Vulkan driver allows in one allocation
        # (16 GB RX 9070 XT), so a "high" edit always fails. Cap edits at ~2 MP.
        pixels = min(pixels, C.EDIT_MAX_PIXELS)
    ar = aspect_override if aspect_override in C.ASPECTS else plan.get("aspect")
    if ar in C.ASPECTS:
        return size_for(C.ASPECTS[ar], pixels)
    if ref_names:
        with Image.open(C.IMG_DIR / ref_names[0]) as im:
            return size_for(im.size, pixels)
    return size_for((1, 1), pixels)


# ---------------------------------------------------------------- sd-server
async def sd_status() -> dict:
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{C.SD_URL}/sdcpp/v1/models/status")
            return {"ok": r.status_code == 200, "detail": r.json()}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def model_selection(model: str) -> dict | None:
    """Per-request component selection so turbo/base can be swapped without restarting."""
    dm = C.MODELS.get(model)
    if not dm or not (C.MODEL_ROOT / dm).exists():
        return None
    return {
        "recipe": C.RECIPES[model],
        "models": {
            "diffusion_model": dm,
            "vae": C.VAE,
            "llm": C.LLM,
            "llm_vision": C.LLM_VISION,
        },
    }


def build_payload(prompt: str, refs: list[str], w: int, h: int, seed: int, model: str, preview: bool) -> dict:
    s = C.SAMPLING[model]
    payload = {
        "prompt": prompt,
        "negative_prompt": "",
        "width": w,
        "height": h,
        "seed": seed,
        "batch_count": 1,
        "ref_images": [image_data_url(n) for n in refs],
        "sample_params": {
            "sample_method": "euler",
            "sample_steps": s["steps"],
            "custom_sigmas": s["sigmas"],
            "guidance": {"txt_cfg": s["cfg"]},
        },
        # Always tile: after sampling the DiT still occupies VRAM, and an untiled
        # decode spills to host memory (35s instead of ~5s at 1024²).
        "vae_tiling_params": {"enabled": True},
        "preview": "proj" if preview else "none",
        "preview_interval": 1,
        "output_format": "png",
        "embed_image_metadata": True,
    }
    # When the app supervises sd-server, the model is chosen by restarting it
    # (see sdserver.ensure_running); per-request selection is only a fallback for an
    # externally started sd-server.
    sel = model_selection(model)
    if sel and not sdserver.owned():
        payload["model"] = sel
    return payload


async def run_job(cid: str, mid: str, plan: dict, refs: list[str], opts: dict,
                  history: list[str] | None = None) -> None:
    LIVE[mid].update(status="queued", stage="queued")
    async with GPU_LOCK:
        await _run_job(cid, mid, plan, refs, opts, history)


async def _run_job(cid: str, mid: str, plan: dict, refs: list[str], opts: dict,
                   history: list[str] | None = None) -> None:
    live = LIVE[mid]
    settings = C.load_settings()
    use_llm = bool(opts.get("use_llm", settings.get("use_llm")))
    if use_llm and not plan.get("enhanced"):
        live.update(status="enhancing", stage="enhancing")
        try:
            enh = await enhance(plan["mode"], plan["user_text"], history or [], refs)
        except Exception:
            enh = None
        if enh:
            plan["prompt"] = enh["prompt"]
            plan["enhancer"] = enh["enhancer"]
            # An explicit shape word from the user beats the LLM's suggestion.
            if not plan.get("aspect") and enh.get("aspect"):
                plan["aspect"] = enh["aspect"]
        plan["enhanced"] = True
        await persist_plan(cid, mid, plan)
    model = opts.get("model") or settings["model"]
    if model not in C.SAMPLING or not (C.MODEL_ROOT / C.MODELS[model]).exists():
        model = "turbo"
    quality = opts.get("quality") or settings["quality"]
    w, h = pick_size(plan, refs, quality, opts.get("aspect", "auto"))
    seed = int(opts.get("seed") if opts.get("seed") not in (None, "", -1, "-1") else random.randint(0, 2**31 - 1))
    payload = build_payload(plan["prompt"], refs, w, h, seed, model, bool(settings.get("preview", True)))
    meta = {"mode": plan["mode"], "prompt": plan["prompt"], "user_text": plan.get("user_text", ""),
            "enhancer": plan.get("enhancer"), "aspect": plan.get("aspect"),
            "width": w, "height": h, "seed": seed, "model": model, "quality": quality,
            "refs": refs, "steps": C.SAMPLING[model]["steps"]}
    live.update(status="queued", meta=meta, stage="queued")
    t0 = time.time()
    try:
        if sdserver.owned() and sdserver.current_model != model:
            live.update(status="loading", stage="loading")
        await sdserver.ensure_running(model)
        try:
            j = await sd_generate(payload, live, t0)
        except SDServerLost as e:
            # sd-server crashed (e.g. Vulkan OOM). Restart it and retry the job once.
            live.update(status="queued", stage="loading", preview=None)
            if not await sdserver.ensure_running(model):
                raise RuntimeError("画像エンジンを再起動できませんでした") from e
            j = await sd_generate(payload, live, t0)
        if j["status"] == "failed" and sdserver.owned():
            # A failed job can leave sd-server unusable: e.g. after a Vulkan "device lost"
            # (GPU reset) every later job fails with "VAE encode failed". Start a fresh
            # sd-server (new GPU context) and retry once, whatever the message was.
            err = str((j.get("error") or {}).get("message", ""))
            log.warning("job failed (%s); restarting sd-server and retrying once", err)
            live.update(status="queued", stage="loading", preview=None)
            if await sdserver.ensure_running(model, force=True):
                j = await sd_generate(payload, live, t0)
        if j["status"] == "completed":
            imgs = [save_image_bytes(base64.b64decode(im["b64_json"])) for im in j["result"]["images"]]
            started, done = j.get("started") or 0, j.get("completed") or 0
            meta["gen_seconds"] = round((done - started) / 1000, 1) if started and done else None
            meta["total_seconds"] = round(time.time() - t0, 1)
            await finish(cid, mid, status="done", images=imgs, meta=meta)
        elif j["status"] == "cancelled":
            await finish(cid, mid, status="cancelled", meta=meta, text="キャンセルしました。")
        else:
            err = (j.get("error") or {}).get("message", "unknown error")
            await finish(cid, mid, status="error", meta=meta, text=f"生成に失敗しました: {err}")
    except Exception as e:  # noqa: BLE001
        await finish(cid, mid, status="error", meta=meta, text=f"エラー: {e or type(e).__name__}")


class SDServerLost(Exception):
    pass


async def sd_generate(payload: dict, live: dict, t0: float) -> dict:
    """Submit one job to sd-server and poll it to a terminal state."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=600)) as c:
            r = await c.post(f"{C.SD_URL}/sdcpp/v1/img_gen", json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"sd-server {r.status_code}: {r.text[:500]}")
            job = r.json()
            live["job_id"] = job["id"]
            while True:
                await asyncio.sleep(0.6)
                jr = await c.get(f"{C.SD_URL}/sdcpp/v1/jobs/{job['id']}")
                if jr.status_code >= 400:
                    raise RuntimeError(f"job poll {jr.status_code}: {jr.text[:300]}")
                j = jr.json()
                prog = j.get("progress") or {}
                live.update(status=j["status"], stage=prog.get("stage"), detail=prog.get("detail"),
                            completed=prog.get("completed"), total=prog.get("total"),
                            unit=prog.get("unit"), queue_position=j.get("queue_position"),
                            elapsed=round(time.time() - t0, 1))
                pv = j.get("preview")
                if pv and pv.get("b64_json"):
                    live["preview"] = "data:image/png;base64," + pv["b64_json"]
                if j["status"] in ("completed", "failed", "cancelled"):
                    return j
    except httpx.TransportError as e:
        raise SDServerLost(str(e) or type(e).__name__) from e


async def persist_plan(cid: str, mid: str, plan: dict) -> None:
    async with lock_for(cid):
        conv = load_conv(cid)
        for m in conv["messages"]:
            if m["id"] == mid and "request" in m:
                m["request"]["plan"] = plan
        save_conv(conv)


async def finish(cid: str, mid: str, status: str, meta: dict, images: list[str] | None = None,
                 text: str | None = None) -> None:
    async with lock_for(cid):
        conv = load_conv(cid)
        for m in conv["messages"]:
            if m["id"] == mid:
                m["status"] = status
                m["meta"] = meta
                if images is not None:
                    m["images"] = images
                if text is not None:
                    m["text"] = text
                elif status == "done":
                    m["text"] = describe(meta)
        save_conv(conv)
    LIVE.pop(mid, None)


def describe(meta: dict) -> str:
    kind = "画像を編集しました" if meta["mode"] == "edit" else "画像を生成しました"
    return kind


# ---------------------------------------------------------------- planning
def previous_image(conv: dict) -> str | None:
    for m in reversed(conv["messages"]):
        if m["role"] == "assistant" and m.get("images"):
            return m["images"][-1]
    return None


def make_plan(conv: dict, text: str, attachments: list[str], forced: str) -> dict:
    prev = previous_image(conv)
    plan = heuristic_plan(text, len(attachments), prev is not None, forced)
    plan["user_text"] = text.strip()
    refs: list[str] = []
    if plan["mode"] == "edit":
        if plan.get("use_previous") and prev:
            refs.append(prev)
        if plan.get("use_attachments"):
            refs.extend(attachments)
        refs = refs[:10]
        if not refs:
            plan["mode"] = "generate"
    if not plan["prompt"]:
        plan["prompt"] = "Improve this image while keeping its content." if refs else "A beautiful image."
        plan["enhanced"] = True
    return {"plan": plan, "refs": refs}


def user_history(conv: dict) -> list[str]:
    return [m["text"] for m in conv["messages"] if m["role"] == "user" and m.get("text")]


# ---------------------------------------------------------------- API
log = logging.getLogger("app")
WARM: dict = {"state": "idle"}


async def warm_up() -> None:
    """Make the first real job after a (re)start as fast as the others.

    sd-server reads its ~13 GB of weights lazily, on the first job, which made that
    job take ~2.5 min instead of ~30 s. A tiny 256x256, 1-step edit with a gray
    reference image touches every part (text encoder + vision tower, DiT, VAE
    encode/decode), so all weights are loaded before anyone is waiting for them.
    Real jobs queue behind it on GPU_LOCK.
    """
    if not C.WARMUP or not await sdserver.ensure_running():
        return
    WARM["state"] = "running"
    t0 = time.time()
    try:
        async with GPU_LOCK:
            from PIL import Image
            buf = io.BytesIO()
            Image.new("RGB", (256, 256), (128, 128, 128)).save(buf, "PNG")
            payload = build_payload("a plain gray square", [], 256, 256, 1, sdserver.current_model, False)
            payload["ref_images"] = ["data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()]
            payload["sample_params"].update(sample_steps=1, custom_sigmas=[1.0, 0.0])
            payload["embed_image_metadata"] = False
            j = await sd_generate(payload, {}, t0)
            log.warning("warm-up %s in %.0f s", j.get("status"), time.time() - t0)
    except Exception as e:  # noqa: BLE001
        log.warning("warm-up failed: %s", e)
    finally:
        WARM.update(state="done", seconds=round(time.time() - t0, 1))


@app.on_event("startup")
async def _start_engine() -> None:
    asyncio.create_task(warm_up() if C.WARMUP else sdserver.ensure_running())
    asyncio.create_task(sdserver.watchdog(lambda: GPU_LOCK.locked()))


@app.on_event("shutdown")
async def _stop_engine() -> None:
    sdserver.stop_if_owned()


@app.on_event("startup")
async def _mark_interrupted() -> None:
    """Jobs don't survive an app restart; don't leave spinners forever."""
    for p in C.CONV_DIR.glob("*.json"):
        try:
            conv = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        changed = False
        for m in conv["messages"]:
            if m.get("status") == "pending":
                m["status"] = "error"
                m["text"] = "アプリの再起動で中断されました。「再生成」でやり直せます。"
                changed = True
        if changed:
            save_conv(conv)


@app.get("/api/status")
async def api_status():
    sd, ol = await asyncio.gather(sd_status(), ollama_available())
    return {"sd": sd, "ollama": ol, "settings": C.load_settings(), "warm": WARM.get("state"),
            "models": {k: (C.MODEL_ROOT / v).exists() for k, v in C.MODELS.items()}}


@app.post("/api/settings")
async def api_settings(body: dict):
    s = C.load_settings()
    for k in C.DEFAULT_SETTINGS:
        if k in body:
            s[k] = body[k]
    C.save_settings(s)
    return s


@app.get("/api/conversations")
async def list_convs():
    out = []
    for p in C.CONV_DIR.glob("*.json"):
        try:
            c = json.loads(p.read_text(encoding="utf-8"))
            out.append({"id": c["id"], "title": c.get("title") or "新しいチャット", "updated": c.get("updated", 0), "source": c.get("source", "web")})
        except Exception:
            continue
    out.sort(key=lambda x: x["updated"], reverse=True)
    return out


@app.post("/api/conversations")
async def create_conv():
    return new_conversation()


@app.get("/api/conversations/{cid}")
async def get_conv(cid: str):
    conv = load_conv(cid)
    for m in conv["messages"]:
        if m["id"] in LIVE:
            m["live"] = LIVE[m["id"]]
    return conv


@app.patch("/api/conversations/{cid}")
async def rename_conv(cid: str, body: dict):
    async with lock_for(cid):
        conv = load_conv(cid)
        conv["title"] = str(body.get("title", ""))[:80]
        save_conv(conv)
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
async def delete_conv(cid: str):
    p = _conv_path(cid)
    if p.exists():
        p.unlink()
    return {"ok": True}


@app.post("/api/conversations/{cid}/messages")
async def post_message(
    cid: str,
    text: str = Form(""),
    options: str = Form("{}"),
    files: list[UploadFile] = File(default=[]),
):
    opts = json.loads(options or "{}")
    attachments = []
    for f in files:
        data = await f.read()
        if data:
            try:
                attachments.append(save_image_bytes(data))
            except Exception:
                raise HTTPException(400, f"画像を読み込めませんでした: {f.filename}")
    # Images re-attached from earlier messages (e.g. "この画像を編集" button).
    for name in opts.get("reuse_images", []):
        if (C.IMG_DIR / Path(name).name).exists():
            attachments.append(Path(name).name)
    if not text.strip() and not attachments:
        raise HTTPException(400, "メッセージが空です")
    return await submit_turn(cid, text, attachments, opts)


def new_conversation(cid: str | None = None, source: str = "web") -> dict:
    conv = {"id": cid or uuid.uuid4().hex[:16], "title": "", "created": time.time(),
            "source": source, "messages": []}
    save_conv(conv)
    return conv


async def submit_turn(cid: str, text: str, attachments: list[str], opts: dict) -> dict:
    """Record a user turn, plan it and start the job. Shared by the web UI and the Telegram bot."""
    async with lock_for(cid):
        conv = load_conv(cid)
        history = user_history(conv)
        planned = make_plan(conv, text, attachments, opts.get("mode", "auto"))
        user_msg = {"id": uuid.uuid4().hex[:12], "role": "user", "text": text, "images": attachments,
                    "time": time.time()}
        asst = {"id": uuid.uuid4().hex[:12], "role": "assistant", "text": "", "images": [],
                "status": "pending", "time": time.time(), "meta": {"mode": planned["plan"]["mode"]},
                "request": {"plan": planned["plan"], "refs": planned["refs"], "opts": opts}}
        conv["messages"] += [user_msg, asst]
        if not conv.get("title"):
            conv["title"] = (text.strip() or "画像の編集")[:30]
        save_conv(conv)
    LIVE[asst["id"]] = {"status": "planning", "stage": "queued"}
    asyncio.create_task(run_job(cid, asst["id"], planned["plan"], planned["refs"], opts, history))
    return {"user": user_msg, "assistant": asst}


@app.post("/api/conversations/{cid}/messages/{mid}/regenerate")
async def regenerate(cid: str, mid: str, body: dict | None = None):
    async with lock_for(cid):
        conv = load_conv(cid)
        src = next((m for m in conv["messages"] if m["id"] == mid), None)
        if not src or "request" not in src:
            raise HTTPException(404, "message not found")
        req = json.loads(json.dumps(src["request"]))
        req["opts"] = {**req.get("opts", {}), **(body or {}), "seed": -1}
        asst = {"id": uuid.uuid4().hex[:12], "role": "assistant", "text": "", "images": [],
                "status": "pending", "time": time.time(), "meta": {"mode": req["plan"]["mode"]},
                "request": req}
        conv["messages"].append(asst)
        save_conv(conv)
    LIVE[asst["id"]] = {"status": "planning", "stage": "queued"}
    asyncio.create_task(run_job(cid, asst["id"], req["plan"], req["refs"], req["opts"]))
    return {"assistant": asst}


@app.get("/api/live/{mid}")
async def live(mid: str):
    return LIVE.get(mid) or {"status": "gone"}


@app.post("/api/live/{mid}/cancel")
async def cancel(mid: str):
    st = LIVE.get(mid)
    if not st or not st.get("job_id"):
        return {"ok": False}
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"{C.SD_URL}/sdcpp/v1/jobs/{st['job_id']}/cancel")
    return {"ok": True}


@app.get("/images/{name}")
async def image(name: str):
    p = C.IMG_DIR / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "max-age=31536000, immutable"})


@app.on_event("startup")
async def _start_telegram() -> None:
    if C.TELEGRAM_TOKEN:
        import telegram_bot
        asyncio.create_task(telegram_bot.run(sys.modules[__name__]))


app.mount("/", StaticFiles(directory=C.APP_DIR / "static", html=True), name="static")
