"""Text conversation through NanoGPT (OpenAI-compatible), with image generation as a tool.

Without a slash command, every message goes to the LLM together with one tool,
`create_image`. The model answers in text, or calls the tool when the user wants a
picture drawn or changed; the bot then runs the normal Qwen-Image pipeline. This is
how ChatGPT decides between answering and drawing.

Chat turns are stored in the same conversation file as image turns (assistant
messages with meta.mode == "chat"), so the history the LLM sees includes both, and
the web UI shows them too.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
import uuid

import httpx
from PIL import Image

import config as C

log = logging.getLogger("chat")

SYSTEM = """あなたは Telegram の Bot の中で動くアシスタントです。普通の質問や雑談には、ユーザーと同じ言語で、自然に短めに答えます。
この Bot は、ローカルの画像生成モデル Qwen-Image-2.1 で画像を「新しく作る」ことと「編集する」ことができます。
ユーザーが画像を描いてほしい・作ってほしい・変えてほしいときは、文章で説明せずに create_image ツールを呼んでください。
- 直前に作った画像や添付された写真への変更依頼（「もっと明るく」「背景を夜に」「この猫に帽子を」など）は mode="edit"。
- それ以外の新しい絵は mode="generate"。
- instruction には、会話の流れを踏まえて、それだけで意味が通じる画像の指示を書きます。ユーザーの言語のまま書き、「」で囲まれた文字やロゴの文言は一字一句そのまま残します。
画像について話しているだけ（感想を聞く、写真の中身を説明してもらう、など）のときはツールを呼ばずに文章で答えます。
ツールを呼ぶときの返答は、空か一言だけにしてください。"""

# For "/chat": no tool is offered, and the prompt must not mention one either, or the
# model writes the call out as text ("create_image(...)", "<tool_call>...").
SYSTEM_TEXT_ONLY = """あなたは Telegram の Bot の中で動くアシスタントです。ユーザーと同じ言語で、自然に短めに答えます。
このやり取りでは文章だけで答えてください。絵を頼まれた場合は、どんな絵になるかを言葉で描写します。"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "create_image",
        "description": "Generate a new image, or edit the attached photo(s) / the last generated image, "
                       "with the local Qwen-Image-2.1 model.",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string",
                                "description": "Self-contained image instruction in the user's language."},
                "mode": {"type": "string", "enum": ["generate", "edit"],
                         "description": "edit = change the attached photo(s) or the last generated image"},
            },
            "required": ["instruction", "mode"],
        },
    },
}]


def enabled() -> bool:
    return bool(C.NANOGPT_API_KEY)


def model_label(model_id: str) -> str:
    for mid, label in C.NANOGPT_MODEL_CHOICES.values():
        if mid == model_id:
            return label
    return model_id


def _data_url(path) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1280, 1280))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def history_messages(conv: dict) -> list[dict]:
    """The conversation so far, as text, for the LLM (images described, not attached)."""
    out: list[dict] = []
    for m in conv["messages"][-C.CHAT_HISTORY_TURNS * 2:]:
        if m["role"] == "user":
            t = m.get("text") or ""
            if m.get("images"):
                t = f"[写真{len(m['images'])}枚を添付] " + t
            out.append({"role": "user", "content": t or "[写真]"})
        else:
            meta = m.get("meta") or {}
            if meta.get("mode") == "chat":
                t = m.get("text") or ""
            elif m.get("status") == "done" and m.get("images"):
                verb = "編集" if meta.get("mode") == "edit" else "生成"
                t = f"[画像を{verb}しました。指示: {meta.get('user_text') or ''}]"
            else:
                t = f"[画像の処理に失敗: {m.get('text') or ''}]"
            if t:
                out.append({"role": "assistant", "content": t})
    return out


async def ask(conv: dict, text: str, images: list, model: str, allow_tool: bool = True) -> dict:
    """Return {"text": str, "image": {"instruction", "mode"} | None}."""
    content: list | str = text or "（写真を送りました）"
    if images:
        content = [{"type": "text", "text": text or "（写真を送りました）"}] + \
                  [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in images[:4]]
    messages = [{"role": "system", "content": SYSTEM if allow_tool else SYSTEM_TEXT_ONLY}] + history_messages(conv) + \
               [{"role": "user", "content": content}]
    body = {"model": model, "messages": messages, "max_tokens": 2000, "temperature": 0.8}
    if allow_tool:
        body["tools"] = TOOLS
        body["tool_choice"] = "auto"
    t0 = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=15)) as c:
        r = await c.post(f"{C.NANOGPT_BASE_URL}/chat/completions", json=body,
                         headers={"Authorization": f"Bearer {C.NANOGPT_API_KEY}"})
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {}).get("message") or r.text
        except Exception:  # noqa: BLE001
            msg = r.text
        raise RuntimeError(f"NanoGPT {r.status_code}: {str(msg)[:300]}")
    msg = r.json()["choices"][0]["message"]
    log.info("chat %s in %.1fs", model, time.time() - t0)
    image = None
    for call in msg.get("tool_calls") or []:
        fn = call.get("function") or {}
        if fn.get("name") == "create_image":
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if args.get("instruction"):
                image = {"instruction": str(args["instruction"]),
                         "mode": "edit" if args.get("mode") == "edit" else "generate"}
                break
    return {"text": (msg.get("content") or "").strip(), "image": image}


def record_chat(S, cid: str, user_text: str, user_images: list[str], answer: str, model: str) -> None:
    """Store a text-only exchange in the conversation file."""
    conv = S.load_conv(cid)
    now = time.time()
    conv["messages"] += [
        {"id": uuid.uuid4().hex[:12], "role": "user", "text": user_text, "images": user_images, "time": now},
        {"id": uuid.uuid4().hex[:12], "role": "assistant", "text": answer, "images": [], "status": "done",
         "time": now, "meta": {"mode": "chat", "llm": model}},
    ]
    if not conv.get("title"):
        conv["title"] = (user_text.strip() or "会話")[:30]
    S.save_conv(conv)
