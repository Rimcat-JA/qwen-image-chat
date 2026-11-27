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
import re
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


def history_messages(conv: dict, with_tools: bool = True) -> list[dict]:
    """The conversation so far for the LLM (images are described, not attached).

    Past image turns are given as what they were: a create_image tool call plus its
    result. Writing them as assistant *text* ("[画像を編集しました。指示: …]") made
    models copy that text instead of calling the tool, so no image was made.
    """
    out: list[dict] = []
    for m in conv["messages"][-C.CHAT_HISTORY_TURNS * 2:]:
        if m["role"] == "user":
            t = m.get("text") or ""
            if m.get("images"):
                t = f"（写真{len(m['images'])}枚を添付）" + t
            out.append({"role": "user", "content": t or "（写真）"})
            continue
        meta = m.get("meta") or {}
        if meta.get("mode") == "chat":
            if m.get("text"):
                out.append({"role": "assistant", "content": m["text"]})
            continue
        ok = m.get("status") == "done" and m.get("images")
        result = ("画像をユーザーに送りました。" if ok else f"失敗しました: {m.get('text') or ''}")
        if with_tools:
            call_id = "call_" + m.get("id", uuid.uuid4().hex[:12])
            args = {"instruction": meta.get("user_text") or "", "mode": "edit" if meta.get("mode") == "edit" else "generate"}
            out.append({"role": "assistant", "content": "", "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "create_image", "arguments": json.dumps(args, ensure_ascii=False)}}]})
            out.append({"role": "tool", "tool_call_id": call_id, "content": result})
        else:  # no tools in this request: keep it short and unlike a reply to copy
            out.append({"role": "assistant", "content": "（画像を送りました）" if ok else "（画像の処理に失敗しました）"})
    return out


_TEXT_CALLS = [
    # "[画像を編集しました。指示: …]"  (imitating an older history format)
    re.compile(r"^\s*[\[［]\s*画像を(?P<verb>生成|編集)しました。?\s*指示[:：]\s*(?P<ins>.+?)\s*[\]］]?\s*$", re.S),
    # "create_image(mode="edit", instruction="…")"
    re.compile(r"create_image\s*\(.*?mode\s*=\s*[\"'](?P<mode>\w+)[\"'].*?instruction\s*=\s*[\"'](?P<ins>.+?)[\"']\s*\)", re.S),
    re.compile(r"create_image\s*\(.*?instruction\s*=\s*[\"'](?P<ins>.+?)[\"'].*?mode\s*=\s*[\"'](?P<mode>\w+)[\"']\s*\)", re.S),
]


def _text_tool_call(text: str) -> dict | None:
    """Some models write the tool call as text instead of calling it. Recover it."""
    m = re.search(r"<tool_call>\s*(?:create_image)?\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if m:
        try:
            a = json.loads(m.group(1))
            a = a.get("arguments", a) if isinstance(a, dict) else {}
            if isinstance(a, str):
                a = json.loads(a)
            if a.get("instruction"):
                return {"instruction": str(a["instruction"]), "mode": "edit" if a.get("mode") == "edit" else "generate"}
        except (json.JSONDecodeError, AttributeError):
            pass
    for pat in _TEXT_CALLS:
        m = pat.search(text)
        if m:
            g = m.groupdict()
            mode = g.get("mode") or ("edit" if g.get("verb") == "編集" else "generate")
            return {"instruction": g["ins"].strip(), "mode": "edit" if mode == "edit" else "generate"}
    return None


async def ask(conv: dict, text: str, images: list, model: str, allow_tool: bool = True) -> dict:
    """Return {"text": str, "image": {"instruction", "mode"} | None}."""
    content: list | str = text or "（写真を送りました）"
    if images:
        content = [{"type": "text", "text": text or "（写真を送りました）"}] + \
                  [{"type": "image_url", "image_url": {"url": _data_url(p)}} for p in images[:4]]
    messages = [{"role": "system", "content": SYSTEM if allow_tool else SYSTEM_TEXT_ONLY}] + \
               history_messages(conv, with_tools=allow_tool) + \
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
    text_out = (msg.get("content") or "").strip()
    if image is None and allow_tool:
        image = _text_tool_call(text_out)
        if image:
            log.warning("model %s wrote the tool call as text; recovered it", model)
            text_out = ""
    return {"text": text_out, "image": image}


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
