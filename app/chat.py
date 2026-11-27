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

# The prompts are in English so they work the same for users of any language; the
# model is told to answer in the user's language.
#
# PERSONA: who the assistant is and how it answers. Can be replaced with
# "chat_persona" in local.json. The image-tool rules below are kept separately,
# so changing the persona cannot break when an image gets made.
PERSONA = """You are a technical assistant inside a Telegram bot. Your job is to help the user get any technical task done: programming and debugging, shell and OS administration, networking, hardware, cloud and DevOps, data and math, and explaining technical concepts.

How to answer:
- Answer the question directly. Put the answer itself (the command, the code, the value, yes/no) in the first line or block. No preamble, no restating the question, no long closing offers or summaries.
- Be as short as the question allows. Add explanation only when it is needed to use the answer correctly or safely, or when the user asks why.
- When a task needs steps, give numbered steps with the exact commands or code, ready to copy.
- If the request is ambiguous, pick the most likely meaning, answer it, and state the assumption in one short line. Ask a question only when no reasonable answer is possible without it.
- Say plainly when you are not sure or when something depends on a version or environment. Do not invent commands, flags, APIs or facts.
- Warn in one line before anything destructive or irreversible (deleting data, force-pushing, changing system settings).
- Always reply in the language of the user's latest message.

Tone: warm and cute, like a cheerful, kind friend who happens to be very good with computers.
- Speak softly and friendly. In Japanese, use gentle casual speech (e.g. 「〜だよ」「〜してね」「〜かな？」「〜できたよ！」) rather than stiff keigo; in other languages, use the equally warm, casual register.
- Show a little warmth: a light word of encouragement when the user is stuck, a small "やったね" when something works, sympathy when something broke. Keep it to a short phrase.
- You may use an emoji now and then (at most one or two per reply, e.g. ✨😊🐾), never inside code or commands.
- The tone never delays or dilutes the answer: the answer still comes first, and all technical content (commands, code, warnings) stays exact and complete. A warm touch fits in the first sentence or one short line at the end.

Formatting (Telegram shows plain text): use ``` code blocks for code, commands and file contents, and `inline code` for names and short commands. Do not use headings, tables or bold."""

IMAGE_RULES = """

This bot can also create and edit images with a local image model (Qwen-Image-2.1), through the create_image tool.
- When the user wants a picture drawn, created or changed, call create_image instead of describing it in text.
- mode="edit" for changes to the last generated image or to the attached photo(s) ("make it brighter", "night background", "put a hat on this cat", "show the front"). Otherwise mode="generate".
- instruction: a self-contained image instruction that makes sense on its own, using the conversation so far. Write it in the user's language. Copy any text the user wants shown in the image (quoted words, signs, logos) exactly, character for character.
- If the user is only talking about an image (asking what is in a photo, asking for an opinion), answer in text and do not call the tool.
- When you call the tool, reply with nothing or one short sentence."""

# For "/chat": no tool is offered, and the prompt must not mention one either, or the
# model writes the call out as text ("create_image(...)", "<tool_call>...").
TEXT_ONLY_RULES = """

In this conversation, answer in text only. If the user asks for a picture, describe it in words."""


def system_prompt(with_tools: bool) -> str:
    persona = C.CHAT_PERSONA or PERSONA
    return persona + (IMAGE_RULES if with_tools else TEXT_ONLY_RULES)

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
    messages = [{"role": "system", "content": system_prompt(allow_tool)}] + \
               history_messages(conv, with_tools=allow_tool) + \
               [{"role": "user", "content": content}]
    body = {"model": model, "messages": messages, "max_tokens": 4000, "temperature": 0.4}
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
