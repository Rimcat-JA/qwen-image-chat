"""Telegram front-end: chat with the same pipeline from your phone, anywhere.

Long polling (getUpdates) means the PC only makes outbound HTTPS requests, so no
port forwarding or public URL is needed. Each Telegram chat maps to one
conversation in data/conversations, so it also shows up in the web UI.

Only user IDs listed in `telegram_allowed_users` (local.json) may use the bot.
Anyone else gets their numeric ID back, which makes the first-time setup easy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import httpx

import chat
import config as C

log = logging.getLogger("telegram")
API = f"https://api.telegram.org/bot{C.TELEGRAM_TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{C.TELEGRAM_TOKEN}"
STATE_FILE = C.DATA_DIR / "telegram.json"

HELP_IMAGE = """🎨 Qwen-Image-2.1 画像生成ボット

• 文章を送る → 画像を生成
• 写真を送る（キャプションに指示）→ その写真を編集
• 生成後に「もっと明るく」「背景を夜にして」など → 直前の画像を編集
• 複数の写真をまとめて送る → 参照画像として合成・編集（最大10枚）
• 「横長」「縦長」「16:9」などで比率を指定

コマンド
/new 新しい会話を始める"""

HELP_CHAT = """💬 会話もできる画像生成ボット

• 普通に話しかける → AI が答えます。絵を頼むと、AI が判断して画像を作ります
• 写真を送る → 写真について話すことも、キャプションの指示で編集することもできます
• 生成後に「もっと明るく」など → 直前の画像を編集
• 「横長」「縦長」「16:9」などで比率を指定

どちらかに決めたいときは先頭にコマンド
/image 〜 必ず画像を作る（写真付きなら編集）
/chat 〜 必ず文章で答える（画像は作らない）

コマンド
/new 新しい会話を始める
/llm 会話に使う AI モデルを選ぶ"""

HELP_COMMON = """
/quality fast|standard|high 画質（約1MP/2MP/4MP）
/model turbo|base 高速/高品質モデル
/enhance on|off プロンプト強化
/settings 現在の設定"""


def split_force(text: str) -> tuple[str | None, str]:
    """'/image 猫' -> ('image', '猫'); '/chat@bot 質問' -> ('chat', '質問'); else (None, text)."""
    head, _, rest = text.partition(" ")
    if "\n" in head:
        head, _, more = head.partition("\n")
        rest = more + (" " + rest if rest else "")
    cmd = head.split("@")[0].lower()
    if cmd in ("/image", "/img"):
        return "image", rest.strip()
    if cmd == "/chat":
        return "chat", rest.strip()
    return None, text


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_to_html(text: str) -> str:
    """The little Markdown an LLM uses -> Telegram HTML: ```lang blocks, `code`, **bold**."""
    out = []
    for i, part in enumerate(re.split(r"```", text)):
        if i % 2:  # inside a fence: first line may name the language
            lang, nl, body = part.partition("\n")
            if not nl or not re.fullmatch(r"[\w+#.-]{0,20}", lang.strip()):
                lang, body = "", part
            body = body.strip("\n")
            cls = f' class="language-{_esc(lang.strip())}"' if lang.strip() else ""
            out.append(f"<pre><code{cls}>{_esc(body)}</code></pre>")
        else:
            s = _esc(part)
            s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)
            s = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", s)
            out.append(s)
    return "".join(out)


def split_markdown(text: str, limit: int = 3500) -> list[str]:
    """Split into Telegram-sized messages at line breaks, closing and reopening a ```
    block that spans two messages."""
    chunks, cur, fence = [], "", None
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            chunks.append(cur + ("\n```" if fence is not None else ""))
            cur = ("```" + fence + "\n") if fence is not None else ""
        while len(line) > limit:  # a single huge line
            chunks.append(cur + line[:limit])
            cur, line = "", line[limit:]
        cur += line + "\n"
        if line.strip().startswith("```"):
            fence = None if fence is not None else line.strip()[3:]
    if cur.strip():
        chunks.append(cur.rstrip("\n"))
    return chunks or [text]


def help_text() -> str:
    return (HELP_CHAT if chat.enabled() else HELP_IMAGE) + HELP_COMMON


COMMANDS_CHAT = [
    {"command": "image", "description": "必ず画像を作る: /image 指示"},
    {"command": "chat", "description": "必ず文章で答える: /chat 質問"},
    {"command": "llm", "description": "会話に使う AI モデルを選ぶ"},
]
COMMANDS = [
    {"command": "new", "description": "新しい会話を始める"},
    {"command": "quality", "description": "画質: fast / standard / high"},
    {"command": "model", "description": "モデル: turbo / base"},
    {"command": "enhance", "description": "プロンプト強化: on / off"},
    {"command": "settings", "description": "現在の設定"},
    {"command": "help", "description": "使い方"},
]

STAGE_JA = {
    "planning": "準備中", "queued": "順番待ち", "enhancing": "プロンプトを最適化中", "waiting": "順番待ち",
    "loading": "モデルを読み込み中", "preparing": "準備中", "conditioning": "指示と画像を解析中",
    "sampling": "生成中", "vae_encode": "参照画像を読み込み中", "vae_decode": "仕上げ中", "encoding": "保存中",
}


class Bot:
    def __init__(self, server_module):
        self.S = server_module
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(70, connect=15))
        self.state: dict = {}
        self.groups: dict[str, dict] = {}  # media_group_id -> buffered album
        if STATE_FILE.exists():
            try:
                self.state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self.state = {}

    # ---------------------------------------------------------------- helpers
    def save_state(self) -> None:
        STATE_FILE.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")

    def chat_state(self, chat_id: int) -> dict:
        st = self.state.setdefault(str(chat_id), {})
        defaults = C.load_settings()
        for k in ("model", "quality", "use_llm"):
            st.setdefault(k, defaults[k])
        return st

    async def call(self, method: str, **params):
        files = params.pop("_files", None)
        for attempt in range(3):
            try:
                if files:
                    r = await self.client.post(f"{API}/{method}", data=params, files=files)
                else:
                    r = await self.client.post(f"{API}/{method}", json=params)
                d = r.json()
                if d.get("ok"):
                    return d["result"]
                retry = (d.get("parameters") or {}).get("retry_after")
                if retry:
                    await asyncio.sleep(retry + 0.5)
                    continue
                log.warning("telegram %s failed: %s", method, d.get("description"))
                return None
            except httpx.HTTPError as e:
                log.warning("telegram %s error: %s", method, e)
                await asyncio.sleep(2 * (attempt + 1))
        return None

    def conv_id(self, chat_id: int) -> str:
        st = self.chat_state(chat_id)
        cid = st.get("conv")
        if cid and (C.CONV_DIR / f"{cid}.json").exists():
            return cid
        return self.new_conv(chat_id)

    def new_conv(self, chat_id: int) -> str:
        conv = self.S.new_conversation(source="telegram")
        conv["title"] = "Telegram"
        self.S.save_conv(conv)
        self.chat_state(chat_id)["conv"] = conv["id"]
        self.save_state()
        return conv["id"]

    async def download_photo(self, file_id: str) -> str | None:
        f = await self.call("getFile", file_id=file_id)
        if not f:
            return None
        r = await self.client.get(f"{FILE_API}/{f['file_path']}")
        if r.status_code != 200:
            return None
        return self.S.save_image_bytes(r.content)

    # ---------------------------------------------------------------- main loop
    async def run(self) -> None:
        me = await self.call("getMe")
        if not me:
            log.error("Telegram token rejected; bot disabled")
            return
        log.warning("Telegram bot @%s started (allowed users: %s)", me.get("username"),
                    sorted(C.TELEGRAM_ALLOWED) or "none yet")
        await self.call("setMyCommands", commands=(COMMANDS_CHAT if chat.enabled() else []) + COMMANDS)
        for uid in C.TELEGRAM_NOTIFY:
            await self.call("sendMessage", chat_id=uid, disable_notification=False,
                            text=f"🟢 起動しました（{time.strftime('%m/%d %H:%M')}）。画像の生成・編集を受け付けています。")
        await self.mark_interrupted()
        offset = None
        while True:
            try:
                params = {"timeout": 50, "allowed_updates": ["message", "callback_query"]}
                if offset is not None:
                    params["offset"] = offset
                updates = await self.call("getUpdates", **params)
                for u in updates or []:
                    offset = u["update_id"] + 1
                    asyncio.create_task(self.handle(u))
                if updates is None:
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("poll error: %s", e)
                await asyncio.sleep(5)

    async def handle(self, u: dict) -> None:
        try:
            if "callback_query" in u:
                await self.on_callback(u["callback_query"])
            elif "message" in u:
                await self.on_message(u["message"])
        except Exception as e:  # noqa: BLE001
            log.exception("handler error: %s", e)

    def allowed(self, user_id: int) -> bool:
        return user_id in C.TELEGRAM_ALLOWED

    # ---------------------------------------------------------------- messages
    async def on_message(self, m: dict) -> None:
        chat_id = m["chat"]["id"]
        user_id = (m.get("from") or {}).get("id", 0)
        if not self.allowed(user_id):
            await self.call("sendMessage", chat_id=chat_id, text=(
                f"このボットは許可されたユーザーだけが使えます。\nあなたのユーザーID: {user_id}\n\n"
                "持ち主の方は、この数字を local.json の telegram_allowed_users に追加して、アプリを再起動してください。"))
            return

        text = (m.get("text") or m.get("caption") or "").strip()
        if text.startswith("/") and split_force(text)[0] is None:
            await self.on_command(chat_id, text)
            return

        file_id = None
        if m.get("photo"):
            file_id = m["photo"][-1]["file_id"]  # largest size
        elif (m.get("document") or {}).get("mime_type", "").startswith("image/"):
            file_id = m["document"]["file_id"]

        gid = m.get("media_group_id")
        if gid:  # albums arrive as separate messages: collect them briefly
            g = self.groups.setdefault(gid, {"files": [], "text": "", "chat": chat_id, "msg": m["message_id"]})
            if file_id:
                g["files"].append(file_id)
            if text:
                g["text"] = text
            g["t"] = time.time()
            if len(g["files"]) == 1:
                asyncio.create_task(self.flush_group(gid))
            return

        await self.turn(chat_id, m["message_id"], text, [file_id] if file_id else [])

    async def flush_group(self, gid: str) -> None:
        while time.time() - self.groups[gid]["t"] < 1.5:
            await asyncio.sleep(0.3)
        g = self.groups.pop(gid)
        await self.turn(g["chat"], g["msg"], g["text"], g["files"])

    async def on_command(self, chat_id: int, text: str) -> None:
        cmd, _, arg = text.partition(" ")
        cmd = cmd.split("@")[0].lower()
        arg = arg.strip().lower()
        st = self.chat_state(chat_id)
        if cmd in ("/start", "/help"):
            await self.call("sendMessage", chat_id=chat_id, text=help_text())
        elif cmd == "/new":
            self.new_conv(chat_id)
            await self.call("sendMessage", chat_id=chat_id, text="🆕 新しい会話を始めました。何を描きましょう？")
        elif cmd == "/quality" and arg in C.QUALITY:
            st["quality"] = arg
            self.save_state()
            note = ""
            if C.QUALITY[arg] > C.GEN_MAX_PIXELS:
                note = f"\n（このPCの VRAM では約{C.GEN_MAX_PIXELS / 1e6:.1f}MP までに抑えて生成します）"
            await self.call("sendMessage", chat_id=chat_id, text=f"画質を {arg} にしました。{note}")
        elif cmd == "/model" and arg in C.SAMPLING:
            st["model"] = arg
            self.save_state()
            await self.call("sendMessage", chat_id=chat_id, text=f"モデルを {arg} にしました。")
        elif cmd == "/enhance" and arg in ("on", "off"):
            st["use_llm"] = arg == "on"
            self.save_state()
            await self.call("sendMessage", chat_id=chat_id, text=f"プロンプト強化を {arg} にしました。")
        elif cmd == "/llm" and chat.enabled():
            await self.on_llm(chat_id, st, text.partition(" ")[2].strip())
        elif cmd == "/settings":
            lines = [f"画像モデル: {st['model']}", f"画質: {st['quality']}",
                     f"プロンプト強化: {'on' if st['use_llm'] else 'off'}"]
            if chat.enabled():
                lines.append(f"会話 AI: {chat.model_label(st.get('llm') or C.NANOGPT_MODEL)}"
                             + ("" if st.get("llm") else "（既定）"))
            await self.call("sendMessage", chat_id=chat_id, text="\n".join(lines))
        else:
            await self.call("sendMessage", chat_id=chat_id, text="不明なコマンドです。\n\n" + help_text())

    async def on_llm(self, chat_id: int, st: dict, arg: str) -> None:
        """/llm            -> list the models
           /llm glm        -> use a listed model (or any NanoGPT model id) in this chat
           /llm default    -> back to the default from local.json"""
        if arg.lower() in ("default", "既定", "reset"):
            st.pop("llm", None)
            self.save_state()
            await self.call("sendMessage", chat_id=chat_id,
                            text=f"会話 AI を既定（{chat.model_label(C.NANOGPT_MODEL)}）に戻しました。")
            return
        if arg:
            mid = C.NANOGPT_MODEL_CHOICES.get(arg.lower(), (arg, arg))[0]
            st["llm"] = mid
            self.save_state()
            await self.call("sendMessage", chat_id=chat_id, text=f"会話 AI を {chat.model_label(mid)} にしました。")
            return
        current = st.get("llm") or C.NANOGPT_MODEL
        lines = ["会話に使う AI モデル（このチャットだけに適用）", ""]
        for name, (mid, label) in C.NANOGPT_MODEL_CHOICES.items():
            mark = "▶ " if mid == current else "　 "
            lines.append(f"{mark}/llm {name} — {label}" + ("  ※既定" if mid == C.NANOGPT_MODEL else ""))
        lines += ["", "/llm default で既定に戻します。NanoGPT のモデル ID を直接指定することもできます。"]
        await self.call("sendMessage", chat_id=chat_id, text="\n".join(lines))

    async def send_text(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        """Send an LLM answer: ``` blocks become Telegram code blocks (monospace, tap to copy).
        Falls back to plain text if Telegram rejects the formatting."""
        for i, chunk in enumerate(split_markdown(text)):
            params = {"chat_id": chat_id, "text": markdown_to_html(chunk), "parse_mode": "HTML",
                      "link_preview_options": {"is_disabled": True}}
            if reply_to and i == 0:
                params["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
            if not await self.call("sendMessage", **params):
                params.pop("parse_mode")
                params["text"] = chunk
                await self.call("sendMessage", **params)

    async def _typing(self, chat_id: int, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.call("sendChatAction", chat_id=chat_id, action="typing")
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                pass

    async def turn(self, chat_id: int, reply_to: int, text: str, file_ids: list[str]) -> None:
        # "/image ..." always makes a picture, "/chat ..." always answers in text;
        # anything else is decided by the LLM (when a NanoGPT key is configured).
        force, text = split_force(text)
        names = []
        for fid in file_ids[:10]:
            n = await self.download_photo(fid)
            if n:
                names.append(n)
        if not text and not names:
            if force:
                await self.call("sendMessage", chat_id=chat_id, text=f"使い方: /{force} の後に続けて書いてください。")
            return
        st = self.chat_state(chat_id)
        cid = self.conv_id(chat_id)
        mode = "auto"
        if force == "chat" and not chat.enabled():
            await self.call("sendMessage", chat_id=chat_id,
                            text="会話機能は設定されていません（local.json の nanogpt_api_key）。")
            return
        if force != "image" and chat.enabled():
            model = st.get("llm") or C.NANOGPT_MODEL
            stop = asyncio.Event()
            typing = asyncio.create_task(self._typing(chat_id, stop))
            try:
                res = await chat.ask(self.S.load_conv(cid), text, [C.IMG_DIR / n for n in names], model,
                                     allow_tool=force is None)
            except Exception as e:  # noqa: BLE001
                log.warning("chat failed: %s", e)
                await self.send_text(chat_id, f"⚠️ 会話 AI でエラーが起きました: {e}", reply_to)
                return
            finally:
                stop.set()
                await typing
            if not res["image"]:
                answer = res["text"] or "（返答が空でした）"
                async with self.S.lock_for(cid):
                    chat.record_chat(self.S, cid, text, names, answer, model)
                await self.send_text(chat_id, answer, reply_to)
                return
            if res["text"]:
                await self.send_text(chat_id, res["text"], reply_to)
            text, mode = res["image"]["instruction"], res["image"]["mode"]
        await self.call("sendChatAction", chat_id=chat_id, action="upload_photo")
        opts = {"mode": mode, "aspect": "auto", "quality": st["quality"], "model": st["model"],
                "use_llm": st["use_llm"]}
        res = await self.S.submit_turn(cid, text, names, opts)
        await self.follow(chat_id, reply_to, cid, res["assistant"])

    # ---------------------------------------------------------------- progress + result
    async def follow(self, chat_id: int, reply_to: int | None, cid: str, asst: dict) -> None:
        mid = asst["id"]
        verb = "編集" if asst["meta"].get("mode") == "edit" else "生成"
        params = {"chat_id": chat_id, "text": f"⏳ 画像を{verb}しています…"}
        if reply_to:
            params["reply_parameters"] = {"message_id": reply_to, "allow_sending_without_reply": True}
        status = await self.call("sendMessage", **params)
        if status:
            # Remember the progress message, so that after an app restart it can be
            # marked as interrupted instead of showing "準備中" forever.
            self.state.setdefault("_pending", {})[mid] = {
                "chat": chat_id, "msg": status["message_id"], "cid": cid, "verb": verb}
            self.save_state()
        last_text, last_edit = "", 0.0
        while mid in self.S.LIVE:
            live = self.S.LIVE.get(mid) or {}
            label = STAGE_JA.get(live.get("stage") or live.get("status"), "処理中")
            if live.get("stage") == "queued" and self.S.WARM.get("state") == "running":
                label = "起動直後のため、モデルを読み込み中（初回だけ1〜2分かかります）"
            if live.get("stage") == "sampling" and live.get("total"):
                done, total = live.get("completed") or 0, live["total"]
                label = f"生成中 {'▓' * done}{'░' * (total - done)} {done}/{total}" if total <= 12 else \
                    f"生成中 {done}/{total}"
            msg = f"⏳ 画像を{verb}しています…\n{label}"
            if live.get("elapsed"):
                msg += f"（{int(live['elapsed'])}秒）"
            if status and msg != last_text and time.time() - last_edit > 3:
                await self.call("editMessageText", chat_id=chat_id, message_id=status["message_id"], text=msg)
                last_text, last_edit = msg, time.time()
            if time.time() - last_edit > 4:
                await self.call("sendChatAction", chat_id=chat_id, action="upload_photo")
            await asyncio.sleep(1.5)

        if self.state.get("_pending", {}).pop(mid, None):
            self.save_state()
        conv = self.S.load_conv(cid)
        m = next((x for x in conv["messages"] if x["id"] == mid), None)
        ok = bool(m and m.get("status") == "done" and m.get("images"))
        meta = (m or {}).get("meta", {})
        # Keep the progress message and turn it into a final status line (not deleted),
        # so the chat shows what happened even before the photo upload finishes.
        final = (f"✅ 画像の{verb}が完了しました（{meta.get('total_seconds')}秒）。画像を送信します…" if ok
                 else "⚠️ " + ((m or {}).get("text") or f"{verb}に失敗しました"))
        if status:
            await self.call("editMessageText", chat_id=chat_id, message_id=status["message_id"], text=final)
        elif not ok:
            await self.call("sendMessage", chat_id=chat_id, text=final)
        if not ok:
            return
        caption = (f"{'✏️ 編集' if meta.get('mode') == 'edit' else '🎨 生成'} · {meta.get('width')}×{meta.get('height')}"
                   f" · {'Base' if meta.get('model') == 'base' else 'Turbo'} · {meta.get('total_seconds')}秒")
        kb = {"inline_keyboard": [[
            {"text": "↻ 別案", "callback_data": f"r|{cid}|{mid}"},
            {"text": "🔍 高精細で", "callback_data": f"h|{cid}|{mid}"},
            {"text": "📄 原寸", "callback_data": f"f|{cid}|{mid}"},
            {"text": "📝 プロンプト", "callback_data": f"p|{cid}|{mid}"},
        ]]}
        path = C.IMG_DIR / m["images"][-1]
        sent = await self.call("sendPhoto", chat_id=chat_id, caption=caption, reply_markup=json.dumps(kb),
                               _files={"photo": (path.name, path.read_bytes(), "image/png")})
        if not sent:  # e.g. too large for a photo: fall back to a file
            sent = await self.call("sendDocument", chat_id=chat_id, caption=caption, reply_markup=json.dumps(kb),
                                   _files={"document": (path.name, path.read_bytes(), "image/png")})
        if status:
            await self.call("editMessageText", chat_id=chat_id, message_id=status["message_id"],
                            text=(f"✅ 画像の{verb}が完了しました（{meta.get('total_seconds')}秒）" if sent else
                                  f"⚠️ 画像の{verb}は完了しましたが、送信に失敗しました。Web UI の履歴から確認できます。"))

    async def mark_interrupted(self) -> None:
        """Progress messages of jobs that were running when the app stopped: say so, offer a retry."""
        pending = self.state.pop("_pending", {})
        if not pending:
            return
        self.save_state()
        for mid, p in pending.items():
            kb = {"inline_keyboard": [[{"text": "↻ やり直す", "callback_data": f"r|{p['cid']}|{mid}"}]]}
            await self.call("editMessageText", chat_id=p["chat"], message_id=p["msg"], reply_markup=kb,
                            text=f"⚠️ アプリの再起動で、画像の{p['verb']}が中断されました。"
                                 "下のボタンでやり直すか、もう一度送ってください。")

    async def on_callback(self, q: dict) -> None:
        user_id = q["from"]["id"]
        chat_id = q["message"]["chat"]["id"]
        if not self.allowed(user_id):
            await self.call("answerCallbackQuery", callback_query_id=q["id"], text="許可されていません")
            return
        kind, cid, mid = q["data"].split("|")
        conv = self.S.load_conv(cid)
        m = next((x for x in conv["messages"] if x["id"] == mid), None)
        if not m:
            await self.call("answerCallbackQuery", callback_query_id=q["id"], text="見つかりません")
            return
        if kind == "p":
            await self.call("answerCallbackQuery", callback_query_id=q["id"])
            meta = m.get("meta", {})
            await self.call("sendMessage", chat_id=chat_id, text=(
                f"📝 使用したプロンプト\n{meta.get('prompt', '')[:3500]}\n\nseed: {meta.get('seed')}"))
        elif kind == "f":
            await self.call("answerCallbackQuery", callback_query_id=q["id"], text="原寸画像を送ります")
            path = C.IMG_DIR / m["images"][-1]
            await self.call("sendDocument", chat_id=chat_id,
                            _files={"document": (f"qwen-image-{m['meta'].get('seed')}.png", path.read_bytes(), "image/png")})
        elif kind in ("r", "h"):
            await self.call("answerCallbackQuery", callback_query_id=q["id"],
                            text="別案を作ります" if kind == "r" else
                            (f"高精細で作り直します（このPCでは編集は約{C.EDIT_MAX_PIXELS / 1e6:.1f}MPまで。数分かかります）"
                             if m.get("meta", {}).get("mode") == "edit" else
                             f"高精細（約{min(C.QUALITY['high'], C.GEN_MAX_PIXELS) / 1e6:.1f}MP）で作り直します"))
            body = {"quality": "high"} if kind == "h" else {}
            res = await self.S.regenerate(cid, mid, body)
            await self.follow(chat_id, q["message"]["message_id"], cid, res["assistant"])


async def run(server_module) -> None:
    if not C.TELEGRAM_TOKEN:
        return
    await Bot(server_module).run()
