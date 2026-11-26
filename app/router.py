"""Turn a chat turn (text + optional images + history) into a generation plan.

The plan decides:
  * mode: "generate" (text-to-image) or "edit" (reference-image conditioned)
  * which images to use as references (attachments and/or the previous result)
  * the prompt to send to Qwen-Image-2.1
  * the aspect ratio

The mode decision is a deterministic keyword heuristic (predictable, instant).
Prompt rewriting is a separate, optional step (see enhancer.py).
"""
from __future__ import annotations

import re

# Words that signal "modify what we already have".
EDIT_HINTS = [
    "変え", "変更", "替え", "かえて", "にして", "追加", "加えて", "足して", "入れて", "消して", "削除",
    "除去", "取り除", "外して", "なくして", "もっと", "少し", "ちょっと", "さらに", "明るく", "暗く",
    "大きく", "小さく", "色を", "背景", "置き換", "差し替", "修正", "直して", "編集", "着せ", "持たせ",
    "風に", "っぽく", "スタイルに", "バージョン", "に変換", "塗って", "向きを", "表情", "髪", "服を",
    "make it", "change", "add ", "remove", "replace", "turn it", "turn the", "edit", "instead",
    "more ", "less ", "brighter", "darker", "recolor", "convert",
]
# Words that refer back to an existing image.
REF_HINTS = [
    "さっき", "先ほど", "前の", "上の", "この画像", "その画像", "今の", "それ", "これ", "同じ", "この絵",
    "この写真", "この子", "this image", "that image", "the image", "it ", "same",
]
# Words that signal "start over with something new".
NEW_HINTS = [
    "新しく", "新しい画像", "別の", "違う画像", "描いて", "描け", "生成して", "生成", "作って", "作成",
    "画像を", "イラストを", "写真を", "絵を", "new image", "generate", "draw", "create", "a picture of",
    "an image of", "paint",
]

ASPECT_WORDS = [
    (r"(16\s*[:：対x×]\s*9|ワイド|シネマ|横長ワイド|youtube|サムネ|wallpaper for pc|デスクトップ)", "16:9"),
    (r"(9\s*[:：対x×]\s*16|スマホ|待ち受け|ストーリー|縦長ワイド|tiktok|reels)", "9:16"),
    (r"(4\s*[:：対x×]\s*3)", "4:3"),
    (r"(3\s*[:：対x×]\s*4)", "3:4"),
    (r"(3\s*[:：対x×]\s*2|横長|landscape|風景画)", "3:2"),
    (r"(2\s*[:：対x×]\s*3|縦長|portrait|ポスター|全身)", "2:3"),
    (r"(1\s*[:：対x×]\s*1|正方形|スクエア|square|アイコン|icon)", "1:1"),
]


def detect_aspect(text: str) -> str | None:
    t = text.lower()
    for pat, ar in ASPECT_WORDS:
        if re.search(pat, t):
            return ar
    return None


def _count(text: str, words: list[str]) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)


def heuristic_plan(text: str, n_attachments: int, has_previous: bool, forced_mode: str = "auto") -> dict:
    plan = {
        "mode": "generate",
        "use_attachments": n_attachments > 0,
        "use_previous": False,
        "prompt": text.strip(),
        "aspect": detect_aspect(text),
        "router": "heuristic",
    }
    if forced_mode == "generate":
        plan["use_attachments"] = False
        return plan
    if forced_mode == "edit":
        plan["mode"] = "edit"
        plan["use_previous"] = n_attachments == 0 and has_previous
        if n_attachments == 0 and not has_previous:
            plan["mode"] = "generate"
        return plan

    edit = _count(text, EDIT_HINTS)
    ref = _count(text, REF_HINTS)
    new = _count(text, NEW_HINTS)

    if n_attachments > 0:
        plan["mode"] = "edit"
        # "Combine this with the previous picture" style requests.
        if has_previous and ref > 0 and any(w in text for w in ("さっき", "先ほど", "前の", "上の")):
            plan["use_previous"] = True
        return plan

    if has_previous:
        # A follow-up phrased as a request ("花火を上げて", "水彩画風に", "add a hat") that
        # doesn't ask for a new picture most likely modifies the last result, like in ChatGPT.
        if new == 0 and (REQUEST_END.search(text.strip()) or EN_IMPERATIVE.match(text.strip())):
            edit += 1
        if ref > 0 or (edit > 0 and new == 0) or (edit > new):
            plan["mode"] = "edit"
            plan["use_previous"] = True
        elif len(text.strip()) <= 12 and edit > 0:
            plan["mode"] = "edit"
            plan["use_previous"] = True
    return plan


# Japanese request endings: 〜て / 〜に / 〜く / 〜でください (a bare 〜で like "横長で" is a condition, not a request) (+ ください・ほしい etc.)
REQUEST_END = re.compile(r"((て|に|く)(ください|下さい|ほしい|欲しい|みて|ちょうだい)?|でください|で下さい|お願い(します|いたします)?|おねがい(します)?)[。．.!！?？\s]*$")
EN_IMPERATIVE = re.compile(r"^(please\s+)?(add|remove|put|give|make|turn|change|replace|let|place|erase|delete)\b", re.I)
