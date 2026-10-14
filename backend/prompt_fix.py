"""
Normalize user prompts before ComfyUI.

- Fix messy spacing / light typos
- Translate non-English (and Hinglish-ish) text to clear English
- Optional edit/motion framing (disabled when raw_prompt / frame=False)
"""

from __future__ import annotations

import re
from typing import Literal

Mode = Literal["img", "vid"]


def _raw_prompt_enabled() -> bool:
    try:
        from backend.config import get_settings

        return bool(get_settings().raw_prompt)
    except Exception:
        return True

# Common informal / misspelled tokens → English
_TYPO_MAP = {
    "pls": "please",
    "plz": "please",
    "u": "you",
    "ur": "your",
    "r": "are",
    "dont": "don't",
    "doesnt": "doesn't",
    "cant": "can't",
    "wont": "won't",
    "im": "I'm",
    "teh": "the",
    "adn": "and",
    "taht": "that",
    "waht": "what",
    "becoz": "because",
    "bcoz": "because",
    "cuz": "because",
    "pic": "photo",
    "pics": "photos",
    "img": "image",
    "vid": "video",
    "selfie": "selfie photo",
    "hd": "high detail",
    "4k": "high resolution",
    "realstic": "realistic",
    "realisitc": "realistic",
    "photorealisitc": "photorealistic",
    "beutiful": "beautiful",
    "beatiful": "beautiful",
    "smille": "smile",
    "smilling": "smiling",
    "backround": "background",
    "backgrond": "background",
    "cloths": "clothes",
    "clothe": "clothes",
    "jewlery": "jewelry",
    "jewellry": "jewelry",
    "colur": "color",
    "colour": "color",
    "chang": "change",
    "chng": "change",
    "remov": "remove",
    "addd": "add",
}

# Longer Hinglish / informal phrases first (order matters)
_HINGLISH_PHRASES: list[tuple[str, str]] = [
    (r"\bkar\s*do\b", "do"),
    (r"\bkar\s*dena\b", "do"),
    (r"\bkar\s*de\b", "do"),
    (r"\bbana\s*do\b", "make"),
    (r"\bbanao\b", "make"),
    (r"\bdikhao\b", "show"),
    (r"\blagao\b", "apply"),
    (r"\bhatao\b", "remove"),
    (r"\bhata\s*do\b", "remove"),
    (r"\bchange\s*kar\s*do\b", "change"),
    (r"\bremove\s*kar\s*do\b", "remove"),
    (r"\badd\s*kar\s*do\b", "add"),
    (r"\bthoda\s+sa\b", "slightly"),
    (r"\bthoda\b", "slightly"),
    (r"\bzyada\b", "more"),
    (r"\bbilkul\b", "exactly"),
    (r"\bwaisa\s+hi\b", "the same"),
    (r"\bjaisa\s+hai\b", "as is"),
    (r"\bphoto\s+ko\b", "the photo"),
    (r"\bimage\s+ko\b", "the image"),
    (r"\buski\b", "her/his"),
    (r"\buska\b", "her/his"),
    (r"\biski\b", "this"),
    (r"\biska\b", "this"),
    (r"\bmujhe\b", "I want"),
    (r"\bmera\b", "my"),
    (r"\bmeri\b", "my"),
    (r"\bapka\b", "your"),
    (r"\bapni\b", "your"),
    (r"\bkali\b", "black"),
    (r"\bkala\b", "black"),
    (r"\bsafed\b", "white"),
    (r"\blal\b", "red"),
    (r"\bneela\b", "blue"),
    (r"\bhara\b", "green"),
    (r"\bpila\b", "yellow"),
    (r"\bchehra\b", "face"),
    (r"\bbaal\b", "hair"),
    (r"\bkameez\b", "shirt"),
    (r"\bpant\b", "pants"),
    (r"\bskirt\b", "skirt"),
    (r"\bsmile\s+karwao\b", "make smile"),
    (r"\bnatural\s+lagao\b", "make it look natural"),
]

_NON_LATIN = re.compile(
    r"[\u0900-\u097F\u0980-\u09FF\u0A00-\u0A7F\u0A80-\u0AFF"
    r"\u0B00-\u0B7F\u0B80-\u0BFF\u0C00-\u0C7F\u0C80-\u0CFF"
    r"\u0D00-\u0D7F\u0600-\u06FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF]"
)


def _clean_spaces(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", ". ", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    return text.strip(" \t\r\n,;|")


def _fix_hinglish(text: str) -> str:
    out = text
    for pattern, repl in _HINGLISH_PHRASES:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
    return _clean_spaces(out)


def _fix_typos(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        key = word.lower()
        fixed = _TYPO_MAP.get(key)
        if not fixed:
            return word
        if word.isupper():
            return fixed.upper()
        if word[0].isupper():
            return fixed[:1].upper() + fixed[1:]
        return fixed

    return re.sub(r"[A-Za-z']+", repl, text)


def _needs_translation(text: str) -> bool:
    if _NON_LATIN.search(text):
        return True
    # Mostly Latin but mixed informal Hinglish markers
    low = text.lower()
    hinglish = (
        "karo",
        "kar do",
        "banao",
        "dikhao",
        "lagao",
        "hatao",
        "thoda",
        "zyada",
        "waisa",
        "jaisa",
        "photo ko",
        "image ko",
        "usko",
        "isko",
        "mujhe",
        "mera",
        "meri",
        "apka",
        "apni",
    )
    return any(tok in low for tok in hinglish)


def _translate_to_english(text: str) -> str:
    try:
        from deep_translator import GoogleTranslator

        out = GoogleTranslator(source="auto", target="en").translate(text)
        return (out or text).strip() or text
    except Exception:
        return text


def _polish_english(text: str) -> str:
    """
    Light grammar polish: if text still looks broken, round-trip via translator
    (auto→en often cleans informal / misspelled English).
    """
    letters = re.sub(r"[^A-Za-z]", "", text)
    if len(letters) < 8:
        return text
    # Heuristic: lots of short tokens / missing vowels → try translate polish
    tokens = re.findall(r"[A-Za-z']+", text)
    short = sum(1 for t in tokens if len(t) <= 2)
    if short >= max(2, len(tokens) // 3) or _needs_translation(text):
        try:
            from deep_translator import GoogleTranslator

            polished = GoogleTranslator(source="auto", target="en").translate(text)
            if polished and len(polished.strip()) >= 3:
                return polished.strip()
        except Exception:
            pass
    return text


def _as_edit_instruction(text: str) -> str:
    """Force edit framing so Flux does not invent a new scene."""
    t = text.strip().rstrip(".")
    low = t.lower()
    # Strip leading generation verbs that cause full redraws
    for prefix in (
        "generate ",
        "create ",
        "make an image of ",
        "make a photo of ",
        "draw ",
        "paint ",
        "a photo of ",
        "a picture of ",
        "an image of ",
        "photograph of ",
    ):
        if low.startswith(prefix):
            t = t[len(prefix) :].strip()
            low = t.lower()
            break

    if low.startswith(("change ", "edit ", "replace ", "remove ", "add ", "turn ", "make ")):
        return t[0].upper() + t[1:] if t else t

    return f"Edit this exact photo only: {t}"


def _as_motion_instruction(text: str) -> str:
    t = text.strip().rstrip(".")
    low = t.lower()
    for prefix in ("generate ", "create ", "make a video of ", "animate "):
        if low.startswith(prefix):
            t = t[len(prefix) :].strip()
            low = t.lower()
            break
    if any(
        k in low
        for k in ("move", "walk", "turn", "smile", "look", "camera", "zoom", "pan", "motion")
    ):
        return t[0].upper() + t[1:] if t else t
    return f"Subtle natural motion: {t}. Keep the same person and scene."


def normalize_prompt(
    raw: str,
    mode: Mode = "img",
    *,
    frame: bool = True,
    recolor: bool = True,
) -> dict[str, str]:
    """
    Returns {original, cleaned, english} where english is what ComfyUI should use.
    Set frame=False for negatives (translate/clean only, no edit/motion wrapper).
    When settings.raw_prompt is True, framing is skipped even if frame=True.
    recolor=False keeps "jacket black" as written instead of "change jacket to black".
    """
    original = (raw or "").strip()
    if not original:
        return {"original": "", "cleaned": "", "english": ""}

    cleaned = _fix_typos(_fix_hinglish(_clean_spaces(original)))
    english = cleaned
    if _needs_translation(cleaned) or _NON_LATIN.search(original):
        # Prefer translating the original when it has native script
        source = original if _NON_LATIN.search(original) else cleaned
        english = _translate_to_english(source)
        english = _fix_typos(_fix_hinglish(_clean_spaces(english)))
    english = _polish_english(english)
    english = _fix_typos(_fix_hinglish(_clean_spaces(english)))
    # Drop leftover romanized fillers Google often leaves behind
    english = re.sub(
        r"\b(kar|dena|bhai|yaar|bas|nahi|haan)\b",
        "",
        english,
        flags=re.IGNORECASE,
    )
    english = _clean_spaces(english)
    # "jacket black" / "hair brown" → "change jacket to black" (skip if already "… to COLOR")
    # Never rewrite "and black" / "her white" / "on red" — those are grammar, not recolors —
    # nor "navy blue pants" / "long black hair", where the colour describes the word after it.
    if recolor and not re.search(
        r"\bto\s+(black|white|red|blue|green|yellow|brown|pink|orange|purple|gray|grey)\b",
        english,
        flags=re.IGNORECASE,
    ):
        _color_stop = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "her",
            "his",
            "their",
            "its",
            "on",
            "to",
            "of",
            "with",
            "for",
            "from",
            "at",
            "in",
            "into",
            "onto",
            "navy",
            "light",
            "dark",
            "pale",
            "bright",
            "deep",
            "sky",
            "royal",
            "baby",
            "hot",
            "pastel",
            "neon",
            "off",
            "jet",
            "pitch",
            "snow",
            "blood",
            "olive",
            "forest",
            "lime",
            "teal",
            "dusty",
            "muted",
            "long",
            "short",
            "all",
        }

        def _recolor(match: re.Match[str]) -> str:
            noun = match.group(1)
            if noun.lower() in _color_stop:
                return match.group(0)
            return f"change {noun} to {match.group(2)}"

        english = re.sub(
            r"\b(?:change\s+)?([A-Za-z][\w'/]{1,38})\s+"
            r"(black|white|red|blue|green|yellow|brown|pink|orange|purple|gray|grey)\b"
            r"(?!\s+(?!(?:and|or|please|now|instead|color|colour|in|on|with|for|but)\b)[A-Za-z])",
            _recolor,
            english,
            flags=re.IGNORECASE,
            count=1,
        )
    english = re.sub(r"\s{2,}", " ", english).strip(" ,.")
    # Orphan verbs left after Hinglish cleanup ("... black do")
    english = re.sub(r"\b(do|kar)\s*$", "", english, flags=re.IGNORECASE).strip(" ,.")
    if not english:
        english = cleaned or original

    apply_frame = frame and not _raw_prompt_enabled()
    if apply_frame:
        if mode == "vid":
            english = _as_motion_instruction(english)
        else:
            english = _as_edit_instruction(english)

    return {
        "original": original,
        "cleaned": cleaned,
        "english": english,
    }
