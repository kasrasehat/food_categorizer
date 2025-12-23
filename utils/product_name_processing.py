from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

from utils.categorizer import _extract_food_name


_EN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")
_HAS_NON_ASCII_LETTER_RE = re.compile(r"[^\x00-\x7F]")

# Script detection (fast, no models needed)
_ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_HAN_RE = re.compile(r"[\u4E00-\u9FFF]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30FF]")
_HANGUL_RE = re.compile(r"[\uAC00-\uD7AF]")

# Common mojibake markers for UTF-8 that was decoded as Latin-1/CP1252
_MOJIBAKE_HINT_RE = re.compile(r"[ÃÂÐÑØÙÝÞ]|â€™|â€œ|â€\x9d")


def _score_scriptiness(s: str) -> int:
    # Prefer candidates that contain “real” non-Latin scripts when present
    return (
        10 * len(_ARABIC_RE.findall(s))
        + 6 * len(_CYRILLIC_RE.findall(s))
        + 6 * len(_HAN_RE.findall(s))
        + 6 * len(_HIRAGANA_KATAKANA_RE.findall(s))
        + 6 * len(_HANGUL_RE.findall(s))
    )


def fix_mojibake(text: str) -> str:
    """
    Repair common mojibake cases like 'Ø§Ù...' (Arabic UTF-8 bytes decoded as Latin-1/CP1252).
    Safe: if it can't confidently improve, it returns the original string.
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return s

    # If it already contains Arabic (or other scripts), don't touch it.
    if _score_scriptiness(s) > 0:
        return s

    # Only attempt repair when it looks suspicious
    if not _MOJIBAKE_HINT_RE.search(s):
        return s

    candidates: List[str] = [s]
    for src_enc in ("latin-1", "cp1252"):
        # latin-1 can always roundtrip bytes 0-255; cp1252 can fail on some controls -> ignore
        enc_errors = "strict" if src_enc == "latin-1" else "ignore"
        try:
            b = s.encode(src_enc, errors=enc_errors)
        except Exception:
            continue
        for dst_enc in ("utf-8",):
            try:
                repaired = b.decode(dst_enc, errors="strict")
                candidates.append(repaired)
            except Exception:
                continue

    # Choose the candidate that most increases presence of non-Latin scripts.
    best = s
    best_score = _score_scriptiness(s)
    for c in candidates[1:]:
        sc = _score_scriptiness(c)
        if sc > best_score:
            best, best_score = c, sc
    return best


def detect_language(text: str) -> str:
    """
    Lightweight language detector:
    - Script-based detection for Arabic/Cyrillic/CJK/Korean/Japanese
    - If Latin script: try NLTK stopword scoring if available
    Returns a short code: ar, ru, zh, ja, ko, en, es, fr, de, it, pt, or unknown.
    """
    s = fix_mojibake(text)
    if not s or not s.strip():
        return "unknown"

    if _ARABIC_RE.search(s):
        return "ar"
    if _CYRILLIC_RE.search(s):
        return "ru"
    if _HAN_RE.search(s):
        return "zh"
    if _HIRAGANA_KATAKANA_RE.search(s):
        return "ja"
    if _HANGUL_RE.search(s):
        return "ko"

    # Latin-ish: try NLTK stopwords if installed + corpus available
    tokens = [t.lower() for t in _EN_TOKEN_RE.findall(s)]
    if not tokens:
        return "unknown"

    try:
        from nltk.corpus import stopwords  # type: ignore
    except Exception:
        return "unknown"

    langs = {
        "en": "english",
        "es": "spanish",
        "fr": "french",
        "de": "german",
        "it": "italian",
        "pt": "portuguese",
    }
    best = ("unknown", 0)
    for code, name in langs.items():
        try:
            sw = set(stopwords.words(name))
        except Exception:
            continue
        score = sum(1 for t in tokens if t in sw)
        if score > best[1]:
            best = (code, score)

    # Require some signal; otherwise unknown
    return best[0] if best[1] >= 2 else "unknown"


def _normalize_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = fix_mojibake(s)
    s = unicodedata.normalize("NFKC", s)
    # common separators/symbols to spaces
    s = re.sub(r"[_@#/$\\|]+", " ", s)
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_english_part(text: str) -> str:
    """
    Extract an English-only candidate from the input.
    """
    base = _normalize_text(text)
    tokens = _EN_TOKEN_RE.findall(base)
    candidate = " ".join(tokens).strip().lower()
    # prevent garbage outputs like "x" / "g"
    if len(candidate) < 2:
        return ""
    return candidate


def basic_clean_english(text: str) -> str:
    """
    Clean a string assuming it's English-ish already.
    Uses existing `_extract_food_name` and extra normalization.
    """
    s = _normalize_text(text)
    s = _extract_food_name(s)  # lowercases, removes units/numbers/punct (existing logic)
    # remove leftover underscores etc and collapse
    s = re.sub(r"[_@#/$\\|]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def choose_processing_plan(raw: str) -> Tuple[str, bool]:
    """
    Returns (candidate, needs_llm).

    - If there's an English part, use it.
    - If it's mixed language, still prefer English-only part.
    - If there's no English part, mark for LLM translation.
    """
    raw_norm = _normalize_text(raw)
    english_part = extract_english_part(raw_norm)
    if english_part:
        # If mixed language or english already, we keep english tokens and clean further.
        return basic_clean_english(english_part), False

    # No English tokens detected -> needs LLM translation.
    return raw_norm, True


def finalize_processed_name(s: str) -> str:
    """
    Final pass after either English extraction or LLM translation.
    """
    s = basic_clean_english(s)
    # If still contains non-ascii (LLM failure), keep only english tokens
    if _HAS_NON_ASCII_LETTER_RE.search(s):
        s = extract_english_part(s) or s
        s = basic_clean_english(s)
    return s


def prepare_translation_inputs(names: List[str]) -> Tuple[List[str], List[int], List[str], List[str]]:
    """
    For a list of raw names:
    - returns (needs_translation_texts, indices, initial_processed)
    where initial_processed has same length as names (filled with best-effort candidates or "")
    and indices map translation_texts back to the input rows.
    """
    translation_texts: List[str] = []
    translation_indices: List[int] = []
    translation_langs: List[str] = []
    initial_processed: List[str] = [""] * len(names)

    for i, raw in enumerate(names):
        candidate, needs_llm = choose_processing_plan(raw)
        if needs_llm:
            translation_texts.append(candidate)
            translation_indices.append(i)
            translation_langs.append(detect_language(candidate))
        else:
            initial_processed[i] = finalize_processed_name(candidate)

    return translation_texts, translation_indices, initial_processed, translation_langs


