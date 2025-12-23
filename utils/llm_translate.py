from __future__ import annotations

import json
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from utils.embedding import OpenAITextEmbeddings


def translate_to_english_batch(
    texts: List[str],
    *,
    source_languages: Optional[List[str]] = None,
    cfg: Optional[Dict] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Translate product names to English via LLM in batches.

    Returns a list aligned to `texts`.
    """
    cfg = cfg or {}
    resolved_key = OpenAITextEmbeddings._resolve_api_key(api_key)

    # Setup LLM (mirrors user's desired snippet)
    model_name = (cfg.get("rag_llm_model") or "gpt-4o-mini")
    chosen_model = model_name or "gpt-4o-mini"
    alias_map = {
        "gpt40": "gpt-4o",
        "gpt-40": "gpt-4o",
        "gpt4o": "gpt-4o",
        "gpt4o-mini": "gpt-4o-mini",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt5-low": "gpt-5",
        "gpt5-medium": "gpt-5",
        "gpt5-high": "gpt-5",
        "gpt5-nano": "gpt-5-nano",
        "gpt-5-nano": "gpt-5-nano",
    }
    chosen_model = alias_map.get(chosen_model, chosen_model)

    _llm_kwargs = {
        "model": chosen_model,
        "temperature": 0.1,
        "openai_api_key": resolved_key,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }
    if chosen_model == "gpt-5":
        _llm_kwargs["reasoning"] = {"effort": "low"}

    llm = ChatOpenAI(**_llm_kwargs)

    def _system_for_lang(lang: str) -> str:
        prefix = "Language hint: unknown.\n"
        if lang and lang != "unknown":
            prefix = f"Language hint: the input is likely {lang}.\n"
            if lang == "ar":
                prefix = "Language hint: the input is likely Arabic (ar).\n"
            elif lang == "ru":
                prefix = "Language hint: the input is likely Russian/Cyrillic (ru).\n"
            elif lang == "zh":
                prefix = "Language hint: the input is likely Chinese (zh).\n"
            elif lang == "ja":
                prefix = "Language hint: the input is likely Japanese (ja).\n"
            elif lang == "ko":
                prefix = "Language hint: the input is likely Korean (ko).\n"

        return (
            "You are a product-name normalizer.\n"
            + prefix
            + "Translate the input to English ONLY.\n"
            "Return a JSON object with exactly one key: \"english\".\n"
            "Rules:\n"
            "- Output only the product name in English.\n"
            "- Remove quantities/units (e.g., 250g, 1L, 2 pcs), packaging notes, and extra symbols.\n"
            "- If the input contains both English and another language, keep only the English meaning.\n"
            "- Do not include explanations.\n"
        )

    if source_languages is None:
        source_languages = ["unknown"] * len(texts)
    elif len(source_languages) != len(texts):
        raise ValueError("source_languages must be the same length as texts (or omitted).")

    message_batches = [
        [
            SystemMessage(content=_system_for_lang(source_languages[i])),
            HumanMessage(content=f'Input: "{t}"\nReturn JSON.'),
        ]
        for i, t in enumerate(texts)
    ]

    responses = llm.batch(message_batches)

    out: List[str] = []
    for i, r in enumerate(responses):
        content = getattr(r, "content", "") or ""
        english = ""
        try:
            obj = json.loads(content) if isinstance(content, str) else {}
            english = str(obj.get("english", "")).strip()
        except Exception:
            english = ""
        out.append(english if english else texts[i])
    return out


