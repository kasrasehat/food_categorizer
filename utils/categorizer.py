from __future__ import annotations

import asyncio
import os
import re
from collections import Counter
from typing import Optional, List
from langchain_community.vectorstores import Chroma

from utils.lfspan import get_vectordb_state
from utils.logging_config import get_logger

logger = get_logger(__name__)


_UNITS_RE = re.compile(
    r"\b("
    r"g|gr|gram|grams|kg|kilogram|kilograms|mg|ml|l|liter|litre|"
    r"tbsp|tsp|tablespoon|teaspoon|cup|cups|oz|ounce|ounces|lb|lbs|pound|pounds|"
    r"pcs|pc|piece|pieces|pack|packs|packet|packets"
    r")\b",
    re.IGNORECASE,
)

_SIZES_RE = re.compile(
    r"\b("
    r"xs|s|sm|small|"
    r"m|md|medium|"
    r"l|lg|large|"
    r"xl|x\-?l|xxl|2xl|3xl|"
    r"xlarge|x\-?large|extra\s+large|"
    r"regular|reg"
    r")\b",
    re.IGNORECASE,
)


def _extract_food_name(text: str) -> str:
    """
    Very lightweight normalization to approximate the food name from a free-form label.

    Examples:
      - "2x Chicken Burger 250g" -> "chicken burger"
      - "Coca Cola 330 ml can"   -> "coca cola can"
    """
    s = (text or "").strip().lower()
    if not s:
        return ""

    # Remove bracketed parts (often brand / packaging notes)
    s = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", s)

    # Drop numeric quantities / separators
    s = re.sub(r"\b\d+(?:\.\d+)?\b", " ", s)
    # s = s.replace("x", " ")

    # Remove units
    s = _UNITS_RE.sub(" ", s)

    # Remove common size descriptors (often appear in beverages/menus)
    s = _SIZES_RE.sub(" ", s)

    # Keep letters, digits, spaces, hyphens; collapse whitespace
    s = re.sub(r"[^\w\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _mmr_search_by_vector(
    vectordb: Chroma,
    embedding: List[float],
    k: int,
    fetch_k: int,
):
    """MMR search using a precomputed embedding vector for efficiency."""
    docs = vectordb.max_marginal_relevance_search_by_vector(embedding, k=k, fetch_k=fetch_k)
    return list(docs)


def categorize_food_text(text: str) -> dict:
    """
    Categorize a food label using your vector DB (Chroma) + voting.

    Flow:
    - extract/normalize a food name from `text`
    - embed it with `OpenAITextEmbeddings`
    - retrieve top-N similar foods from vectordb (default 13)
    - vote categories across the top-K (default 5) results
    """
    debug = True
    if debug:
        logger.info("[CATEGORIZE] input_text=%r", text)

    q = _extract_food_name(text)
    if not q:
        if debug:
            logger.info("[CATEGORIZE] empty_normalized_query -> fallback other")
        return {"category": "other", "probability": 0.0}

    vectordb_state = get_vectordb_state()
    k = 17
    fetch_k = 13

    if k is None:
        k = int(os.getenv("TEXT_DEFAULT_K", 17))
    if fetch_k is None:
        fetch_k = int(os.getenv("TEXT_DEFAULT_FETCH_K", 13))
    
    vote_k = max(1, fetch_k)

    if debug:
        logger.info(
            "[CATEGORIZE] normalized_query=%r k=%s fetch_k=%s vote_k=%s",
            q,
            k,
            fetch_k,
            vote_k,
        )

    try:
        embedding = vectordb_state["text_embedder"].embed_documents([q])[0]
    except Exception as e:
        logger.warning("[CATEGORIZE] Embedding failed for %r: %s", q, e, exc_info=True)
        return {"category": "other", "probability": 0.0}

    try:
        docs = _mmr_search_by_vector(
            vectordb_state["text_db"],
            embedding,
            int(k),
            int(fetch_k),
        )
    except Exception as e:
        logger.warning("[CATEGORIZE] Vector search failed for %r: %s", q, e, exc_info=True)
        return {"category": "other", "probability": 0.0}

    if not docs:
        if debug:
            logger.info("[CATEGORIZE] no_retrieval_results -> fallback other")
        return {"category": "other", "probability": 0.0}

    top_for_vote = docs[: min(vote_k, len(docs))]

    if debug:
        logger.info(
            "[CATEGORIZE] retrieved=%d using_for_vote=%d",
            len(docs),
            len(top_for_vote),
        )
        # Log a small, safe preview of matches to help debug retrieval quality.
        for i, d in enumerate(top_for_vote[: min(fetch_k, len(top_for_vote))]):
            page = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", None) or {}
            logger.info(
                "[CATEGORIZE] match[%d] page_content=%r meta_group=%r",
                i,
                page[:120],
                meta.get("group"),
            )

    categories: list[str] = []
    for d in top_for_vote:
        meta = getattr(d, "metadata", None) or {}
        cat = meta.get("group")
        if cat:
            categories.append(cat)

    if not categories:
        if debug:
            logger.info("[CATEGORIZE] no_categories_in_metadata -> fallback other")
        return {"category": "other", "probability": 0.0}

    counts = Counter(categories)
    best_cat, best_count = counts.most_common(1)[0]
    prob = best_count / float(len(top_for_vote))

    logger.info(
        "[CATEGORIZE] q=%r vote_k=%d best=%r prob=%.3f counts=%s",
        q,
        len(top_for_vote),
        best_cat,
        prob,
        dict(counts),
    )
    return {
        "category": best_cat,
        "probability": prob,
    }


async def categorize_food_batch(texts: List[str]) -> List[dict]:
    """
    Batch version of `categorize_food_text` for lower latency in API endpoints.

    Uses the same flow per item:
    - extract/normalize a food name from `text`
    - embed it with `OpenAITextEmbeddings`
    - retrieve top-N similar foods from vectordb
    - vote categories across the top-K results

    Returns a list of dicts (same length as `texts`), each with:
      {"category": str, "probability": float}
    """
    debug = True

    results: List[dict] = [{"category": "other", "probability": 0.0} for _ in texts]
    if not texts:
        return results

    # Normalize queries first
    normalized: List[str] = []
    for t in texts:
        if debug:
            logger.info("[CATEGORIZE_BATCH] input_text=%r", t)
        q = _extract_food_name(t)
        normalized.append(q)

    vectordb_state = get_vectordb_state()
    k = 17
    fetch_k = 13

    if k is None:
        k = int(os.getenv("TEXT_DEFAULT_K", 17))
    if fetch_k is None:
        fetch_k = int(os.getenv("TEXT_DEFAULT_FETCH_K", 13))

    vote_k = max(1, fetch_k)

    # Only embed/search non-empty queries
    idxs: List[int] = [i for i, q in enumerate(normalized) if q]
    if not idxs:
        return results

    queries: List[str] = [normalized[i] for i in idxs]
    if debug:
        logger.info(
            "[CATEGORIZE_BATCH] batch_size=%d non_empty=%d k=%s fetch_k=%s vote_k=%s",
            len(texts),
            len(queries),
            k,
            fetch_k,
            vote_k,
        )

    try:
        embeddings: List[List[float]] = await asyncio.to_thread(
            vectordb_state["text_embedder"].embed_documents, queries
        )
    except Exception as e:
        logger.warning("[CATEGORIZE_BATCH] Embedding failed for batch: %s", e, exc_info=True)
        return results

    async def _search_one(embedding: List[float]):
        return await asyncio.to_thread(
            _mmr_search_by_vector,
            vectordb_state["text_db"],
            embedding,
            int(k),
            int(fetch_k),
        )

    docs_list = await asyncio.gather(*[_search_one(emb) for emb in embeddings], return_exceptions=True)

    for local_i, docs_or_exc in enumerate(docs_list):
        i = idxs[local_i]
        q = normalized[i]

        if debug:
            logger.info(
                "[CATEGORIZE_BATCH] normalized_query=%r k=%s fetch_k=%s vote_k=%s",
                q,
                k,
                fetch_k,
                vote_k,
            )

        if isinstance(docs_or_exc, Exception):
            logger.warning(
                "[CATEGORIZE_BATCH] Vector search failed for %r: %s",
                q,
                docs_or_exc,
                exc_info=True,
            )
            continue

        docs = docs_or_exc
        if not docs:
            if debug:
                logger.info("[CATEGORIZE_BATCH] no_retrieval_results -> fallback other")
            continue

        top_for_vote = docs[: min(vote_k, len(docs))]

        if debug:
            logger.info(
                "[CATEGORIZE_BATCH] retrieved=%d using_for_vote=%d",
                len(docs),
                len(top_for_vote),
            )
            for j, d in enumerate(top_for_vote[: min(fetch_k, len(top_for_vote))]):
                page = getattr(d, "page_content", "") or ""
                meta = getattr(d, "metadata", None) or {}
                logger.info(
                    "[CATEGORIZE_BATCH] match[%d] page_content=%r meta_group=%r",
                    j,
                    page[:120],
                    meta.get("group"),
                )

        categories: list[str] = []
        for d in top_for_vote:
            meta = getattr(d, "metadata", None) or {}
            cat = meta.get("group")
            if cat:
                categories.append(cat)

        if not categories:
            if debug:
                logger.info("[CATEGORIZE_BATCH] no_categories_in_metadata -> fallback other")
            continue

        counts = Counter(categories)
        best_cat, best_count = counts.most_common(1)[0]
        prob = best_count / float(len(top_for_vote))

        logger.info(
            "[CATEGORIZE_BATCH] q=%r vote_k=%d best=%r prob=%.3f counts=%s",
            q,
            len(top_for_vote),
            best_cat,
            prob,
            dict(counts),
        )

        results[i] = {"category": best_cat, "probability": prob}

    return results


