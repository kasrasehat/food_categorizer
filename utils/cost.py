from typing import Dict

# Embedding model pricing (USD per 1M tokens)
_RAG_EMBED_PRICING: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
}

_RAG_COST_STATE: Dict[str, float] = {
    "embedding_tokens": 0.0,
    "embedding_cost_usd": 0.0,
    "llm_input_tokens": 0.0,
    "llm_output_tokens": 0.0,
    "llm_cost_usd": 0.0,
}


def _estimate_tokens(text: str) -> int:
    """
    Very rough token estimator used only for pricing embeddings.
    Assumes ~4 characters per token.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _add_rag_embedding_usage(model_name: str, tokens: int) -> float:
    """
    Record embedding token usage and return cost for this call.
    """
    if tokens <= 0:
        return 0.0
    price = _RAG_EMBED_PRICING.get(model_name)
    if not price:
        return 0.0
    cost = tokens / 1_000_000.0 * price
    _RAG_COST_STATE["embedding_tokens"] += float(tokens)
    _RAG_COST_STATE["embedding_cost_usd"] += cost
    return cost


