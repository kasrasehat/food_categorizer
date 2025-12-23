from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings

from utils.cost import _add_rag_embedding_usage, _estimate_tokens
from utils.logging_config import get_logger


logger = get_logger(__name__)

class OpenAITextEmbeddings:
    """LangChain-style embeddings for text using OpenAI's text-embedding-ada-002."""
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        resolved_key = self._resolve_api_key(explicit_key=api_key)
        # Mask the API key for security
        masked_key = resolved_key[:8] + "..." + resolved_key[-4:] if len(resolved_key) > 12 else resolved_key
        logger.info(f"[OK] OpenAI client initialized with key: {masked_key}")
        self.model_name = model_name
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=resolved_key
        )

    @staticmethod
    def _resolve_api_key(explicit_key: Optional[str] = None) -> str:
        """Resolve OpenAI API key from explicit arg, env vars, or a misformatted .env line.

        Handles cases where .env accidentally splits the key across multiple lines.
        """
        if explicit_key and explicit_key.strip():
            return explicit_key.strip().strip('"').strip("'")

        env_key = os.getenv("OPENAI_API_KEY")
        if env_key and env_key.strip():
            return env_key.strip().strip('"').strip("'")

        # Fallback: try to recover key from a broken .env by concatenating following lines
        try:
            candidate_paths = [
                Path.cwd() / ".env",
                Path(__file__).parent / ".env",
                Path(__file__).parent.parent / ".env",
            ]
            for p in candidate_paths:
                if p.exists():
                    with p.open("r", encoding="utf-8") as f:
                        lines = [ln.rstrip("\n") for ln in f]
                    for i, ln in enumerate(lines):
                        if ln.startswith("OPENAI_API_KEY="):
                            value = ln.split("=", 1)[1].strip()
                            # concatenate subsequent lines that don't contain another KEY=
                            j = i + 1
                            while j < len(lines) and ("=" not in lines[j] or lines[j].startswith("#") or not lines[j].strip()):
                                if lines[j].strip() and not lines[j].lstrip().startswith("#"):
                                    value += lines[j].strip()
                                j += 1
                            value = value.strip().strip('"').strip("'")
                            if value:
                                return value
        except Exception:
            pass

        raise ValueError("OPENAI_API_KEY environment variable is not set. Set it in .env (single line) or pass --openai-api-key.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Estimate token usage for embeddings and track cost
        total_tokens = sum(_estimate_tokens(t) for t in texts)
        cost = _add_rag_embedding_usage(self.model_name, total_tokens)
        if total_tokens > 0:
            logger.info(
                f"[RAG COST] Embeddings model={self.model_name} "
                f"tokens={total_tokens} cost_usd={cost:.6f}"
            )
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        tokens = _estimate_tokens(text)
        cost = _add_rag_embedding_usage(self.model_name, tokens)
        if tokens > 0:
            logger.info(
                f"[RAG COST] Embedding query model={self.model_name} "
                f"tokens={tokens} cost_usd={cost:.6f}"
            )
        return self.embeddings.embed_query(text)