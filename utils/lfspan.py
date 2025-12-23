from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from langchain_community.vectorstores import Chroma

from utils.embedding import OpenAITextEmbeddings
from utils.logging_config import get_logger

logger = get_logger(__name__)

_CACHED_STATE: Dict[str, Any] | None = None

def get_vectordb_state() -> Dict[str, Any]:
    """
    Create or reuse a cached vectordb_state containing:
      - text_embedder
      - text_db

    Environment variables (optional):
      - OPENAI_API_KEY
      - TEXT_DB_DIR
      - RAG_EMBEDDING_MODEL (fallback for embedding_model)
    """
    global _CACHED_STATE
    if _CACHED_STATE is not None:
        return _CACHED_STATE

    # Embeddings (OpenAI)
    model = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
    embedder = OpenAITextEmbeddings(model_name=model, api_key=os.getenv("OPENAI_API_KEY"))

    # Resolve Chroma persistence directory (env → defaults)
    repo_root = Path(__file__).resolve().parents[1]
    default_text_dir = repo_root / "data" / "vectordbs" / "chromadb_food_group"
    text_dir = Path(os.getenv("CHROMA_FOOD_GROUP_DB_DIR", str(default_text_dir)))

    collection_name = os.getenv("TEXT_COLLECTION_NAME", "food_group")

    logger.info("[RAG] Loading Chroma DB persist_directory=%s collection=%s", text_dir, collection_name)
    text_db = Chroma(
        persist_directory=str(text_dir),
        collection_name=collection_name,
        embedding_function=embedder,
    )

    _CACHED_STATE = {
        "text_embedder": embedder,
        "text_db": text_db,
    }
    return _CACHED_STATE