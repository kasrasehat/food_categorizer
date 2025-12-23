import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import numpy as np
from PIL import Image

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


# Load environment variables from .env file
def load_environment():
    """Load environment variables with robust path resolution."""
    script_dir = Path(__file__).parent

    # Possible .env file locations (in order of preference)
    possible_env_paths = [
        script_dir.parent / ".env",  # Parent directory (main project)
        script_dir / ".env",         # Same directory as script
        Path.cwd() / ".env",         # Current working directory
    ]

    env_loaded = False
    try:
        from dotenv import load_dotenv as _load
    except Exception:
        _load = None

    for env_path in possible_env_paths:
        if env_path.exists() and _load is not None:
            try:
                result = _load(env_path, override=True)
                if result:
                    print(f"✓ Loaded .env from: {env_path}")
                    env_loaded = True
                    break
                else:
                    print(f"⚠️  .env file found at {env_path} but no variables loaded")
            except Exception as e:
                print(f"⚠️  Error loading .env from {env_path}: {e}")
        else:
            print(f"🐛 DEBUG: .env not found at: {env_path}")

    if not env_loaded:
        print("⚠️  No .env file found. Please ensure .env exists with required variables")
        print("   Expected locations:")
        for path in possible_env_paths:
            print(f"   - {path}")

    return env_loaded

# Initialize environment loading
load_environment()


# ---------------- Embedding Interfaces ----------------

class OpenAITextEmbeddings:
    """LangChain-style embeddings for text using OpenAI's text-embedding-ada-002."""
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        resolved_key = self._resolve_api_key(explicit_key=api_key)
        # Mask the API key for security
        masked_key = resolved_key[:8] + "..." + resolved_key[-4:] if len(resolved_key) > 12 else resolved_key
        print(f"✓ OpenAI client initialized with key: {masked_key}")
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
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

# ---------------- Utilities ----------------

def read_json(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSON file that contains a list of dicts (like portion.json).
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure we always return a list of dicts
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # If someone accidentally stored a dict-of-dicts, take the values
        return list(data.values())
    raise ValueError(f"Unsupported JSON structure in {path}: expected list or dict, got {type(data)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks",
        default="F:/my codes py/food_categorizer/data/food_group.json",
        help="Path to chunks JSON file (expects a list or dict-of-dicts, like portion.json).",
    )
    ap.add_argument("--db-dir", default="F:/my codes py/food_categorizer/data/vectordbs/chromadb_food_group", help="Chroma persistence directory.")
    ap.add_argument("--collection-name", default="food_group", help="Override collection name.")
    ap.add_argument("--index-col", default="name", choices=["Yield Name"],
                    help="Which column to index. If omitted, you will be prompted.")
    ap.add_argument("--openai-api-key", default=None, help="OpenAI API key override (optional)")
    args = ap.parse_args()

    index_col = args.index_col
    if not index_col:
        print("Choose index column")

    chunks_path = Path(args.chunks)
    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    rows = read_json(chunks_path)
    print(f"→ Using collection_name = {args.collection_name}")

    embedder = OpenAITextEmbeddings(api_key=args.openai_api_key)
    print(f"→ Using embeddings model: {embedder.embeddings.model}")

     # Build Documents with page_content = column we embed.
    docs: List[Document] = []
    for i, r in enumerate(rows):
        if index_col == "name":
            val = r.get("name")
            
        else:
            val = r.get(index_col) or ""
            val = val.strip()
        if not val:
            continue
        page_content = val

        # Start with all fields from the record and add custom fields
        raw_meta = dict(r)
        raw_meta["_index_col"] = index_col

        # Simplify metadata: keep only simple types; convert lists/dicts to strings
        meta: Dict[str, Any] = {}
        for k, v in raw_meta.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                meta[k] = v
            elif isinstance(v, list):
                try:
                    meta[k] = ", ".join(str(x) for x in v)
                except Exception:
                    continue
            elif isinstance(v, dict):
                try:
                    # Store nested dicts as JSON strings so they are serializable as metadata
                    meta[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    continue
            # Skip other complex types

        doc = Document(page_content=page_content, metadata=meta)
        docs.append(doc)

    if not docs:
        raise SystemExit("No valid documents to index (check your data and chosen index column).")

    # Create / load collection
    collection_name = args.collection_name or f"chunks_{index_col}"
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        persist_directory=str(db_dir),
        collection_name=collection_name,
    )
    vectordb.persist()
    print(f"✅ Built collection '{collection_name}' with {len(docs)} docs at {db_dir}")


if __name__ == "__main__":
    main()