#!/usr/bin/env python3
"""
Semantic Chunking Pipeline for MDC-Challenge-2025
Creates chunks from Document objects using semantic chunking + ChromaDB storage
"""

import os
import sys
from typing import List, Optional
import threading
from sentence_transformers import SentenceTransformer
import numpy as np


# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.helpers import initialize_logging, timer_wrap
from src.semantic_chunking import save_chunk_objs_to_chroma, save_chunks_to_chroma
from src.models import EmbeddingResult
from api.utils.duckdb_utils import get_duckdb_helper
from pathlib import Path

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

DEFAULT_CHROMA_CONFIG = os.path.join(project_root, "configs", "chunking.yaml")

# Default database path
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"

DEFAULT_CACHE_DIR = Path(
    os.path.join(project_root, "offline_models")
)

# Lazy-load embedder to avoid repeated heavy loads
_embedder_lock = threading.Lock()
_EMBEDDER: Optional[SentenceTransformer] = None

def _get_embedder(model_path: str) -> SentenceTransformer:
    global _EMBEDDER
    if _EMBEDDER is None:
        with _embedder_lock:
            if _EMBEDDER is None:
                _EMBEDDER = SentenceTransformer(model_path)
    return _EMBEDDER

@timer_wrap
def embed_text(texts: List[str], model_name: Optional[str] = None, batch_size: int = 100) -> np.ndarray:
    embedder = _get_embedder(str(model_name or DEFAULT_CACHE_DIR))
    embeddings = embedder.encode(texts, convert_to_numpy=True, batch_size=batch_size)
    # Ensure embeddings array is not empty
    if embeddings is None or (isinstance(embeddings, np.ndarray) and embeddings.size == 0):
        logger.error("No embeddings generated for %d texts", len(texts))
        return []
    return embeddings

@timer_wrap
def get_embedding_result(text: str, collection_name: str = None, model_name: Optional[str] = None, save_to_chroma: bool = False) -> EmbeddingResult:
    if save_to_chroma:
        res = save_chunks_to_chroma([text], collection_name, model_name)
        if res:
            return EmbeddingResult(
                success=True,
                embeddings=res.embeddings,
                id=res.id,
                model_name=model_name,
                collection_name=collection_name,
                error=None
            )
        else:
            return EmbeddingResult(
                success=False,
                error="Failed to save to chroma",
                model_name=model_name,
                collection_name=collection_name,
                embeddings=None,
                id=None
            )
    try:
        embeddings = embed_text([text])
    except Exception as e:
        logger.error("Error embedding text: %s", e)
        return EmbeddingResult(
            success=False,
            error=str(e),
            model_name=model_name,
            collection_name=collection_name,
            id=None
        )
    return EmbeddingResult(
        success=True,
        embeddings=embeddings,
        model_name=model_name,
        collection_name=collection_name,
        id=None
    )

def embed_chunks_from_duckdb(db_path: str = DEFAULT_DUCKDB_PATH, collection_name: str = "mdc_training_data", cfg_path: str = DEFAULT_CHROMA_CONFIG):
    db_helper = get_duckdb_helper(db_path)
    chunks = db_helper.get_chunks_by_chunk_ids()
    db_helper.close()
    # save chunk objects to chroma
    return save_chunk_objs_to_chroma(chunks, collection_name, cfg_path)
