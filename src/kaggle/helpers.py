# src/kaggle/helpers.py

"""
Helper functions for the Kaggle competition.
"""

from typing import List
import os, sys
import time
import inspect
from functools import wraps
import logging
import numpy as np
from pathlib import Path
import hashlib

root_dir = Path(__file__).parent
print(root_dir)
sys.path.append(str(root_dir))

from .models import CitationEntity
from .baml_wrapper import extract_cites

logger = logging.getLogger(__name__)

def initialize_logging(filename: str) -> logging.Logger:
    """
    Initialize logging for a given filename.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    return logger

def ensure_dir(path) -> Path:
    """
    Create a directory if it doesn't exist. Returns a Path.
    Expands '~' and env vars.
    """
    p = Path(os.path.expanduser(os.path.expandvars(str(path))))
    p.mkdir(parents=True, exist_ok=True)
    return p

def ensure_dirs(*paths) -> list[Path]:
    """
    Create multiple directories if they don't exist. Returns list[Path].
    """
    return [ensure_dir(p) for p in paths]

def timer_wrap(func):
    if inspect.iscoroutinefunction(func):
        # Handle async functions
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__}...")
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
            return result
        return async_wrapper
    elif inspect.isasyncgenfunction(func):
        # Handle async generator functions
        @wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__} (async generator)...")
            start_time = time.time()
            async for item in func(*args, **kwargs):
                yield item
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Async generator {func.__name__} took {elapsed_time:.4f} seconds to complete.")
        return async_gen_wrapper
    else:
        # Handle sync functions
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            print(f"Executing {func.__name__}...")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function {func.__name__} took {elapsed_time:.4f} seconds to complete.")
            return result
        return sync_wrapper

def num_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    import tiktoken
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def compute_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of XML file for debugging reference."""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return ""


def sliding_window_chunks(text: str, window_size: int = 300, overlap: int = 30) -> List[str]:
    """
    Split the input text into sliding window chunks based on word count.
    """
    logger.info(f"Creating chunks with window size {window_size} and overlap {overlap}")
    # Normalize whitespace and split into words
    words = text.replace('\n', ' ').split()
    chunks = []
    start = 0
    total_words = len(words)
    # Create chunks with specified overlap
    while start < total_words:
        end = min(start + window_size, total_words)
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Move start by window_size minus overlap
        start += window_size - overlap
    # Merge any chunks smaller than twice the overlap into the previous chunk
    min_size = overlap * 2
    refined_chunks: List[str] = []
    for ch in chunks:
        word_count = len(ch.split())
        if refined_chunks and word_count < min_size:
            refined_chunks[-1] += " " + ch
        else:
            refined_chunks.append(ch)
    chunks = refined_chunks
    logger.info(f"Successfully created {len(chunks)} chunks after merging small fragments")
    return chunks

def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between A [N,D] and B [M,D]."""
    A = A.astype(np.float32, copy=False)
    B = B.astype(np.float32, copy=False)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T  # [N, M]

# --- Embedding with local BGE-small ---
def load_bge_model(local_dir: str | Path):
    """
    Load a local, pre-downloaded BGE-small v1.5 SentenceTransformers model (offline-friendly).
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(local_dir))  # local path works offline
    return model

def embed_texts(model, texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode texts with L2-normalized embeddings for cosine similarity.
    """
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,  # recommended for BGE
        show_progress_bar=True,
    )
    return emb

def extract_entities_baml(doc: List[str], doc_id: str) -> List[CitationEntity]:
    """
    Extract citation entities from the document text using the BAML client.
    """
    logger.info(f"Extracting citation entities using BAML client for {doc_id}.")
    citations = extract_cites(doc)
    if citations:
        entities = [{
            "data_citation": entity.data_citation,
            "document_id": doc_id,
            "evidence": entity.evidence,
        } for entity in citations]
        citation_entities = [CitationEntity.model_validate(entity) for entity in entities]
        return citation_entities
    else:
        logger.warning("No citation entities found using BAML client.")
        return []