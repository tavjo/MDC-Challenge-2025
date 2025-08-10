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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

logger = logging.getLogger(__name__)


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