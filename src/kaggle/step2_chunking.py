"""
Sliding-window chunking utilities for Kaggle.

Exports a minimal set of functions used by `get_citation_context.py` and the
dataset construction pipeline.
"""

from __future__ import annotations

from typing import List
import logging, os
import pandas as pd
from pathlib import Path
import sys

# Allow importing sibling kaggle helpers/models when used as a standalone script
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

try:
    from src.kaggle.models import Document, Chunk, ChunkMetadata
    from src.kaggle.helpers import sliding_window_chunks, num_tokens
    from src.kaggle.duckdb import get_duckdb_helper
except Exception:
    from .models import Document, Chunk, ChunkMetadata  # type: ignore
    from .helpers import sliding_window_chunks, num_tokens, timer_wrap  # type: ignore
    from .duckdb import get_duckdb_helper  # type: ignore

logger = logging.getLogger(__name__)

@timer_wrap
def create_chunks_from_document(
    document: Document,
    chunk_size: int = 300,
    overlap: int = 30,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    doc = document

    if isinstance(doc.full_text, list):
        text = "\n\n".join(doc.full_text)
    else:
        text = doc.full_text
    parts = sliding_window_chunks(text, window_size=chunk_size, overlap=overlap)
    processed_chunks = []
    for part in parts:
        # TODO: This messes somewhat with the ordering of chunks since the ones added to the list if they are too large will not be in order.
        if num_tokens(part) <= 1500:
            processed_chunks.append(part)
        else:
            processed_chunks.extend(sliding_window_chunks(part, chunk_size, overlap))
    parts = processed_chunks

    previous_id = None
    for idx, part in enumerate(parts):
        chunk_id = f"{doc.doi}_chunk_{idx:05d}"
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            previous_chunk_id=previous_id,
            next_chunk_id=None,
            token_count=num_tokens(part),
            citation_entities=None,
        )
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=doc.doi,
            text=part,
            score=None,
            chunk_metadata=metadata,
        )
        chunks.append(chunk)
        if previous_id is not None:
            # fill next on previous
            chunks[-2].chunk_metadata.next_chunk_id = chunk_id
        previous_id = chunk_id
    return chunks

@timer_wrap
def create_chunks_summary_csv(chunks: List[Chunk], export: bool = True, output_path: str = "chunks_for_embedding_summary.csv") -> str:
    """
    Export chunk summary statistics to CSV.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path for output CSV file
        
    Returns:
        Path to the exported CSV file
    """
    logger.info(f"Exporting chunk summary to CSV: {output_path}")
    
    summary_rows = []
    for chunk in chunks:
        row = chunk.chunk_metadata.model_dump()
        row['text_length'] = len(chunk.text)
        row['text_preview'] = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    if export:
        summary_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Exported chunk summary to CSV: {output_path}")
    return summary_df 

@timer_wrap
def chunk_documents(chunk_size: int = 300, overlap: int = 30, db_path: str | None = None, output_dir: str = "kaggle/temp") -> List[Chunk]:
    db_helper = get_duckdb_helper(db_path)
    documents = db_helper.get_all_documents()
    chunks = []
    for doc in documents: # TODO: implement optional parallel processing
        chunks.extend(create_chunks_from_document(doc, chunk_size, overlap)) # TODO: implement optional parallel processing
    db_helper.bulk_insert_chunks(chunks)
    db_helper.close()
    # generate summary csv
    filename = os.path.join(output_dir, "chunks_for_embedding_summary.csv")
    create_chunks_summary_csv(chunks, export=True, output_path=filename)
    return chunks



__all__ = [
    "create_chunks_from_document",
    "create_chunks_summary_csv",
    "chunk_documents"
]


