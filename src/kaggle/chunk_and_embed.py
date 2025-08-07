#!/usr/bin/env python3
"""
Semantic Chunking Pipeline for MDC-Challenge-2025
Creates chunks from Document objects using semantic chunking + ChromaDB storage
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Local imports
from src.models import Document, Chunk, ChunkMetadata, ChunkingResult, DocumentChunkingResult, EmbeddingResult, EmbeddingPayload
from src.helpers import initialize_logging, timer_wrap, sliding_window_chunks
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
from api.utils.duckdb_utils import DuckDBHelper
import tiktoken

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "embed_chunks": "/embed_chunks"
}

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------

@timer_wrap
def load_input_data_from_duckdb(db_path: str = "artifacts/mdc_challenge.db") -> List[Document]:
    """
    Load Document instances from DuckDB database.
    
    Args:
        db_path: Path to DuckDB database file
        
    Returns:
        List of Document instances
    """
    logger.info(f"Loading data from DuckDB: {db_path}")
    
    try:
        db_helper = DuckDBHelper(db_path)
        documents = db_helper.get_all_documents()
        db_helper.close()
        
        logger.info(f"✅ Loaded {len(documents)} documents from DuckDB")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Failed to load data from DuckDB: {str(e)}")
        raise


# ---------------------------------------------------------------------------
# Chunk Creation (Using Semantic Chunking)
# ---------------------------------------------------------------------------

@timer_wrap
def create_chunks_from_document(document: Document,
                                chunk_size: int = 300, chunk_overlap: int = 30) -> List[Chunk]:
    """
    Create chunks from document.
    
    Args:
        document: Document object
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of Chunk objects
    """
    logger.info(f"Creating semantic chunks for {document.doi}")
    
    chunks = []
    
    doc = document
    # Standardize text format
    if isinstance(doc.full_text, list):
        text = " ".join(doc.full_text)
    else:
        text = doc.full_text

    # Use semantic chunking with overrides from pipeline
    raw_chunks = sliding_window_chunks(
        text, chunk_size=chunk_size, overlap=chunk_overlap
        )
    # Fallback: ensure no chunk exceeds the hard token limit
    # import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")
    processed_chunks = []
    for chunk_text in raw_chunks:
        token_ids = tok.encode(chunk_text)
        if len(token_ids) <= 1500: 
            processed_chunks.append(chunk_text)
        else:
            logger.warning(f"Chunk too large ({len(token_ids)} tokens); splitting into smaller chunks using sliding window method.")
            processed_chunk = sliding_window_chunks(chunk_text, chunk_size)
            processed_chunks.extend(processed_chunk)
    # Preprocess text on all final chunks
    text_chunks = [pc for pc in processed_chunks]
    
    # Create Chunk objects with metadata
    for i, chunk_text in enumerate(text_chunks):
        chunk_id = f"{doc.doi}_{i}"
        
        # Use tiktoken for accurate token counting
        # import tiktoken
        tok = tiktoken.get_encoding("cl100k_base")
        token_count = len(tok.encode(chunk_text))
        
        chunk_meta = ChunkMetadata(
            chunk_id=chunk_id,
            token_count=token_count,
            citation_entities=None,
            previous_chunk_id=None,  # Will be set by link_adjacent_chunks
            next_chunk_id=None
        )
        
        chunks.append(Chunk(
            chunk_id=chunk_id,
            document_id=doc.doi,
            text=chunk_text,
            chunk_metadata=chunk_meta
        ))
    
    logger.info(f"✅ Created {len(chunks)} chunks from {document.doi}")
    return chunks 

# ---------------------------------------------------------------------------
# Chunk Linking (Adapted from Deprecated Script)
# ---------------------------------------------------------------------------

@timer_wrap
def link_adjacent_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """
    Link adjacent chunks with previous/next chunk IDs.
    
    Args:
        chunks: List of Chunk objects
        
    Returns:
        List of Chunk objects with linked IDs
    """
    logger.info(f"Linking adjacent chunks for {len(chunks)} chunks")
    
    # Group chunks by document
    # by_document = defaultdict(list)
    # for i, chunk in enumerate(chunks):
    #     document_id = chunk.document_id
    #     by_document[document_id].append((i, chunk.text, chunk.chunk_metadata.model_dump()))

    doc_chunks = []
    for i, chunk in enumerate(chunks):
        doc_chunks.append((i, chunk.text, chunk.chunk_metadata.model_dump(), chunk.document_id))
    
    # Link chunks within each document
    linked_chunks = []
    # for document_id, doc_chunks in by_document.items():
    # Sort by chunk position (assuming order from semantic chunking)
    doc_chunks.sort(key=lambda x: x[0])
    
    for i, (original_idx, text, metadata, document_id) in enumerate(doc_chunks):
        # Set previous chunk ID
        if i > 0:
            prev_metadata = doc_chunks[i-1][2]
            metadata['previous_chunk_id'] = prev_metadata['chunk_id']
        
        # Set next chunk ID
        if i < len(doc_chunks) - 1:
            next_metadata = doc_chunks[i+1][2]
            metadata['next_chunk_id'] = next_metadata['chunk_id']
        
        new_chunk = Chunk(
            chunk_id=metadata['chunk_id'],
            document_id=document_id,
            text=text,
            chunk_metadata=ChunkMetadata.model_validate(metadata)
        )
        linked_chunks.append(new_chunk)
    
    logger.info(f"✅ Linked {len(linked_chunks)} chunks")
    return linked_chunks 


# ---------------------------------------------------------------------------
# DuckDB Storage Functions
# ---------------------------------------------------------------------------

@timer_wrap
def save_chunks_to_duckdb(chunks: List[Chunk], db_path: str = "artifacts/mdc_challenge.db") -> bool:
    """
    Save chunks to DuckDB database.
    
    Args:
        chunks: List of Chunk objects to save
        db_path: Path to DuckDB database file
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info(f"Saving {len(chunks)} chunks to DuckDB: {db_path}")
    
    try:
        db_helper = DuckDBHelper(db_path)
        # Use batch upsert for better performance
        if db_helper.store_chunks_batch(chunks):
            logger.info(f"✅ Successfully batch-upserted {len(chunks)} chunks to DuckDB")
            db_helper.close()
            return True
        else:
            logger.error(f"❌ Failed to save chunks to DuckDB")
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to save chunks to DuckDB: {str(e)}")
        return False

# ---------------------------------------------------------------------------
# Export Functions
# ---------------------------------------------------------------------------

@timer_wrap
def export_chunks_to_json(chunks: List[Chunk], output_path: str = "chunks_for_embedding.json") -> str:
    """
    Export chunks to JSON for visual inspection using model_dump() approach.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path for output JSON file
        
    Returns:
        Path to the exported JSON file
    """
    logger.info(f"Exporting {len(chunks)} chunks to JSON: {output_path}")
    
    # Use model_dump() to get JSON-serializable dicts
    to_export = [chunk.model_dump() for chunk in chunks]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_export, f, indent=4, ensure_ascii=False)
    
    logger.info(f"✅ Exported chunks to JSON: {output_path}")
    return output_path

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

# ---------------------------------------------------------------------------
# Main Pipeline Function
# ---------------------------------------------------------------------------

@timer_wrap
def prepare_document(
    document: Document,
    chunk_size: int,
    chunk_overlap: int,
    cfg_path: str,
    collection_name: str,
) -> dict:
    """
    Phase 1: CPU/I-O-light work for one document.
    Returns a dict with document, chunks, stats, and params.
    """
    start = datetime.now().isoformat()

    # 2) Create & link chunks
    chunks = create_chunks_from_document(document, chunk_size, chunk_overlap)
    chunks = link_adjacent_chunks(chunks)
    total_tokens = sum(ck.chunk_metadata.token_count for ck in chunks)
    avg_tokens = total_tokens / len(chunks) if chunks else 0.0

    return {
        "document": document,
        "chunks": chunks,
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens": avg_tokens,
        "pipeline_started_at": start,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "cfg_path": cfg_path,
        "collection_name": collection_name,
    }

def build_document_result(prep: dict) -> Tuple[DocumentChunkingResult, pd.DataFrame]:
    """
    Phase 2: single-threaded persistence. Returns DocumentChunkingResult.
    Rolls back on DuckDB errors; cleans up on ChromaDB errors.
    """
    doc: Document = prep["document"]
    chunks: List[Chunk] = prep["chunks"]
    start = prep["pipeline_started_at"]
    now = datetime.now().isoformat()
    summary_df = create_chunks_summary_csv(chunks, export=False)

    base = dict(
        document_id=doc.doi,
        success=False,
        chunk_size=prep["chunk_size"],
        chunk_overlap=prep["chunk_overlap"],
        cfg_path=prep["cfg_path"],
        collection_name=prep["collection_name"],
        total_chunks=prep["total_chunks"],
        total_tokens=prep["total_tokens"],
        avg_tokens_per_chunk=prep["avg_tokens"],
        pipeline_started_at=start,
        pipeline_completed_at=now,
    )

    # Persistence of chunks is handled in batch Phase 3; no per-document DB or Chroma writes here
    base["success"] = True
    return DocumentChunkingResult.model_validate(base), summary_df


def summarize_run(doc_results: List[DocumentChunkingResult]) -> ChunkingResult:
    """
    Summarize list of DocumentChunkingResult into a single ChunkingResult.
    """
    total_documents = len(doc_results)
    total_chunks = sum(r.total_chunks for r in doc_results)
    total_tokens = sum(r.total_tokens for r in doc_results)
    avg_tokens_per_chunk = total_tokens / total_chunks if total_chunks else 0.0
    error_messages = [r.error for r in doc_results if r.error]
    error = "; ".join(error_messages) if error_messages else None

    return ChunkingResult(
        success=True if error is None else False,
        total_documents=total_documents,
        total_chunks=total_chunks,
        total_tokens=total_tokens,
        avg_tokens_per_chunk=avg_tokens_per_chunk,
        output_path=None,
        output_files=None,
        error=error,
        pipeline_completed_at=datetime.now().isoformat()
    )

def embed_chunks(chunks: List[Chunk], collection_name: str, cfg_path: str, base_api_url: str = API_ENDPOINTS["base_api_url"]) -> EmbeddingResult:
    """
    Embed chunks using the /embed_chunks endpoint.
    """
    import requests
    url = f"{base_api_url}{API_ENDPOINTS['embed_chunks']}"
    # get chunk text
    chunk_texts = [chunk.text for chunk in chunks]
    payload = EmbeddingPayload(
        text=chunk_texts,
        collection_name=collection_name,
        cfg_path=cfg_path,
        local_model=True,
        model_name="BAAI/bge-small-en-v1.5",
        save_to_chroma=True,
        ids=[chunk.chunk_id for chunk in chunks],
        metadata=[{"document_id": chunk.document_id} for chunk in chunks]
    )
    response = requests.post(url, json=payload.model_dump(exclude_none=True))
    if response.status_code == 200:
        return EmbeddingResult.model_validate(response.json())
    else:
        logger.error(f"❌ Failed to embed chunks: {response.status_code} {response.text}")
        return EmbeddingResult(success=False, embeddings=[])


@timer_wrap
def run_val_chunking_pipeline(output_dir: str = "tmp/",
                                 chunk_size: int = 300,
                                 chunk_overlap: int = 30,
                                 collection_name: Optional[str] = None,
                                 cfg_path: str = "configs/chunking.yaml",
                                 subset: bool = False,
                                 subset_size: Optional[int] = None,
                                 use_duckdb: bool = True,
                                 db_path: str = "artifacts/mdc_challenge.db",
                                 max_workers: int = 8,
                                 base_api_url: str = API_ENDPOINTS["base_api_url"],
                                 ) -> ChunkingResult:
    """
    Run the complete semantic chunking pipeline.
    
    Args:
        documents_path: Path to documents JSON file (used when use_duckdb=False)
        citation_entities_path: Path to citation entities JSON file (used when use_duckdb=False)
        output_dir: Directory for output files
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        collection_name: ChromaDB collection name
        cfg_path: Path to chunking configuration file
        subset: Whether to use a subset of documents
        subset_size: Size of subset to use
        use_duckdb: Whether to use DuckDB for data I/O (default: True)
        db_path: Path to DuckDB database file
        
    Returns:
        ChunkingResult object with pipeline results
    """
    # 1) Load input data
    if use_duckdb:
        docs = load_input_data_from_duckdb(db_path)
    
    if subset:
        if subset_size is None:
            subset_size = 20
            logger.warning(f"No subset size provided using default size of {subset_size}.")
        np.random.seed(42)
        # choose randomly from docs
        logger.info(f"Choosing {subset_size} documents randomly from {len(docs)} documents.")
        docs = np.random.choice(docs, size=subset_size, replace=False)
        if subset_size > len(docs):
            logger.warning(f"Requested {subset_size} docs but only {len(docs)} documents available; sampling all.")
            subset_size = len(docs)


    # 2) Phase 1: document preparation
    phase1_start = time.time()
    logger.info(f"Phase 1: starting preparation of {len(docs)} docs (threshold {max_workers})")
    prepped = []
    summary_dfs = []
    if len(docs) < max_workers:
        logger.info(f"Phase 1: using ThreadPoolExecutor with {min(max_workers, len(docs))} threads for {len(docs)} docs")
        with ThreadPoolExecutor(max_workers=min(max_workers, len(docs))) as exe:
            futures = {
                exe.submit(prepare_document, doc, chunk_size, chunk_overlap, cfg_path, collection_name): doc
                for doc in docs
            }
            for fut in as_completed(futures):
                try:
                    prepped.append(fut.result())
                except Exception as e:
                    doc = futures[fut]
                    prepped.append({
                        "document": doc,
                        "chunks": [],
                        "total_chunks": 0,
                        "total_tokens": 0,
                        "avg_tokens": 0,
                        "pipeline_started_at": datetime.now().isoformat(),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "cfg_path": cfg_path,
                        "collection_name": collection_name,
                        "prep_error": str(e)
                    })
    else:
        logger.info(f"Phase 1: using ProcessPoolExecutor with {max_workers} processes for {len(docs)} docs")
        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {
                exe.submit(prepare_document, doc, chunk_size, chunk_overlap, cfg_path, collection_name): doc
                for doc in docs
            }
            for fut in as_completed(futures):
                try:
                    prepped.append(fut.result())
                except Exception as e:
                    doc = futures[fut]
                    prepped.append({
                        "document": doc,
                        "chunks": [],
                        "total_chunks": 0,
                        "total_tokens": 0,
                        "avg_tokens": 0,
                        "pipeline_started_at": datetime.now().isoformat(),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "cfg_path": cfg_path,
                        "collection_name": collection_name,
                        "prep_error": str(e)
                    })
    logger.info(f"Phase 1 complete in {time.time() - phase1_start:.2f}s")

    # 3) Phase 2: build document results
    phase2_start = time.time()
    logger.info("Phase 2: building document results and summaries")
    doc_results: List[DocumentChunkingResult] = []
    for prep in prepped:
        if "prep_error" in prep:
            doc_results.append(DocumentChunkingResult(
                document_id=prep["document"].doi,
                success=False,
                error=prep["prep_error"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                cfg_path=cfg_path,
                collection_name=collection_name,
                total_chunks=0,
                total_tokens=0,
                avg_tokens_per_chunk=0.0,
                pipeline_started_at=prep["pipeline_started_at"],
                pipeline_completed_at=datetime.now().isoformat()
            ))
        else:
            doc_to_commit, summary_df = build_document_result(prep)
            doc_results.append(doc_to_commit)
            summary_dfs.append(summary_df)
    logger.info(f"Phase 2 complete in {time.time() - phase2_start:.2f}s")
    if len(summary_dfs) > 0:
        summary = pd.concat(summary_dfs)
        summary.to_csv(Path(os.path.join(project_root, output_dir, "chunks_for_embedding_summary.csv")), index=False)
    else:
        logger.warning("No summary data to export.")

    # 4) Phase 3: batch persistence of validated chunks
    phase3_start = time.time()
    logger.info("Phase 3: batch persistence starting")
    valid_chunks: List[Chunk] = []
    for prep in prepped:
        if prep.get("validation_passed"):
            valid_chunks.extend(prep["chunks"])
    logger.info(f"Batch storing {len(valid_chunks)} validated chunks to DuckDB")
    save_chunks_to_duckdb(valid_chunks, db_path)
    logger.info(f"Batch storing {len(valid_chunks)} validated chunks to ChromaDB")
    if collection_name is None:
        collection_name = "mdc_val_data"
    embed_chunks(valid_chunks, collection_name=collection_name, cfg_path=cfg_path, base_api_url=base_api_url)
    logger.info(f"Phase 3 complete in {time.time() - phase3_start:.2f}s")
    
    final_res = summarize_run(doc_results)

    # 5) Summarize into one ChunkingResult
    return final_res

if __name__ == "__main__":
    # Run with default parameters
    results = run_val_chunking_pipeline(subset=False, use_duckdb=True, db_path=str(project_root / "artifacts" / "mdc_challenge.db"), cfg_path=str(project_root / "configs" / "chunking.yaml"), collection_name="mdc_val_data")
    # Prepare output directory under project root
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "chunk_embed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "chunking_results.json")
    if results.success:
        logger.info("✅ Semantic chunking completed successfully!")
        logger.info(f"Chunking results: {results.model_dump()}")
        # save results to a json file in reports/chunk_embed
        # with open(output_file, "w") as f:
        #     json.dump(results.model_dump(), f)
    else:
        logger.error(f"❌ Pipeline failed: {results.error}")
        logger.error(f"Chunking results: {results.model_dump()}")
        # # save results to a json file in reports/chunk_embed
        # with open(output_file, "w") as f:
        #     json.dump(results.model_dump(), f) 