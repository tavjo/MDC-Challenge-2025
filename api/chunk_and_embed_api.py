"""
FastAPI application for the MDC Challenge 2025 Chunking and Embedding Microservice

This module provides a minimal API focused on running the semantic chunking pipeline
with DuckDB integration as specified in the chunk_and_embedding_api.md plan.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import Document, ChunkingResult
from src.run_semantic_chunking import run_semantic_chunking_pipeline
from src.helpers import initialize_logging

# Initialize logging
logger = initialize_logging("chunk_and_embed_api")

# Create FastAPI app
app = FastAPI(
    title="Chunking & Embedding Microservice",
    description="API for running semantic chunking pipeline with DuckDB and ChromaDB integration",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default database path
DEFAULT_DB_PATH = "artifacts/mdc_challenge.db"

@app.post("/run_semantic_chunking", response_model=ChunkingResult)
async def run_pipeline(
    config_path: Optional[str] = Query(None, description="Path to chunking config file"),
    db_path: Optional[str] = Query(DEFAULT_DB_PATH, description="Path to DuckDB database"),
    collection_name: Optional[str] = Query("semantic_chunks", description="ChromaDB collection name"),
    chunk_size: Optional[int] = Query(200, description="Target chunk size in tokens"),
    chunk_overlap: Optional[int] = Query(20, description="Overlap between chunks in tokens"),
    subset: Optional[bool] = Query(False, description="Whether to use a subset of documents"),
    subset_size: Optional[int] = Query(None, description="Size of subset to use")
):
    """
    Trigger the semantic chunking pipeline.
    
    This endpoint runs the complete chunking pipeline using DuckDB for data I/O
    and ChromaDB for embedding storage.
    """
    try:
        logger.info("Starting semantic chunking pipeline via API")
        
        # Build parameters for the pipeline
        pipeline_params = {
            "output_dir": "Data",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection_name": collection_name,
            "cfg_path": config_path or "configs/chunking.yaml",
            "subset": subset,
            "subset_size": subset_size,
            "use_duckdb": True,
            "db_path": db_path
        }
        
        # Run the pipeline
        result = run_semantic_chunking_pipeline(**pipeline_params)
        
        logger.info(f"Pipeline completed with success: {result.success}")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

@app.post("/chunk/documents", response_model=ChunkingResult)
async def chunk_specific_documents(
    documents: List[Document],
    db_path: Optional[str] = Query(DEFAULT_DB_PATH, description="Path to DuckDB database"),
    collection_name: Optional[str] = Query("semantic_chunks", description="ChromaDB collection name"),
    chunk_size: Optional[int] = Query(200, description="Target chunk size in tokens"),
    chunk_overlap: Optional[int] = Query(20, description="Overlap between chunks in tokens")
):
    """
    Process specific documents through the chunking pipeline.
    
    This endpoint temporarily stores the provided documents in DuckDB and then
    runs the chunking pipeline on them.
    """
    try:
        logger.info(f"Processing {len(documents)} specific documents")
        
        # Connect to DuckDB and temporarily store documents
        conn = duckdb.connect(db_path)
        
        # Insert documents temporarily (we'll use INSERT OR REPLACE to handle duplicates)
        for doc in documents:
            doc_row = doc.to_duckdb_row()
            conn.execute("""
                INSERT OR REPLACE INTO documents 
                (doi, has_dataset_citation, full_text, total_char_length, 
                 parsed_timestamp, total_chunks, total_tokens, avg_tokens_per_chunk,
                 file_hash, file_path, citation_entities, n_pages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_row["doi"],
                doc_row["has_dataset_citation"],
                doc_row["full_text"],
                doc_row["total_char_length"],
                doc_row["parsed_timestamp"],
                doc_row["total_chunks"],
                doc_row["total_tokens"],
                doc_row["avg_tokens_per_chunk"],
                doc_row["file_hash"],
                doc_row["file_path"],
                doc_row["citation_entities"],
                doc_row["n_pages"],
                doc_row["created_at"],
                doc_row["updated_at"]
            ))
        
        conn.commit()
        conn.close()
        
        # Run the pipeline on the stored documents
        pipeline_params = {
            "output_dir": "Data",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection_name": collection_name,
            "cfg_path": "configs/chunking.yaml",
            "use_duckdb": True,
            "db_path": db_path
        }
        
        result = run_semantic_chunking_pipeline(**pipeline_params)
        
        logger.info(f"Document processing completed with success: {result.success}")
        return result
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/health")
async def health_check(db_path: Optional[str] = Query(DEFAULT_DB_PATH, description="Path to DuckDB database")):
    """
    Simple health check for the microservice.
    
    Checks DuckDB connection and write permissions, and assumes ChromaDB is working.
    """
    try:
        # Check DuckDB connection and write permissions
        conn = duckdb.connect(db_path)
        
        # Test basic connection
        conn.execute("SELECT 1")
        
        # Test write permissions with a rollback to avoid side effects
        conn.execute("BEGIN TRANSACTION")
        conn.execute("CREATE TABLE IF NOT EXISTS health_check (i INTEGER)")
        conn.execute("INSERT INTO health_check VALUES (1)")
        conn.execute("ROLLBACK")
        
        conn.close()
        duckdb_ok = True
        
    except Exception as e:
        logger.error(f"DuckDB health check failed: {str(e)}")
        duckdb_ok = False
    
    # Assume ChromaDB is working (would need more complex check in production)
    chromadb_ok = True
    
    status = "healthy" if duckdb_ok and chromadb_ok else "unhealthy"
    
    return {
        "status": status,
        "duckdb_connected": duckdb_ok,
        "chromadb_connected": chromadb_ok,
        "embedding_model": "text-embedding-3-small",
        "db_path": db_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 