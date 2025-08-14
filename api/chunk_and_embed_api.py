"""
FastAPI application for the MDC Challenge 2025 Chunking and Embedding Microservice

This module provides a minimal API focused on running the semantic chunking pipeline
with DuckDB integration as specified in the chunk_and_embedding_api.md plan.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import (
    ChunkingResult, EmbeddingResult, ChunkingPipelinePayload, RetrievalPayload, BatchRetrievalResult, EmbeddingPayload, DatasetConstructionResult, DatasetConstructionPayload, NeighborhoodStatsPayload,
    LoadChromaDataPayload, LoadChromaDataResult,
    ValRetrievalPayload, BulkParseRequest
    )
from api.services.chunking_and_embedding_services import run_semantic_chunking_pipeline
from api.services.retriever_services import batch_retrieve_top_chunks, load_embeddings, batch_retrieve_top_chunks_val
from src.helpers import initialize_logging, timer_wrap, export_docs
from src.semantic_chunking import sliding_window_chunk_text
# from api.utils.duckdb_utils import get_duckdb_helper
from api.services.embeddings_services import get_embedding_result
from api.services.dataset_construction_service import construct_datasets_from_retrieval_results
from api.services.neighborhood_stats import run_neighborhood_stats_pipeline as run_neighborhood_stats_pipeline_service
from api.utils.duckdb_utils import DuckDBHelper
# from api.services.document_parsing_service import build_document_objects, build_document_object

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

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
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
# Remove module-level helper instantiation to defer until requests
# DUCKDB_HELPER = get_duckdb_helper(DEFAULT_DUCKDB_PATH)
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"

@app.post("/load_embeddings", response_model=LoadChromaDataResult, response_model_exclude_none=True)
async def load_chroma_data(payload: LoadChromaDataPayload) -> LoadChromaDataResult:
    """
    Load embeddings with associated metadata and text from ChromaDB
    """
    try:
        params = {
            "collection_name": payload.collection_name,
            "cfg_path": payload.cfg_path or DEFAULT_CHROMA_CONFIG,
            "include": payload.include or ["embeddings"]
        }
        return load_embeddings(**params)
    except Exception as e:
        logger.error(f"Load embeddings failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Load embeddings failed: {str(e)}")

@app.post("/run_neighborhood_stats_pipeline", response_model=bool)
async def run_neighborhood_stats_pipeline(payload: NeighborhoodStatsPayload) -> bool:
    """
    Run the neighborhood stats pipeline.
    """
    logger.info("Running neighborhood stats pipeline")
    pipeline_params = {
        "db_path": payload.db_path or DEFAULT_DUCKDB_PATH,
        "collection_name": payload.collection_name or "dataset-aggregates-train",
        "k": payload.k,
        "cfg_path": payload.cfg_path or DEFAULT_CHROMA_CONFIG,
        "max_workers": payload.max_workers or 1
    }
    try:
        return run_neighborhood_stats_pipeline_service(**pipeline_params)
    except Exception as e:
        logger.error(f"Neighborhood stats pipeline failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Neighborhood stats pipeline failed: {str(e)}")


@app.post("/construct_datasets", response_model=DatasetConstructionResult)
async def construct_datasets(payload: DatasetConstructionPayload) -> DatasetConstructionResult:
    """
    Construct datasets from retrieval results.
    """
    params = {
        "retrieval_results_path": payload.retrieval_results_path,
        "db_path": payload.db_path or DEFAULT_DUCKDB_PATH,
        "mask_token": payload.mask_token
    }
    try:
        logger.info(f"Constructing datasets from retrieval results.")
        return construct_datasets_from_retrieval_results(**params)
    except Exception as e:
        logger.error(f"Dataset construction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Dataset construction failed: {str(e)}")

@app.post("/create_chunks", response_model=List[str])
async def create_chunks(
    text: str,
    cfg_path: Optional[str] = None,
    model_name: Optional[str] = None,
    chunk_size: Optional[int] = 300,
    chunk_overlap: Optional[int] = 20
):
    """
    Create chunks from a document.
    """
    try:
        logger.info(f"Creating chunk from text.")
        if cfg_path is None:
            cfg_path = DEFAULT_CHROMA_CONFIG
        if model_name is None:
            chunks = sliding_window_chunk_text(text, cfg_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            chunks = sliding_window_chunk_text(text, cfg_path,chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Chunk creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chunk creation failed: {str(e)}")

@app.post("/batch_create_chunks", response_model=Dict[str, List[str]])
async def batch_create_chunks(
    texts: List[str],
    cfg_path: Optional[str] = None,
    chunk_size: Optional[int] = 200,
    chunk_overlap: Optional[int] = 20
) -> Dict[str, List[str]]:
    """
    Create chunks for a list of texts/documents.
    """
    try:
        logger.info(f"Creating chunks for {len(texts)} texts.")
        chunks = {}
        for i, text in enumerate(texts):
            chunks[i] = (sliding_window_chunk_text(text, cfg_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Chunk creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chunk creation failed: {str(e)}")

@app.post("/embed_chunks", response_model=EmbeddingResult)
async def embed_chunks(payload: EmbeddingPayload) -> EmbeddingResult:
    """
    Create embeddings for chunks.
    """
    embed_params = {
        "texts": payload.text,
        "cfg_path": payload.cfg_path or DEFAULT_CHROMA_CONFIG,
        "model_name": payload.model_name,
        "collection_name": payload.collection_name,
        "save_to_chroma": payload.save_to_chroma,
        "ids": payload.ids,
        "metadata": payload.metadata
    }
    try:
        logger.info(f"Creating embeddings for {payload.text[:10]}...")
        response = get_embedding_result(**embed_params)
        return response
    except Exception as e:
        logger.error(f"Embeddings failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embeddings failed: {str(e)}")
    
# New Batch Retrieval endpoint
@app.post("/batch_retrieve_val", response_model=BatchRetrievalResult)
async def batch_retrieve_val(payload: ValRetrievalPayload) -> BatchRetrievalResult:
    """
    Batch retrieve top-k chunks for multiple keys.
    """
    try:
        logger.info(f"Starting batch retrieval for {len(payload.doc_ids)} documents")
        return batch_retrieve_top_chunks_val(
            query_embeddings=payload.query_embeddings,
            max_workers=payload.max_workers or 1,
            collection_name=payload.collection_name or "mdc_val_data",
            k=payload.k,
            cfg_path=payload.cfg_path or DEFAULT_CHROMA_CONFIG,
            symbolic_boost=payload.symbolic_boost or 0.15,
            use_fusion_scoring=payload.use_fusion_scoring or True,
            analyze_chunk_text=payload.analyze_chunk_text or False,
            doc_ids=payload.doc_ids or []
        )
    except Exception as e:
        logger.error("Batch retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# New Batch Retrieval endpoint
@app.post("/batch_retrieve", response_model=BatchRetrievalResult)
async def batch_retrieve(payload: RetrievalPayload) -> BatchRetrievalResult:
    """
    Batch retrieve top-k chunks for multiple keys.
    """
    try:
        logger.info(f"Starting batch retrieval for {len(payload.query_texts)} keys")
        return batch_retrieve_top_chunks(
            query_texts=payload.query_texts,
            max_workers=payload.max_workers or 1,
            collection_name=payload.collection_name or "mdc_training_data",
            k=payload.k,
            cfg_path=payload.cfg_path or DEFAULT_CHROMA_CONFIG,
            symbolic_boost=payload.symbolic_boost or 0.15,
            use_fusion_scoring=payload.use_fusion_scoring or True,
            analyze_chunk_text=payload.analyze_chunk_text or False,
            doc_id_map=payload.doc_id_map or {}
        )
    except Exception as e:
        logger.error("Batch retrieval failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_semantic_chunking", response_model=ChunkingResult)
async def run_pipeline(
    payload: ChunkingPipelinePayload
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
            # Use provided output_dir or default to 'Data'
            "output_dir": payload.output_dir or "Data",
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
            "collection_name": payload.collection_name or "mdc_training_data",
            "cfg_path": payload.cfg_path or DEFAULT_CHROMA_CONFIG,
            "subset": payload.subset,
            "subset_size": payload.subset_size,
            "db_path": payload.db_path or DEFAULT_DUCKDB_PATH,
        }
        # Pass through max_workers
        pipeline_params["max_workers"] = payload.max_workers or 8

        # Run the pipeline
        result = run_semantic_chunking_pipeline(**pipeline_params)
        
        logger.info(f"Pipeline completed with success: {result.success}")
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

# # Parse a single document
# @app.get("/parse_doc")
# async def parse_doc(pdf_path: str):
#     helper = DuckDBHelper(DEFAULT_DUCKDB_PATH)
#     try:
#         document = build_document_object(pdf_path=pdf_path)
#         helper.store_document(document)
#         return {"message": "Document parsed and stored successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         helper.close()

# Parse multiple documents at once
# @app.post("/bulk_parse_docs")
# async def parse_docs(payload: BulkParseRequest):
#     pdf_paths = payload.pdf_paths
#     export_file = payload.export_file
#     export_path = payload.export_path
#     subset = payload.subset
#     subset_size = payload.subset_size
#     max_workers = payload.max_workers
#     params = {
#         "pdf_paths": pdf_paths,
#         "subset": subset,
#         "subset_size": subset_size,
#         "max_workers": max_workers
#     }
#     helper = DuckDBHelper(DEFAULT_DUCKDB_PATH)
#     try:
#         documents = build_document_objects(**params)
#         if not documents:
#             logger.error(f"No document objects built: {pdf_paths}")
#             raise HTTPException(status_code=400, detail="No document objects built")
#         success = helper.batch_upsert_documents(documents)
#         if not success:
#             logger.error(f"Failed to store documents: {documents}")
#             raise HTTPException(status_code=500, detail="Failed to store documents")
#         # also export as a json file
#         if export_file:
#             export_docs(documents, output_file=export_file, output_dir=export_path)
#         elif export_path:
#             export_docs(documents, output_dir=export_path)
#         elif export_file and export_path:
#             export_docs(documents, output_file=export_file, output_dir=export_path)
#         else:
#             export_docs(documents)
#         return {"message": f"{len(documents)} Documents parsed and stored successfully"}
#     except Exception as e:
#         logger.error(f"Error parsing documents: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
#     finally:
#         helper.close()


@app.get("/health")
async def health_check(db_path: Optional[str] = Query(DEFAULT_DUCKDB_PATH, description="Path to DuckDB database"), model_name: Optional[str] = Query("bge-small-en-v1.5", description="Model name")):
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
        "embedding_model": model_name,
        "db_path": db_path
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 