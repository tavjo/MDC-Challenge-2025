#!/usr/bin/env python3
"""
Semantic Chunking Pipeline for MDC-Challenge-2025
Creates chunks from Document objects using semantic chunking + ChromaDB storage
"""

import json
import os
import sys
# from pathlib import Path
from typing import Optional

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import ChunkingResult, ChunkingPipelinePayload
from src.helpers import initialize_logging, timer_wrap
import requests

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "create_chunks": "/create_chunks",
    "batch_create_chunks": "/batch_create_chunks",
    "embed_chunk": "/embed_chunk",
    "embed_chunks": "/embed_chunks",
    "run_semantic_chunking": "/run_semantic_chunking",
    "chunk_specific_documents": "/chunk/documents"
}

# Default database path
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"

class SemanticChunkingPipeline:
    def __init__(self,
                 base_api_url: str = API_ENDPOINTS["base_api_url"], 
                 subset: bool = False,
                 subset_size: Optional[int] = None,
                 use_duckdb: bool = True, 
                 db_path: Optional[str] = None, 
                 cfg_path: Optional[str] = None,
                 collection_name: Optional[str] = None,
                 chunk_size: Optional[int] = 300,
                 chunk_overlap: Optional[int] = 2,
                 local_model: Optional[bool] = False,
                 max_workers: int = 8):
        self.subset = subset
        self.subset_size = subset_size
        # if self.subset:
        #     self.subset_size = 5 if not self.subset_size else self.subset_size
        self.use_duckdb = use_duckdb
        self.db_path = db_path if db_path is not None else None
        self.cfg_path = cfg_path if cfg_path is not None else None
        self.collection_name = collection_name if collection_name is not None else None
        self.chunk_size = chunk_size if chunk_size is not None else None
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else None
        self.local_model = local_model if local_model is not None else False
        self.max_workers = max_workers
        self.base_api_url = base_api_url
        self.endpoints = API_ENDPOINTS

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_api_url}{self.endpoints[endpoint]}"
    
    def _construct_payload(self) -> ChunkingPipelinePayload:
        return ChunkingPipelinePayload(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            collection_name=self.collection_name,
            cfg_path=self.cfg_path,
            db_path=self.db_path,
            subset=self.subset,
            subset_size=self.subset_size,
            local_model=self.local_model
            ,max_workers=self.max_workers
        )
    def run_pipeline(self) -> ChunkingResult:
        # let's the do the simplest iteration for now
        payload = self._construct_payload()
        full_url = self._get_api_url("run_semantic_chunking")
        logger.info(f"Running semantic chunking pipeline with full url: {full_url}")
        response = requests.post(full_url, json=payload.model_dump(exclude_none=True))
        if response.status_code == 200:
            return ChunkingResult.model_validate(response.json())
        else:
            raise Exception(f"Pipeline failed: {response.text}")
    
@timer_wrap
def main():
    # initialize the pipeline
    semantic_chunker = SemanticChunkingPipeline(
        subset = False,
        cfg_path = DEFAULT_CHROMA_CONFIG,
        db_path = DEFAULT_DUCKDB_PATH,
        collection_name = "mdc_training_data"
    )

    # Run with default parameters
    results = semantic_chunker.run_pipeline()
    
    if results.success:
        logger.info("✅ Semantic chunking completed successfully!")
        # save results to a json file
        with open("chunking_results.json", "w") as f:
            json.dump(results.model_dump(), f)
    else:
        logger.error(f"❌ Pipeline failed: {results.error}")
        # save results to a json file
        with open("chunking_results.json", "w") as f:
            json.dump(results.model_dump(), f) 

if __name__ == "__main__":
    main()