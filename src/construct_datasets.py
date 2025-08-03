#!/usr/bin/env python3
"""
Dataset Construction Pipeline for MDC-Challenge-2025
Constructs Dataset objects from retrieval results and creates embeddings for them
"""

import json
import os
import sys
from typing import Optional, List
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import DatasetConstructionPayload, DatasetConstructionResult, EmbeddingPayload, EmbeddingResult
from src.helpers import initialize_logging, timer_wrap
import requests

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "construct_datasets": "/construct_datasets",
    "embed_chunks": "/embed_chunks"
}

# Default paths and configurations
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_RETRIEVAL_RESULTS_PATH = "reports/retrieval/retrieval_results.json"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"


class DatasetConstructionPipeline:
    def __init__(self,
                 base_api_url: str = API_ENDPOINTS["base_api_url"],
                 retrieval_results_path: str = DEFAULT_RETRIEVAL_RESULTS_PATH,
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 mask_token: str = "<DATASET_ID>",
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 cfg_path: Optional[str] = None,
                 model_name: Optional[str] = None,
                 local_model: bool = False):
        self.base_api_url = base_api_url
        self.retrieval_results_path = retrieval_results_path
        self.db_path = db_path
        self.mask_token = mask_token
        self.collection_name = collection_name
        self.cfg_path = cfg_path
        self.model_name = model_name
        self.local_model = local_model
        self.endpoints = API_ENDPOINTS

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_api_url}{self.endpoints[endpoint]}"

    def _construct_dataset_payload(self) -> DatasetConstructionPayload:
        return DatasetConstructionPayload(
            retrieval_results_path=self.retrieval_results_path,
            db_path=self.db_path,
            mask_token=self.mask_token
        )

    def _get_dataset_texts_from_db(self) -> tuple[List[str], List[str], List[str]]:
        """
        Retrieve the masked text and document_ids from constructed datasets in the database.
        Returns: (texts, dataset_ids, document_ids)
        """
        try:
            from api.utils.duckdb_utils import DuckDBHelper
            
            # Query the database for constructed datasets
            db_helper = DuckDBHelper(self.db_path)
            
            try:
                # Query datasets table to get text, dataset_id, and doc_id
                result = db_helper.engine.execute("SELECT dataset_id, document_id, text FROM datasets")
                rows = result.fetchall()
                
                if not rows:
                    logger.warning("No datasets found in database")
                    return [], [], []
                
                dataset_ids = []
                document_ids = []
                texts = []
                
                for row in rows:
                    dataset_id, document_id, text = row
                    dataset_ids.append(dataset_id)
                    document_ids.append(document_id)
                    texts.append(text)
                
                logger.info(f"Retrieved {len(texts)} dataset texts from database")
                return texts, dataset_ids, document_ids
                
            finally:
                db_helper.close()
            
        except Exception as e:
            logger.error(f"Failed to get dataset texts from database: {str(e)}")
            return [], [], []

    def _construct_embedding_payload(self, texts: List[str], dataset_ids: List[str], document_ids: List[str]) -> EmbeddingPayload:
        return EmbeddingPayload(
            text=texts,
            collection_name=self.collection_name,
            cfg_path=self.cfg_path,
            model_name=self.model_name,
            local_model=self.local_model,
            save_to_chroma=True,  # As specified by user
            ids=dataset_ids,  # dataset_id will be the ChromaDB ID
            metadata=[{"document_id": doc_id} for doc_id in document_ids]
        )

    def construct_datasets(self) -> DatasetConstructionResult:
        """Call the /construct_datasets endpoint."""
        payload = self._construct_dataset_payload()
        full_url = self._get_api_url("construct_datasets")
        
        logger.info(f"Constructing datasets with URL: {full_url}")
        logger.info(f"Using retrieval results from: {self.retrieval_results_path}")
        
        try:
            response = requests.post(full_url, json=payload.model_dump(exclude_none=True))
            if response.status_code == 200:
                result = DatasetConstructionResult.model_validate(response.json())
                logger.info("✅ Dataset construction completed successfully!")
                return result
            else:
                logger.error(f"Dataset construction failed with status {response.status_code}: {response.text}")
                return DatasetConstructionResult(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Dataset construction request failed: {str(e)}")
            return DatasetConstructionResult(
                success=False,
                error=str(e)
            )

    def embed_datasets(self, texts: List[str], dataset_ids: List[str], document_ids: List[str]) -> EmbeddingResult:
        """Call the /embed_chunks endpoint to create embeddings for dataset texts."""
        if not texts:
            logger.warning("No texts to embed")
            return EmbeddingResult(success=True, embeddings=[])
        
        payload = self._construct_embedding_payload(texts, dataset_ids, document_ids)
        full_url = self._get_api_url("embed_chunks")
        
        logger.info(f"Creating embeddings with URL: {full_url}")
        logger.info(f"Embedding {len(texts)} dataset texts to collection: {self.collection_name}")
        
        try:
            response = requests.post(full_url, json=payload.model_dump(exclude_none=True))
            if response.status_code == 200:
                result = EmbeddingResult.model_validate(response.json())
                logger.info("✅ Dataset embedding completed successfully!")
                return result
            else:
                logger.error(f"Dataset embedding failed with status {response.status_code}: {response.text}")
                return EmbeddingResult(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Dataset embedding request failed: {str(e)}")
            return EmbeddingResult(
                success=False,
                error=str(e)
            )

    def run_pipeline(self) -> dict:
        """Run the complete dataset construction and embedding pipeline."""
        results = {
            "dataset_construction": None,
            "embedding": None,
            "overall_success": False
        }
        
        # Step 1: Construct datasets
        logger.info("🚀 Starting dataset construction pipeline...")
        construction_result = self.construct_datasets()
        results["dataset_construction"] = construction_result.model_dump()
        
        if not construction_result.success:
            logger.error("❌ Dataset construction failed, skipping embedding step")
            return results
        
        # Step 2: Get dataset texts for embedding
        logger.info("📝 Retrieving dataset texts for embedding...")
        texts, dataset_ids, document_ids = self._get_dataset_texts_from_db()
        
        if not texts:
            logger.warning("⚠️ No dataset texts found for embedding")
            results["overall_success"] = construction_result.success
            return results
        
        # Step 3: Create embeddings
        logger.info("🔄 Creating embeddings for dataset texts...")
        embedding_result = self.embed_datasets(texts, dataset_ids, document_ids)
        # results["embedding"] = embedding_result.model_dump()
        
        # Overall success if both steps succeeded
        results["overall_success"] = construction_result.success and embedding_result.success
        
        if results["overall_success"]:
            logger.info("✅ Complete pipeline finished successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        return results


@timer_wrap
def main():
    """Main function to run the dataset construction pipeline."""
    
    # Initialize the pipeline with default parameters
    logger.info("Initializing dataset construction pipeline...")
    
    pipeline = DatasetConstructionPipeline(
        retrieval_results_path=DEFAULT_RETRIEVAL_RESULTS_PATH,
        db_path=DEFAULT_DUCKDB_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
        cfg_path=DEFAULT_CHROMA_CONFIG,
        local_model=True
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Prepare output directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "dataset_construction")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dataset_construction_results.json")
    
    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary logging
    if results["overall_success"]:
        logger.info("🎉 Dataset construction and embedding pipeline completed successfully!")
    else:
        logger.error("💥 Pipeline failed - check the logs and results file for details")
        
        # Log specific failures
        if results["dataset_construction"] and not results["dataset_construction"]["success"]:
            error = results["dataset_construction"].get("error", "Unknown error")
            logger.error(f"Dataset construction error: {error}")
        
        if results["embedding"] and not results["embedding"]["success"]:
            error = results["embedding"].get("error", "Unknown error")
            logger.error(f"Embedding error: {error}")


if __name__ == "__main__":
    main()
