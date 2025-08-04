#!/usr/bin/env python3
"""
Dimensionality Reduction Pipeline Runner for MDC-Challenge-2025
Loads embeddings from ChromaDB and runs dimensionality reduction pipeline
"""

import json
import os
import sys
from typing import Optional, Dict
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import LoadChromaDataPayload, LoadChromaDataResult
from src.helpers import initialize_logging, timer_wrap
from src.dimensionality_reduction import Reducer
import requests

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "load_embeddings": "/load_embeddings"
}

# Default paths and configurations
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"

# UMAP parameters
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_N_COMPONENTS = 2
DEFAULT_RANDOM_SEED = 42


class DimensionalityReductionPipeline:
    def __init__(self,
                 base_api_url: str = API_ENDPOINTS["base_api_url"],
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 cfg_path: Optional[str] = None,
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 n_neighbors: int = DEFAULT_N_NEIGHBORS,
                 min_dist: float = DEFAULT_MIN_DIST,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 random_seed: int = DEFAULT_RANDOM_SEED):
        self.base_api_url = base_api_url
        self.collection_name = collection_name
        self.cfg_path = cfg_path
        self.db_path = db_path
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.random_seed = random_seed
        self.endpoints = API_ENDPOINTS

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_api_url}{self.endpoints[endpoint]}"

    def _construct_load_embeddings_payload(self) -> LoadChromaDataPayload:
        return LoadChromaDataPayload(
            collection_name=self.collection_name,
            cfg_path=self.cfg_path,
            include=["embeddings"]  # Only need embeddings for dimensionality reduction
        )

    def load_embeddings_from_api(self) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from ChromaDB via API call."""
        payload = self._construct_load_embeddings_payload()
        full_url = self._get_api_url("load_embeddings")
        
        logger.info(f"Loading embeddings with URL: {full_url}")
        logger.info(f"Using collection: {self.collection_name}")
        
        try:
            response = requests.post(full_url, json=payload.model_dump(exclude_none=True))
            if response.status_code == 200:
                result = LoadChromaDataResult.model_validate(response.json())
                if result.success and result.results:
                    logger.info(f"✅ Successfully loaded {len(result.results)} embeddings")
                    
                    # Convert embeddings to numpy arrays if they aren't already
                    dat = result.results
                    dataset_embeddings = {id: embeddings for id, embeddings in zip(dat["ids"], dat["embeddings"])}
                    
                    return dataset_embeddings
                else:
                    error_msg = result.error or "No embeddings returned"
                    logger.error(f"❌ Failed to load embeddings: {error_msg}")
                    return None
            else:
                logger.error(f"Load embeddings failed with status {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Load embeddings request failed: {str(e)}")
            return None

    def run_dimensionality_reduction(self, dataset_embeddings: Dict[str, np.ndarray]) -> bool:
        """Run the dimensionality reduction pipeline on the loaded embeddings."""
        logger.info("🔄 Starting dimensionality reduction pipeline...")
        logger.info(f"Processing {len(dataset_embeddings)} datasets with parameters:")
        logger.info(f"  - n_neighbors: {self.n_neighbors}")
        logger.info(f"  - min_dist: {self.min_dist}")
        logger.info(f"  - n_components: {self.n_components}")
        logger.info(f"  - random_seed: {self.random_seed}")
        
        try:
            # Initialize the reducer
            reducer = Reducer(
                base_api_url=self.base_api_url,
                collection_name=self.collection_name,
                cfg_path=self.cfg_path,
                db_path=self.db_path,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=self.n_components,
                random_seed=self.random_seed
            )
            
            # Run the complete pipeline
            results = reducer.run_pipeline()
            
            if results["overall_success"]:
                logger.info("✅ Dimensionality reduction pipeline completed successfully!")
            else:
                logger.error("❌ Dimensionality reduction pipeline failed!")
                
            return results["overall_success"]
            
        except Exception as e:
            logger.error(f"Dimensionality reduction pipeline failed with error: {str(e)}")
            return False

    def run_pipeline(self) -> dict:
        """Run the complete pipeline: load embeddings + dimensionality reduction."""
        results = {
            "load_embeddings": None,
            "dimensionality_reduction": None,
            "overall_success": False
        }
        
        # Step 1: Load embeddings from ChromaDB
        logger.info("🚀 Starting dimensionality reduction pipeline...")
        logger.info("📥 Loading embeddings from ChromaDB...")
        
        dataset_embeddings = self.load_embeddings_from_api()
        results["load_embeddings"] = {"success": dataset_embeddings is not None}
        
        if dataset_embeddings is None:
            logger.error("❌ Failed to load embeddings, skipping dimensionality reduction step")
            return results
        
        # Step 2: Run dimensionality reduction pipeline
        logger.info("🔄 Running dimensionality reduction on loaded embeddings...")
        dim_reduction_success = self.run_dimensionality_reduction(dataset_embeddings)
        results["dimensionality_reduction"] = {"success": dim_reduction_success}
        
        # Overall success if both steps succeeded
        results["overall_success"] = (dataset_embeddings is not None) and dim_reduction_success
        
        if results["overall_success"]:
            logger.info("✅ Complete dimensionality reduction pipeline finished successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        return results


@timer_wrap
def main():
    """Main function to run the dimensionality reduction pipeline."""
    
    # Initialize the pipeline with default parameters
    logger.info("Initializing dimensionality reduction pipeline...")
    
    pipeline = DimensionalityReductionPipeline(
        collection_name=DEFAULT_COLLECTION_NAME,
        cfg_path=DEFAULT_CHROMA_CONFIG,
        db_path=DEFAULT_DUCKDB_PATH,
        n_neighbors=DEFAULT_N_NEIGHBORS,
        min_dist=DEFAULT_MIN_DIST,
        n_components=DEFAULT_N_COMPONENTS,
        random_seed=DEFAULT_RANDOM_SEED
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Prepare output directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "dimensionality_reduction")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "dimensionality_reduction_pipeline_results.json")
    
    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary logging
    if results["overall_success"]:
        logger.info("🎉 Dimensionality reduction pipeline completed successfully!")
    else:
        logger.error("💥 Pipeline failed - check the logs and results file for details")
        
        # Log specific failures
        if results["load_embeddings"] and not results["load_embeddings"]["success"]:
            logger.error("Embedding loading failed")
        
        if results["dimensionality_reduction"] and not results["dimensionality_reduction"]["success"]:
            logger.error("Dimensionality reduction computation failed")


if __name__ == "__main__":
    main()