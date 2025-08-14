#!/usr/bin/env python3
"""
Clustering Pipeline for MDC-Challenge-2025
Loads embeddings from ChromaDB and runs clustering pipeline
"""

import json
import os
import sys
from typing import Optional, Dict
# from pathlib import Path
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import LoadChromaDataPayload, LoadChromaDataResult
from src.helpers import initialize_logging, timer_wrap
from src.clustering import run_clustering_pipeline
import requests

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "load_embeddings": "/load_embeddings"
}

# Default paths and configurations
DEFAULT_DUCKDB_PATH = os.path.join(project_root, "artifacts", "mdc_challenge.db")
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"

# Clustering parameters
DEFAULT_K_NEIGHBORS = 3
DEFAULT_SIMILARITY_THRESHOLD = None
DEFAULT_THRESHOLD_METHOD = "degree_target"
DEFAULT_RESOLUTION = 1
DEFAULT_MIN_CLUSTER_SIZE = 3
DEFAULT_MAX_CLUSTER_SIZE = 9999
DEFAULT_SPLIT_FACTOR = 1.3
DEFAULT_RANDOM_SEED = 42
DEFAULT_TARGET_N = 60
DEFAULT_TOL = 2


class ClusteringPipeline:
    def __init__(self,
                 base_api_url: str = API_ENDPOINTS["base_api_url"],
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 cfg_path: Optional[str] = None,
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 k_neighbors: int = DEFAULT_K_NEIGHBORS,
                 similarity_threshold: Optional[float] = DEFAULT_SIMILARITY_THRESHOLD,
                 threshold_method: str = DEFAULT_THRESHOLD_METHOD,
                 resolution: float = DEFAULT_RESOLUTION,
                 min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
                 max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
                 split_factor: float = DEFAULT_SPLIT_FACTOR,
                 random_seed: int = DEFAULT_RANDOM_SEED,
                 target_n: int = DEFAULT_TARGET_N,
                 tol: int = DEFAULT_TOL):
        self.base_api_url = base_api_url
        self.collection_name = collection_name
        self.cfg_path = cfg_path
        self.db_path = db_path
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.threshold_method = threshold_method
        self.resolution = resolution
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.split_factor = split_factor
        self.random_seed = random_seed
        self.target_n = target_n
        self.tol = tol
        self.endpoints = API_ENDPOINTS

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_api_url}{self.endpoints[endpoint]}"

    def _construct_load_embeddings_payload(self) -> LoadChromaDataPayload:
        return LoadChromaDataPayload(
            collection_name=self.collection_name,
            cfg_path=self.cfg_path,
            include=["embeddings"]  # Only need embeddings for clustering
        )

    def load_embeddings_from_api(self) -> Optional[np.ndarray]:
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
                    dataset_embeddings = np.array(dat["embeddings"])
                    logger.info(f"✅ Embeddings shape: {dataset_embeddings.shape}")
                    
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

    def run_clustering(self, dataset_embeddings: np.ndarray) -> bool:
        """Run the clustering pipeline on the loaded embeddings."""
        logger.info("🔄 Starting clustering pipeline...")
        logger.info(f"Clustering {len(dataset_embeddings)} datasets with parameters:")
        logger.info(f"  - k_neighbors: {self.k_neighbors}")
        logger.info(f"  - similarity_threshold: {self.similarity_threshold}")
        logger.info(f"  - threshold_method: {self.threshold_method}")
        logger.info(f"  - resolution: {self.resolution}")
        logger.info(f"  - min_cluster_size: {self.min_cluster_size}")
        logger.info(f"  - random_seed: {self.random_seed}")
        
        try:
            success = run_clustering_pipeline(
                dataset_embeddings=dataset_embeddings,
                k_neighbors=self.k_neighbors,
                similarity_threshold=self.similarity_threshold,
                threshold_method=self.threshold_method,
                # resolution=self.resolution,
                min_cluster_size=self.min_cluster_size,
                max_cluster_size=self.max_cluster_size,
                split_factor=self.split_factor,
                random_seed=self.random_seed,
                target_n=self.target_n,
                tol=self.tol,
                # db_path=self.db_path,
                output_dir="reports/clustering"
            )
            
            if success:
                logger.info("✅ Clustering pipeline completed successfully!")
            else:
                logger.error("❌ Clustering pipeline failed!")
                
            return success
            
        except Exception as e:
            logger.error(f"Clustering pipeline failed with error: {str(e)}")
            return False

    def run_pipeline(self) -> dict:
        """Run the complete pipeline: load embeddings + clustering."""
        results = {
            "load_embeddings": None,
            "clustering": None,
            "overall_success": False
        }
        
        # Step 1: Load embeddings from ChromaDB
        logger.info("🚀 Starting clustering pipeline...")
        logger.info("📥 Loading embeddings from ChromaDB...")
        
        dataset_embeddings = self.load_embeddings_from_api()
        results["load_embeddings"] = {"success": dataset_embeddings is not None}
        
        if dataset_embeddings is None:
            logger.error("❌ Failed to load embeddings, skipping clustering step")
            return results
        
        # # Prepare embeddings matrix
        # dataset_ids = list(dataset_embeddings.keys())
        # embedding_matrix = np.array([dataset_embeddings[dataset_id] for dataset_id in dataset_ids])
        
        logger.info(f"Embedding matrix shape: {dataset_embeddings.shape}")
        
        # Step 2: Run clustering pipeline
        logger.info("🔄 Running clustering on loaded embeddings...")
        clustering_success = self.run_clustering(dataset_embeddings)
        results["clustering"] = {"success": clustering_success}
        
        # Overall success if both steps succeeded
        results["overall_success"] = (dataset_embeddings is not None) and clustering_success
        
        if results["overall_success"]:
            logger.info("✅ Complete clustering pipeline finished successfully!")
        else:
            logger.error("❌ Pipeline completed with errors")
        
        return results


@timer_wrap
def main():
    """Main function to run the clustering pipeline."""
    
    # Initialize the pipeline with default parameters
    logger.info("Initializing clustering pipeline...")
    
    pipeline = ClusteringPipeline(
        collection_name=DEFAULT_COLLECTION_NAME,
        cfg_path=DEFAULT_CHROMA_CONFIG,
        db_path=DEFAULT_DUCKDB_PATH,
        k_neighbors=DEFAULT_K_NEIGHBORS,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        threshold_method=DEFAULT_THRESHOLD_METHOD,
        resolution=DEFAULT_RESOLUTION,
        min_cluster_size=DEFAULT_MIN_CLUSTER_SIZE,
        max_cluster_size=DEFAULT_MAX_CLUSTER_SIZE,
        split_factor=DEFAULT_SPLIT_FACTOR,
        random_seed=DEFAULT_RANDOM_SEED
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Prepare output directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "clustering")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "clustering_pipeline_results.json")
    
    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary logging
    if results["overall_success"]:
        logger.info("🎉 Clustering pipeline completed successfully!")
    else:
        logger.error("💥 Pipeline failed - check the logs and results file for details")
        
        # Log specific failures
        if results["load_embeddings"] and not results["load_embeddings"]["success"]:
            logger.error("Embedding loading failed")
        
        if results["clustering"] and not results["clustering"]["success"]:
            logger.error("Clustering computation failed")


if __name__ == "__main__":
    main()