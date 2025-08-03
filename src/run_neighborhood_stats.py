#!/usr/bin/env python3
"""
Neighborhood Stats Pipeline for MDC-Challenge-2025
Runs neighborhood statistics computation for dataset embeddings
"""

import json
import os
import sys
from typing import Optional
# from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.models import NeighborhoodStatsPayload
from src.helpers import initialize_logging, timer_wrap
import requests

filename = os.path.basename(__file__)
logger = initialize_logging(filename)

API_ENDPOINTS = {
    "base_api_url": "http://localhost:8000",
    "run_neighborhood_stats_pipeline": "/run_neighborhood_stats_pipeline"
}

# Default paths and configurations
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"
DEFAULT_K = 3
DEFAULT_MAX_WORKERS = 8


class NeighborhoodStatsPipeline:
    def __init__(self,
                 base_api_url: str = API_ENDPOINTS["base_api_url"],
                 db_path: str = DEFAULT_DUCKDB_PATH,
                 collection_name: str = DEFAULT_COLLECTION_NAME,
                 k: int = DEFAULT_K,
                 cfg_path: Optional[str] = None,
                 max_workers: int = DEFAULT_MAX_WORKERS):
        self.base_api_url = base_api_url
        self.db_path = db_path
        self.collection_name = collection_name
        self.k = k
        self.cfg_path = cfg_path
        self.max_workers = max_workers
        self.endpoints = API_ENDPOINTS

    def _get_api_url(self, endpoint: str) -> str:
        return f"{self.base_api_url}{self.endpoints[endpoint]}"

    def _construct_neighborhood_stats_payload(self) -> NeighborhoodStatsPayload:
        return NeighborhoodStatsPayload(
            db_path=self.db_path,
            collection_name=self.collection_name,
            k=self.k,
            cfg_path=self.cfg_path,
            max_workers=self.max_workers
        )

    def run_neighborhood_stats(self) -> bool:
        """Call the /run_neighborhood_stats_pipeline endpoint."""
        payload = self._construct_neighborhood_stats_payload()
        full_url = self._get_api_url("run_neighborhood_stats_pipeline")
        
        logger.info(f"Running neighborhood stats pipeline with URL: {full_url}")
        logger.info(f"Using collection: {self.collection_name}")
        logger.info(f"Finding {self.k} nearest neighbors with {self.max_workers} workers")
        
        try:
            response = requests.post(full_url, json=payload.model_dump(exclude_none=True))
            if response.status_code == 200:
                result = response.json()
                if result:
                    logger.info("✅ Neighborhood stats pipeline completed successfully!")
                    return True
                else:
                    logger.error("❌ Neighborhood stats pipeline returned False")
                    return False
            else:
                logger.error(f"Neighborhood stats pipeline failed with status {response.status_code}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Neighborhood stats pipeline request failed: {str(e)}")
            return False

    def run_pipeline(self) -> dict:
        """Run the complete neighborhood stats pipeline."""
        results = {
            "neighborhood_stats": None,
            "overall_success": False
        }
        
        # Run neighborhood stats pipeline
        logger.info("🚀 Starting neighborhood stats pipeline...")
        success = self.run_neighborhood_stats()
        results["neighborhood_stats"] = {"success": success}
        results["overall_success"] = success
        
        if results["overall_success"]:
            logger.info("✅ Neighborhood stats pipeline finished successfully!")
        else:
            logger.error("❌ Neighborhood stats pipeline completed with errors")
        
        return results


@timer_wrap
def main():
    """Main function to run the neighborhood stats pipeline."""
    
    # Initialize the pipeline with default parameters
    logger.info("Initializing neighborhood stats pipeline...")
    
    pipeline = NeighborhoodStatsPipeline(
        db_path=DEFAULT_DUCKDB_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
        k=DEFAULT_K,
        cfg_path=DEFAULT_CHROMA_CONFIG,
        max_workers=DEFAULT_MAX_WORKERS
    )
    
    # Run the complete pipeline
    results = pipeline.run_pipeline()
    
    # Prepare output directory
    base_dir = os.getcwd()
    output_dir = os.path.join(base_dir, "reports", "neighborhood_stats")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "neighborhood_stats_results.json")
    
    # Save results
    logger.info(f"Saving results to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary logging
    if results["overall_success"]:
        logger.info("🎉 Neighborhood stats pipeline completed successfully!")
    else:
        logger.error("💥 Pipeline failed - check the logs and results file for details")
        
        # Log specific failures
        if results["neighborhood_stats"] and not results["neighborhood_stats"]["success"]:
            logger.error("Neighborhood stats computation failed")


if __name__ == "__main__":
    main()