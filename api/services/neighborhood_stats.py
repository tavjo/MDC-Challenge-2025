"""
api/services/neighborhood_stats.py
-----------------
Network-based clustering using k-NN similarity graphs and Leiden algorithm.

This module provides functions for:
1. Computing neighborhood embedding statistics from ChromaDB
2. Updating dataset records and engineered features in DuckDB
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import numpy as np
# import pandas as pd
# import igraph as ig
import chromadb
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.helpers import initialize_logging, timer_wrap
from src.models import EngineeredFeatures
from api.utils.duckdb_utils import DuckDBHelper
import threading, yaml

# Initialize logging
filename = os.path.basename(__file__)
logger = initialize_logging(filename)

# Default database path
DEFAULT_DUCKDB_PATH = "artifacts/mdc_challenge.db"
DEFAULT_CHROMA_CONFIG = "configs/chunking.yaml"
DEFAULT_COLLECTION_NAME = "dataset-aggregates-train"

# Thread-local retriever to avoid multiple ChromaDB initializations per thread
_thread_local = threading.local()
# Shared cache of Chroma collections
_chroma_cache: dict[str, chromadb.Collection] = {}
# Lock to guard shared Chroma collection cache
_chroma_cache_lock = threading.Lock()

def get_collection(cfg_path, collection_name):
    if not hasattr(_thread_local, "chroma_collection"):
        cfg = _load_cfg(cfg_path)
        _thread_local.chroma_collection = _get_chroma_collection(cfg, collection_name)
    return _thread_local.chroma_collection

@timer_wrap
def _load_cfg(cfg_path: os.PathLike | None = None) -> Dict[str, Any]:
    """Load YAML config matching existing semantic_chunking.py pattern."""
    default_path = Path("configs/chunking.yaml")
    path = Path(cfg_path or default_path).expanduser()
    
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
        
    with path.open("r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh) or {}
    
    if "vector_store" not in cfg:
        raise KeyError("YAML must define a 'vector_store' section")
    return cfg


@timer_wrap
def _get_chroma_collection(cfg: Dict[str, Any], collection_name: str):
    """Return a *shared* Collection instance (thread-safe)"""
    # Guard shared cache with a lock
    with _chroma_cache_lock:
        if collection_name in _chroma_cache:
            return _chroma_cache[collection_name]

        chroma_path = (Path(__file__).resolve().parents[2] / cfg["vector_store"].get("path", "./local_chroma")).expanduser()
        chroma_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(chroma_path))
        _chroma_cache[collection_name] = client.get_or_create_collection(collection_name)
        return _chroma_cache[collection_name]

@timer_wrap
def compute_neighborhood_embedding_stats(
    dataset_embedding: np.ndarray,
    collection,
    # collection_name: str,
    k: int = 3
) -> Dict[str, float]:
    """
    Find k-nearest neighbors to dataset embedding and compute:
    - mean, max, variance of neighbor similarities
    - mean, max, variance of neighbor embedding norms
    
    Args:
        dataset_embedding: The embedding vector for the dataset
        chroma_client: ChromaDB client instance
        collection_name: Name of the ChromaDB collection
        k: Number of nearest neighbors to find
        
    Returns:
        Dictionary with neighborhood statistics
    """
    try:
        # collection = chroma_client.get_collection(collection_name)
        
        # Query for k nearest neighbors
        results = collection.query(
            query_embeddings=[dataset_embedding.tolist()],
            n_results=k,
            include=["embeddings", "distances"]
        )
        
        if not results["embeddings"] or not results["embeddings"][0]:
            logger.warning("No neighbors found for dataset embedding")
            return {}
        
        neighbor_embeddings = np.array(results["embeddings"][0])
        distances = np.array(results["distances"][0])
        
        # Convert distances to similarities (assuming cosine distance)
        similarities = 1 - distances
        
        # Compute neighbor embedding norms
        neighbor_norms = np.linalg.norm(neighbor_embeddings, axis=1)
        
        stats = {
            "neighbor_similarity_mean": float(np.mean(similarities)),
            "neighbor_similarity_max": float(np.max(similarities)),
            "neighbor_similarity_var": float(np.var(similarities)),
            "neighbor_norm_mean": float(np.mean(neighbor_norms)),
            "neighbor_norm_max": float(np.max(neighbor_norms)),
            "neighbor_norm_var": float(np.var(neighbor_norms))
        }
        
        logger.info(f"Computed neighborhood stats for {k} neighbors")
        return stats
        
    except Exception as e:
        logger.error(f"Error computing neighborhood embedding stats: {e}")
        return {}

@timer_wrap
def add_neighborhood_features_to_duckdb(
    dataset_neighborhood_stats: Dict[str, Dict[str, float]],
    db_helper: DuckDBHelper
) -> bool:
    """
    Add new feature values from neighborhood embeddings stats to 
    engineered_feature_values table in DuckDB.
    
    Args:
        dataset_neighborhood_stats: Dict mapping dataset_id to neighborhood stats
        db_helper: DuckDBHelper instance
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Adding neighborhood features for {len(dataset_neighborhood_stats)} datasets")
        
        # Get all datasets to map dataset_id to document_id
        datasets = db_helper.get_all_datasets()
        dataset_to_doc = {ds.dataset_id: ds.document_id for ds in datasets}
        
        # Create EngineeredFeatures objects
        features_list = []
        
        for dataset_id, stats in dataset_neighborhood_stats.items():
            if dataset_id in dataset_to_doc:
                # Create EngineeredFeatures object with neighborhood stats
                features = EngineeredFeatures(
                    dataset_id=dataset_id,
                    document_id=dataset_to_doc[dataset_id],
                    **stats  # Unpack all the neighborhood statistics
                )
                features_list.append(features)
        
        if features_list:
            success = db_helper.upsert_engineered_features_batch(features_list)
            if success:
                logger.info(f"Successfully added neighborhood features for {len(features_list)} datasets")
                return True
            else:
                logger.error("Failed to add neighborhood features")
                return False
        else:
            logger.warning("No datasets found to add neighborhood features")
            return True
            
    except Exception as e:
        logger.error(f"Error adding neighborhood features: {e}")
        return False


@timer_wrap
def export_neighborhood_stats_report(
    dataset_neighborhood_stats: Dict[str, Dict[str, float]],
    output_dir: str = "reports/neighborhood_stats"
) -> bool:
    """
    Export summary report in JSON to reports/clustering directory.
    
    Args:
        dataset_neighborhood_stats: Neighborhood statistics for each dataset
        output_dir: Output directory for the report
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for unique filename
        report_file = output_path / f"neighborhood_stats_report.json"
        
        # Build report
        report = {
            "timestamp": datetime.now().isoformat(),
            "neighborhood_stats_summary": {
                "datasets_with_stats": len(dataset_neighborhood_stats),
                "avg_neighbor_similarity_mean": np.mean([
                    stats.get("neighbor_similarity_mean", 0) 
                    for stats in dataset_neighborhood_stats.values()
                ]) if dataset_neighborhood_stats else 0
            }
        }
        
        # Write report to file
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Clustering report exported to {report_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error exporting clustering report: {e}")
        return False


@timer_wrap
def run_neighborhood_stats_pipeline(
        db_path: str = DEFAULT_DUCKDB_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        k: int = 3,
        output_dir: str = "reports/neighborhood_stats",
        cfg_path: str = DEFAULT_CHROMA_CONFIG,
        max_workers: int = 4
) -> bool:
    """
    Run the neighborhood stats pipeline.
    
    Args:
        db_path: Path to DuckDB database
        collection_name: ChromaDB collection name
        k: Number of nearest neighbors to find
        output_dir: Output directory for reports
        cfg_path: Path to configuration file
        max_workers: Maximum number of worker threads for parallel processing
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Running neighborhood stats pipeline")
    logger.info("Initializing DuckDB helper")
    db_helper = DuckDBHelper(db_path)
    # get all dataset embeddings from ChromaDB
    logger.info("Initializing ChromaDB collection")
    collection = get_collection(cfg_path, collection_name)
    logger.info("Retrieving dataset embeddings from ChromaDB")
    dat = collection.get(include=["embeddings"])
    logger.info(f"Found {len(dat['ids'])} dataset embeddings")
    id_embeddings = {id: embeddings for id, embeddings in zip(dat["ids"], dat["embeddings"])}
    # compute neighborhood stats for each dataset using multi-threading
    logger.info("Computing neighborhood stats for each dataset")
    workers = min(max_workers, len(id_embeddings)) if id_embeddings else 1
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(compute_neighborhood_embedding_stats, np.array(embedding), collection, k): dataset_id
            for dataset_id, embedding in id_embeddings.items()
        }
        neighborhood_stats = {futures[fut]: fut.result() for fut in as_completed(futures)}
    logger.info(f"Found {len(neighborhood_stats)} neighborhood stats")
    # add neighborhood stats to DuckDB
    logger.info("Adding neighborhood stats to DuckDB")
    add_neighborhood_features_to_duckdb(neighborhood_stats, db_helper)
    # export neighborhood stats report
    logger.info("Exporting neighborhood stats report")
    return export_neighborhood_stats_report(neighborhood_stats, output_dir)


@timer_wrap
def main():
    """
    Main function to demonstrate the clustering pipeline.
    This is primarily for testing purposes.
    """
    logger.info("Starting neighborhood stats pipeline")
    res = run_neighborhood_stats_pipeline()
    if res:
        logger.info("Neighborhood stats pipeline completed successfully")
    else:
        logger.error("Neighborhood stats pipeline failed")



if __name__ == "__main__":
    main()