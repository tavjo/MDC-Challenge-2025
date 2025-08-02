#!/usr/bin/env python3
"""
Clustering Pipeline Script for MDC-Challenge-2025
Runs the complete network-based clustering pipeline for dataset embeddings
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.clustering import run_full_clustering_pipeline
from src.helpers import initialize_logging

filename = os.path.basename(__file__)
logger = initialize_logging(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Run network-based clustering pipeline for dataset embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--collection-name",
        default="dataset-aggregates-train",
        help="ChromaDB collection name for dataset embeddings"
    )
    
    parser.add_argument(
        "--db-path",
        default="artifacts/mdc_challenge.db",
        help="Path to DuckDB database"
    )
    
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=30,
        help="Number of neighbors for k-NN graph construction"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold for edge creation (auto-determined if not provided)"
    )
    
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden clustering resolution parameter"
    )
    
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="Minimum cluster size"
    )
    
    parser.add_argument(
        "--output-dir",
        default="reports/clustering",
        help="Output directory for clustering reports"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting clustering pipeline with parameters:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    try:
        # Run the full clustering pipeline
        result = run_full_clustering_pipeline(
            collection_name=args.collection_name,
            db_path=args.db_path,
            k_neighbors=args.k_neighbors,
            similarity_threshold=args.similarity_threshold,
            resolution=args.resolution,
            min_cluster_size=args.min_cluster_size,
            output_dir=args.output_dir
        )
        
        if result["status"] == "success":
            logger.info("Clustering pipeline completed successfully!")
            logger.info(f"Processed {result['datasets_processed']} datasets")
            logger.info(f"Created {result['clusters_created']} clusters")
            logger.info(f"Execution time: {result['execution_time_seconds']:.2f} seconds")
            logger.info(f"Report saved to: {result['report_file']}")
        else:
            logger.error(f"Clustering pipeline failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()